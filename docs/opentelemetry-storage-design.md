# Milvus Storage OpenTelemetry 设计

## 1. 状态、目标与基线

状态：设计提案，接口、埋点和传播适配尚未实现或完成运行验证。本文只描述 **milvus-storage repo** 的改造；宿主侧见 [Milvus 设计](opentelemetry-milvus-design.md)。本文路径相对 Storage repo。

Storage 基线为 `43b7793406136208928ad20dc1596b510c7b629a`；对照 Milvus `5963da22e8156ad275bb6d1431aeeaa0cfb2c70a`，其构建仍固定 Storage `3ae6ac4`。实施需重新核对锁定依赖，不能假设独立 checkout 已部署。

目标：

1. 将 Storage 的 metadata、任务读取、I/O、格式处理、结果组装关联到宿主 trace。
2. 保持 `Reader`、`ChunkReader`、格式与文件系统既有业务入口参数尽量不变，通过新增 tracing 能力接入。
3. 支持普通线程、Folly Future/fiber，以及 Arrow、CRT、Tokio 边界的显式传播。
4. 不依赖 Milvus 的 `OpContext`、配置中心或 tracer 包，独立宿主和测试可直接使用。

不改变数据格式、缓存/调度/取消语义，不承诺无需格式引擎 hook 即可测量精确 codec CPU 时间，不把远程 Talon 服务内部 spans 纳入本次交付。

## 2. 当前架构

已有 OTel C++ 构建依赖，但没有 Storage 请求级 tracing 链路。详细异步执行语义见 [async-read-design.md](async-read-design.md)。

| 路径 | 当前执行形态 | tracing 需处理的边界 |
|---|---|---|
| 同步 chunk/take/full scan | 部分路径使用 ThreadPoolHolder；full scan 保持同步组织 | 线程池提交、逐次读取 |
| async chunk/take | 规划任务、组合 SemiFuture、collectAll | 入口捕获、惰性执行、fan-out/fan-in |
| Parquet | FollyArrowExecutor 驱动 Arrow generator；部分 open 是延迟执行的阻塞操作 | Arrow 任务、完成回调、metadata I/O |
| Vortex | C++ Promise 桥接，共享 Tokio runtime 执行 open/scan/collect | C++/Rust、spawn、poll、blocking pool |
| CRT I/O | NonBlockingRandomAccessFile、ReadAtAsyncInto、回调完成 | 提交到完成的请求状态 |
| 其他格式 | 可能使用同步 ready-future fallback | 不能将返回 Future 等同于非阻塞 |

Storage 当前不选择调用方 Folly executor；本改造也不增加调度 executor。Vortex 内部新建任务、Arrow I/O pool 等不会因为外层存在一个 context 就自动继承。

## 3. 公共边界契约（v1）

本节是两份设计的公共契约定义。拟新增 `cpp/include/milvus-storage/tracing.h`，实现放在 Storage 动态库内部。以下是接口草图，错误处理、导出宏等以实现评审为准。

```cpp
namespace milvus_storage::tracing {

struct TraceParent {
    std::array<uint8_t, 16> trace_id{};
    std::array<uint8_t, 8> span_id{};
    uint8_t trace_flags = 0;
    std::string tracestate;
    bool is_remote = false;
};

using ProviderPtr = opentelemetry::nostd::shared_ptr<
    opentelemetry::trace::TracerProvider>;

void SetTracerProvider(ProviderPtr provider);

class TraceScope;  // 不可复制/移动；通过 C++17+ 返回值消除构造
[[nodiscard]] TraceScope AttachParent(const TraceParent& parent);

}
```

### 3.1 Provider

- Storage 保存注入的 provider，取得 instrumentation scope 为 `milvus-storage`、version 为自身构建版本的 Tracer；不调用宿主全局 `SetTracerProvider`，不自行启动 exporter。
- 未注入或传空时默认不记录/导出；调用方应用负责初始化 SDK。测试宿主可注入 in-memory exporter 对应 provider。
- 发布新 provider 快照须并发安全；每次操作固定自己的 provider/tracer，内部子操作使用同一配置快照。
- provider 更新不迁移已有 spans。宿主负责在相关在途工作完成后 flush/shutdown 旧 provider；Storage 不主动关闭共享实例。
- C++ provider 类型要求 OTel ABI/version/编译选项匹配；不作为稳定 C ABI，也不跨 Rust 传递。显式注入避免猜测不同动态库中的 global singleton 是否一致。

### 3.2 AttachParent 的精确定义

`AttachParent` **不创建 span，不调用父 span End，也不绑定 Reader**。它为当前执行上下文设置 Storage parent，并在析构时恢复以前的上下文。

实现使用 Folly `RequestContext` 的 Storage 专属 key：保存当前上下文，建立浅拷贝，用不可变、拥有数据的 parent 覆盖该 key，退出时恢复。不得直接改写可能被多个任务继承的共享可变 RequestData。

嵌套作用域按栈顺序恢复；guard 不可复制、移动或转交其他执行流。Folly 管理的 fiber 挂起时 guard 可以留在 fiber 栈上，由运行时保存/恢复 RequestContext。对不支持该机制的线程/协程，必须显式传递快照。

无效/空 parent 建立“无父上下文”边界，遮蔽外层 parent，默认不自行创建根 span；不能悄悄沿用旧 TLS 或宿主残留 parent。未采样的有效 parent 仍须传播，不能因为 `IsRecording()==false` 就改成新根 trace；实际记录由 provider 采样策略决定。

本接口只更新 **Storage 当前上下文**，不把普通 OTel `WithActiveSpan` guard 留在 fiber 栈上。所有 Storage 埋点从该上下文读取 parent，显式设置 `StartSpanOptions.parent`。第三方仅依赖 OTel TLS `GetCurrentSpan()` 的埋点不自动覆盖。

### 3.3 入口捕获与使用示例

```cpp
{
    auto scope = tracing::AttachParent(parent);
    reader->get_chunks(indices);  // 每次调用创建独立子操作
    reader->take(rows);
}

auto future = [&] {
    auto scope = tracing::AttachParent(parent);
    return reader->get_chunks_async(indices);
}();  // scope 已结束，future 持有入口时捕获的上下文
ConsumeOnExecutor(std::move(future), executor);  // 示意
```

入口必须在返回前捕获 parent，即使真正工作延迟到 future 消费时。不能根据消费线程重新选择 parent。惰性工作在实际开始执行时创建工作 span；如需统计调用到执行的延迟，另行记录 start delay。未消费且无工作发生的 future 不伪造 I/O span。

逐次读取/stream 的上下文属于读取操作，不属于共享 Reader 或文件。请求专属 stream wrapper 可以由宿主保留 parent，在每次调用时重新激活；共享 Reader 永远不记住第一个调用者。

## 4. 模块所有权与交互

| 模块 | 新职责 | 不承担的职责 |
|---|---|---|
| 公共 tracing API | provider 注入、parent 激活 | Milvus 配置、OpContext |
| 内部 tracing runtime | 捕获/恢复不可变 context，完成状态，span helper | 改变调度或取消行为 |
| Reader / 任务层 | 操作入口、子任务、fan-in spans | 网络重试细节 |
| 格式层 | metadata、格式读取与结果物化 | 未观测到的引擎内部 CPU 分解 |
| 文件系统层 | 逻辑读取及请求完成观测 | 将缓存读取认作必然发生网络请求 |
| Rust bridge | Tokio/FFI 上下文传播，调用 C++ 埋点能力 | 独立初始化第二套 SDK/exporter |
| Milvus adapter | 将 parent 送至 Storage 入口 | Storage 内部异步实现 |

拟新增私有 `OperationTrace`，持有 parent 快照、provider/tracer、当前 span、必要的观测计数和完成状态。任务状态按不可变上下文派生；不要把所有并行 worker 的 current span 放进同一个可变槽位。

`Finish(status)` 必须至多执行一次；只为存在多条竞争完成路径的操作增加相应同步。span 和 context 释放不应持有整个 Reader 或数据缓冲区形成引用环。不可为了等待 span 结束增加业务阻塞。

## 5. 异步与 fiber 传播

### 5.1 为什么使用 RequestContext

普通 TLS 在 Fiber A 挂起时不会自动清除；Fiber B 可能读到 A，A 恢复后又可能读到 B。RAII guard 留在各自 fiber 栈上不能修复线程局部状态。

本地 Folly 源码的 `FiberManagerInternal-inl.h` 在运行 fiber 时恢复 `rcontext_`，在相应挂起/切换路径保存 RequestContext；`Request.h` 提供浅拷贝 scope。实现必须针对实际锁定版本验证该行为。

```text
Fiber A：AttachParent(A) → RequestContext A → 挂起
Fiber B：RequestContext B → Storage 读取 B
Fiber A：恢复 RequestContext A → Storage 读取 A
```

这只保证经过支持的调度路径；`runInMainContext`、自定义 awaiter、跨 runtime 和脱离请求的后台任务需逐个定义上下文边界。Folly coroutine `Task` 的传播能力也按锁定版本和实际 await/start 时机验证，不等同于所有 C++ coroutine 自动支持。

### 5.2 边界处理矩阵

| 边界 | 捕获位置 | 恢复位置与限制 |
|---|---|---|
| ThreadPoolHolder / 自定义 executor | 提交前捕获操作 context | 闭包执行内建立新 context scope |
| Folly continuation | 构建 continuation 时固定操作 context | continuation 执行时恢复；不依赖完成线程 TLS |
| Folly fiber | 入口 AttachParent 创建隔离 RequestContext | runtime 挂起/恢复切换；嵌套不可共享修改 |
| Arrow executor | 操作专属 executor/任务适配器保存 context | SpawnReal 包装任务；不能只读取 CRT 提交线程 TLS |
| Arrow/CRT callback | 提交请求时保存 context 和完成状态 | 回调内恢复；成功、失败、同步完成都处理 |
| C++ → Rust | FFI 提交时捕获 owning context handle | Rust 操作状态持有，不能只存在 C++ 完成 callback 中 |
| Tokio future | 构造/派生任务时复制 context | 每次 poll 前 attach，poll 返回时 detach |
| Tokio spawn / spawn_blocking | 每次提交显式捕获 | 子 future/闭包中恢复；不是只包装最外层 task |
| Rust → C++ 文件系统 | 对应读取操作 context | FFI 调用前恢复；异步 C++ 请求再次捕获 |

用 RequestContext 作为载体不意味着自动同步 OTel TLS。第一版全部 Storage-owned 埋点显式使用 parent；不全局替换 OTel RuntimeContextStorage，不用 RequestData 回调维持跨线程共享的 OTel attach token。

### 5.3 Rust 内部桥接

拟新增内部 context handle 操作：`capture`、`retain`、`release`、`attach`、`detach`；Rust 需要创建 spans 时，再增加受限的 `start_span`/`end_span` 桥接。

- handle 指向 C++ 拥有、不可变且可跨线程引用的上下文；Rust wrapper 的 Clone/Drop 对应 retain/release。
- attach 返回的 token 只在本次执行片段有效，同一次 poll/闭包内按栈顺序 detach；token 不能保留到下一次 poll，也不能跨线程转交。
- future wrapper 在进入 poll 时 attach，Ready/Pending/错误退出时 detach。panic 路径必须恢复；已有跨 FFI panic/异常隔离保持不变。
- context 随 open/scan 状态显式传递，覆盖 Vortex runtime adapter 创建的实际子任务。若引擎存在不可适配边界，该路径标记未覆盖，不能声称全部 I/O 已串联。
- C++ 反向 FFI 不能抛异常穿过边界；使用现有 `LoonFFIResult` 约定。内部桥接和公共符号导出范围要区分；需要公开的新符号按 binding 构建更新 exports map。

Rust `tracing-opentelemetry` 不是第一版依赖。后续若需要现有引擎内部 tracing spans，可独立评估；引入时仍须设置 parent 并使用按 poll 激活的 instrument wrapper。

### 5.4 完成、错误与取消

active scope 退出只恢复执行上下文，不结束操作。I/O span 从提交到实际完成，不能在返回 Future 时结束。同步阻塞 I/O 的 scope 覆盖调用期间，底层若能让出 fiber 仍使用 RequestContext 语义。

失败、短读、异常转换、提交失败均记录实际结果，不改变原错误分类。当前没有完整取消桥接：丢弃 Folly future 不代表 Tokio/CRT 工作已停止。操作继续时保留其完成状态；只有实际取消完成才标记取消。不得通过析构未消费 continuation 提前终止后台 I/O span。

## 6. 埋点与测量口径

| Span/信息 | 测量范围 | 落点 |
|---|---|---|
| `storage.read` | 一次实际读取工作的 wall time | `cpp/src/reader.cpp` |
| `storage.metadata.lookup` | cache 查询和命中信息 | `format/format_reader_cache.cpp` |
| `storage.metadata.wait` | follower 等待 leader | metadata cache |
| `storage.metadata.load` | leader 加载 metadata | cache loader / 格式 open |
| `storage.read_task` | 一个规划任务，包括 I/O 和格式处理 | column group / 格式层 |
| `storage.fs.read` | 逻辑文件读取，提交到完成 | 文件系统 wrapper / async 接口 |
| `storage.backend.request` | 实际 backend 请求与可见重试信息 | backend 请求层，分阶段覆盖 |
| `storage.decode_materialize` | 有明确执行边界的格式处理阶段 wall time | Parquet/Vortex 适配层 |
| `storage.assemble` | Storage 结果重排、复制、组装 | reader / column group 层 |

只有能隔离真实解码/物化边界时才命名 `decode_materialize`；如果包含 I/O 等待，应使用 `storage.format.read` 并标注范围。精确解压/解码 CPU 时间需要引擎内部 hook 或独立 profiling，不用总时间减去 I/O 总时长推导。

属性采用固定 span 名加属性：format、操作类型、row group 数、请求行数、列数、请求/返回字节、offset、backend、cache 状态、错误分类。对象路径仅按配置保留有限信息，不默认记录凭证、签名 URL、完整查询或返回数据。

文件系统已有 `Observable` 可复用观测边界，但必须同时覆盖 `ReadAt` 两种重载、`ReadAsync`、`ReadManyAsync`、`ReadAtAsyncInto`，并区分 metadata/head。wrapper 不能丢失 `NonBlockingRandomAccessFile` 能力、改变 dynamic_cast 判断或降级为阻塞实现。

同一逻辑操作与 backend 请求可形成父子 spans，但字节汇总只在定义的唯一层计数，不能将两层累加。Talon/其他缓存命中时，逻辑文件读取不一定对应对象存储网络请求。

并行 spans 时间重叠，不将 duration 之和当作端到端延迟。Milvus 的排队、内存许可等待和 Arrow → Milvus Chunk 转换由宿主文档定义。

## 7. 缓存、共享状态与采样

metadata singleflight 中 leader 记录一次实际 load/I/O；每个 follower 在自己的 trace 记录 wait，并在可获得 leader SpanContext 时附加 link。不能把同一次物理 I/O 复制为多个请求的子操作。leader context 只保存在在途加载状态，不进入长期缓存条目。

成功缓存的 metadata、Reader、文件句柄不保存请求 parent。后台预热从宿主传入独立 parent，后续请求只记录自身命中/等待。

默认只为有效父上下文创建操作链；follow-parent 采样由宿主 provider 决定。未记录时避免构造昂贵属性和逐请求日志。tracing 禁用后的成本及未采样传播的必要成本分别测量，不承诺未经基准验证的零开销。

提供细粒度 I/O spans 开关与每操作 span 数量预算；达到预算后记录聚合信息和丢弃数量，保留操作级结果。不得为每行/每页无限创建 spans。观测配置使用库显式配置/既有 property 机制，不从库内部读取环境变量；新增 property 按仓库注册与 C ABI 常量规范处理。

## 8. 改动文件规划

| 位置 | 改动 |
|---|---|
| 新增 `cpp/include/milvus-storage/tracing.h`、私有 tracing 实现 | 公共契约、context、operation span helper |
| `cpp/src/reader.cpp` | 入口捕获、fan-out/fan-in、assemble |
| `cpp/src/format/column_group_reader.cpp`、`column_group_lazy_reader.cpp` | 同步线程池与异步任务传播 |
| `cpp/src/format/format_reader_cache.cpp` | lookup/wait/load 与 singleflight link |
| `cpp/src/format/parquet/folly_arrow_executor.cpp`、`parquet_format_reader.cpp` | executor 与格式操作传播 |
| `cpp/include/milvus-storage/filesystem/observable.h`、`cpp/src/filesystem/` | 逻辑 I/O、backend 请求、非阻塞能力保留 |
| `cpp/src/ffi/filesystem_c.cpp` | callback 生命周期与错误路径 |
| `cpp/src/format/vortex/`、`cpp/src/format/bridge/rust/` | open/scan/collect/poll/spawn/反向 FFI |
| `cpp/CMakeLists.txt`、Conan/exports 配置 | 明确 OTel API 依赖与必要符号，测试 SDK 独立链接 |
| `cpp/test/` | 独立宿主、context 与传播回归测试 |

路径为实施落点，不代表每个文件必然修改。新增 FFI 若对外公开，要完整处理各 binding 的可见性；不要把 Rust 私有控制接口误当成公共业务 ABI。

## 9. 独立验证与交付阶段

| 测试 | 通过条件 |
|---|---|
| 人工 parent + in-memory exporter | trace ID 保留，parent span ID 正确，instrumentation scope 正确 |
| 嵌套 AttachParent / 空 parent | 正确恢复与遮蔽，不继承残留上下文 |
| 共享 Reader 并发 A/B 请求 | 各操作和内部任务不串 trace |
| 两 Folly fibers 同线程交错挂起 | 挂起前后及嵌套 scope 的 parent 正确，另一个 fiber 不受污染 |
| Future 在其他线程消费/不消费 | parent 在入口确定；未执行不产生虚假 I/O |
| inline callback / async callback / 提交失败 | 完成恰好一次，无泄漏、死锁或提前 End |
| Arrow 由 CRT 完成线程再次调度 | 解码/格式子任务仍有正确 parent |
| Tokio poll、子 spawn、blocking task | 每个边界恢复且退出清理；panic/错误路径也验证 |
| cache leader/follower、cache hit | 物理工作仅一次，wait/link 正确，无长期 parent 缓存 |
| 丢弃 future、真实取消、短读与分类错误 | 不改变业务结果，span 与真实工作寿命一致 |
| 开关、采样、span 预算与 provider 更新 | 配置语义正确，量化额外 CPU/分配/尾延迟 |
| 各文件系统与 async 能力 | wrapper 不破坏非阻塞接口，不重复计数字节 |

阶段 S1：公共契约、RequestContext、同步读取、metadata、基本 I/O、独立宿主测试。

阶段 S2：Folly/Arrow/CRT 完整 async 路径、Vortex Tokio/FFI 传播、错误与丢弃语义验证。S1 不宣称完成这些异步路径。

阶段 S3：与 Milvus 联调实际锁定版本和对象存储故障路径；完成测量预算后逐步启用。未覆盖格式/引擎内部路径明确列出，不将外层 span 成功视为内部追踪完成。

编译、格式化、生成和测试遵循仓库规定的 Milvus builder 容器、共享 Conan cache 挂载和输出目录所有权要求。本设计文档不代表已运行这些检查。

## 10. 参考

- [Milvus 宿主改造](opentelemetry-milvus-design.md)
- [当前异步读取设计](async-read-design.md)
- [OTel instrumentation scope](https://opentelemetry.io/docs/concepts/instrumentation-scope/)
- [OTel C++ instrumentation](https://opentelemetry.io/docs/languages/cpp/instrumentation/)
- [OTel tracing SDK](https://opentelemetry.io/docs/specs/otel/trace/sdk/)
- [Folly RequestContext](https://github.com/facebook/folly/blob/main/folly/io/async/Request.h)
- [Folly fiber 调度](https://github.com/facebook/folly/blob/main/folly/fibers/FiberManagerInternal-inl.h)
- [Rust tracing 的异步使用限制](https://docs.rs/tracing/latest/tracing/struct.Span.html)

参考上游 main/latest 仅解释机制，实现与验收以实际锁定依赖版本为准。
