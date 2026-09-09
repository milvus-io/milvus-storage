# Milvus Storage OpenTelemetry

本文统一说明 Storage 的 tracing 接入、上下文传播、实现边界及验证结果，路径均相对 milvus-storage 仓库。宿主负责 provider、采样与 exporter 生命周期；Milvus 侧接入方案由 Milvus 仓库维护，不属于本文的改动范围。

当前已实现 C++ 接入、Folly/Arrow/CRT 与 Rust 传播适配，并完成下述容器内筛选回归。**相对引入 Telemetry 前版本的完整无性能回退验收仍未完成**，不能将功能测试或局部优化数据当作生产性能承诺。

- 引入前基线：`43b7793406136208928ad20dc1596b510c7b629a`。
- 接入与初期修复：`6966d49`；局部优化：`fb59cbf`。
- 合入上游 Lance dataset 复用：`8ecbf49`（上游 `e7f4477`）。
- 异常处理修复及最近一次筛选回归：`14dc509`。

## 接入与公共契约

公共接口位于 [tracing.h](../cpp/include/milvus-storage/tracing.h)，实现位于 Storage 动态库中。接入不要求修改 Reader、ChunkReader、格式或文件系统的业务入口参数。

```cpp
#include "milvus-storage/tracing.h"

// provider 由宿主 SDK 创建，必须与 Storage 的 OTel C++ ABI、版本和编译选项一致。
milvus_storage::tracing::SetTracerProvider(provider);
milvus_storage::tracing::SetTraceOptions({.io_spans = true,
                                         .max_spans_per_operation = 256});
{
    auto scope = milvus_storage::tracing::AttachParent(parent);
    auto table = reader->take(rows);
}
auto future = [&] {
    auto scope = milvus_storage::tracing::AttachParent(parent);
    return reader->take_async(rows);
}();
// 此处可在宿主选择的 executor 上消费，parent 和 provider 已固定。
```

`TraceParent` 拥有 trace ID、span ID、flags、tracestate 和 remote 标志。`TraceOptions` 提供 I/O spans 开关及每操作 span 预算，默认分别为 `true` 和 `256`。

### Provider

- Storage 保存注入的 provider，取得 instrumentation scope 为 `milvus-storage`、version 为自身构建版本的 Tracer；不调用宿主全局 `SetTracerProvider`，不自行启动 exporter。
- 未注入或传空时默认不记录/导出；调用方应用负责初始化 SDK。测试宿主可注入 in-memory exporter 对应 provider。
- 发布新 provider 快照须并发安全；每次操作固定自己的 provider/tracer，内部子操作使用同一配置快照。
- provider 更新不迁移已有 spans。宿主负责在相关在途工作完成后 flush/shutdown 旧 provider；Storage 不主动关闭共享实例。
- C++ provider 类型要求 OTel ABI/version/编译选项匹配；不作为稳定 C ABI，也不跨 Rust 传递。显式注入避免猜测不同动态库中的 global singleton 是否一致。

### Parent 作用域

`AttachParent` **不创建 span，不调用父 span End，也不绑定 Reader**。它为当前执行上下文设置 Storage parent，并在析构时恢复以前的上下文。

实现使用 Folly `RequestContext` 的 Storage 专属 key：保存当前上下文，建立浅拷贝，用不可变、拥有数据的 parent 覆盖该 key，退出时恢复。不得直接改写可能被多个任务继承的共享可变 RequestData。

嵌套作用域按栈顺序恢复；guard 不可复制、移动或转交其他执行流。Folly 管理的 fiber 挂起时 guard 可以留在 fiber 栈上，由运行时保存/恢复 RequestContext。对不支持该机制的线程/协程，必须显式传递快照。

无效/空 parent 建立“无父上下文”边界，遮蔽外层 parent，默认不自行创建根 span；不能悄悄沿用旧 TLS 或宿主残留 parent。未采样的有效 parent 仍须传播，不能因为 `IsRecording()==false` 就改成新根 trace；实际记录由 provider 采样策略决定。

本接口只更新 **Storage 当前上下文**，不把普通 OTel `WithActiveSpan` guard 留在 fiber 栈上。所有 Storage 埋点从该上下文读取 parent，显式设置 `StartSpanOptions.parent`。第三方仅依赖 OTel TLS `GetCurrentSpan()` 的埋点不自动覆盖。

入口必须在返回前捕获 parent，即使真正工作延迟到 future 消费时。不能根据消费线程重新选择 parent。惰性工作在实际开始执行时创建工作 span；如需统计调用到执行的延迟，另行记录 start delay。未消费且无工作发生的 future 不伪造 I/O span。

逐次读取/stream 的上下文属于读取操作，不属于共享 Reader 或文件；每次 `ReadNext` 使用调用时的 parent。请求专属 stream wrapper 可以由宿主保留 parent，在每次调用时重新激活；共享 Reader 永远不记住第一个调用者。

## 执行架构与上下文传播

Storage 复用已有 OTel C++ 构建依赖；详细异步执行语义见 [async-read-design.md](async-read-design.md)。

| 路径 | 当前执行形态 | tracing 处理的边界 |
|---|---|---|
| 同步 chunk/take/full scan | 部分路径使用 ThreadPoolHolder；full scan 保持同步组织 | 线程池提交、逐次读取 |
| async chunk/take | 规划任务、组合 SemiFuture、collectAll | 入口捕获、惰性执行、fan-out/fan-in |
| Parquet | FollyArrowExecutor 驱动 Arrow generator；部分 open 是延迟执行的阻塞操作 | Arrow 任务、完成回调、metadata I/O |
| Vortex | C++ Promise 桥接，共享 Tokio runtime 执行 open/scan/collect | C++/Rust、spawn、poll、blocking pool |
| CRT I/O | NonBlockingRandomAccessFile、ReadAtAsyncInto、回调完成 | 提交到完成的请求状态 |
| 其他格式 | 可能使用同步 ready-future fallback | 不能将返回 Future 等同于非阻塞 |

Storage 当前不选择调用方 Folly executor；tracing 也不增加调度 executor。Vortex 内部新建任务、Arrow I/O pool 等不会因为外层存在一个 context 就自动继承。

内部 `OperationTrace` 保存不可变的上下文和配置快照、span 与完成状态。并行任务派生各自上下文，不共享可变的 current-span 槽位；`Finish(status)` 至多完成一次，不以业务阻塞等待 span 结束，也不通过持有整个 Reader 或数据缓冲区形成引用环。

### RequestContext 与 fiber

普通 TLS 在 Fiber A 挂起时不会自动清除；Fiber B 可能读到 A，A 恢复后又可能读到 B。RAII guard 留在各自 fiber 栈上不能修复线程局部状态。

本地 Folly 源码的 `FiberManagerInternal-inl.h` 在运行 fiber 时恢复 `rcontext_`，在相应挂起/切换路径保存 RequestContext；`Request.h` 提供浅拷贝 scope。实现必须针对实际锁定版本验证该行为。

```text
Fiber A：AttachParent(A) → RequestContext A → 挂起
Fiber B：RequestContext B → Storage 读取 B
Fiber A：恢复 RequestContext A → Storage 读取 A
```

这只保证经过支持的调度路径；`runInMainContext`、自定义 awaiter、跨 runtime 和脱离请求的后台任务需逐个定义上下文边界。Folly coroutine `Task` 的传播能力也按锁定版本和实际 await/start 时机验证，不等同于所有 C++ coroutine 自动支持。

### 边界处理矩阵

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

### Rust 内部桥接

私有 CXX 桥接提供 `capture_trace_context` 与 `attach_trace_context`。`SharedPtr<TraceContext>` 拥有不可变上下文，`UniquePtr<TraceAttachment>` 在析构时恢复作用域；没有公开稳定 C 业务 ABI。

- handle 指向 C++ 拥有、不可变且可跨线程引用的上下文；Rust wrapper 的 Clone/Drop 对应 retain/release。
- attach 返回的 token 只在本次执行片段有效，同一次 poll/闭包内按栈顺序 detach；token 不能保留到下一次 poll，也不能跨线程转交。
- future wrapper 在进入 poll 时 attach，Ready/Pending/错误退出时 detach。panic 路径必须恢复；已有跨 FFI panic/异常隔离保持不变。
- context 随 open/scan 状态显式传递，覆盖 Vortex runtime adapter 创建的实际子任务。若引擎存在不可适配边界，该路径标记未覆盖，不能声称全部 I/O 已串联。
- C++ 反向 FFI 不能抛异常穿过边界；使用现有 `LoonFFIResult` 约定。内部桥接和公共符号导出范围要区分；需要公开的新符号按 binding 构建更新 exports map。

Rust `tracing-opentelemetry` 不是第一版依赖。后续若需要现有引擎内部 tracing spans，可独立评估；引入时仍须设置 parent 并使用按 poll 激活的 instrument wrapper。

## 埋点、完成与错误

| 范围 | 实现 |
| --- | --- |
| Reader | `storage.open`、`storage.read`、`storage.read_task`；同步线程池闭包和 Folly continuation 显式恢复上下文 |
| 结果组装 | `storage.assemble` 覆盖 take 的最终列投影、补空列和 Table 组装 |
| metadata | `storage.metadata.lookup` 的 hit/miss/in_flight，leader 的 `storage.metadata.load`，follower 的 `storage.metadata.wait` 和 leader link；长期条目仅存 metadata |
| 格式 | Parquet、Vortex、Lance、Iceberg、Paimon 的 `storage.format.read`；含 I/O 等待，不能当作 codec CPU 时间 |
| 逻辑文件读取 | FileSystemProxy 打开的文件覆盖两个 Read/ReadAt 重载、ReadAsync、ReadManyAsync、ReadAtAsyncInto；metadata/head 单列 |
| Arrow | 操作专属 FollyArrowExecutor 保存操作快照，即使后续 Spawn 来自 CRT 完成线程也恢复相同 parent |
| CRT | GetObjectAsync/HeadObjectAsync 提交至 source future 完成的 `storage.backend.request`；callback 恢复上下文；提交异常转换为 UnknownError 并完成 source future，由 observer 结束 span |
| Rust | CXX 私有 owning handle；共享 Tokio runtime 的 block_on/spawn/spawn_blocking；future 每 poll attach/detach；Vortex Executor 的 async、CPU、blocking I/O 子任务委托原执行器，保留其调度和 profiling labels |
| 反向 FFI | Rust 在 poll/闭包内调用 C++ 文件系统；异步完成 callback 重新激活提交时上下文 |

惰性 Future 在实际工作开始时创建 span。未消费且未开始工作的 Future 不导出虚假工作 span。Arrow source future 完成观测独立于消费者，丢弃返回的 Future 不会提前结束后台 I/O span。Vortex 原生操作直接在完成 callback 中结束 span，返回原始 Future，不增加 consumer executor 调度。其他 Folly 操作通过完成 continuation 观测最终结果；若消费者丢弃已开始的操作，最后一个工作上下文释放时结束尚未完成的 span，并设置 `storage.completion.unobserved=true`，不把它当作真实取消。

tracing wrapper 已捕获的同步/提交异常，以及 Folly 完成 continuation 收到的异常，转为不包含异常原文的 `UnknownError`，通过 Status/Result 或已完成的失败 Future 返回。批量提交异常为每个输入 range 返回一个失败 Future。原有 Status/Result 的扩展分类保持不变；无上下文快速转发路径保持原实现。

CXX 将异常转换为 Result；Rust guard 在 Pending、Ready 和 panic unwind 时析构。attach token 不跨 poll/线程保留。桥接没有引入 Rust OTel SDK 或 exporter，也未新增公开 C 业务 ABI，FFI 导出表保持现有范围。

格式层使用 `storage.format.read` 表示包含 I/O 等待的操作 wall time。只有能隔离真实解码/物化边界时才适合命名 `decode_materialize`；当前未提供精确 codec CPU 时间。并行 spans 会重叠，不能将 duration 相加作为端到端延迟，也不能用总耗时减 I/O 时长推导解码 CPU。

## 缓存、采样与观测配置

metadata singleflight 中 leader 记录一次实际 load/I/O；每个 follower 在自己的 trace 记录 wait，并在可获得 leader SpanContext 时附加 link。不能把同一次物理 I/O 复制为多个请求的子操作。leader context 只保存在在途加载状态，不进入长期缓存条目。

成功缓存的 metadata、Reader、文件句柄不保存请求 parent。后台预热从宿主传入独立 parent，后续请求只记录自身命中/等待。

默认只为有效父上下文创建操作链；follow-parent 采样由宿主 provider 决定。未记录时避免构造昂贵属性和逐请求日志。tracing 禁用后的成本及未采样传播的必要成本分别测量，不承诺未经基准验证的零开销。

`io_spans=false` 抑制细粒度逻辑 I/O 和 backend spans，保留操作及汇总。`max_spans_per_operation` 包含操作本身，最小有效值为 1。达到预算后复用父上下文，保留 `storage.spans.dropped` 以及根操作的 `storage.io.reads`、`storage.io.requested_bytes`、`storage.io.returned_bytes`。预算由一次操作的所有并行任务共享。

字节仅在逻辑文件层计数，backend span 不再累计字节；不能把根汇总与子 span 的字节重复相加。批量读取记录一个逻辑 span，返回字节为实际完成结果之和。短读记录实际字节，不擅自把合法 EOF 变成错误；原有 FFI 对短读的错误判定保持不变。

固定属性包括 `storage.operation`、`storage.format`、`storage.backend`、cache 状态、offset、请求/返回字节和 range_count。错误仅导出分类及已有扩展状态的 retryable，不导出原错误消息、对象路径、凭证、签名 URL、查询或数据。

## 实现位置与局部优化

| 位置 | 职责 |
| --- | --- |
| `cpp/src/tracing/` | context、span 生命周期及文件 wrapper |
| `cpp/src/reader.cpp`、`cpp/src/format/column_group*reader.cpp` | 操作入口、任务传播及结果组装 |
| `cpp/src/format/format_reader_cache.cpp` | singleflight lookup/wait/load 与 link |
| `cpp/src/format/parquet/`、`cpp/src/format/vortex/` | Arrow executor、格式及原生异步完成 |
| `cpp/src/filesystem/s3/s3_filesystem.cpp` | CRT async head/get 请求完成 |
| `cpp/src/format/bridge/rust/`、`cpp/src/ffi/filesystem_c.cpp` | Tokio/CXX/反向 FFI 的上下文与生命周期 |
| `cpp/test/tracing/tracing_test.cpp`、`cpp/benchmark/` | 回归测试和基准入口 |

`6966d49` 已消除无上下文时多余的完成 continuation、属性设置与 Folly TLS 查询，将 Rust future 的内部 pin 从额外 heap allocation 改为 wrapper 内 pin。`fb59cbf` 的后续优化如下：

- 文件读取及 Run/RunAsync 使用 `HasContext()` 判断当前上下文，避免仅检查空值时构造 owning shared_ptr；文件同步/异步读取的 recording 部分单独放入不内联的函数，缩小关闭路径的栈帧。
- 子操作在分配 SpanState 前检查 null provider、I/O 开关和 span 预算；根操作仍捕获固定配置，保留后续注入 provider 时的行为。
- SpanState 首次启动完成后用 release/acquire 标志发布 span 指针或禁用决定，后续 Start 不再重复取 mutex。标志放在现有 bool 字段旁的 padding 中；仍保留首次启动和结束的互斥保护。
- metadata 无上下文时不分配 leader OperationTrace；无上下文 follower 直接等待已有结果。带 tracing 的 follower 可等待未带 tracing 的 leader，此时不创建无效 link。
- Rust capture 的 opaque handle 直接持有 immutable Context；attachment 的私有派生对象直接持有 ContextScope，删除两处独立 Impl 分配。CXX 函数签名及 Rust 侧调度行为保持原有约定。

Scope 复用、跳过所有未采样子 span，以及修改 executor/重复装箱路径，局部优化中均未实施。其他 RequestContext key 的隔离、空上下文遮蔽和宿主 sampler 的决策继续由现有路径保障。

## 覆盖边界与待验收项

- 本次没有 Milvus checkout 改动，也没有更改 Milvus 锁定的 Storage 版本。跨仓库部署和对象存储故障联调仍待完成。
- backend request spans 当前覆盖 CRT async head/get；其他 provider、同步 SDK 的内部重试、Talon 远程服务内部仍只有逻辑层/外层 span，不能据此宣称网络请求全部可见。
- Vortex 使用 Storage 提供的 runtime adapter 的子任务已接入。Lance/DataFusion、Iceberg/Paimon 等第三方自行 spawn 的任务、外部 Arrow executor 和引擎内部 codec spans 不自动继承；未替换全局 OTel TLS，也没有安装第三方引擎 hook。
- 已有 API 不提供完整取消桥接；丢弃 Future 不表示远程或 Tokio 工作已取消。
- Python 构建的现有 FFI-only 符号策略保持不变；本公共 provider 接口是常规 C++ 宿主接口，不是 Python/Rust 可直接传递的稳定 ABI。

跨仓库部署需在 Milvus 实际锁定的 Storage 依赖版本上验证；本地 checkout 的功能不代表已部署。后续联调仍需覆盖对象存储故障、网络 exporter、生产并发、真实取消及尾延迟，不能仅凭外层 span 成功就认定引擎内部追踪完整。

## 构建与功能验证

所有编译、格式化和测试均在 `storage-tracing-dev` 开发容器中执行，镜像为 `milvusdb/milvus-env:ubuntu22.04-20260714-c135601`。按要求将 `/data/yuruiz/milvus/.docker/amd64-ubuntu22.04-conan2` 挂到 `/root/.conan2`，另挂载当前构建使用的依赖缓存。功能构建开启 ASAN、coverage、FIU，默认关闭 CRT；性能测量另用独立 Release 构建。

以下命令在开发容器内执行。不同源码树共享 Cargo target 时的根 crate 清理要求见后文“性能验证方法”。

```sh
cmake --build cpp/build/Release --target milvus_test Test_FFI tracing_benchmark -j 6
cd cpp
. build/Release/generators/conanrun.sh
export ASAN_OPTIONS=detect_leaks=0
build/Release/test/milvus_test --gtest_filter='StorageTracingTest.*:FollyArrowExecutorTest.*:*APIWriterReaderTest*:*FormatReaderMetadataCacheParamTest*:*FormatReaderTest*:*LocalFileSystem*:*CloudFSMetrics*-*ParquetOpenAsyncIsLazyAndSupportsThreadedExecutor*'
build/Release/test/Test_FFI
```

各阶段测试范围不同，以下数量分别记录，不能相加或当作一次全量测试：

| 阶段 | C++ 通过 / 条件跳过 | tracing 通过 | C FFI 通过 |
| --- | ---: | ---: | ---: |
| 初次接入 | 325 / 69 | 17 | 83 |
| 初期修复、强制重编 Rust bridge 后 | 329 / 69 | 21 | 83 |
| 局部优化 `fb59cbf` | 334 / 69 | 26 | 83 |
| 上游合并 `8ecbf49`，增加 Lance 测试范围 | 397 / 76 | 26 | 83 |
| 异常修复 `14dc509`，上述命令的筛选范围 | 337 / 69 | 29 | 83 |

`14dc509` 新增同步异常、延迟执行异常和文件提交异常回归，检查失败结果、批量 Future 完成数量及 span 正常结束。error-handling ratchet 对比上游 baseline 通过，throw 数量为 21，未修改规则或 baseline。C++ clang-format 18、Rust rustfmt（涉及 Rust 改动时）和 `git diff --check` 已通过。

测试覆盖 parent 隔离/嵌套/fiber 恢复、provider 快照/未采样传播、惰性 Future、完成竞争、native Future 丢弃/同步完成、I/O 重载与开关、Arrow 跨线程提交、singleflight link、共享 Reader 的 Parquet/Vortex 并发，以及错误分类与敏感消息不导出。

初次接入时另有编译覆盖：Rust bridge 的 `s3-crt-async` feature 编译，以及 C++ CRT 源文件在 AWS SDK 1.11.842 头文件下的条件路径语法编译。后者不是完整 CRT 链接或真实 S3 请求测试。

当前环境的 `ParquetOpenAsyncIsLazyAndSupportsThreadedExecutor` 参数化用例被排除：组合运行会崩溃，不含任何 Storage tracing 代码的独立探针确认，头文件给出的 `sizeof(folly::ThreadedExecutor)` 为 256，而预编译构造函数写到对象起始位置之后的第 415 字节。预编译依赖使用 GCC 14，容器编译器是 GCC 12；已确认对象布局不一致，尚未确定造成布局差异的全部编译选项。需在依赖与编译环境一致后复验，不能将本次筛选回归等同于全量通过。

### 功能验证证据

以下 `/tmp` 路径是当时验证机器上的原始记录，不是仓库分发文件：

- 初次接入：`/tmp/storage-tracing-build-confirm.log`、`/tmp/storage-tracing-regression-final.log`、`/tmp/storage-tracing-ffi-final.log`。Folly ABI 独立探针：`/tmp/storage-threaded-probe.cpp`、`/tmp/storage-threaded-probe.log`。
- 初期修复：`/tmp/storage-telemetry-perf-20260909/optimized-tests-verified.log`、`optimized-ffi.log`。
- 局部优化：`/tmp/storage-telemetry-opt-6966d49/` 下 `tests-build.log`、`tests.log`、`ffi.log`、`release-build.log`、`inputs.json`。
- 上游合并：同一优化目录下 `merge-build.log`、`merge-tests.log`、`merge-ffi.log`。构建期间容器曾以 137 退出，恢复后完成增量构建和测试。
- 异常修复：`/tmp/storage-ratchet-build-verified.log`、`/tmp/storage-ratchet-tests.log`、`/tmp/storage-ratchet-ffi.log`。

## 性能验证方法

### 验收口径

比较引入 Telemetry 之前的 Storage 与当前工作树，主要验收场景是未注入 parent/provider 的默认路径。另行测量有 parent 但无 provider、未采样、已采样三种模式，不能把它们混入默认关闭模式的结果。

“未发现回退”需要配对测量和足够窄的置信区间支持，单次更快或差异不显著都不能证明零开销。脚本中的 2% 是测量灵敏度边界，不是用户同意的回退额度。置信区间跨过该边界时结果为 `inconclusive`；区间下界大于 0 时标记 `detected_regression`。区间为各场景各自的 95% 区间，并非整个测试矩阵的同时置信区间。

### 测量对象

- 基线：`43b7793406136208928ad20dc1596b510c7b629a`，仅添加同一份 benchmark harness 和对应 CMake target。
- 被测版本：当时包含 Telemetry 及初期修复的工作树快照（提交前测量），不是本文最新 head；每次运行保存 executable 和实际构建目录下 shared library 的 SHA-256。后续局部优化另以 `6966d49` 为基线。
- 独立 Release 构建：`WITH_ASAN=OFF`、`WITH_UT=OFF`、`WITH_FIU=OFF`、`WITH_TALON=OFF`、`WITH_BENCHMARK=ON`，无 coverage flags。不使用已有 ASAN/coverage 构建的性能数据。
- 开发容器：`milvusdb/milvus-env:ubuntu22.04-20260714-c135601`，按验证约定挂载 Conan 缓存；两版本使用相同工具链与依赖。
- CPU：Xeon Gold 6338；固定亲和性 `8,9`，Storage/Tokio/benchmark executor 各配置两个 worker。系统使用 `ondemand` 调频，未修改宿主调频策略，仍存在共享机器噪声。
- 数据：同一批固定种子的 65,536 行，`int64 id`、`float64 value`、32 维 `float32` 向量；Parquet/Vortex 各一个 column group。基线生成一次，所有版本复用同一文件与 manifest，记录数据 SHA-256。
- 25 个场景：两种格式的同步/异步 take（64/1,024 行，metadata 冷/热）、完整扫描、同步/异步 chunk 读取，以及 64/4,096/65,536 字节本地 `ReadAt`。
- metadata 冷表示每次创建 Reader/cache；文件数据已在 OS page cache 中，不代表冷磁盘。测试检查 take 返回 ID/顺序及 chunk 行数一致。
- 主指标：每操作 wall time 和整个进程的 CPU time；配对比值取几何均值，并计算 log-ratio 的 Student t 区间。进程峰值 RSS 单独保存，不等于每操作分配量。
- 已采样模式使用容量为 1 的内存 exporter 与 `SimpleSpanProcessor`，包括本地 span 构造/处理成本，不包括网络导出。

模式 0 的进程从未调用 `AttachParent`；不代表启用过 tracing 后再次关闭的所有历史状态。模式 1/2/3 在每次操作外建立 parent scope，其耗时计入操作，因此小块 `ReadAt` 也包含每次注入 parent 的成本。

### 可复现入口

源码为 `cpp/benchmark/benchmark_telemetry_read.cpp`，运行器为 `cpp/benchmark/run_telemetry_comparison.py`。所有构建、格式化和执行命令均须在开发容器内运行。

当前版构建 `telemetry_read_benchmark` target；基线也编译同一 harness，但将 target 的 `STORAGE_TELEMETRY_BENCHMARK` 定义替换为 `OPENTELEMETRY_STL_VERSION=2017`，不编译宿主 tracing 接入代码。将数据目录路径传给 `STORAGE_BENCH_DATA`，仅首次用基线可执行文件设置 `STORAGE_BENCH_PREPARE=1` 生成数据。

不要在不同源码树之间直接复用根 crate 的 Cargo 编译结果。本次隔离构建共享了 Rust 依赖缓存，切换源码前必须执行以下定向清理，再构建对应 CMake target，并检查日志中的 `Compiling rust-bridge` 源码路径。仅复用第三方依赖，保存各版本单独链接的库；两个构建不得同时运行。

```sh
cargo clean --manifest-path "$SOURCE/cpp/src/format/bridge/rust/Cargo.toml" \
  --target-dir "$CARGO_TARGET" --release \
  --target x86_64-unknown-linux-gnu -p rust-bridge
cmake --build "$BUILD" --target telemetry_read_benchmark -j8

. cpp/build/Release/generators/conanrun.sh
python3 cpp/benchmark/run_telemetry_comparison.py \
  --baseline "$BASELINE_BUILD/benchmark/telemetry_read_benchmark" \
  --current "$CURRENT_BUILD/benchmark/telemetry_read_benchmark" \
  --data "$DATA" --output "$NEW_OUTPUT" \
  --pairs 7 --seconds 0.35 --warmup 0.3 --casewise --cpus 8,9 --mode 0
```

`--output` 必须是新目录，避免覆盖历史证据。模式 `0/1/2/3` 分别为无 parent/provider、有 parent 无 provider、未采样、采样。每个场景紧邻执行七对 AB/BA，避免先跑完一个版本再跑另一个版本的长时间漂移。`inputs.json` 保存输入，逐次 `.json/.log/.resources.json` 保存原始记录，`pairs.json` 保存配对，`summary.json` 保存统计。

自动验收可加 `--require-no-regression`：任一场景的 wall/CPU 结果检测到回退，或置信区间仍不足以排除超出灵敏度边界的回退，保存结果后返回非零退出码。它不会把高噪声的“不显著”当作通过。

### 早期记录与排除项

原始构建、输入和运行记录保存在 `/tmp/storage-telemetry-perf-20260909`。早期 `initial-disabled-v1` 至 `v5` 均排除：前几次有 harness 错误，v5 虽运行完成但复用了错误的 Rust 桥接产物；原因记录在 `INVALID-EARLY-RUNS.md`，原始记录保留。

引入前后完整矩阵的无回退验收尚未完成。以下局部优化结果以已含 Telemetry 的 `6966d49` 为基线；不代表相对引入前基线 `43b7793` 的结果。本地读取不能外推到 S3/Talon、网络 exporter、生产并发或 p95/p99 尾延迟。

## 局部优化测量结果

本节比较 `6966d49` 与局部优化版本 `fb59cbf`，原始输入、源码指纹及数据位于 `/tmp/storage-telemetry-opt-6966d49`，后文相对证据路径均以该目录为根。

测量早于上游合并 `8ecbf49` 与异常修复 `14dc509`，未重新测量这些后续版本，不能用本节数据宣称最新 head 已通过性能验收。

### 分配与 jemalloc 实际占用

前后均为独立 Release `-O3` 构建，关闭 ASAN/coverage/FIU，使用同一 Conan 依赖和同一份 `libjemalloc_pic.a`。计时与 C++ new 计数使用分别编译的探针；表中请求字节不等于 RSS 或 jemalloc active/resident。

| 探针 | C++ new 次数：优化前 → 后 | 请求字节：优化前 → 后 |
| --- | ---: | ---: |
| 有 parent、null provider，根操作及 4 个子操作 | 27 → 23 | 2,576 → 1,776 |
| 未采样，根操作及 4 个子操作 | 51 → 51 | 3,656 → 3,656 |
| 采样，根操作及 4 个子操作 | 79 → 79 | 6,268 → 6,268 |
| 有 parent，Rust capture + attach | 8 → 6 | 384 → 352 |
| 无 parent 的文件读取 | 0 → 0 | 0 → 0 |

另用 jemalloc `stats.allocated/active/resident` 测量同时持有 50,000 个 Rust Context handle 的占用：vector 预先 reserve，读取统计前 flush 当前线程 tcache 并更新 epoch。5 轮中首轮各有 1,952 B 的惰性初始化；后 4 轮结果稳定：

| 统计增量 | 优化前 | 优化后 |
| --- | ---: | ---: |
| 持有期间 allocated | 3,200,000 B | 1,600,000 B |
| 持有期间 active | 3,186,688 B | 1,585,152 B |
| 释放后 allocated | 0 B | 0 B |

该固定 handle 场景的 allocated 减少 50%。resident 受 allocator 保留页影响，不能据此推导整个服务 RSS 或完整读取请求下降 50%。原始数据：`probes/before-heap.json`、`probes/after-heap.json`；源码 `probes/heap.cpp`。

### 时间对照

运行相同探针、同一数据，固定 CPU 8、每个进程预热 300 ms。5 对前后进程按 AB/BA 交替；每进程内部重复测量，使用进程内中位数计算配对 log-ratio 的 95% Student t 区间。系统仍是 ondemand 调频，因此保留全部原始数据和不确定结果。

| 场景 | 耗时配对变化 | 95% 区间 |
| --- | ---: | --- |
| 无 parent，根操作及 4 个子操作 | -25.3% | -38.3%～-9.6% |
| 有 parent、null provider，根操作及 4 个子操作 | -12.9% | -16.6%～-9.0% |
| 未采样，根操作及 4 个子操作 | -7.0% | -17.0%～+4.2% |
| 采样，根操作及 4 个子操作 | -2.6% | -6.2%～+1.1% |
| 已启动 span，再次 Start，null provider | -72.5% | -76.3%～-68.1% |
| 已启动 span，再次 Start，未采样 | -55.8% | -60.6%～-50.3% |
| 已启动 span，再次 Start，采样 | -57.8% | -67.0%～-46.2% |

未采样/采样根操作整体改善未达到统计显著；文件微基准也不能确认整体提速。上述小操作结果不能直接外推为 Reader 吞吐提升。汇总为 `probes/summary.json`，原始数据、探针和构建/运行脚本保存在 `probes/`。

### 端到端读取

同一批 Parquet/Vortex 文件，固定 CPU 8,9，7 对 AB/BA，每次最小测量 0.35 s、预热 0.25 s；模式为默认无 parent/provider。六个代表性场景的 wall time 如下，CPU 指标的判定一致：

| 场景 | 配对变化 | 95% 区间 | 结论 |
| --- | ---: | --- | --- |
| Parquet 同步 take 64，冷 metadata | -5.17% | -16.19%～+7.31% | 不确定 |
| Parquet 异步 take 1024，热 metadata | -2.69% | -10.44%～+5.74% | 不确定 |
| Vortex 同步 take 64，冷 metadata | -9.74% | -23.14%～+5.99% | 不确定 |
| Vortex 异步 take 64，热 metadata | -1.98% | -5.01%～+1.15% | 区间上界低于 2% 灵敏度边界 |
| 本地 ReadAt 64 B | +0.22% | -1.30%～+1.77% | 区间上界低于 2% 灵敏度边界 |
| 本地 ReadAt 4 KiB | -1.14% | -2.73%～+0.49% | 区间上界低于 2% 灵敏度边界 |

六项均未检出统计显著回退，但三项仍无法排除超过 2% 的回退，因此不能给出全面“无回退”结论。2% 是测量灵敏度边界，不是用户允许的回退额度。相对 Telemetry 引入前版本的零回退验收仍未完成；剩余问题需要固定频率、可 profiling 的环境，不能用多次噪声测量挑选有利结果。

复现入口仍为 `cpp/benchmark/run_telemetry_comparison.py`。本轮输入与所有原始记录见 `end-to-end/inputs.json`、`end-to-end/pairs.json`、`end-to-end/summary.json`。

## 历史 helper 微基准

`tracing_benchmark` 比较无 parent/provider、有效 parent + null provider、未采样和采样四种模式。它只测量作用域及四个子操作，不代表完整读取；默认关闭路径的快路径也不能替代性能验证。以下保留初次接入的历史记录，不能用于评价后续优化版本。

基准命令：`tracing_benchmark --benchmark_filter=BM_StorageTracing --benchmark_min_time=0.1s --benchmark_repetitions=3 --benchmark_out=/tmp/storage-tracing-benchmark.json --benchmark_out_format=json`。2026-09-09 三次测量的 wall time 中位数如下（每次包含 parent scope 及四个子操作）：

| 模式 | ns/operation |
| --- | ---: |
| 无 parent/provider | 1,584 |
| 有 parent、null provider | 19,826 |
| 未采样 parent | 34,884 |
| 采样 parent | 53,605 |

这些数值来自开启 ASAN/coverage 且启用 CPU scaling 的验证环境，仅用于本实现四种模式的成本对照，不能作为生产吞吐或尾延迟承诺。原始结果保存在 `/tmp/storage-tracing-benchmark.json`。

## 参考

- [当前异步读取设计](async-read-design.md)
- [OTel instrumentation scope](https://opentelemetry.io/docs/concepts/instrumentation-scope/)
- [OTel C++ instrumentation](https://opentelemetry.io/docs/languages/cpp/instrumentation/)
- [OTel tracing SDK](https://opentelemetry.io/docs/specs/otel/trace/sdk/)
- [Folly RequestContext](https://github.com/facebook/folly/blob/main/folly/io/async/Request.h)
- [Folly fiber 调度](https://github.com/facebook/folly/blob/main/folly/fibers/FiberManagerInternal-inl.h)
- [Rust tracing 的异步使用限制](https://docs.rs/tracing/latest/tracing/struct.Span.html)

参考上游 main/latest 仅解释机制，实现与验收以实际锁定依赖版本为准。
