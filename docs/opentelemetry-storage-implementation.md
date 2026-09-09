# Storage tracing 接入与覆盖范围

本实现对应 [Storage 设计](opentelemetry-storage-design.md)。宿主的配置、exporter 生命周期及 Milvus 依赖版本升级仍由宿主负责；本仓库不读取 tracing 环境变量、不安装全局 provider。

## 接入

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

`TraceParent` 拥有 trace ID、span ID、flags、tracestate 和 remote 标志。`AttachParent` 不创建或结束父 span，不更改 OTel TLS。空/无效 parent 遮蔽外层 parent。作用域不可复制/移动，必须在同一执行流内按栈顺序析构；Folly fiber 可在作用域内挂起。

Storage 使用专属 Folly RequestContext key，激活时浅复制并覆盖自己的不可变数据，保留其他 key。共享 Reader、metadata 缓存条目、文件句柄以及全局 runtime 不保存请求 parent。stream 每次 `ReadNext` 使用调用时的 parent。

`SetTracerProvider(nullptr)` 禁用后续操作的记录；已开始或已捕获的操作仍使用其快照。宿主须等待旧 provider 对应的工作结束后自行 flush/shutdown。instrumentation scope 为 `milvus-storage`，version 来自 CMake 项目版本。

## 埋点与传播

| 范围 | 实现 |
| --- | --- |
| Reader | `storage.open`、`storage.read`、`storage.read_task`；同步线程池闭包和 Folly continuation 显式恢复上下文 |
| 结果组装 | `storage.assemble` 覆盖 take 的最终列投影、补空列和 Table 组装 |
| metadata | `storage.metadata.lookup` 的 hit/miss/in_flight，leader 的 `storage.metadata.load`，follower 的 `storage.metadata.wait` 和 leader link；长期条目仅存 metadata |
| 格式 | Parquet、Vortex、Lance、Iceberg、Paimon 的 `storage.format.read`；含 I/O 等待，不能当作 codec CPU 时间 |
| 逻辑文件读取 | FileSystemProxy 打开的文件覆盖两个 Read/ReadAt 重载、ReadAsync、ReadManyAsync、ReadAtAsyncInto；metadata/head 单列 |
| Arrow | 操作专属 FollyArrowExecutor 保存操作快照，即使后续 Spawn 来自 CRT 完成线程也恢复相同 parent |
| CRT | GetObjectAsync/HeadObjectAsync 提交至 source future 完成的 `storage.backend.request`；callback 恢复上下文；提交抛异常记录错误并继续传播原异常 |
| Rust | CXX 私有 owning handle；共享 Tokio runtime 的 block_on/spawn/spawn_blocking；future 每 poll attach/detach；Vortex Executor 的 async、CPU、blocking I/O 子任务委托原执行器，保留其调度和 profiling labels |
| 反向 FFI | Rust 在 poll/闭包内调用 C++ 文件系统；异步完成 callback 重新激活提交时上下文 |

惰性 Future 在实际工作开始时创建 span。未消费且未开始工作的 Future 不导出虚假工作 span。Arrow source future 完成观测独立于消费者，丢弃返回的 Future 不会提前结束后台 I/O span。Vortex 原生操作直接在完成 callback 中结束 span，返回原始 Future，不增加 consumer executor 调度。其他 Folly 操作通过完成 continuation 观测最终结果；若消费者丢弃已开始的操作，最后一个工作上下文释放时结束尚未完成的 span，并设置 `storage.completion.unobserved=true`，不把它当作真实取消。

CXX 将异常转换为 Result；Rust guard 在 Pending、Ready 和 panic unwind 时析构。attach token 不跨 poll/线程保留。桥接没有引入 Rust OTel SDK 或 exporter，也未新增公开 C 业务 ABI，FFI 导出表保持现有范围。

## 配置、属性与测量

`io_spans=false` 抑制细粒度逻辑 I/O 和 backend spans，保留操作及汇总。`max_spans_per_operation` 包含操作本身，最小有效值为 1。达到预算后复用父上下文，保留 `storage.spans.dropped` 以及根操作的 `storage.io.reads`、`storage.io.requested_bytes`、`storage.io.returned_bytes`。预算由一次操作的所有并行任务共享。

字节仅在逻辑文件层计数，backend span 不再累计字节；不能把根汇总与子 span 的字节重复相加。批量读取记录一个逻辑 span，返回字节为实际完成结果之和。短读记录实际字节，不擅自把合法 EOF 变成错误；原有 FFI 对短读的错误判定保持不变。

固定属性包括 `storage.operation`、`storage.format`、`storage.backend`、cache 状态、offset、请求/返回字节和 range_count。错误仅导出分类及已有扩展状态的 retryable，不导出原错误消息、对象路径、凭证、签名 URL、查询或数据。

`tracing_benchmark` 独立目标比较无 parent/provider、有效 parent + null provider、未采样 parent、采样 parent 四种运行成本。它测量作用域及四个子操作的 CPU/wall time，不代表完整读取的延迟或内存占用；实际服务的分配、并发和尾延迟仍需宿主联调测量。

完整读取的引入前/后 Release 对比使用 `telemetry_read_benchmark` 和配对运行器，场景、验收口径及本轮结果见 [Telemetry 性能验证](opentelemetry-storage-performance.md)。默认关闭路径有独立快路径；这本身不能替代性能验证，也不表示有 parent 或启用采样时没有成本。

## 明确的边界

- 本次没有 Milvus checkout 改动，也没有更改 Milvus 锁定的 Storage 版本。跨仓库部署和对象存储故障联调属于设计中的 S3。
- backend request spans 当前覆盖 CRT async head/get；其他 provider、同步 SDK 的内部重试、Talon 远程服务内部仍只有逻辑层/外层 span，不能据此宣称网络请求全部可见。
- Vortex 使用 Storage 提供的 runtime adapter 的子任务已接入。Lance/DataFusion、Iceberg/Paimon 等第三方自行 spawn 的任务、外部 Arrow executor 和引擎内部 codec spans 不自动继承；未替换全局 OTel TLS，也没有安装第三方引擎 hook。
- 已有 API 不提供完整取消桥接；丢弃 Future 不表示远程或 Tokio 工作已取消。
- Python 构建的现有 FFI-only 符号策略保持不变；本公共 provider 接口是常规 C++ 宿主接口，不是 Python/Rust 可直接传递的稳定 ABI。

## 验证

编译、格式化和测试均在 `milvusdb/milvus-env:ubuntu22.04-20260714-c135601` 容器内执行，挂载要求的 Conan 缓存到 `/root/.conan2`，另挂载当前构建使用的依赖缓存。当前构建开启 ASAN、coverage、FIU，默认关闭 CRT。

```sh
cmake --build cpp/build/Release --target milvus_test Test_FFI tracing_benchmark -j 6
cd cpp
. build/Release/generators/conanrun.sh
export ASAN_OPTIONS=detect_leaks=0
build/Release/test/milvus_test --gtest_filter='StorageTracingTest.*:FollyArrowExecutorTest.*:*APIWriterReaderTest*:*FormatReaderMetadataCacheParamTest*:*FormatReaderTest*:*LocalFileSystem*:*CloudFSMetrics*-*ParquetOpenAsyncIsLazyAndSupportsThreadedExecutor*'
build/Release/test/Test_FFI
```

初次接入时的验证结果：三个构建目标均通过；上述筛选回归运行 394 项，其中 **325 项通过、69 项按原有格式/配置条件跳过**，包含新增的 **17 项 tracing 测试全部通过**；C FFI **83 项通过、0 失败**。`VortexOpenAsyncUsesTokioRuntime/1` 与新增 consumer executor 调度测试均通过。C++ clang-format 18、Rust rustfmt 和 `git diff --check` 通过。本轮修复后的 21 项 tracing 用例及回归结果记录在性能验证文档中。

测试覆盖 parent 隔离/嵌套/fiber 恢复、provider 快照/未采样传播、惰性 Future、完成竞争、native Future 丢弃/同步完成、I/O 重载与开关、Arrow 跨线程提交、singleflight link、共享 Reader 的 Parquet/Vortex 并发，以及错误分类与敏感消息不导出。

本次原始日志：`/tmp/storage-tracing-build-confirm.log`、`/tmp/storage-tracing-regression-final.log`、`/tmp/storage-tracing-ffi-final.log`；环境探针为 `/tmp/storage-threaded-probe.cpp` 与 `/tmp/storage-threaded-probe.log`。

额外编译覆盖：Rust bridge 的 `s3-crt-async` feature 编译，以及 C++ CRT 源文件在 AWS SDK 1.11.842 头文件下的条件路径语法编译。后者不是完整 CRT 链接或真实 S3 请求测试。

当前环境的 `ParquetOpenAsyncIsLazyAndSupportsThreadedExecutor` 参数化用例被排除：组合运行会崩溃，不含任何 Storage tracing 代码的独立探针确认，头文件给出的 `sizeof(folly::ThreadedExecutor)` 为 256，而预编译构造函数写到对象起始位置之后的第 415 字节。预编译依赖使用 GCC 14，容器编译器是 GCC 12；已确认对象布局不一致，尚未确定造成布局差异的全部编译选项。需在依赖与编译环境一致后复验，不能将本次筛选回归等同于全量通过。

基准命令：`tracing_benchmark --benchmark_filter=BM_StorageTracing --benchmark_min_time=0.1s --benchmark_repetitions=3 --benchmark_out=/tmp/storage-tracing-benchmark.json --benchmark_out_format=json`。2026-09-09 三次测量的 wall time 中位数如下（每次包含 parent scope 及四个子操作）：

| 模式 | ns/operation |
| --- | ---: |
| 无 parent/provider | 1,584 |
| 有 parent、null provider | 19,826 |
| 未采样 parent | 34,884 |
| 采样 parent | 53,605 |

这些数值来自开启 ASAN/coverage 且启用 CPU scaling 的验证环境，仅用于本实现四种模式的成本对照，不能作为生产吞吐或尾延迟承诺。原始结果保存在 `/tmp/storage-tracing-benchmark.json`。
