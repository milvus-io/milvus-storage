# Telemetry 局部优化与验证

本轮先将现有 Telemetry 接入、修复和 benchmark 提交为 `6966d49`，再优化 tracing 内部实现。所有前后对照均以该提交为基线。实现改动集中在 6 个文件：runtime、文件 tracing、metadata cache 和 Rust 私有桥接；另新增 5 项回归测试。

## 实现

- 文件读取及 Run/RunAsync 使用 `HasContext()` 判断当前上下文，避免仅检查空值时构造 owning shared_ptr；文件同步/异步读取的 recording 部分单独放入不内联的函数，缩小关闭路径的栈帧。
- 子操作在分配 SpanState 前检查 null provider、I/O 开关和 span 预算；根操作仍捕获固定配置，保留后续注入 provider 时的行为。
- SpanState 首次启动完成后用 release/acquire 标志发布 span 指针或禁用决定，后续 Start 不再重复取 mutex。标志放在现有 bool 字段旁的 padding 中；仍保留首次启动和结束的互斥保护。
- metadata 无上下文时不分配 leader OperationTrace；无上下文 follower 直接等待已有结果。带 tracing 的 follower 可等待未带 tracing 的 leader，此时不创建无效 link。
- Rust capture 的 opaque handle 直接持有 immutable Context；attachment 的私有派生对象直接持有 ContextScope，删除两处独立 Impl 分配。CXX 函数签名及 Rust 侧调度行为保持原有约定。

Scope 复用、跳过所有未采样子 span，以及修改 executor/重复装箱路径，本轮均未实施。其他 RequestContext key 的隔离、空上下文遮蔽和宿主 sampler 的决策继续由现有路径保障。

## 构建与功能验证

使用 `storage-tracing-dev`，镜像 `milvusdb/milvus-env:ubuntu22.04-20260714-c135601`，按约定挂载 Conan 缓存。所有编译、格式化和测试在容器内执行。切换源码树前仅清理 Rust 根 crate，再重编；日志确认各自的 `Compiling rust-bridge` 源码路径。

- C++ 筛选回归：403 项，334 通过、69 项原有条件跳过。
- 26 项 tracing 测试全部通过；新增覆盖并发惰性启动、AlwaysOn 对未采样 parent 的处理、其他 RequestContext key 的恢复、Rust token 跨线程生命周期、带/不带 tracing 的 metadata 调用混用。
- C FFI：83 项通过、0 失败。
- 原有 Folly `ThreadedExecutor` ABI 问题用例继续排除，具体原因见 [实现说明](opentelemetry-storage-implementation.md)。本次不是全量测试或远端存储验证。

日志与源码指纹位于 `/tmp/storage-telemetry-opt-6966d49`：`tests-build.log`、`tests.log`、`ffi.log`、`release-build.log`、`inputs.json`。

## 分配与 jemalloc 实际占用

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

## 时间对照

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

## 端到端读取

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
