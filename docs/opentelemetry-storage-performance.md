# Telemetry 完整读取性能验证

提交 `6966d49` 后的局部优化、jemalloc 实际占用和前后对照结果见 [优化验证](opentelemetry-storage-optimization.md)。

## 验收口径

比较引入 Telemetry 之前的 Storage 与当前工作树，主要验收场景是未注入 parent/provider 的默认路径。另行测量有 parent 但无 provider、未采样、已采样三种模式，不能把它们混入默认关闭模式的结果。

“未发现回退”需要配对测量和足够窄的置信区间支持，单次更快或差异不显著都不能证明零开销。脚本中的 2% 是测量灵敏度边界，不是用户同意的回退额度。置信区间跨过该边界时结果为 `inconclusive`；区间下界大于 0 时标记 `detected_regression`。区间为各场景各自的 95% 区间，并非整个测试矩阵的同时置信区间。

## 测量对象

- 基线：`43b7793406136208928ad20dc1596b510c7b629a`，仅添加同一份 benchmark harness 和对应 CMake target。
- 当前：包含 Telemetry 及本轮修复的未提交工作树；每次运行保存 executable 和实际构建目录下 shared library 的 SHA-256。
- 独立 Release 构建：`WITH_ASAN=OFF`、`WITH_UT=OFF`、`WITH_FIU=OFF`、`WITH_TALON=OFF`、`WITH_BENCHMARK=ON`，无 coverage flags。不使用已有 ASAN/coverage 构建的性能数据。
- 开发容器：`milvusdb/milvus-env:ubuntu22.04-20260714-c135601`，按验证约定挂载 Conan 缓存；两版本使用相同工具链与依赖。
- CPU：Xeon Gold 6338；固定亲和性 `8,9`，Storage/Tokio/benchmark executor 各配置两个 worker。系统使用 `ondemand` 调频，未修改宿主调频策略，仍存在共享机器噪声。
- 数据：同一批固定种子的 65,536 行，`int64 id`、`float64 value`、32 维 `float32` 向量；Parquet/Vortex 各一个 column group。基线生成一次，所有版本复用同一文件与 manifest，记录数据 SHA-256。
- 25 个场景：两种格式的同步/异步 take（64/1,024 行，metadata 冷/热）、完整扫描、同步/异步 chunk 读取，以及 64/4,096/65,536 字节本地 `ReadAt`。
- metadata 冷表示每次创建 Reader/cache；文件数据已在 OS page cache 中，不代表冷磁盘。测试检查 take 返回 ID/顺序及 chunk 行数一致。
- 主指标：每操作 wall time 和整个进程的 CPU time；配对比值取几何均值，并计算 log-ratio 的 Student t 区间。进程峰值 RSS 单独保存，不等于每操作分配量。
- 已采样模式使用容量为 1 的内存 exporter 与 `SimpleSpanProcessor`，包括本地 span 构造/处理成本，不包括网络导出。

模式 0 的进程从未调用 `AttachParent`；不代表启用过 tracing 后再次关闭的所有历史状态。模式 1/2/3 在每次操作外建立 parent scope，其耗时计入操作，因此小块 `ReadAt` 也包含每次注入 parent 的成本。

## 可复现入口

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

## 本轮记录

原始构建、输入和运行记录保存在 `/tmp/storage-telemetry-perf-20260909`。早期 `initial-disabled-v1` 至 `v5` 均排除：前几次有 harness 错误，v5 虽运行完成但复用了错误的 Rust 桥接产物；原因记录在 `INVALID-EARLY-RUNS.md`，原始记录保留。

本轮消除了默认无上下文时多余的 trace 完成 continuation、属性设置与 Folly TLS 查询，并将 Rust future 的内部 pin 从额外 heap allocation 改为 wrapper 内 pin。保留空上下文遮蔽、跨线程恢复和已有操作 provider 快照语义。

语义回归在强制重编 Rust bridge 后通过：21 项 tracing 用例全部通过，筛选 C++ 回归 329 项通过、69 项原有条件跳过，C FFI 83 项通过。日志为 `optimized-tests-verified.log`、`optimized-ffi.log`。被排除的 Folly `ThreadedExecutor` ABI 用例和验证范围见 [实现说明](opentelemetry-storage-implementation.md)。

本地读取结果不能外推到 S3/Talon、网络 exporter、生产并发或 p95/p99 尾延迟；这些需要对应环境的负载测试。
