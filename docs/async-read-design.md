# Async Read Design: Cross-Column-Group Parallel I/O

## Problem

The current read path serializes I/O across column groups (CGs):

```
load_internal() / ReaderImpl::take()
  └─ for each CG (serial, sync barrier between CGs):
       └─ get_chunks() / take()
            └─ split into N tasks → thread pool → wait all
                 └─ read_chunks_from_files() / take_rows_from_files()
                      └─ FormatReader::read_with_range() (blocking I/O)
```

The thread pool is a fixed-size singleton (`folly::IOThreadPoolExecutor`).
Each CG monopolizes the entire pool, then waits for completion before the next CG starts. CGs never overlap.

Additionally, the current `split_chunks` / `split_row_indices` split by item count,
not by file locality. This causes redundant I/O — the same file may be cloned and
opened by multiple tasks independently.

## Solution: Centralized Task Planning + Async Execution

Two key changes:

1. **Centralized planning in reader.cpp**: reader.cpp has the global view (all CGs,
   thread pool size). It collects tasks from all CGs, decides whether to split, then
   submits them all at once. CG readers no longer make parallelism decisions internally.

2. **`_async` methods propagated from CG reader down to FormatReader**:
   The async chain runs through all layers. If the FormatReader implementation natively
   supports async reads, it can be called directly.

### Architecture

```
reader.cpp (calling thread):
  1. Collect natural tasks from all CGs  → flat task list
  2. Compare task count vs available threads → maybe split large tasks
  3. Submit all tasks via folly::via       → flat queue in thread pool
  4. collectAll + aggregate results
```

```
┌─────────────────────────────────────────────────────────────┐
│ reader.cpp — centralized scheduler                          │
│                                                             │
│  ┌─ CG0.get_natural_tasks() → [{file2, range[0,100]},      │
│  │                              {file5, range[200,300]}]    │
│  ├─ CG1.get_natural_tasks() → [{file0, range[0,500]}]      │
│  └─ CG2.get_natural_tasks() → [{file1, range[0,50]}]       │
│                                                             │
│  flat task list: 4 tasks                                    │
│  available threads: 8 → 4 < 8, split large tasks           │
│  final task list: 8 tasks                                   │
│                                                             │
│  submit all 8 tasks via folly::via(executor)                │
└─────────────────────────────────────────────────────────────┘
         │  │  │  │  │  │  │  │
         ▼  ▼  ▼  ▼  ▼  ▼  ▼  ▼
     ┌──────────────────────────────┐
     │   folly::IOThreadPoolExecutor │
     │   8 threads, flat task queue  │
     └──────────────────────────────┘
```

### Async Call Chain

```
ChunkReaderImpl::get_chunks()          ← centralized scheduling entry in reader.cpp
  └─ ColumnGroupReader::get_chunks_async()
       └─ FormatReader::read_with_range_async()  ← lowest-level async

ReaderImpl::take()                     ← centralized scheduling entry in reader.cpp
  └─ ColumnGroupLazyReader::take_async()
       └─ FormatReader::take_async()             ← lowest-level async
```

### Why It Works

- **No nesting**: reader.cpp submits leaf-level tasks directly. Tasks never spawn sub-tasks.
- **No deadlock**: thread pool threads only do I/O work, never wait on other tasks.
- **File-aware splitting**: by default, without splitting, each file is only opened once.
  When the thread pool has idle threads, a same-file task may be split for higher parallelism.
- **Adaptive granularity**: when threads > tasks, large tasks are split (at row group
  boundaries). When threads <= tasks, merged ranges are kept intact for I/O merging benefit.

## Interface Change Summary

### Change Principles

- reader.cpp layer (`ChunkReaderImpl::get_chunks`, `ReaderImpl::take`): centralized scheduling, calls `_async` methods
- ColumnGroupReader / ColumnGroupLazyReader:
  - Existing sync methods drop the `parallelism` parameter, becoming pure single-threaded sync execution
  - New `_async` methods that internally call FormatReader's `_async` methods
  - New `get_natural_tasks` for task planning
- FormatReader: new `_async` methods (lowest-level async boundary)

### Interface Comparison by Layer

| Layer | Current Interface | After Refactoring |
|-------|------------------|-------------------|
| FormatReader | `read_with_range(start, end)` | Keep + add `read_with_range_async(executor, start, end)` |
| FormatReader | `take(row_indices)` | Keep + add `take_async(executor, row_indices)` |
| ColumnGroupReader | `get_chunks(indices, parallelism)` | `get_chunks(indices)` (drop parallelism) + add `get_chunks_async(ChunkTask)` + `get_natural_tasks` + `get_chunk_info` |
| ColumnGroupLazyReader | `take(indices, parallelism)` | `take(indices)` (drop parallelism) + add `take_async(TakeTask)` + `get_natural_tasks` |
| ChunkReaderImpl | `get_chunks(indices, parallelism)` | Internally refactored to centralized scheduling + calls `get_chunks_async` |
| ReaderImpl | `take(indices, parallelism)` | Internally refactored to centralized scheduling + calls `take_async` |

## FormatReader Async Interface

FormatReader is the lowest level of the async chain. New `_async` methods execute on
a cloned reader. If the FormatReader implementation natively supports async (e.g.,
the underlying storage supports async reads), native async can be used directly;
otherwise, sync methods are wrapped with `folly::via(executor)`.

```cpp
// format_reader.h — new async interface
class FormatReader {
  public:
  // ... existing sync interface preserved ...

  // Async read of a specified row range (must be called on a cloned reader)
  [[nodiscard]] virtual folly::SemiFuture<arrow::Result<std::shared_ptr<arrow::RecordBatchReader>>>
  read_with_range_async(std::shared_ptr<folly::ThreadPoolExecutor> executor,
                        uint64_t start_offset, uint64_t end_offset) {
      // Default implementation: wrap sync method
      return folly::via(executor.get(),
          [this, start_offset, end_offset, executor]()
              -> arrow::Result<std::shared_ptr<arrow::RecordBatchReader>> {
              return this->read_with_range(start_offset, end_offset);
          });
  }

  // Async take (must be called on a cloned reader)
  [[nodiscard]] virtual folly::SemiFuture<arrow::Result<std::shared_ptr<arrow::Table>>>
  take_async(std::shared_ptr<folly::ThreadPoolExecutor> executor,
             const std::vector<int64_t>& row_indices) {
      return folly::via(executor.get(),
          [this, row_indices, executor]()
              -> arrow::Result<std::shared_ptr<arrow::Table>> {
              return this->take(row_indices);
          });
  }
};
```

**Note**: FormatReader is NOT thread-safe. `_async` methods must be called on instances
produced by `clone_reader()`. Cloning is performed by the upper layer (ColumnGroupReader /
ColumnGroupLazyReader) before submitting the async operation.

## Task Planning: `get_natural_tasks`

CG readers expose a planning method that returns work units at file × merged_range
granularity. This is pure metadata computation, no I/O involved.

### ColumnGroupReader

ChunkTask carries file-level metadata (file_index and merged row range),
so `get_chunks_async` can use it directly without redundant grouping.

```cpp
// column_group_reader.h
struct ChunkTask {
    size_t file_index;                   // which file this task belongs to
    std::vector<int64_t> chunk_indices;  // chunks in this task (same file, contiguous range)
    uint64_t range_start;                // merged row range start offset in file
    uint64_t range_end;                  // merged row range end offset in file
};

// Returns tasks grouped by file × merged_range.
// Each task contains contiguous chunks from the same file.
virtual std::vector<ChunkTask> get_natural_tasks(
    const std::vector<int64_t>& chunk_indices) = 0;
```

Implementation:

```cpp
std::vector<ChunkTask> ColumnGroupReaderImpl::get_natural_tasks(
    const std::vector<int64_t>& chunk_indices) {

    // Group by file, then merge contiguous chunks within each file
    std::map<size_t, std::vector<int64_t>> file_groups;
    for (auto idx : chunk_indices) {
        file_groups[chunk_infos_[idx].file_index].push_back(idx);
    }

    std::vector<ChunkTask> tasks;
    for (auto& [file_idx, chunks] : file_groups) {
        // Break at non-contiguous positions within a file
        auto& first_info = chunk_infos_[chunks[0]];
        ChunkTask current;
        current.file_index = file_idx;
        current.chunk_indices.push_back(chunks[0]);
        current.range_start = first_info.row_offset_in_file;
        current.range_end = first_info.row_offset_in_file + first_info.number_of_rows;

        for (size_t i = 1; i < chunks.size(); i++) {
            auto& prev = chunk_infos_[chunks[i - 1]];
            auto& curr = chunk_infos_[chunks[i]];

            if (curr.row_offset_in_file == prev.row_offset_in_file + prev.number_of_rows) {
                // contiguous → merge into current task
                current.chunk_indices.push_back(chunks[i]);
                current.range_end = curr.row_offset_in_file + curr.number_of_rows;
            } else {
                // non-contiguous → start new task
                tasks.push_back(std::move(current));
                current = ChunkTask{
                    .file_index = file_idx,
                    .chunk_indices = {chunks[i]},
                    .range_start = curr.row_offset_in_file,
                    .range_end = curr.row_offset_in_file + curr.number_of_rows,
                };
            }
        }
        tasks.push_back(std::move(current));
    }
    return tasks;
}
```

### ColumnGroupLazyReader

```cpp
// column_group_lazy_reader.h
struct TakeTask {
    uint32_t file_index;                    // which file this task belongs to
    std::vector<int64_t> row_indices;       // global row indices in this task (same file)
    std::vector<size_t> original_positions; // each row's position in the caller's row_indices array
};

virtual std::vector<TakeTask> get_natural_tasks(
    const std::vector<int64_t>& row_indices) = 0;
```

`get_natural_tasks` groups row indices by file while recording each row's position
in the original `row_indices` array, enabling `ReaderImpl::take` to restore row order
after cross-CG merging:

```cpp
std::vector<TakeTask> ColumnGroupLazyReaderImpl::get_natural_tasks(
    const std::vector<int64_t>& row_indices) {

    const auto& files = column_group_->files;
    // file_idx -> [(global_row_index, original_position)]
    std::map<uint32_t, std::vector<std::pair<int64_t, size_t>>> file_groups;

    for (size_t pos = 0; pos < row_indices.size(); ++pos) {
        auto [file_idx, _] = get_index_and_offset_of_file(files, row_indices[pos]).ValueOrDie();
        file_groups[file_idx].push_back({row_indices[pos], pos});
    }

    std::vector<TakeTask> tasks;
    for (auto& [file_idx, rows_and_positions] : file_groups) {
        TakeTask task;
        task.file_index = file_idx;
        task.row_indices.reserve(rows_and_positions.size());
        task.original_positions.reserve(rows_and_positions.size());
        for (auto& [row, pos] : rows_and_positions) {
            task.row_indices.push_back(row);
            task.original_positions.push_back(pos);
        }
        tasks.push_back(std::move(task));
    }
    return tasks;
}
```

## CG Reader Async Execution Interface

CG readers no longer decide parallelism. They receive a pre-planned chunk/row list
and execute it as a single work unit. No `parallelism` parameter, no internal splitting.
Internally they call FormatReader's `_async` methods.

### ColumnGroupReader

```cpp
// column_group_reader.h

// Synchronous method: no parallelism, single-threaded execution
virtual arrow::Result<std::vector<std::shared_ptr<arrow::RecordBatch>>>
get_chunks(const std::vector<int64_t>& chunk_indices) = 0;

// Async method: accepts a pre-grouped ChunkTask (from get_natural_tasks),
// so no redundant file grouping or range merging is needed internally.
virtual folly::SemiFuture<arrow::Result<std::vector<std::shared_ptr<arrow::RecordBatch>>>>
get_chunks_async(const ChunkTask& task,
                 std::shared_ptr<folly::ThreadPoolExecutor> executor) = 0;
```

Async implementation — since `ChunkTask` already carries `file_index` and merged
`range_start/range_end`, `get_chunks_async` uses them directly without redundant
file grouping or range merging:

```cpp
folly::SemiFuture<arrow::Result<std::vector<std::shared_ptr<arrow::RecordBatch>>>>
ColumnGroupReaderImpl::get_chunks_async(
    const ChunkTask& task,
    std::shared_ptr<folly::ThreadPoolExecutor> executor) {

    // ChunkTask already provides: file_index, chunk_indices, range_start, range_end.
    // No file grouping or range merging needed — the caller (reader.cpp)
    // already did that via get_natural_tasks().

    auto cloned = format_readers_[task.file_index]->clone_reader();
    if (!cloned.ok()) {
        return folly::makeSemiFuture(
            arrow::Result<std::vector<std::shared_ptr<arrow::RecordBatch>>>(cloned.status()));
    }

    auto reader = cloned.MoveValueUnsafe();
    auto chunk_idxs = task.chunk_indices;
    auto* self = this;

    return reader->read_with_range_async(executor, task.range_start, task.range_end)
        .deferValue([reader, chunk_idxs, self](auto&& rb_reader_result)
            -> arrow::Result<std::vector<std::shared_ptr<arrow::RecordBatch>>> {
            ARROW_ASSIGN_OR_RAISE(auto rb_reader, std::move(rb_reader_result));
            ARROW_ASSIGN_OR_RAISE(auto rbs, rb_reader->ToRecordBatches());

            // Slice record batches back to individual chunks
            // (same logic as existing read_chunks_from_files)
            std::vector<std::shared_ptr<arrow::RecordBatch>> result;
            result.reserve(chunk_idxs.size());
            size_t rbs_idx = 0;
            size_t rbs_offset = 0;
            for (auto chunk_index : chunk_idxs) {
                const auto& chunk_info = self->chunk_infos_[chunk_index];
                auto rb = rbs[rbs_idx]->Slice(rbs_offset, chunk_info.number_of_rows);
                result.push_back(std::move(rb));
                rbs_offset += chunk_info.number_of_rows;
                if (rbs_offset == rbs[rbs_idx]->num_rows()) {
                    rbs_idx++;
                    rbs_offset = 0;
                }
            }
            return result;
        });
}
```

Simplified sync method (parallelism removed, directly calls `read_chunks_from_files`):

```cpp
arrow::Result<std::vector<std::shared_ptr<arrow::RecordBatch>>>
ColumnGroupReaderImpl::get_chunks(const std::vector<int64_t>& chunk_indices) {
    // Direct sync read, no parallel splitting
    ARROW_ASSIGN_OR_RAISE(auto chunk_rb_map, read_chunks_from_files(chunk_indices));

    std::vector<std::shared_ptr<arrow::RecordBatch>> result;
    result.reserve(chunk_indices.size());
    for (auto idx : chunk_indices) {
        result.push_back(chunk_rb_map[idx]);
    }
    return result;
}
```

### ColumnGroupLazyReader

```cpp
// column_group_lazy_reader.h

// Synchronous method: parallelism removed
virtual arrow::Result<std::shared_ptr<arrow::Table>>
take(const std::vector<int64_t>& row_indices) = 0;

// Async method: accepts a pre-planned TakeTask (single file), executes via
// FormatReader::take_async. No internal file grouping — the caller (reader.cpp)
// already did that via get_natural_tasks.
// Row order in the returned Table matches task.row_indices (no reordering within
// a single file). Row reordering (restoring original row_indices order) is the
// caller's (ReaderImpl::take) responsibility.
virtual folly::SemiFuture<arrow::Result<std::shared_ptr<arrow::Table>>>
take_async(const TakeTask& task,
           std::shared_ptr<folly::ThreadPoolExecutor> executor) = 0;
```

Async implementation (internally calls FormatReader's `_async`):

`TakeTask` already carries `file_index`, so `take_async` uses it directly without
redundant file grouping. Global-to-file-local row index conversion happens here.

```cpp
folly::SemiFuture<arrow::Result<std::shared_ptr<arrow::Table>>>
ColumnGroupLazyReaderImpl::take_async(
    const TakeTask& task,
    std::shared_ptr<folly::ThreadPoolExecutor> executor) {

    // prepare must complete before async submission (has mutex, opens metadata)
    auto prepare_status = prepare_format_readers(task.row_indices);
    if (!prepare_status.ok()) {
        return folly::makeSemiFuture(
            arrow::Result<std::shared_ptr<arrow::Table>>(prepare_status));
    }

    // Convert global row indices to file-local indices
    const auto& cg_files = column_group_->files;
    std::vector<int64_t> rows_in_file;
    rows_in_file.reserve(task.row_indices.size());
    for (auto global_row : task.row_indices) {
        auto [file_idx, row_in_file] =
            get_index_and_offset_of_file(cg_files, global_row).ValueOrDie();
        rows_in_file.push_back(row_in_file);
    }

    auto cloned = loaded_format_readers_[task.file_index]->clone_reader();
    if (!cloned.ok()) {
        return folly::makeSemiFuture(
            arrow::Result<std::shared_ptr<arrow::Table>>(cloned.status()));
    }
    auto reader = cloned.MoveValueUnsafe();

    return reader->take_async(executor, rows_in_file)
        .deferValue([reader](auto&& table_result)
            -> arrow::Result<std::shared_ptr<arrow::Table>> {
            return std::move(table_result);
        });
}
```

Simplified sync method (parallelism removed):

```cpp
arrow::Result<std::shared_ptr<arrow::Table>>
ColumnGroupLazyReaderImpl::take(const std::vector<int64_t>& row_indices) {
    ARROW_RETURN_NOT_OK(prepare_format_readers(row_indices));
    return take_rows_from_files(row_indices);
}
```

## Centralized Scheduling in reader.cpp

Only `ChunkReaderImpl::get_chunks` and `ReaderImpl::take` are modified.

### ChunkReaderImpl::get_chunks

```cpp
arrow::Result<std::vector<std::shared_ptr<arrow::RecordBatch>>>
ChunkReaderImpl::get_chunks(const std::vector<int64_t>& chunk_indices, size_t parallelism) {

    if (parallelism <= 1) {
        // Use sync method directly (parallelism removed)
        return chunk_reader_->get_chunks(chunk_indices);
    }

    auto executor = ThreadPoolHolder::GetThreadPool(parallelism);
    size_t available_threads = executor->numThreads();

    // Phase 1: collect natural tasks (already grouped by file × merged_range)
    auto all_tasks = chunk_reader_->get_natural_tasks(chunk_indices);

    // Phase 2: if threads are underutilized, split the largest task.
    // Splitting preserves file_index and recalculates range boundaries.
    while (all_tasks.size() < available_threads) {
        auto it = std::max_element(all_tasks.begin(), all_tasks.end(),
            [](auto& a, auto& b) {
                return a.chunk_indices.size() < b.chunk_indices.size();
            });
        if (it->chunk_indices.size() <= 1) break;

        size_t mid = it->chunk_indices.size() / 2;

        // The split point's chunk determines the range boundary
        auto& mid_info = chunk_reader_->get_chunk_info(it->chunk_indices[mid]);

        ChunkTask right;
        right.file_index = it->file_index;
        right.chunk_indices.assign(it->chunk_indices.begin() + mid, it->chunk_indices.end());
        right.range_start = mid_info.row_offset_in_file;
        right.range_end = it->range_end;

        it->chunk_indices.resize(mid);
        it->range_end = mid_info.row_offset_in_file;

        all_tasks.push_back(std::move(right));
    }

    // Phase 3: submit all tasks (pass ChunkTask directly, no re-grouping)
    std::vector<folly::SemiFuture<arrow::Result<std::vector<std::shared_ptr<arrow::RecordBatch>>>>> futures;
    // Track chunk_indices per task for result mapping
    std::vector<std::vector<int64_t>> task_chunk_lists;

    for (auto& task : all_tasks) {
        task_chunk_lists.push_back(task.chunk_indices);
        futures.push_back(
            chunk_reader_->get_chunks_async(task, executor));
    }

    // Phase 4: collectAll and map results back to original chunk order
    auto results = folly::collectAll(std::move(futures)).get();

    std::unordered_map<int64_t, std::shared_ptr<arrow::RecordBatch>> all_rbs;
    for (size_t i = 0; i < results.size(); ++i) {
        auto& tryResult = results[i];
        if (tryResult.hasException()) {
            return arrow::Status::IOError(tryResult.exception().what().toStdString());
        }
        ARROW_ASSIGN_OR_RAISE(auto rbs, std::move(tryResult.value()));

        // Map each result batch back to its chunk index
        auto& chunk_list = task_chunk_lists[i];
        assert(rbs.size() == chunk_list.size());
        for (size_t j = 0; j < rbs.size(); ++j) {
            all_rbs[chunk_list[j]] = std::move(rbs[j]);
        }
    }

    // Reorder results to match original chunk_indices order
    // Then apply slice logic (existing lines 500-542)
    // ...
}
```

### ReaderImpl::take

```cpp
arrow::Result<std::shared_ptr<arrow::Table>> ReaderImpl::take(
    const std::vector<int64_t>& row_indices, size_t parallelism,
    const std::shared_ptr<std::vector<std::string>>& needed_columns) {

    // ... validation, resolve columns, create lazy_readers (unchanged) ...

    auto executor = ThreadPoolHolder::GetThreadPool(parallelism);
    size_t available_threads = executor->numThreads();

    // Phase 1: collect natural tasks from all CGs, with original_positions
    //
    // Different CGs have different file layouts. get_natural_tasks groups by
    // file_index, which reorders rows relative to the original row_indices.
    // original_positions records each row's position in the original array,
    // used in Phase 5 for reordering to ensure cross-CG row alignment.
    struct Task {
        size_t cg_idx;
        TakeTask take_task;  // contains file_index, row_indices, original_positions
    };
    std::vector<Task> all_tasks;

    for (size_t i = 0; i < lazy_readers.size(); ++i) {
        auto natural = lazy_readers[i]->get_natural_tasks(row_indices);
        for (auto& task : natural) {
            all_tasks.push_back({i, std::move(task)});
        }
    }

    // Phase 2: if threads are underutilized, split the largest task
    //
    // Unlike the get_chunks path, take uses naive midpoint splitting rather
    // than row-group-boundary splitting. Reason: take is random access —
    // FormatReader::take() internally locates row groups by row index, and
    // rows are scattered. Even if the same row group is read by two tasks,
    // that's an inherent cost of splitting within a file, and row group
    // boundary alignment has limited benefit. Also, getting row group info
    // requires prepare_format_readers (which involves I/O), making it
    // unsuitable for the planning phase.
    //
    // When splitting, row_indices and original_positions are split in sync,
    // preserving their 1:1 correspondence. file_index stays the same
    // (both halves still belong to the same file).
    while (all_tasks.size() < available_threads) {
        auto it = std::max_element(all_tasks.begin(), all_tasks.end(),
            [](auto& a, auto& b) {
                return a.take_task.row_indices.size() < b.take_task.row_indices.size();
            });
        if (it->take_task.row_indices.size() <= 1) break;

        size_t mid = it->take_task.row_indices.size() / 2;
        Task right{
            it->cg_idx,
            TakeTask{
                it->take_task.file_index,
                {it->take_task.row_indices.begin() + mid, it->take_task.row_indices.end()},
                {it->take_task.original_positions.begin() + mid, it->take_task.original_positions.end()}
            }
        };
        it->take_task.row_indices.resize(mid);
        it->take_task.original_positions.resize(mid);
        all_tasks.push_back(std::move(right));
    }

    // Phase 3: submit all tasks (call _async methods, passing TakeTask)
    // Use parallel vectors to track each future's CG index and original_positions
    std::vector<size_t> task_cg_indices;
    std::vector<std::vector<size_t>> task_positions;
    std::vector<folly::SemiFuture<arrow::Result<std::shared_ptr<arrow::Table>>>> futures;
    task_cg_indices.reserve(all_tasks.size());
    task_positions.reserve(all_tasks.size());
    futures.reserve(all_tasks.size());

    for (auto& task : all_tasks) {
        task_cg_indices.push_back(task.cg_idx);
        task_positions.push_back(std::move(task.take_task.original_positions));
        futures.push_back(
            lazy_readers[task.cg_idx]->take_async(task.take_task, executor));
    }

    // Phase 4: collectAll, group results by CG, and collect original_positions
    auto all_results = folly::collectAll(std::move(futures)).get();

    std::vector<std::vector<std::shared_ptr<arrow::Table>>> per_cg_tables(lazy_readers.size());
    std::vector<std::vector<size_t>> per_cg_positions(lazy_readers.size());

    for (size_t i = 0; i < all_results.size(); ++i) {
        auto& tryResult = all_results[i];
        if (tryResult.hasException()) {
            return arrow::Status::IOError(
                tryResult.exception().what().toStdString());
        }
        ARROW_ASSIGN_OR_RAISE(auto table, std::move(tryResult.value()));
        size_t cg_idx = task_cg_indices[i];
        per_cg_tables[cg_idx].push_back(std::move(table));
        per_cg_positions[cg_idx].insert(
            per_cg_positions[cg_idx].end(),
            task_positions[i].begin(), task_positions[i].end());
    }

    // Phase 5: merge per-CG tables, reorder rows to match original row_indices,
    // then align columns
    //
    // Each CG's concatenated row order is determined by get_natural_tasks'
    // file grouping. Different CGs have different file layouts → different
    // row orders. Use original_positions to build a reorder index and
    // arrow::compute::Take to restore the original order.
    std::vector<std::shared_ptr<arrow::Table>> tables;
    for (size_t cg = 0; cg < lazy_readers.size(); ++cg) {
        if (per_cg_tables[cg].empty()) continue;
        ARROW_ASSIGN_OR_RAISE(auto concatenated,
            arrow::ConcatenateTables(per_cg_tables[cg]));

        // reorder[original_pos] = offset_in_concatenated
        auto& positions = per_cg_positions[cg];
        std::vector<int64_t> reorder(row_indices.size());
        for (size_t i = 0; i < positions.size(); ++i) {
            reorder[positions[i]] = static_cast<int64_t>(i);
        }
        auto indices_array = std::make_shared<arrow::Int64Array>(
            row_indices.size(), arrow::Buffer::Wrap(reorder));
        ARROW_ASSIGN_OR_RAISE(auto reordered,
            arrow::compute::Take(concatenated, indices_array));
        tables.push_back(reordered.table());
    }

    // ... existing column alignment + projection logic ...
}
```

## Scenario Walkthrough

### Scenario 1: 10 rows scattered across 7 files, 3 CGs, 8 threads

```
Phase 1: get_natural_tasks
  CG0: {f2:[r3,r5]}, {f7:[r1]}, {f12:[r8,r9]} → 3 tasks
  CG1: {f0:[r3,r5]}, {f3:[r1]}, {f9:[r8,r9]} → 3 tasks
  CG2: {f1:[r3,r5]}, {f4:[r1]}               → 2 tasks
  Total: 8 tasks

Phase 2: 8 tasks == 8 threads → no splitting needed

Phase 3: submit 8 tasks, all 8 threads busy
  Each task: clone 1 reader, read 1 file, 0 redundant I/O
```

### Scenario 2: 10 rows all in 1 file (contiguous), 3 CGs, 8 threads

```
Phase 1: get_natural_tasks
  CG0: {f42:[r0..r9]} → 1 task
  CG1: {f42:[r0..r9]} → 1 task
  CG2: {f42:[r0..r9]} → 1 task
  Total: 3 tasks

Phase 2: 3 tasks < 8 threads → split
  Split CG0 task (10 rows) → 2 tasks of 5
  Split CG1 task (10 rows) → 2 tasks of 5
  Split CG2 task (10 rows) → 2 tasks of 5
  Now 6 tasks < 8 → split again
  Split one 5-row task → 3 + 2
  Split another 5-row task → 3 + 2
  Now 8 tasks == 8 threads → done

Phase 3: submit 8 tasks, all 8 threads busy
  Each task: clone reader, read 2-3 row groups in parallel
```

### Scenario 3: 1000 chunks across 50 files, 3 CGs, 8 threads

```
Phase 1: get_natural_tasks
  CG0: 50 file groups, some with multiple merged ranges → ~60 tasks
  CG1: similar → ~55 tasks
  CG2: similar → ~58 tasks
  Total: ~173 tasks

Phase 2: 173 > 8 → no splitting, keep merged ranges for I/O merging

Phase 3: submit 173 tasks, thread pool queues them
  8 threads process tasks as they complete, natural load balancing
  I/O merging preserved within each task
```

## Design Notes

### Responsibility Split

| Layer | Responsibility |
|-------|---------------|
| reader.cpp (ChunkReaderImpl / ReaderImpl) | Global scheduling: collect tasks, decide splitting, submit, aggregate results |
| ColumnGroupReader / ColumnGroupLazyReader | Three things: report natural tasks + sync execution + async execution (calls FormatReader async) |
| FormatReader | Lowest-level I/O: provides sync and async interfaces. Async defaults to wrapping sync; subclasses can override with native async |

CG readers don't know about other CGs or the thread pool size.
reader.cpp doesn't know about file internals — it gets pre-computed tasks.

### SemiFuture vs Future

Use `SemiFuture` at interface boundaries (lazy, no executor affinity).
The caller decides where continuations run via `.via(executor)`.
This is folly best practice.

### Executor Lifetime

`ThreadPoolHolder::GetThreadPool()` returns `shared_ptr<ThreadPoolExecutor>`.
When the singleton exists, this is safe. When it creates a temporary pool, the shared_ptr
could be destroyed before futures complete. Fix: **capture `executor` by value
(shared_ptr copy) in lambda/deferValue closures** to extend lifetime.

### ARROW_RETURN_NOT_OK Incompatibility

`ARROW_RETURN_NOT_OK` expands to `return arrow::Status(...)`, which is incompatible with
functions returning `SemiFuture<Result<T>>`. In async functions, handle errors explicitly:

```cpp
auto status = some_arrow_call();
if (!status.ok()) {
    return folly::makeSemiFuture(arrow::Result<T>(status));
}
```

### Memory Control

`load_internal` Phase 1 (min-heap + `memory_usage_limit_`) determines which chunks to load
**before** task planning. Memory upper bound is unchanged.

### Error Handling

Use `collectAll` to wait for all futures uniformly, then check exceptions one by one:

```cpp
auto all_results = folly::collectAll(std::move(futures)).get();
for (size_t i = 0; i < all_results.size(); ++i) {
    auto& tryResult = all_results[i];
    if (tryResult.hasException()) {
        return arrow::Status::IOError(
            tryResult.exception().what().toStdString());
    }
    ARROW_ASSIGN_OR_RAISE(auto result, std::move(tryResult.value()));
    // use result
}
```

Note: `collectAll` waits for all futures to complete (even if some fail),
avoiding the head-of-line blocking issue caused by sequential `.get()` calls.
