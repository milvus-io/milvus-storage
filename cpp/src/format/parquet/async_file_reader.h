// Copyright 2026 Zilliz
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <arrow/filesystem/filesystem.h>
#include <folly/futures/Future.h>
#include <parquet/arrow/reader.h>
#include <mutex>

namespace milvus_storage::parquet::detail {

// Keeps issued reads alive until an operation has drained, including early failures.
struct PendingReads {
  void Add(arrow::Future<> read);
  arrow::Future<> Drain();

  std::mutex mutex;
  std::vector<arrow::Future<>> reads;
};

// Owns an independently opened reader; encrypted footer metadata must not be shared.
struct OpenedPackedFile {
  std::shared_ptr<::parquet::arrow::FileReader> reader;
  bool encrypted_footer;
  std::shared_ptr<PendingReads> pending_reads;
};

// Defers open, size and footer I/O, retaining inputs and inheriting the consumer executor.
folly::SemiFuture<arrow::Result<OpenedPackedFile>> OpenPackedFileAsync(
    std::shared_ptr<arrow::fs::FileSystem> fs,
    std::string path,
    ::parquet::ReaderProperties reader_properties,
    ::parquet::ArrowReaderProperties arrow_properties,
    std::shared_ptr<::parquet::FileMetaData> plaintext_metadata = nullptr);

// Materializes only the selected leaf columns after asynchronous pre-buffering.
folly::SemiFuture<arrow::Result<std::shared_ptr<arrow::Table>>> ReadPackedRowGroupsAsync(
    OpenedPackedFile opened, std::vector<int> row_groups, std::vector<std::string> columns);

}  // namespace milvus_storage::parquet::detail
