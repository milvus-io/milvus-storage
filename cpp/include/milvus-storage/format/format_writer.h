// Copyright 2023 Zilliz
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <memory>
#include <vector>

#include <arrow/status.h>
#include <arrow/result.h>
#include <arrow/record_batch.h>

#include <string_view>

#include "milvus-storage/column_groups.h"
#include "milvus-storage/common/log.h"

namespace milvus_storage {

class FormatWriter {
  public:
  virtual ~FormatWriter() = default;
  // Every non-OK operation is terminal. Retry by destroying this writer and
  // opening another writer on a new file path.
  /**
   * @brief Write a record batch to the file
   *
   * @param record Record batch to write
   * @return arrow::Status
   */
  [[nodiscard]] virtual arrow::Status Write(const std::shared_ptr<arrow::RecordBatch> record) = 0;

  /**
   * @brief Flush the writer
   *        After call this function, the data should no longer exist in memory
   * @return arrow::Status
   */
  [[nodiscard]] virtual arrow::Status Flush() = 0;

  /**
   * @brief Finish with this writer
   *
   * "I am done", not "finish the file". A healthy writer is finalized and its
   * file published; a writer that has ALREADY FAILED is abandoned instead --
   * releasing whatever it allocated in the store -- and the first failure comes
   * back unchanged. Either way the writer is spent and cannot be used again.
   *
   * Abandoning on failure is not the same as finalizing on failure: nothing is
   * published, so this does not contradict "a failed writer must not be
   * finished". It exists because Close() is the only verb some callers can
   * reach for. An FFI, Go, Java or Python caller has no destructor to fall
   * back on,
   * and every failure path that returned without releasing used to leave an S3
   * multipart upload behind holding parts that no bucket listing can name.
   *
   * What Close() cannot express is abandoning a HEALTHY writer -- publishing a
   * file nobody wants is worse than leaking an upload, because the file is not
   * in the manifest and the manifest is what a GC walks. That is Abort()'s job.
   */
  [[nodiscard]] virtual arrow::Result<api::ColumnGroupFile> Close() = 0;

  /**
   * @brief Give up on this file and release what it already allocated
   *
   * The counterpart of Close(): Close() finishes the file, Abort() abandons it.
   * A writer that has failed cannot be closed -- the format may have written
   * half a footer, and finishing it would publish a file nobody should read --
   * but abandoning it is not free either. An S3 multipart upload holds every
   * part already sent, and those parts are invisible to a bucket listing, so
   * only the writer that created the upload can ever release them.
   *
   * Implementations must be safe to call on an already-failed writer, must not
   * report their own cleanup failures (the caller is already handling one --
   * log and drop), and must be idempotent. In particular Abort() after a
   * SUCCESSFUL Close() is a no-op in every respect, including the recorded
   * status: that is what lets a caller abandon unconditionally without first
   * working out which of the two verbs it owes.
   *
   * It returns void on purpose. A verb whose contract is "always succeeds"
   * has nothing to report, and an arrow::Status return invited two mistakes at
   * once: callers checking a value that never carries information, and
   * implementations smuggling a real failure through it.
   *
   * noexcept for the same reason. Anything that escapes here would escape into
   * a destructor; an allocation failure while abandoning a writer takes the
   * process down rather than leaving a caller to guess what was released.
   *
   * The WRITER'S OWN destructor still must not do storage I/O -- it can run at
   * shutdown, after the executor and client are gone. Its owner calls Abort()
   * explicitly from its own failure path (R2.7b).
   */
  virtual void Abort() noexcept = 0;
};  // class FormatWriter

/// Run one abandonment step, absorbing anything it throws.
///
/// Abort() is noexcept so that AbandonUnlessClosed can call it from a
/// destructor, and void so that a caller already handling their own failure is
/// never handed a second one. But "does not report" is not "cannot happen":
/// abandonment reaches into arrow streams and, below them, filesystem
/// implementations we do not own. An arrow::io::OutputStream::Abort() may throw
/// for reasons that have nothing to do with us -- this repository's own tests
/// contain one that does.
///
/// Without this the noexcept would turn that into std::terminate, which is the
/// opposite of what a best-effort cleanup verb should do, and would kill the
/// process before the FFI boundary could convert the failure into an error
/// code. Log it and carry on; that IS the contract.
///
/// An allocation failure inside the logging still ends the process, which is
/// deliberate and consistent with the rest of this library: OOM is not handled
/// anywhere, it just ends things.
template <typename Step>
void AbandonQuietly(std::string_view what, Step&& step) noexcept {
  try {
    step();
  } catch (const std::exception& e) {
    LOG_STORAGE_WARNING_ << "Failed to abandon " << what << ": " << e.what();
  } catch (...) {
    LOG_STORAGE_WARNING_ << "Failed to abandon " << what << " due to a non-standard exception";
  }
}

}  // namespace milvus_storage
