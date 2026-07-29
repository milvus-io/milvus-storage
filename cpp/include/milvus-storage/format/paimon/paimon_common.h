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

#include <exception>
#include <string>
#include <unordered_map>

#include <arrow/status.h>

#include "milvus-storage/filesystem/fs.h"

namespace milvus_storage::paimon {

std::unordered_map<std::string, std::string> ToStorageOptions(const ArrowFileSystemConfig& config);

std::string ToStandardUri(const std::string& milvus_uri);

std::string ToMilvusUri(const std::string& standard_uri, const std::string& address);

/// Classify a Paimon bridge exception into an arrow Status without inverting
/// transient-vs-terminal semantics. The bridge FFI carries messages only, so
/// the Rust side marks terminal input-state errors with stable prefixes:
///
/// - "[paimon:error=invalid]" (expired snapshot, corrupt descriptor) maps
///   to Status::Invalid — retrying cannot help, the collection must be
///   refreshed/rebuilt;
/// - "[paimon:error=not-implemented]" (e.g. bitmap64 deletion vectors) maps to
///   Status::NotImplemented;
/// - everything else keeps the caller-provided default (typically IOError,
///   which stays retryable).
///
/// The classification is one-directional on purpose: an unmarked message is
/// never promoted to Invalid, so transient storage failures cannot become
/// terminal.
arrow::Status ClassifyPaimonError(const std::string& context, const std::exception& error);

}  // namespace milvus_storage::paimon
