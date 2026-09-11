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

#include <cstdint>
#include <optional>
#include <string>

namespace milvus_storage {

/// Configure the shared Rust Tokio runtime before first Rust bridge use.
///
/// Returns std::nullopt on success, or the failure reason if either thread
/// count is zero, if the runtime was already configured, or if the runtime was
/// already initialized.
std::optional<std::string> ConfigureRustRuntime(uint32_t worker_threads, uint32_t max_blocking_threads);

}  // namespace milvus_storage
