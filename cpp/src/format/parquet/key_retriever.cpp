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

#include "milvus-storage/format/parquet/key_retriever.h"
#include <cassert>

namespace milvus_storage::parquet {

KeyRetriever::KeyRetriever(const std::function<std::string(const std::string&)>& callback)
    : key_retriever_callback_(callback) {
  assert(key_retriever_callback_);
}

std::string KeyRetriever::GetKey(const std::string& key_metadata) { return key_retriever_callback_(key_metadata); }

}  // namespace milvus_storage::parquet