// Copyright 2024 Zilliz
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
#include <arrow/filesystem/filesystem.h>

template <template <typename> class P, typename T>
class PolymorphicWrapper {
  public:
  PolymorphicWrapper(P<T> obj) : obj_(obj) {}
  inline P<T> get() const { return obj_; }

  private:
  P<T> obj_;
};

using FileSystemWrapper = PolymorphicWrapper<std::shared_ptr, arrow::fs::FileSystem>;
using OutputStreamWrapper = PolymorphicWrapper<std::shared_ptr, arrow::io::OutputStream>;
