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

#include <algorithm>
#include <charconv>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include <boost/algorithm/string/trim.hpp>

namespace milvus_storage::api::convert {

/// Convert a string to a typed value.
/// Returns {true, value} on success, {false, T{}} on failure.
template <typename T>
std::pair<bool, T> convertFunc(const std::string& str);

// --- integer types via std::from_chars ---

template <typename I>
std::pair<bool, I> convertIntFunc(const std::string& str) {
  I result{};
  auto [ptr, ec] = std::from_chars(str.data(), str.data() + str.size(), result);
  return {ec == std::errc{} && ptr == str.data() + str.size(), result};
}

template <>
inline std::pair<bool, int32_t> convertFunc<int32_t>(const std::string& str) {
  return convertIntFunc<int32_t>(str);
}

template <>
inline std::pair<bool, int64_t> convertFunc<int64_t>(const std::string& str) {
  return convertIntFunc<int64_t>(str);
}

template <>
inline std::pair<bool, uint32_t> convertFunc<uint32_t>(const std::string& str) {
  return convertIntFunc<uint32_t>(str);
}

template <>
inline std::pair<bool, uint64_t> convertFunc<uint64_t>(const std::string& str) {
  return convertIntFunc<uint64_t>(str);
}

// --- bool ---

template <>
inline std::pair<bool, bool> convertFunc<bool>(const std::string& str) {
  std::string lower = str;
  std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
  return {lower == "true" || lower == "false", lower == "true"};
}

// --- vector<string> (comma-separated) ---

template <typename I>
std::pair<bool, std::vector<I>> convertVectorFunc(const std::string& str) {
  std::vector<I> result;
  if (!str.empty()) {
    size_t start = 0;
    size_t end = str.find(',');
    while (end != std::string::npos) {
      result.push_back(boost::trim_copy(str.substr(start, end - start)));
      start = end + 1;
      end = str.find(',', start);
    }
    result.push_back(boost::trim_copy(str.substr(start)));
  }
  return {true, result};
}

template <>
inline std::pair<bool, std::vector<std::string>> convertFunc<std::vector<std::string>>(const std::string& str) {
  return convertVectorFunc<std::string>(str);
}

}  // namespace milvus_storage::api::convert
