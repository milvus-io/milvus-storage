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

#include <cstddef>
#include <cstdint>
#include <list>
#include <map>
#include <mutex>
#include <optional>
#include <shared_mutex>

namespace milvus_storage {

template <typename K, typename V>
class LRUMemoryCache final {
  public:
  explicit LRUMemoryCache(uint64_t capacity_bytes = 0) noexcept : capacity_bytes_(capacity_bytes) {}

  void set_capacity(uint64_t capacity_bytes) {
    std::unique_lock lock(mutex_);
    capacity_bytes_ = capacity_bytes;
    evict_until_within_capacity();
  }

  [[nodiscard]] std::optional<V> get(const K& key) {
    std::unique_lock lock(mutex_);
    auto it = cache_.find(key);
    if (it == cache_.end()) {
      return std::nullopt;
    }

    lru_list_.splice(lru_list_.begin(), lru_list_, it->second.lru_it);
    return it->second.value;
  }

  bool put(const K& key, const V& value, uint64_t size_bytes) {
    std::unique_lock lock(mutex_);

    auto existing = cache_.find(key);
    if (existing != cache_.end()) {
      current_bytes_ -= existing->second.size_bytes;
      lru_list_.erase(existing->second.lru_it);
      cache_.erase(existing);
    }

    if (size_bytes == 0 || size_bytes > capacity_bytes_) {
      return false;
    }

    lru_list_.push_front(key);
    cache_.emplace(key, Entry{value, size_bytes, lru_list_.begin()});
    current_bytes_ += size_bytes;

    evict_until_within_capacity();
    return true;
  }

  [[nodiscard]] size_t size() const {
    std::shared_lock lock(mutex_);
    return cache_.size();
  }

  void remove(const K& key) {
    std::unique_lock lock(mutex_);
    auto it = cache_.find(key);
    if (it == cache_.end()) {
      return;
    }

    current_bytes_ -= it->second.size_bytes;
    lru_list_.erase(it->second.lru_it);
    cache_.erase(it);
  }

  void clear() {
    std::unique_lock lock(mutex_);
    cache_.clear();
    lru_list_.clear();
    current_bytes_ = 0;
  }

  private:
  struct Entry {
    V value;
    uint64_t size_bytes;
    typename std::list<K>::iterator lru_it;
  };

  void evict_until_within_capacity() {
    while (current_bytes_ > capacity_bytes_ && !lru_list_.empty()) {
      auto last_key = lru_list_.back();
      auto it = cache_.find(last_key);
      if (it != cache_.end()) {
        current_bytes_ -= it->second.size_bytes;
        cache_.erase(it);
      }
      lru_list_.pop_back();
    }
  }

  mutable std::shared_mutex mutex_;
  std::list<K> lru_list_;
  std::map<K, Entry> cache_;
  uint64_t capacity_bytes_;
  uint64_t current_bytes_ = 0;
};

}  // namespace milvus_storage
