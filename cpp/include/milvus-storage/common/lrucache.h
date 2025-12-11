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
#include <string>
#include <mutex>
#include <map>
#include <shared_mutex>
#include <functional>
#include <list>
#include <optional>

#include <arrow/status.h>
#include <arrow/result.h>

namespace milvus_storage {

template <typename K, typename V>
class LRUCache {
  public:
  explicit LRUCache(size_t capacity = 16) noexcept : capacity_(capacity) {}

  // Set the capacity of cache
  void set_capacity(size_t capacity) {
    std::unique_lock lock(mutex_);
    capacity_ = capacity;
    // evict if current size exceeds new capacity
    while (cache_.size() > capacity_) {
      auto last_key = lru_list_.back();
      cache_.erase(last_key);
      lru_list_.pop_back();
    }
  }

  // Get the value associated with the key, returns std::nullopt if not found
  [[nodiscard]] std::optional<V> get(const K& key) {
    std::unique_lock lock(mutex_);
    auto it = cache_.find(key);
    if (it != cache_.end()) {
      // Move to front in LRU list
      lru_list_.splice(lru_list_.begin(), lru_list_, it->second.second);
      return it->second.first;
    }
    return std::nullopt;
  }

  // Put a value into the cache
  void put(const K& key, const V& value) {
    std::unique_lock lock(mutex_);

    // Check if key already exists
    auto it = cache_.find(key);
    if (it != cache_.end()) {
      // Update existing entry and move to front
      it->second.first = value;
      lru_list_.splice(lru_list_.begin(), lru_list_, it->second.second);
      return;
    }

    // Insert new entry at front
    lru_list_.push_front(key);
    cache_[key] = std::make_pair(value, lru_list_.begin());

    // Evict if exceed capacity
    while (cache_.size() > capacity_) {
      auto last_key = lru_list_.back();
      cache_.erase(last_key);
      lru_list_.pop_back();
    }
  }

  // Get the size of cached entries
  [[nodiscard]] size_t size() const {
    std::shared_lock lock(mutex_);
    return cache_.size();
  }

  // Remove the cache entry by key
  void remove(const K& key) {
    std::unique_lock lock(mutex_);
    auto it = cache_.find(key);
    if (it != cache_.end()) {
      lru_list_.erase(it->second.second);
      cache_.erase(it);
    }
  }

  // Clean all cache entries
  void clean() {
    std::unique_lock lock(mutex_);
    cache_.clear();
    lru_list_.clear();
  }

  private:
  mutable std::shared_mutex mutex_;
  // LRU list holds keys with most-recent at front
  std::list<K> lru_list_;
  // map key -> (value, iterator to list)
  // don't use unordered_map, because hash function may slow down the performance
  std::map<K, std::pair<V, typename std::list<K>::iterator>> cache_;
  size_t capacity_;
};

}  // namespace milvus_storage
