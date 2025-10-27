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

#include <arrow/status.h>
#include <arrow/result.h>

namespace milvus_storage {

template <typename K, typename V>
class LRUCache {
  public:
  static LRUCache& getInstance() {
    static LRUCache instance;
    return instance;
  }

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

  // Get the value associated with the key
  [[nodiscard]] arrow::Result<V> get(const K& key, std::function<arrow::Result<V>(const K&)> CreateFunc) {
    std::unique_lock write_lock(mutex_);
    // check if key exists
    auto it = cache_.find(key);
    if (it != cache_.end()) {
      // need to move to front in LRU list; upgrade to unique lock
      // re-find the entry after upgrading lock
      auto it2 = cache_.find(key);
      if (it2 != cache_.end()) {
        lru_list_.splice(lru_list_.begin(), lru_list_, it2->second.second);
        return it2->second.first;
      }
      // if disappeared, fallthrough to create
    }

    // not exist, create new instance
    ARROW_ASSIGN_OR_RAISE(auto instance, CreateFunc(key));
    // insert into LRU front and map
    lru_list_.push_front(key);
    cache_[key] = std::make_pair(instance, lru_list_.begin());

    // evict if exceed capacity
    while (cache_.size() > capacity_) {
      auto last_key = lru_list_.back();
      cache_.erase(last_key);
      lru_list_.pop_back();
    }

    return instance;
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
  LRUCache(size_t capacity = 16) : capacity_(capacity) {}
  ~LRUCache() = default;

  mutable std::shared_mutex mutex_;
  // LRU list holds keys with most-recent at front
  std::list<K> lru_list_;
  // map key -> (value, iterator to list)
  // don't use unordered_map, because hash function may slow down the performance
  std::map<K, std::pair<V, typename std::list<K>::iterator>> cache_;
  size_t capacity_;
};

}  // namespace milvus_storage
