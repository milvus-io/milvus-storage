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

#include "milvus-storage/ffi_c.h"

#include <stdlib.h>
#include <string>
#include <unordered_map>
#include <vector>
#include <memory>
#include <algorithm>
#include <cctype>
#include <charconv>
#include <functional>
#include <iostream>

#include "milvus-storage/ffi_internal/result.h"
#include "milvus-storage/filesystem/fs.h"

using namespace milvus_storage;
using namespace milvus_storage::api;
// ==================== Properties C Implementation ====================

FFIResult properties_create(const char* const* keys,
                            const char* const* values,
                            size_t count,
                            ::Properties* properties) {
  // used to make sure no duplicate keys
  std::unordered_set<std::string_view> key_set;
  if (!properties) {
    RETURN_ERROR(LOON_INVALID_ARGS, "properties should not be empty");
  }

  properties->properties = nullptr;
  properties->count = 0;

  if (count == 0 || !keys || !values) {
    RETURN_ERROR(LOON_INVALID_ARGS, "Invalid keys/values");
  }

  properties->properties = static_cast<Property*>(malloc(sizeof(Property) * count));
  if (!properties->properties) {
    RETURN_ERROR(LOON_MEMORY_ERROR, "Failed to malloc [size=", sizeof(Property) * count, "]");
  }

  for (size_t i = 0; i < count; ++i) {
    properties->properties[i].key = nullptr;
    properties->properties[i].value = nullptr;

    if (keys[i] && key_set.find(keys[i]) == key_set.end()) {
      size_t key_len = strlen(keys[i]) + 1;
      properties->properties[i].key = static_cast<char*>(malloc(key_len));
      if (properties->properties[i].key) {
        strcpy(properties->properties[i].key, keys[i]);
      }

      key_set.insert(keys[i]);
    } else {
      properties_free(properties);
      if (keys[i]) {
        RETURN_ERROR(LOON_INVALID_PROPERTIES, "Duplicate key: ", keys[i], " at index: ", i);
      } else {
        RETURN_ERROR(LOON_INVALID_PROPERTIES, "The key index: ", i, " is invalid");
      }
    }

    if (values[i]) {
      size_t value_len = strlen(values[i]) + 1;
      properties->properties[i].value = static_cast<char*>(malloc(value_len));
      if (properties->properties[i].value) {
        strcpy(properties->properties[i].value, values[i]);
      }
    } else {
      properties_free(properties);
      RETURN_ERROR(LOON_INVALID_PROPERTIES, "The value index: ", i, " is invalid, key: ", keys[i]);
    }
  }

  properties->count = count;
  RETURN_SUCCESS();
}

const char* properties_get(const ::Properties* properties, const char* key) {
  if (!properties || !properties->properties || !key) {
    return nullptr;
  }

  for (size_t i = 0; i < properties->count; ++i) {
    if (properties->properties[i].key && strcmp(properties->properties[i].key, key) == 0) {
      return properties->properties[i].value;
    }
  }

  return nullptr;
}

void properties_free(::Properties* properties) {
  if (!properties) {
    return;
  }

  if (properties->properties) {
    for (size_t i = 0; i < properties->count; ++i) {
      free(properties->properties[i].key);
      free(properties->properties[i].value);
    }
    free(properties->properties);
    properties->properties = nullptr;
  }
  properties->count = 0;
}
