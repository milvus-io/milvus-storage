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

#include "arrow/c/abi.h"
#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdarg.h>

#define TABLE_NAME "test_table"
#define FIELD_INT64_NAME "int64_field"
#define FIELD_INT32_NAME "int32_field"
#define FIELD_STRING_NAME "string_field"

// arrow schema metadata format:
//
// int32: number of key/value pairs (noted N below)
// int32: byte length of key 0
// key 0 (not null-terminated)
// int32: byte length of value 0
// value 0 (not null-terminated)
// ...
// int32: byte length of key N - 1
// key N - 1 (not null-terminated)
// int32: byte length of value N - 1
// value N - 1 (not null-terminated)
char* create_arrow_schema_meta(int32_t count, ...) {
  if (count <= 0) {
    return NULL;
  }

  va_list args1, args2;
  va_start(args1, count);
  va_copy(args2, args1);

  size_t total_size = sizeof(int32_t);

  for (int i = 0; i < count; i++) {
    char* key = va_arg(args1, char*);
    char* value = va_arg(args1, char*);

    total_size += sizeof(int32_t) + strlen(key);
    total_size += sizeof(int32_t) + strlen(value);
  }
  va_end(args1);

  // alloc the buffer
  char* buffer = (char*)malloc(total_size);
  if (buffer == NULL) {
    va_end(args2);
    return NULL;
  }

  char* current = buffer;

  // write the COUNT of kv
  memcpy(current, &count, sizeof(int32_t));
  current += sizeof(int32_t);

  // write each key-value pair
  for (int i = 0; i < count; i++) {
    char* key = va_arg(args2, char*);
    char* value = va_arg(args2, char*);

    // write key
    int32_t key_len = (int32_t)strlen(key);
    memcpy(current, &key_len, sizeof(int32_t));
    current += sizeof(int32_t);
    memcpy(current, key, key_len);
    current += key_len;

    // write value
    int32_t value_len = (int32_t)strlen(value);
    memcpy(current, &value_len, sizeof(int32_t));
    current += sizeof(int32_t);
    memcpy(current, value, value_len);
    current += value_len;
  }

  va_end(args2);
  return buffer;
}

void field_schema_release(struct ArrowSchema* schema) {
  assert(schema != NULL);
  assert(schema->format != NULL);
  assert(strncmp(schema->format, "l", 1) == 0 || strncmp(schema->format, "i", 1) == 0 ||
         strncmp(schema->format, "u", 1) == 0);

  // no need free schema->format, it's a static const char*
  if (schema->name != NULL) {
    free((void*)(schema->name));
  }
  if (schema->metadata != NULL) {
    free((void*)(schema->metadata));
  }

  assert(schema->dictionary == NULL);
  assert(schema->private_data == NULL);
  schema->release = NULL;
}

void struct_schema_release(struct ArrowSchema* schema) {
  if (schema->children != NULL) {
    // release each child schema
    for (int64_t i = 0; i < schema->n_children; i++) {
      if (schema->children[i] != NULL) {
        if (schema->children[i]->release != NULL) {
          schema->children[i]->release(schema->children[i]);
        }
        free(schema->children[i]);
      }
    }
    free(schema->children);
  }
  if (schema->name != NULL) {
    free((void*)(schema->name));
  }
  if (schema->metadata != NULL) {
    free((void*)(schema->metadata));
  }

  assert(schema->dictionary == NULL);
  assert(schema->private_data == NULL);
  schema->release = NULL;
}

struct ArrowSchema* create_test_field_schema(const char* format, const char* name, int nullable, int field_offset) {
  struct ArrowSchema* schema = malloc(sizeof(struct ArrowSchema));
  if (schema == NULL) {
    return NULL;
  }

  schema->format = format;
  schema->name = name ? strdup(name) : NULL;
  schema->flags = nullable ? ARROW_FLAG_NULLABLE : 0;

  schema->n_children = 0;
  schema->children = NULL;
  schema->dictionary = NULL;
  // work around to make sure each field has a milvus:field_id
  assert(field_offset < 9);
  char fid = '1' + field_offset;
  char fid_str[2] = {fid, '\0'};
  schema->metadata = create_arrow_schema_meta(1, "PARQUET:field_id", fid_str);

  schema->release = field_schema_release;
  schema->private_data = NULL;
  return schema;
}

struct ArrowSchema* create_test_struct_schema() {
  struct ArrowSchema* table_schema = malloc(sizeof(struct ArrowSchema));
  if (table_schema == NULL) {
    return NULL;
  }

  table_schema->format = "+s";
  table_schema->name = strdup(TABLE_NAME);
  table_schema->flags = 0;

  // 3 children fields
  table_schema->n_children = 3;
  table_schema->children = malloc(sizeof(struct ArrowSchema*) * 3);

  // first field: int64
  table_schema->children[0] = create_test_field_schema("l", FIELD_INT64_NAME, 0, 0);

  // second field: int32
  table_schema->children[1] = create_test_field_schema("i", FIELD_INT32_NAME, 1, 1);

  // third field: utf8 string
  table_schema->children[2] = create_test_field_schema("u", FIELD_STRING_NAME, 1, 2);

  table_schema->dictionary = NULL;
  table_schema->metadata = NULL;
  table_schema->release = struct_schema_release;
  table_schema->private_data = NULL;

  return table_schema;
}

void release_string_array_data(struct ArrowArray* array) {
  assert(array->n_buffers == 3);
  if (array->buffers[0] != NULL) {
    // release the null bitmap
    free((uint8_t*)array->buffers[0]);
  }
  if (array->buffers[1] != NULL) {
    // release the offsets array
    free((int32_t*)array->buffers[1]);
  }
  if (array->buffers[2] != NULL) {
    // release the data buffer
    free((char*)array->buffers[2]);
  }
  free(array->buffers);
}

void release_primitive_array_data(struct ArrowArray* array) {
  assert(array->n_buffers == 2);
  if (array->buffers[0] != NULL) {
    // release the null bitmap
    free((uint8_t*)array->buffers[0]);
  }
  if (array->buffers[1] != NULL) {
    // release the data buffer
    free((uint8_t*)array->buffers[1]);
  }
  free(array->buffers);
}

void release_root_array(struct ArrowArray* array) {
  if (array->children != NULL) {
    for (int64_t i = 0; i < array->n_children; i++) {
      if (array->children[i] != NULL) {
        if (array->children[i]->release != NULL) {
          array->children[i]->release(array->children[i]);
        }
        free(array->children[i]);
        array->children[i] = NULL;
      }
    }
    free(array->children);
  }

  if (array->buffers != NULL)
    free(array->buffers);
  array->release = NULL;
}

struct ArrowArray* create_int64_array(const int64_t* data,
                                      int64_t length,
                                      const uint8_t* null_bitmap,
                                      int64_t null_count) {
  struct ArrowArray* array = malloc(sizeof(struct ArrowArray));
  if (array == NULL)
    return NULL;

  array->length = length;
  array->null_count = null_count;
  array->offset = 0;
  array->n_buffers = 2;
  array->n_children = 0;
  array->children = NULL;
  array->dictionary = NULL;

  // main buffer
  array->buffers = malloc(sizeof(const void*) * array->n_buffers);
  // null bitmap buffer
  if (null_bitmap) {
    // align 8
    array->buffers[0] = malloc((length + 7) / 8);
    memcpy((void*)array->buffers[0], null_bitmap, (length + 7) / 8);
  } else {
    // non-null
    array->buffers[0] = NULL;
  }

  // main buffer
  array->buffers[1] = malloc(sizeof(int64_t) * length);
  memcpy((void*)array->buffers[1], data, sizeof(int64_t) * length);

  array->release = release_primitive_array_data;
  array->private_data = NULL;
  return array;
}

struct ArrowArray* create_int32_array(const int32_t* data,
                                      int64_t length,
                                      const uint8_t* null_bitmap,
                                      int64_t null_count) {
  struct ArrowArray* array = malloc(sizeof(struct ArrowArray));
  if (array == NULL)
    return NULL;

  array->length = length;
  array->null_count = null_count;
  array->offset = 0;
  array->n_buffers = 2;
  array->n_children = 0;
  array->children = NULL;
  array->dictionary = NULL;

  array->buffers = malloc(sizeof(const void*) * array->n_buffers);

  if (null_bitmap) {
    array->buffers[0] = malloc((length + 7) / 8);
    memcpy((void*)array->buffers[0], null_bitmap, (length + 7) / 8);
  } else {
    array->buffers[0] = NULL;
  }

  array->buffers[1] = malloc(sizeof(int32_t) * length);
  memcpy((void*)array->buffers[1], data, sizeof(int32_t) * length);

  array->release = release_primitive_array_data;
  array->private_data = NULL;
  return array;
}

struct ArrowArray* create_string_array(const char** data,
                                       int64_t length,
                                       const uint8_t* null_bitmap,
                                       int64_t null_count) {
  struct ArrowArray* array = malloc(sizeof(struct ArrowArray));
  if (array == NULL)
    return NULL;

  array->length = length;
  array->null_count = null_count;
  array->offset = 0;
  array->n_buffers = 3;
  array->n_children = 0;
  array->children = NULL;
  array->dictionary = NULL;

  array->buffers = malloc(sizeof(const void*) * array->n_buffers);

  // null bitmap buffer
  if (null_bitmap) {
    array->buffers[0] = malloc((length + 7) / 8);
    memcpy((void*)array->buffers[0], null_bitmap, (length + 7) / 8);
  } else {
    array->buffers[0] = NULL;
  }

  // calculate total length and offsets
  int32_t total_length = 0;
  int32_t* offsets = malloc(sizeof(int32_t) * (length + 1));
  offsets[0] = 0;

  for (int64_t i = 0; i < length; i++) {
    total_length += (int32_t)strlen(data[i]);
    offsets[i + 1] = total_length;
  }

  // offsets buffer
  array->buffers[1] = offsets;

  // data buffer
  char* string_data = malloc(total_length);
  int32_t pos = 0;
  for (int64_t i = 0; i < length; i++) {
    int32_t len = (int32_t)strlen(data[i]);
    memcpy(string_data + pos, data[i], len);
    pos += len;
  }
  array->buffers[2] = string_data;
  array->release = release_string_array_data;
  array->private_data = NULL;
  return array;
}

struct ArrowArray* create_struct_array(struct ArrowArray** children, int64_t n_children, int64_t length) {
  struct ArrowArray* array = malloc(sizeof(struct ArrowArray));
  if (array == NULL)
    return NULL;

  array->length = length;
  array->null_count = 0;
  array->offset = 0;
  array->n_buffers = 1;  // only null bitmap
  array->n_children = n_children;

  // children arrays
  array->children = malloc(sizeof(struct ArrowArray*) * n_children);
  for (int64_t i = 0; i < n_children; i++) {
    array->children[i] = children[i];
  }

  array->dictionary = NULL;

  // null buffers
  array->buffers = malloc(sizeof(const void*) * array->n_buffers);
  array->buffers[0] = NULL;

  array->release = release_root_array;
  array->private_data = NULL;
  return array;
}
