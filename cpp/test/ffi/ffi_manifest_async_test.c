// Copyright 2026 Zilliz
// Licensed under the Apache License, Version 2.0.
#include "milvus-storage/ffi_c.h"
#include "test_runner.h"
#include "async_test_executor.h"
#include <pthread.h>
#include <stdint.h>
#include <time.h>
#include <errno.h>
#include <unistd.h>

static TestExecutor executor;
static LoonIOContextHandle io_context;
struct Completion {
  pthread_mutex_t mutex;
  pthread_cond_t ready;
  int calls;
  int code;
  LoonTransactionHandle transaction;
  int32_t outcome;
  int64_t version;
  int destroy_on_commit;
  TestExecutor* expected_executor;
};
static void begin_complete(uintptr_t token, LoonFFIResult result, LoonTransactionHandle transaction) {
  struct Completion* state = (struct Completion*)token;
  TestExecutor* expected = state->expected_executor ? state->expected_executor : &executor;
  ck_assert(pthread_equal(pthread_self(), expected->threads[0]));
  pthread_mutex_lock(&state->mutex);
  state->calls++;
  state->code = result.err_code;
  if (result.err_code && result.message)
    fprintf(stderr, "Async begin failed: %s\n", result.message);
  state->transaction = transaction;
  loon_ffi_free_result(&result);
  pthread_cond_signal(&state->ready);
  pthread_mutex_unlock(&state->mutex);
}
static void commit_complete(uintptr_t token, LoonFFIResult result, int32_t outcome, int64_t version) {
  struct Completion* state = (struct Completion*)token;
  TestExecutor* expected = state->expected_executor ? state->expected_executor : &executor;
  ck_assert(pthread_equal(pthread_self(), expected->threads[0]));
  pthread_mutex_lock(&state->mutex);
  state->calls++;
  state->code = result.err_code;
  if (state->destroy_on_commit) {
    loon_transaction_destroy(state->transaction);
    state->transaction = 0;
  }
  state->outcome = outcome;
  state->version = version;
  loon_ffi_free_result(&result);
  pthread_cond_signal(&state->ready);
  pthread_mutex_unlock(&state->mutex);
}
static int await_begin(struct Completion* state) {
  struct timespec deadline;
  clock_gettime(CLOCK_REALTIME, &deadline);
  deadline.tv_sec += 10;
  pthread_mutex_lock(&state->mutex);
  int error = 0;
  while (!state->calls && !error) error = pthread_cond_timedwait(&state->ready, &state->mutex, &deadline);
  pthread_mutex_unlock(&state->mutex);
  return error;
}
static void test_async_begin_contract(void) {
  struct Completion state = {PTHREAD_MUTEX_INITIALIZER, PTHREAD_COND_INITIALIZER, 0, 0, 0};
  LoonProperty items[] = {{"fs.storage_type", "remote"},
                          {"fs.address", "127.0.0.1:1"},
                          {"fs.bucket_name", "test"},
                          {"fs.access_key_id", "test"},
                          {"fs.access_key_value", "secret"}};
  LoonProperties properties = {items, sizeof(items) / sizeof(items[0])};
  LoonAsyncHandle operation = (LoonAsyncHandle)(uintptr_t)1;
  LoonAsyncOptions options = {sizeof(options), 1, 0};
  LoonFFIResult result = loon_transaction_begin_async(io_context, "segment", &properties, 0, 0, 0, &options,
                                                      begin_complete, (uintptr_t)&state, &operation);
  ck_assert_int_eq(result.err_code, LOON_INVALID_ARGS);
  loon_ffi_free_result(&result);
  ck_assert(operation == NULL);
  ck_assert_int_eq(state.calls, 0);
  options.flags = 0;
  // Larger future options are accepted. Version zero must not contact the closed endpoint.
  options.struct_size += 16;
  result = loon_transaction_begin_async(io_context, "segment", &properties, 0, 0, 0, &options, begin_complete,
                                        (uintptr_t)&state, &operation);
  ck_assert_msg(result.err_code == 0, "%s", result.message);
  loon_ffi_free_result(&result);
  ck_assert(operation != NULL);
  loon_async_release(operation);  // Early release retains the promised notification.
  ck_assert_int_eq(await_begin(&state), 0);
  ck_assert_int_eq(state.calls, 1);
  ck_assert_int_eq(state.code, 0);
  int64_t version = -1;
  result = loon_transaction_get_read_version(state.transaction, &version);
  ck_assert_int_eq(result.err_code, 0);
  loon_ffi_free_result(&result);
  ck_assert_int_eq(version, 0);
  loon_transaction_destroy(state.transaction);
  loon_async_cancel(NULL);
  loon_async_release(NULL);
  pthread_cond_destroy(&state.ready);
  pthread_mutex_destroy(&state.mutex);
}
static void test_async_begin_local(void) {
  struct Completion state = {PTHREAD_MUTEX_INITIALIZER, PTHREAD_COND_INITIALIZER, 0, 0, 0};
  LoonProperties properties = {NULL, 0};
  LoonAsyncHandle operation = NULL;
  LoonFFIResult result = loon_transaction_begin_async(io_context, "segment", &properties, 0, 0, 0, NULL, begin_complete,
                                                      (uintptr_t)&state, &operation);
  ck_assert_int_eq(result.err_code, 0);
  loon_ffi_free_result(&result);
  ck_assert(operation != NULL);
  ck_assert_int_eq(await_begin(&state), 0);
  ck_assert_int_eq(state.calls, 1);
  ck_assert_int_eq(state.code, 0);
  loon_async_release(operation);
  loon_transaction_destroy(state.transaction);
  pthread_cond_destroy(&state.ready);
  pthread_mutex_destroy(&state.mutex);
}
static void test_async_begin_minio(void) {
  const char* endpoint = getenv("LOON_ASYNC_TEST_ENDPOINT");
  if (!endpoint)
    return;  // Explicit opt-in real MinIO integration.
  struct Completion state = {PTHREAD_MUTEX_INITIALIZER, PTHREAD_COND_INITIALIZER, 0, 0, 0};
  LoonProperty items[] = {{"fs.storage_type", "remote"},
                          {"fs.address", (char*)endpoint},
                          {"fs.bucket_name", "manifest-async-test"},
                          {"fs.access_key_id", "manifesttest"},
                          {"fs.access_key_value", "manifesttestsecret"},
                          {"fs.region", "us-east-1"}};
  LoonProperties properties = {items, sizeof(items) / sizeof(items[0])};
  char path[128];
  snprintf(path, sizeof(path), "begin-%ld-%ld", (long)time(NULL), (long)getpid());
  LoonTransactionHandle sync_transaction = 0;
  LoonFFIResult result = loon_transaction_begin(path, &properties, 0, 0, 0, &sync_transaction);
  ck_assert_msg(result.err_code == 0, "%s", result.message);
  loon_ffi_free_result(&result);
  result = loon_transaction_add_delta_log(sync_transaction, "_delta/delete", 42);
  ck_assert_int_eq(result.err_code, 0);
  loon_ffi_free_result(&result);
  int64_t version = -1;
  result = loon_transaction_commit(sync_transaction, &version);
  ck_assert_msg(result.err_code == 0, "%s", result.message);
  loon_ffi_free_result(&result);
  loon_transaction_destroy(sync_transaction);
  ck_assert_int_eq(version, 1);
  LoonAsyncHandle operation = NULL;
  result = loon_transaction_begin_async(io_context, path, &properties, -1, 0, 0, NULL, begin_complete,
                                        (uintptr_t)&state, &operation);
  ck_assert_msg(result.err_code == 0, "%s", result.message);
  loon_ffi_free_result(&result);
  ck_assert_int_eq(await_begin(&state), 0);
  loon_async_release(operation);
  ck_assert_int_eq(state.code, 0);
  result = loon_transaction_get_read_version(state.transaction, &version);
  ck_assert_int_eq(result.err_code, 0);
  loon_ffi_free_result(&result);
  ck_assert_int_eq(version, 1);
  LoonManifest* manifest = NULL;
  result = loon_transaction_get_manifest(state.transaction, &manifest);
  ck_assert_int_eq(result.err_code, 0);
  loon_ffi_free_result(&result);
  ck_assert_int_eq(manifest->delta_logs.num_delta_logs, 1);
  loon_manifest_destroy(manifest);
  // A transaction must retain a normal filesystem after begin completes.
  result = loon_transaction_add_delta_log(state.transaction, "_delta/sync-after-begin", 1);
  ck_assert_int_eq(result.err_code, 0);
  loon_ffi_free_result(&result);
  result = loon_transaction_commit(state.transaction, &version);
  ck_assert_msg(result.err_code == 0, "%s", result.message);
  loon_ffi_free_result(&result);
  ck_assert_int_eq(version, 2);
  loon_transaction_destroy(state.transaction);
  for (int cancel = 0; cancel < 2; ++cancel) {
    state.calls = 0;
    LoonAsyncOptions options = {sizeof(options), 0, 1000};
    result = loon_transaction_begin_async(io_context, path, &properties, -1, 0, 0, &options, begin_complete,
                                          (uintptr_t)&state, &operation);
    ck_assert_int_eq(result.err_code, 0);
    loon_ffi_free_result(&result);
    ck_assert_int_eq(await_begin(&state), 0);
    ck_assert_int_eq(state.code, 0);
    if (cancel) {
      loon_async_cancel(operation);
    } else {
      struct timespec delay = {1, 100000000};
      nanosleep(&delay, NULL);
    }
    loon_async_release(operation);
    result = loon_transaction_add_delta_log(state.transaction, "_delta/later-sync", 1);
    ck_assert_int_eq(result.err_code, 0);
    loon_ffi_free_result(&result);
    result = loon_transaction_commit(state.transaction, &version);
    ck_assert_msg(result.err_code == 0, "%s", result.message);
    loon_ffi_free_result(&result);
    ck_assert_int_eq(version, 3 + cancel);
    loon_transaction_destroy(state.transaction);
  }
  pthread_cond_destroy(&state.ready);
  pthread_mutex_destroy(&state.mutex);
}
static void test_async_commit_roundtrip(void) {
  const char* endpoint = getenv("LOON_ASYNC_TEST_ENDPOINT");
  struct Completion state = {PTHREAD_MUTEX_INITIALIZER, PTHREAD_COND_INITIALIZER, 0, 0, 0};
  LoonProperty items[] = {{"fs.storage_type", "remote"},
                          {"fs.address", (char*)endpoint},
                          {"fs.bucket_name", "manifest-async-test"},
                          {"fs.access_key_id", "manifesttest"},
                          {"fs.access_key_value", "manifesttestsecret"},
                          {"fs.region", "us-east-1"}};
  LoonProperties properties =
      endpoint ? (LoonProperties){items, sizeof(items) / sizeof(items[0])} : (LoonProperties){NULL, 0};
  char path[128];
  snprintf(path, sizeof(path), "commit-%ld-%ld", (long)time(NULL), (long)getpid());
  LoonAsyncHandle operation = NULL;
  LoonFFIResult result = loon_transaction_begin_async(io_context, path, &properties, 0, 0, 1, NULL, begin_complete,
                                                      (uintptr_t)&state, &operation);
  ck_assert_msg(result.err_code == 0, "%s", result.message);
  loon_ffi_free_result(&result);
  ck_assert_int_eq(await_begin(&state), 0);
  loon_async_release(operation);
  ck_assert_int_eq(state.code, 0);
  LoonTransactionHandle transaction = state.transaction;
  result = loon_transaction_add_delta_log(transaction, "_delta/delete", 42);
  ck_assert_int_eq(result.err_code, 0);
  loon_ffi_free_result(&result);
  const char* columns[] = {"vector"};
  LoonColumnGroupFile file = {.path = "_data/first.parquet", .start_index = 0, .end_index = 10};
  LoonColumnGroup group = {columns, 1, "parquet", &file, 1};
  result = loon_transaction_add_column_group(transaction, &group);
  ck_assert_int_eq(result.err_code, 0);
  loon_ffi_free_result(&result);
  LoonIndexInfo index = {.column_name = "vector",
                         .index_name = "vector_hnsw",
                         .index_type = "HNSW",
                         .path = "_index/hnsw",
                         .field_id = 100,
                         .index_id = 7};
  result = loon_transaction_add_index_info(transaction, &index);
  ck_assert_int_eq(result.err_code, 0);
  loon_ffi_free_result(&result);
  const char* files[] = {"_stats/bloom"};
  result = loon_transaction_update_stat(transaction, "bloom", files, 1, NULL, NULL, 0);
  ck_assert_int_eq(result.err_code, 0);
  loon_ffi_free_result(&result);
  LoonLobFileInfo lob = {"../lobs/value", 100, 10, 9, 128};
  result = loon_transaction_add_lob_file(transaction, &lob);
  ck_assert_int_eq(result.err_code, 0);
  loon_ffi_free_result(&result);
  // Rejected initial scheduling must not consume the transaction.
  pthread_mutex_lock(&executor.mutex);
  executor.reject = 1;
  pthread_mutex_unlock(&executor.mutex);
  result = loon_transaction_commit_async(io_context, transaction, NULL, commit_complete, (uintptr_t)&state, &operation);
  ck_assert_int_eq(result.err_code, LOON_ASYNC_OVERLOADED);
  ck_assert(operation == NULL);
  loon_ffi_free_result(&result);
  ck_assert_int_eq(state.calls, 1);
  pthread_mutex_lock(&executor.mutex);
  executor.reject = 0;
  pthread_mutex_unlock(&executor.mutex);
  state.calls = 0;
  result = loon_transaction_commit_async(io_context, transaction, NULL, commit_complete, (uintptr_t)&state, &operation);
  ck_assert_msg(result.err_code == 0, "%s", result.message);
  loon_ffi_free_result(&result);
  LoonAsyncHandle second = NULL;
  result = loon_transaction_commit_async(io_context, transaction, NULL, commit_complete, (uintptr_t)&state, &second);
  ck_assert_int_eq(result.err_code, LOON_ASYNC_BUSY);
  ck_assert(second == NULL);
  loon_ffi_free_result(&result);
  ck_assert_int_eq(await_begin(&state), 0);
  loon_async_release(operation);
  ck_assert_int_eq(state.calls, 1);
  ck_assert_int_eq(state.code, 0);
  ck_assert_int_eq(state.outcome, LOON_COMMIT_COMMITTED);
  ck_assert_int_eq(state.version, 1);
  loon_transaction_destroy(transaction);
  result = loon_transaction_begin(path, &properties, -1, 0, 0, &transaction);
  ck_assert_msg(result.err_code == 0, "%s", result.message);
  loon_ffi_free_result(&result);
  LoonManifest* manifest = NULL;
  result = loon_transaction_get_manifest(transaction, &manifest);
  ck_assert_int_eq(result.err_code, 0);
  loon_ffi_free_result(&result);
  ck_assert_int_eq(manifest->column_groups.num_of_column_groups, 1);
  ck_assert_int_eq(manifest->indexes.num_indexes, 1);
  ck_assert_int_eq(manifest->delta_logs.num_delta_logs, 1);
  ck_assert_int_eq(manifest->stats.num_stats, 1);
  ck_assert_int_eq(manifest->lob_files.num_files, 1);
  loon_manifest_destroy(manifest);
  // Synchronous begin -> asynchronous commit uses the same filesystem.
  file.path = "_data/second.parquet";
  file.start_index = 10;
  file.end_index = 20;
  LoonColumnGroups groups = {&group, 1};
  result = loon_transaction_append_files(transaction, &groups);
  ck_assert_int_eq(result.err_code, 0);
  loon_ffi_free_result(&result);
  result = loon_transaction_drop_index(transaction, 7);
  ck_assert_int_eq(result.err_code, 0);
  loon_ffi_free_result(&result);
  result = loon_transaction_add_delta_log(transaction, "_delta/second", 2);
  ck_assert_int_eq(result.err_code, 0);
  loon_ffi_free_result(&result);
  state.calls = 0;
  result = loon_transaction_commit_async(io_context, transaction, NULL, commit_complete, (uintptr_t)&state, &operation);
  ck_assert_int_eq(result.err_code, 0);
  loon_ffi_free_result(&result);
  ck_assert_int_eq(await_begin(&state), 0);
  loon_async_release(operation);
  ck_assert_int_eq(state.code, 0);
  ck_assert_int_eq(state.outcome, LOON_COMMIT_COMMITTED);
  ck_assert_int_eq(state.version, 2);
  loon_transaction_destroy(transaction);
  result = loon_transaction_begin(path, &properties, 2, 0, 0, &transaction);
  ck_assert_int_eq(result.err_code, 0);
  loon_ffi_free_result(&result);
  result = loon_transaction_get_manifest(transaction, &manifest);
  ck_assert_int_eq(result.err_code, 0);
  loon_ffi_free_result(&result);
  ck_assert_int_eq(manifest->column_groups.column_group_array[0].num_of_files, 2);
  ck_assert_int_eq(manifest->indexes.num_indexes, 0);
  ck_assert_int_eq(manifest->delta_logs.num_delta_logs, 2);
  loon_manifest_destroy(manifest);
  result = loon_transaction_drop_column(transaction, "vector");
  ck_assert_int_eq(result.err_code, 0);
  loon_ffi_free_result(&result);
  state.calls = 0;
  state.transaction = transaction;
  state.destroy_on_commit = 1;
  result = loon_transaction_commit_async(io_context, transaction, NULL, commit_complete, (uintptr_t)&state, &operation);
  ck_assert_int_eq(result.err_code, 0);
  loon_ffi_free_result(&result);
  ck_assert_int_eq(await_begin(&state), 0);
  loon_async_release(operation);
  ck_assert_int_eq(state.code, 0);
  ck_assert_int_eq(state.version, 3);
  ck_assert_int_eq(state.transaction, 0);
  result = loon_transaction_begin(path, &properties, 3, 0, 0, &transaction);
  ck_assert_int_eq(result.err_code, 0);
  loon_ffi_free_result(&result);
  result = loon_transaction_get_manifest(transaction, &manifest);
  ck_assert_int_eq(result.err_code, 0);
  loon_ffi_free_result(&result);
  ck_assert_int_eq(manifest->column_groups.num_of_column_groups, 0);
  loon_manifest_destroy(manifest);
  loon_transaction_destroy(transaction);
  pthread_cond_destroy(&state.ready);
  pthread_mutex_destroy(&state.mutex);
}
static void test_async_io_context(void) {
  struct Completion state = {PTHREAD_MUTEX_INITIALIZER, PTHREAD_COND_INITIALIZER, 0, 0, 0};
  LoonProperty items[] = {{"fs.storage_type", "remote"},
                          {"fs.address", "127.0.0.1:1"},
                          {"fs.bucket_name", "test"},
                          {"fs.access_key_id", "test"},
                          {"fs.access_key_value", "secret"}};
  LoonProperties properties = {items, sizeof(items) / sizeof(items[0])};
  LoonAsyncHandle operation = NULL;
  LoonFFIResult result = loon_transaction_begin_async(io_context, "segment", &properties, 0, 0, 0, NULL, begin_complete,
                                                      (uintptr_t)&state, &operation);
  ck_assert(result.err_code != 0);
  ck_assert(!operation && state.calls == 0);
  loon_ffi_free_result(&result);
  LoonAsyncExecutor descriptor = test_executor_start(&executor, 1);
  result = loon_io_context_create(NULL, &io_context);
  ck_assert(result.err_code != 0);
  loon_ffi_free_result(&result);
  result = loon_io_context_create(&descriptor, &io_context);
  ck_assert_int_eq(result.err_code, 0);
  loon_ffi_free_result(&result);
  result = loon_io_context_create(&descriptor, NULL);
  ck_assert_int_eq(result.err_code, LOON_INVALID_ARGS);
  loon_ffi_free_result(&result);
  LoonIOContextHandle invalid = (LoonIOContextHandle)(uintptr_t)1;
  descriptor.submit = NULL;
  result = loon_io_context_create(&descriptor, &invalid);
  ck_assert_int_eq(result.err_code, LOON_INVALID_ARGS);
  ck_assert(invalid == NULL);
  loon_ffi_free_result(&result);
  descriptor.submit = test_executor_submit;
  descriptor.struct_size = 0;
  result = loon_io_context_create(&descriptor, &invalid);
  ck_assert_int_eq(result.err_code, LOON_INVALID_ARGS);
  ck_assert(invalid == NULL);
  loon_ffi_free_result(&result);
  descriptor.struct_size = sizeof(descriptor);
  descriptor.reserved = 1;
  result = loon_io_context_create(&descriptor, &invalid);
  ck_assert_int_eq(result.err_code, LOON_INVALID_ARGS);
  ck_assert(invalid == NULL);
  loon_ffi_free_result(&result);
  // External queue rejection must roll back admission without a callback.
  pthread_mutex_lock(&executor.mutex);
  executor.reject = 1;
  pthread_mutex_unlock(&executor.mutex);
  result = loon_transaction_begin_async(io_context, "segment", &properties, 0, 0, 0, NULL, begin_complete,
                                        (uintptr_t)&state, &operation);
  ck_assert_int_eq(result.err_code, LOON_ASYNC_OVERLOADED);
  ck_assert(!operation && state.calls == 0);
  loon_ffi_free_result(&result);
  pthread_mutex_lock(&executor.mutex);
  executor.reject = 0;
  executor.accepts_remaining = 1;
  pthread_mutex_unlock(&executor.mutex);
  // Initial admission succeeds, but final callback enqueue is rejected.
  result = loon_transaction_begin_async(io_context, "segment", &properties, 0, 0, 0, NULL, begin_complete,
                                        (uintptr_t)&state, &operation);
  ck_assert_int_eq(result.err_code, 0);
  loon_ffi_free_result(&result);
  ck_assert_int_eq(await_begin(&state), 0);
  ck_assert_int_eq(state.calls, 1);
  ck_assert_int_eq(state.code, 0);
  loon_async_release(operation);
  loon_transaction_destroy(state.transaction);
  pthread_mutex_lock(&executor.mutex);
  executor.accepts_remaining = -1;
  pthread_mutex_unlock(&executor.mutex);
  pthread_cond_destroy(&state.ready);
  pthread_mutex_destroy(&state.mutex);
}
// Begin and commit may use different contexts/executors. Shutting down one
// context must not stop another, even when they share the same executor.
static void test_async_context_isolation(void) {
  TestExecutor other_executor;
  LoonAsyncExecutor descriptor = test_executor_start(&other_executor, 1);
  LoonIOContextHandle first = NULL, second = NULL;
  LoonFFIResult result = loon_io_context_create(&descriptor, &first);
  ck_assert_int_eq(result.err_code, 0);
  loon_ffi_free_result(&result);
  result = loon_io_context_create(&descriptor, &second);
  ck_assert_int_eq(result.err_code, 0);
  loon_ffi_free_result(&result);
  // The descriptors are copied, so changing the caller's descriptor is harmless.
  descriptor.submit = NULL;
  struct Completion state = {PTHREAD_MUTEX_INITIALIZER, PTHREAD_COND_INITIALIZER, 0, 0, 0};
  state.expected_executor = &other_executor;
  LoonProperties properties = {NULL, 0};
  LoonAsyncHandle operation = NULL;
  char path[128];
  snprintf(path, sizeof(path), "context-%ld-%ld", (long)time(NULL), (long)getpid());
  result = loon_transaction_begin_async(first, path, &properties, 0, 0, 1, NULL, begin_complete, (uintptr_t)&state,
                                        &operation);
  ck_assert_int_eq(result.err_code, 0);
  loon_ffi_free_result(&result);
  // No await: shutdown must drain the accepted callback, including early release.
  loon_async_release(operation);
  loon_io_context_shutdown(first);
  loon_io_context_shutdown(first);
  ck_assert_int_eq(state.calls, 1);
  ck_assert_int_eq(state.code, 0);
  LoonTransactionHandle transaction = state.transaction;
  state.calls = 0;
  result = loon_transaction_begin_async(first, path, &properties, 0, 0, 1, NULL, begin_complete, (uintptr_t)&state,
                                        &operation);
  ck_assert_int_eq(result.err_code, LOON_ASYNC_OVERLOADED);
  ck_assert(operation == NULL);
  ck_assert_int_eq(state.calls, 0);
  loon_ffi_free_result(&result);
  result = loon_transaction_commit_async(NULL, transaction, NULL, commit_complete, (uintptr_t)&state, &operation);
  ck_assert_int_eq(result.err_code, LOON_INVALID_ARGS);
  ck_assert(operation == NULL);
  loon_ffi_free_result(&result);
  result = loon_transaction_commit_async(first, transaction, NULL, commit_complete, (uintptr_t)&state, &operation);
  ck_assert_int_eq(result.err_code, LOON_ASYNC_OVERLOADED);
  ck_assert(operation == NULL);
  ck_assert_int_eq(state.calls, 0);
  loon_ffi_free_result(&result);
  loon_io_context_destroy(first);
  // Rejection must leave the transaction available to another context.
  result = loon_transaction_add_delta_log(transaction, "_delta/context", 1);
  ck_assert_int_eq(result.err_code, 0);
  loon_ffi_free_result(&result);
  state.expected_executor = &executor;
  result = loon_transaction_commit_async(io_context, transaction, NULL, commit_complete, (uintptr_t)&state, &operation);
  ck_assert_int_eq(result.err_code, 0);
  loon_ffi_free_result(&result);
  ck_assert_int_eq(await_begin(&state), 0);
  ck_assert_int_eq(state.code, 0);
  ck_assert_int_eq(state.outcome, LOON_COMMIT_COMMITTED);
  loon_async_release(operation);
  loon_transaction_destroy(transaction);
  state.calls = 0;
  state.expected_executor = &other_executor;
  result = loon_transaction_begin_async(second, path, &properties, 1, 0, 1, NULL, begin_complete, (uintptr_t)&state,
                                        &operation);
  ck_assert_int_eq(result.err_code, 0);
  loon_ffi_free_result(&result);
  loon_io_context_destroy(second);
  ck_assert_int_eq(state.calls, 1);
  ck_assert_int_eq(state.code, 0);
  // Completed operation handles do not retain the IO context.
  loon_async_cancel(operation);
  loon_async_release(operation);
  loon_transaction_destroy(state.transaction);
  descriptor.submit = test_executor_submit;
  result = loon_io_context_create(&descriptor, &second);
  ck_assert_int_eq(result.err_code, 0);
  loon_ffi_free_result(&result);
  loon_io_context_destroy(second);
  test_executor_stop(&other_executor);
  loon_io_context_shutdown(NULL);
  loon_io_context_destroy(NULL);
  pthread_cond_destroy(&state.ready);
  pthread_mutex_destroy(&state.mutex);
}
static void mark_executor_alive(void* value) { *(int*)value = 1; }
void run_manifest_async_suite(void) {
  RUN_TEST(test_async_io_context);
  RUN_TEST(test_async_begin_contract);
  RUN_TEST(test_async_begin_local);
  if (getenv("LOON_ASYNC_TEST_ENDPOINT"))
    RUN_TEST(test_async_begin_minio);
  RUN_TEST(test_async_commit_roundtrip);
  RUN_TEST(test_async_context_isolation);
  loon_io_context_destroy(io_context);
  io_context = NULL;
  int executor_alive = 0;
  ck_assert_int_eq(test_executor_submit(&executor, mark_executor_alive, &executor_alive), 0);
  test_executor_stop(&executor);
  ck_assert_int_eq(executor_alive, 1);
}
#ifdef MANIFEST_ASYNC_STANDALONE
int global_tests_run = 0;
int global_tests_failed = 0;
int main(void) {
  run_manifest_async_suite();
  return global_tests_failed != 0;
}
#endif
