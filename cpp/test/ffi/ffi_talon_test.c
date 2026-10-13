// Copyright 2026 Zilliz. Licensed under the Apache License, Version 2.0.
#include "milvus-storage/ffi_talon_c.h"
#include "async_test_executor.h"
#include "ffi_test_env.h"
#include "test_runner.h"

#include <arpa/inet.h>
#include <errno.h>
#include <netinet/in.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <sys/select.h>
#include <sys/socket.h>
#include <time.h>
#include <unistd.h>

typedef struct {
  pthread_mutex_t mutex;
  pthread_cond_t ready;
  int calls;
  int code;
  LoonTalonResult* result;
  pthread_t callback_thread;
} TalonCompletion;

static void talon_complete(uintptr_t user_data, LoonFFIResult status, LoonTalonResult* result) {
  TalonCompletion* completion = (TalonCompletion*)user_data;
  pthread_mutex_lock(&completion->mutex);
  completion->calls++;
  completion->code = status.err_code;
  completion->result = result;
  completion->callback_thread = pthread_self();
  pthread_cond_signal(&completion->ready);
  pthread_mutex_unlock(&completion->mutex);
  loon_ffi_free_result(&status);
}

static void completion_init(TalonCompletion* completion) {
  memset(completion, 0, sizeof(*completion));
  pthread_mutex_init(&completion->mutex, NULL);
  pthread_cond_init(&completion->ready, NULL);
}

static int completion_wait_for(TalonCompletion* completion, int seconds) {
  struct timespec end;
  clock_gettime(CLOCK_REALTIME, &end);
  end.tv_sec += seconds;
  pthread_mutex_lock(&completion->mutex);
  while (!completion->calls && pthread_cond_timedwait(&completion->ready, &completion->mutex, &end) == 0) {
  }
  int calls = completion->calls;
  pthread_mutex_unlock(&completion->mutex);
  return calls;
}

static int completion_wait(TalonCompletion* completion) { return completion_wait_for(completion, 10); }

static void completion_destroy(TalonCompletion* completion) {
  loon_talon_result_free(completion->result);
  pthread_cond_destroy(&completion->ready);
  pthread_mutex_destroy(&completion->mutex);
}

static LoonTalonClientHandle test_client(void) {
  LoonProperty values[] = {
      {"fs.talon.coordinator", "127.0.0.1:1"}, {"fs.talon.block_size", "8388608"}, {"fs.talon.max_idle_per_addr", "8"}};
  LoonProperties properties = {values, sizeof(values) / sizeof(values[0])};
  LoonTalonClientHandle client = 0;
  LoonFFIResult status = loon_talon_client_create(&properties, &client);
  int code = status.err_code;
  loon_ffi_free_result(&status);
  return code == LOON_SUCCESS ? client : 0;
}

static void test_talon_header_and_empty_results(void) {
  LoonTalonClientHandle client = test_client();
  ck_assert(client != 0);
  LoonTalonResult* result = (LoonTalonResult*)(uintptr_t)1;
  LoonFFIResult status = loon_talon_batch_load(client, NULL, 0, &result);
  ck_assert_int_eq(status.err_code, LOON_SUCCESS);
  ck_assert(result != NULL);
  ck_assert_int_eq(result->operation, LOON_TALON_BATCH_LOAD);
  ck_assert(!result->data.load.failed);
  ck_assert_int_eq(result->data.load.count, 0);
  loon_ffi_free_result(&status);
  loon_talon_result_free(result);
  result = NULL;
  uint64_t zero = 0;
  status = loon_talon_read(client, "s3://bucket/empty", 0, NULL, 0, NULL, &zero, &result);
  ck_assert_int_eq(status.err_code, LOON_SUCCESS);
  ck_assert(result != NULL);
  ck_assert_int_eq(result->operation, LOON_TALON_READ);
  ck_assert_int_eq(result->data.bytes_read, 0);
  loon_ffi_free_result(&status);
  loon_talon_result_free(result);
  LoonTalonLoadRequest load = {"s3://bucket/empty", 0, NULL};
  result = NULL;
  status = loon_talon_load(client, &load, &result);
  ck_assert_int_eq(status.err_code, LOON_SUCCESS);
  ck_assert(result != NULL);
  ck_assert_int_eq(result->operation, LOON_TALON_LOAD);
  ck_assert(!result->data.load.failed);
  ck_assert_int_eq(result->data.load.count, 1);
  ck_assert_int_eq(result->data.load.entries.results[0].size, 0);
  loon_ffi_free_result(&status);
  loon_talon_result_free(result);
  loon_talon_result_free(NULL);
  loon_talon_client_destroy(client);
  loon_talon_client_destroy(0);
}

static void test_talon_invalid_inputs(void) {
  LoonTalonClientHandle client = 11;
  LoonFFIResult status = loon_talon_client_create(NULL, &client);
  ck_assert_int_eq(status.err_code, LOON_INVALID_ARGS);
  ck_assert_int_eq(client, 0);
  loon_ffi_free_result(&status);
  client = test_client();
  ck_assert(client != 0);
  LoonTalonResult* result = (LoonTalonResult*)(uintptr_t)1;
  status = loon_talon_read(client, "s3://bucket/key", 0, NULL, 1, NULL, NULL, &result);
  ck_assert_int_eq(status.err_code, LOON_INVALID_ARGS);
  ck_assert(result == NULL);
  loon_ffi_free_result(&status);
  status = loon_talon_read(client, "s3://bucket/key", 0, NULL, 0, "etag", NULL, &result);
  ck_assert_int_eq(status.err_code, LOON_INVALID_ARGS);
  ck_assert(result == NULL);
  loon_ffi_free_result(&status);
  status = loon_talon_batch_load(client, NULL, 1, &result);
  ck_assert_int_eq(status.err_code, LOON_INVALID_ARGS);
  ck_assert(result == NULL);
  loon_ffi_free_result(&status);
  // A failed batch can have no per-file details; the tag still selects failures.
  LoonTalonLoadRequest invalid = {"invalid", 0, NULL};
  status = loon_talon_batch_load(client, &invalid, 1, &result);
  ck_assert_int_eq(status.err_code, LOON_INVALID_ARGS);
  ck_assert(result && result->operation == LOON_TALON_BATCH_LOAD);
  ck_assert(result->data.load.failed);
  ck_assert_int_eq(result->data.load.count, 0);
  loon_ffi_free_result(&status);
  loon_talon_result_free(result);
  loon_talon_client_destroy(client);
}

static void test_talon_valid_properties_have_no_new_size_caps(void) {
  LoonProperty values[1028] = {{(char*)"fs.talon.coordinator", (char*)"127.0.0.1:1"},
                               {(char*)"fs.talon.block_size", (char*)"8388608"},
                               {(char*)"fs.talon.max_idle_per_addr", (char*)"8"}};
  char keys[1025][32];
  for (size_t i = 0; i < 1025; ++i) {
    snprintf(keys[i], sizeof(keys[i]), "test.extra.%zu", i);
    values[i + 3] = (LoonProperty){keys[i], (char*)"value"};
  }
  char long_key[1026], long_value[16386];
  memset(long_key, 'k', sizeof(long_key) - 1);
  long_key[sizeof(long_key) - 1] = '\0';
  memset(long_value, 'v', sizeof(long_value) - 1);
  long_value[sizeof(long_value) - 1] = '\0';
  values[1027] = (LoonProperty){long_key, long_value};
  LoonProperties properties = {values, 1028};
  LoonTalonClientHandle client = 0;
  LoonFFIResult status = loon_talon_client_create(&properties, &client);
  ck_assert_msg(status.err_code == LOON_SUCCESS, "%s", loon_ffi_get_errmsg(&status));
  ck_assert(client != 0);
  loon_ffi_free_result(&status);
  loon_talon_client_destroy(client);
}

static void test_talon_async_admission_and_lifetime(void) {
  TestExecutor executor;
  LoonAsyncExecutor descriptor = test_executor_start(&executor, 0);
  LoonAsyncContextHandle context = NULL;
  LoonFFIResult status = loon_async_context_create(&descriptor, &context);
  ck_assert_int_eq(status.err_code, LOON_SUCCESS);
  loon_ffi_free_result(&status);
  LoonTalonClientHandle client = test_client();
  ck_assert(client != 0);
  TalonCompletion states[5];
  for (int i = 0; i < 5; ++i) completion_init(&states[i]);
  LoonAsyncHandle operation = {0};
  uint64_t size = 0;
  executor.reject = 1;
  status = loon_talon_read_async(context, client, "s3://bucket/empty", 0, NULL, 0, NULL, &size, talon_complete,
                                 (uintptr_t)&states[0], &operation);
  ck_assert_int_eq(status.err_code, LOON_ASYNC_OVERLOADED);
  ck_assert(operation.internal == NULL && states[0].calls == 0);
  loon_ffi_free_result(&status);
  executor.reject = 0;
  status = loon_talon_read_async(context, client, "s3://bucket/empty", 0, NULL, 0, NULL, &size, talon_complete,
                                 (uintptr_t)&states[0], &operation);
  ck_assert_int_eq(status.err_code, LOON_SUCCESS);
  loon_ffi_free_result(&status);
  status = loon_talon_read_async(context, client, "s3://bucket/empty", 0, NULL, 0, NULL, &size, talon_complete,
                                 (uintptr_t)&states[0], &operation);
  ck_assert_int_eq(status.err_code, LOON_ASYNC_BUSY);
  loon_ffi_free_result(&status);
  loon_async_cancel(&operation);
  loon_async_release(&operation);
  operation.timeout_ms = 1;
  status = loon_talon_batch_load_async(context, client, NULL, 0, talon_complete, (uintptr_t)&states[1], &operation);
  ck_assert_int_eq(status.err_code, LOON_SUCCESS);
  loon_ffi_free_result(&status);
  loon_async_release(&operation);
  operation.timeout_ms = 0;
  char uri[] = "s3://bucket/empty";
  uint8_t read_byte = 0;
  status = loon_talon_read_async(context, client, uri, 0, &read_byte, 1, NULL, &size, talon_complete,
                                 (uintptr_t)&states[2], &operation);
  ck_assert_int_eq(status.err_code, LOON_SUCCESS);
  loon_ffi_free_result(&status);
  loon_async_release(&operation);
  memset(uri, 'x', sizeof(uri) - 1);
  size = 1;
  char load_uri[] = "s3://bucket/empty";
  LoonTalonLoadRequest load = {load_uri, 0, NULL};
  status = loon_talon_load_async(context, client, &load, talon_complete, (uintptr_t)&states[3], &operation);
  ck_assert_int_eq(status.err_code, LOON_SUCCESS);
  loon_ffi_free_result(&status);
  loon_async_release(&operation);
  memset(load_uri, 'x', sizeof(load_uri) - 1);
  char batch_uri[] = "s3://bucket/empty";
  LoonTalonLoadRequest batch[] = {{batch_uri, 0, NULL}};
  status = loon_talon_batch_load_async(context, client, batch, 1, talon_complete, (uintptr_t)&states[4], &operation);
  ck_assert_int_eq(status.err_code, LOON_SUCCESS);
  loon_ffi_free_result(&status);
  loon_async_release(&operation);
  memset(batch_uri, 'x', sizeof(batch_uri) - 1);
  loon_talon_client_destroy(client);
  usleep(5000);
  ck_assert_int_eq(pthread_create(&executor.threads[0], NULL, test_executor_worker, &executor), 0);
  executor.workers = 1;
  ck_assert_int_eq(completion_wait(&states[0]), 1);
  ck_assert_int_eq(completion_wait(&states[1]), 1);
  ck_assert_int_eq(completion_wait(&states[2]), 1);
  ck_assert_int_eq(completion_wait(&states[3]), 1);
  ck_assert_int_eq(completion_wait(&states[4]), 1);
  ck_assert_int_eq(states[0].code, LOON_ASYNC_CANCELLED);
  ck_assert_int_eq(states[1].code, LOON_ASYNC_DEADLINE);
  ck_assert_int_eq(states[2].code, LOON_SUCCESS);
  ck_assert(states[2].result && states[2].result->data.bytes_read == 0);
  ck_assert(pthread_equal(states[2].callback_thread, executor.threads[0]));
  ck_assert_int_eq(states[3].code, LOON_SUCCESS);
  ck_assert(states[3].result && !states[3].result->data.load.failed && states[3].result->data.load.count == 1);
  ck_assert_int_eq(states[4].code, LOON_SUCCESS);
  ck_assert(states[4].result && !states[4].result->data.load.failed && states[4].result->data.load.count == 1);
  loon_async_context_destroy(context);
  test_executor_stop(&executor);
  for (int i = 0; i < 5; ++i) completion_destroy(&states[i]);
}

typedef struct {
  pthread_mutex_t mutex;
  pthread_cond_t ready;
  int listener;
  int accepted;
  int release;
} HeldStatServer;

static void* hold_stat_connection(void* data) {
  HeldStatServer* server = (HeldStatServer*)data;
  fd_set readable;
  FD_ZERO(&readable);
  FD_SET(server->listener, &readable);
  struct timeval timeout = {8, 0};
  int ready = select(server->listener + 1, &readable, NULL, NULL, &timeout);
  int connection = ready == 1 ? accept(server->listener, NULL, NULL) : -1;
  pthread_mutex_lock(&server->mutex);
  server->accepted = connection >= 0;
  pthread_cond_broadcast(&server->ready);
  while (!server->release && connection >= 0) pthread_cond_wait(&server->ready, &server->mutex);
  pthread_mutex_unlock(&server->mutex);
  if (connection >= 0)
    close(connection);
  return NULL;
}

static void test_talon_native_wait_does_not_hold_caller_worker(void) {
  HeldStatServer server = {0};
  pthread_mutex_init(&server.mutex, NULL);
  pthread_cond_init(&server.ready, NULL);
  server.listener = socket(AF_INET, SOCK_STREAM, 0);
  ck_assert(server.listener >= 0);
  struct sockaddr_in address = {0};
  address.sin_family = AF_INET;
  address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  address.sin_port = 0;
  ck_assert_int_eq(bind(server.listener, (struct sockaddr*)&address, sizeof(address)), 0);
  ck_assert_int_eq(listen(server.listener, 1), 0);
  socklen_t address_size = sizeof(address);
  ck_assert_int_eq(getsockname(server.listener, (struct sockaddr*)&address, &address_size), 0);
  pthread_t server_thread;
  ck_assert_int_eq(pthread_create(&server_thread, NULL, hold_stat_connection, &server), 0);
  char coordinator[64];
  snprintf(coordinator, sizeof(coordinator), "127.0.0.1:%u", ntohs(address.sin_port));
  LoonProperty values[] = {
      {"fs.talon.coordinator", coordinator}, {"fs.talon.block_size", "8388608"}, {"fs.talon.max_idle_per_addr", "8"}};
  LoonProperties properties = {values, 3};
  LoonTalonClientHandle client = 0;
  LoonFFIResult status = loon_talon_client_create(&properties, &client);
  int creation_code = status.err_code;
  loon_ffi_free_result(&status);
  TestExecutor executor;
  LoonAsyncExecutor descriptor = test_executor_start(&executor, 1);
  LoonAsyncContextHandle context = NULL;
  status = loon_async_context_create(&descriptor, &context);
  int context_code = status.err_code;
  loon_ffi_free_result(&status);
  TalonCompletion stat, read;
  completion_init(&stat);
  completion_init(&read);
  LoonAsyncHandle stat_operation = {0}, read_operation = {0};
  int rejected_admission = -1, rejected_connection = 0;
  int stat_admission = -1, read_admission = -1, connected = 0, read_completed_while_held = 0;
  if (creation_code == 0 && context_code == 0) {
    pthread_mutex_lock(&executor.mutex);
    executor.reject = 1;
    pthread_mutex_unlock(&executor.mutex);
    status =
        loon_talon_stat_async(context, client, "s3://bucket/key", talon_complete, (uintptr_t)&stat, &stat_operation);
    rejected_admission = status.err_code;
    loon_ffi_free_result(&status);
    usleep(100000);
    pthread_mutex_lock(&server.mutex);
    rejected_connection = server.accepted;
    pthread_mutex_unlock(&server.mutex);
    pthread_mutex_lock(&executor.mutex);
    executor.reject = 0;
    pthread_mutex_unlock(&executor.mutex);
    status =
        loon_talon_stat_async(context, client, "s3://bucket/key", talon_complete, (uintptr_t)&stat, &stat_operation);
    stat_admission = status.err_code;
    loon_ffi_free_result(&status);
    struct timespec end;
    clock_gettime(CLOCK_REALTIME, &end);
    end.tv_sec += 8;
    pthread_mutex_lock(&server.mutex);
    while (!server.accepted && pthread_cond_timedwait(&server.ready, &server.mutex, &end) == 0) {
    }
    connected = server.accepted;
    pthread_mutex_unlock(&server.mutex);
    if (connected) {
      uint64_t zero = 0;
      status = loon_talon_read_async(context, client, "s3://bucket/empty", 0, NULL, 0, NULL, &zero, talon_complete,
                                     (uintptr_t)&read, &read_operation);
      read_admission = status.err_code;
      loon_ffi_free_result(&status);
      read_completed_while_held = completion_wait_for(&read, 3);
    }
  }
  pthread_mutex_lock(&server.mutex);
  server.release = 1;
  pthread_cond_broadcast(&server.ready);
  pthread_mutex_unlock(&server.mutex);
  pthread_join(server_thread, NULL);
  close(server.listener);
  if (context)
    loon_async_context_destroy(context);
  test_executor_stop(&executor);
  loon_async_release(&stat_operation);
  loon_async_release(&read_operation);
  loon_talon_client_destroy(client);
  int read_code = read.code;
  int read_calls = read.calls;
  completion_destroy(&stat);
  completion_destroy(&read);
  pthread_cond_destroy(&server.ready);
  pthread_mutex_destroy(&server.mutex);
  ck_assert_int_eq(creation_code, LOON_SUCCESS);
  ck_assert_int_eq(context_code, LOON_SUCCESS);
  ck_assert_int_eq(rejected_admission, LOON_ASYNC_OVERLOADED);
  ck_assert_int_eq(rejected_connection, 0);
  ck_assert_int_eq(stat_admission, LOON_SUCCESS);
  ck_assert_int_eq(connected, 1);
  ck_assert_int_eq(read_admission, LOON_SUCCESS);
  ck_assert_int_eq(read_completed_while_held, 1);
  ck_assert_int_eq(read_calls, 1);
  ck_assert_int_eq(read_code, LOON_SUCCESS);
}

static void test_talon_real_raw_operations(void) {
  const char* coordinator = getenv("TEST_ENV_TALON_COORDINATOR");
  const char* bucket = getenv("TEST_ENV_BUCKET_NAME");
  if (!is_cloud_env() || !coordinator || !bucket)
    return;
  const char* keys[32] = {0};
  const char* values[32] = {0};
  size_t count = init_test_props(keys, values, 0, 32, FFI_TEST_ROOT_PATH);
  keys[count] = "fs.talon.coordinator";
  values[count++] = coordinator;
  keys[count] = "fs.talon.block_size";
  values[count++] = "8388608";
  keys[count] = "fs.talon.max_idle_per_addr";
  values[count++] = "8";
  LoonProperty pairs[32];
  for (size_t i = 0; i < count; ++i) pairs[i] = (LoonProperty){(char*)keys[i], (char*)values[i]};
  LoonProperties properties = {pairs, count};
  FileSystemHandle fs = 0;
  LoonFFIResult status = loon_filesystem_get(&properties, "", 0, &fs);
  ck_assert_msg(status.err_code == 0, "%s", loon_ffi_get_errmsg(&status));
  loon_ffi_free_result(&status);
  const char payload[] = "Talon C FFI raw test payload";
  char path[128];
  snprintf(path, sizeof(path), "talon-c-ffi-%ld-%ld", (long)time(NULL), (long)getpid());
  status = loon_filesystem_write_file(fs, path, (uint32_t)strlen(path), (const uint8_t*)payload, sizeof(payload) - 1,
                                      NULL, 0);
  ck_assert_msg(status.err_code == 0, "%s", loon_ffi_get_errmsg(&status));
  loon_ffi_free_result(&status);
  LoonTalonClientHandle client = 0;
  status = loon_talon_client_create(&properties, &client);
  ck_assert_msg(status.err_code == 0, "%s", loon_ffi_get_errmsg(&status));
  loon_ffi_free_result(&status);
  char uri[256];
  const char* scheme =
      getenv("TEST_ENV_CLOUD_PROVIDER") && strcmp(getenv("TEST_ENV_CLOUD_PROVIDER"), "azure") == 0 ? "az" : "s3";
  snprintf(uri, sizeof(uri), "%s://%s/%s", scheme, bucket, path);
  LoonTalonResult* result = NULL;
  status = loon_talon_stat(client, uri, &result);
  ck_assert_msg(status.err_code == 0, "%s", loon_ffi_get_errmsg(&status));
  ck_assert(result && result->operation == LOON_TALON_STAT && result->data.stat.size == sizeof(payload) - 1);
  ck_assert(result->data.stat.version && result->data.stat.version[0]);
  char version[256];
  snprintf(version, sizeof(version), "%s", result->data.stat.version);
  loon_ffi_free_result(&status);
  loon_talon_result_free(result);
  TestExecutor copied_executor;
  LoonAsyncExecutor copied_descriptor = test_executor_start(&copied_executor, 0);
  LoonAsyncContextHandle copied_context = NULL;
  status = loon_async_context_create(&copied_descriptor, &copied_context);
  ck_assert_int_eq(status.err_code, LOON_SUCCESS);
  loon_ffi_free_result(&status);
  TalonCompletion copied_completion;
  completion_init(&copied_completion);
  char mutable_version[256];
  snprintf(mutable_version, sizeof(mutable_version), "%s", version);
  LoonTalonLoadRequest copied_request = {uri, sizeof(payload) - 1, mutable_version};
  LoonAsyncHandle copied_operation = {0};
  status = loon_talon_load_async(copied_context, client, &copied_request, talon_complete, (uintptr_t)&copied_completion,
                                 &copied_operation);
  ck_assert_int_eq(status.err_code, LOON_SUCCESS);
  loon_ffi_free_result(&status);
  loon_async_release(&copied_operation);
  copied_request.size = 0;
  mutable_version[0] = mutable_version[0] == 'x' ? 'y' : 'x';
  ck_assert_int_eq(pthread_create(&copied_executor.threads[0], NULL, test_executor_worker, &copied_executor), 0);
  copied_executor.workers = 1;
  ck_assert_int_eq(completion_wait(&copied_completion), 1);
  ck_assert_int_eq(copied_completion.code, LOON_SUCCESS);
  ck_assert(copied_completion.result && !copied_completion.result->data.load.failed &&
            copied_completion.result->data.load.count == 1);
  ck_assert_int_eq(copied_completion.result->data.load.entries.results[0].size, sizeof(payload) - 1);
  loon_async_context_destroy(copied_context);
  test_executor_stop(&copied_executor);
  completion_destroy(&copied_completion);
  LoonTalonLoadRequest request = {uri, sizeof(payload) - 1, version};
  status = loon_talon_load(client, &request, &result);
  ck_assert_msg(status.err_code == 0, "%s", loon_ffi_get_errmsg(&status));
  ck_assert(result && !result->data.load.failed && result->data.load.count == 1 &&
            result->data.load.entries.results[0].blocks == 1);
  loon_ffi_free_result(&status);
  loon_talon_result_free(result);
  uint8_t output[sizeof(payload)] = {0};
  uint64_t size = sizeof(payload) - 1;
  status = loon_talon_read(client, uri, 0, output, size, version, &size, &result);
  ck_assert_msg(status.err_code == 0, "%s", loon_ffi_get_errmsg(&status));
  ck_assert(result && result->data.bytes_read == (int64_t)size);
  ck_assert(memcmp(output, payload, size) == 0);
  loon_ffi_free_result(&status);
  loon_talon_result_free(result);
  LoonTalonLoadRequest good_batch[] = {{uri, size, version}, {uri, 0, ""}};
  status = loon_talon_batch_load(client, good_batch, 2, &result);
  ck_assert_msg(status.err_code == 0, "%s", loon_ffi_get_errmsg(&status));
  ck_assert(result && !result->data.load.failed && result->data.load.count == 2);
  ck_assert_int_eq(result->data.load.entries.results[0].size, size);
  ck_assert_int_eq(result->data.load.entries.results[0].blocks, 1);
  ck_assert_int_eq(result->data.load.entries.results[1].size, 0);
  loon_ffi_free_result(&status);
  loon_talon_result_free(result);
  LoonTalonLoadRequest batch[] = {{uri, size, version}, {uri, size, "missing-version"}, {uri, 0, ""}};
  status = loon_talon_batch_load(client, batch, 3, &result);
  ck_assert(status.err_code != 0);
  ck_assert(result && result->operation == LOON_TALON_BATCH_LOAD);
  ck_assert(result->data.load.failed);
  ck_assert_int_eq(result->data.load.count, 1);
  ck_assert_int_eq(result->data.load.entries.failures[0].index, 1);
  ck_assert(result->data.load.entries.failures[0].error && result->data.load.entries.failures[0].error[0]);
  loon_ffi_free_result(&status);
  loon_talon_result_free(result);
  TestExecutor executor;
  LoonAsyncExecutor descriptor = test_executor_start(&executor, 1);
  LoonAsyncContextHandle context = NULL;
  status = loon_async_context_create(&descriptor, &context);
  ck_assert_int_eq(status.err_code, 0);
  loon_ffi_free_result(&status);
  TalonCompletion states[5];
  for (int i = 0; i < 5; ++i) completion_init(&states[i]);
  LoonAsyncHandle operations[5] = {{0}};
  status = loon_talon_stat_async(context, client, uri, talon_complete, (uintptr_t)&states[0], &operations[0]);
  ck_assert_int_eq(status.err_code, 0);
  loon_ffi_free_result(&status);
  ck_assert_int_eq(completion_wait(&states[0]), 1);
  ck_assert_int_eq(states[0].code, 0);
  status = loon_talon_load_async(context, client, &request, talon_complete, (uintptr_t)&states[1], &operations[1]);
  ck_assert_int_eq(status.err_code, 0);
  loon_ffi_free_result(&status);
  ck_assert_int_eq(completion_wait(&states[1]), 1);
  ck_assert_int_eq(states[1].code, 0);
  status =
      loon_talon_batch_load_async(context, client, batch, 1, talon_complete, (uintptr_t)&states[2], &operations[2]);
  ck_assert_int_eq(status.err_code, 0);
  loon_ffi_free_result(&status);
  ck_assert_int_eq(completion_wait(&states[2]), 1);
  ck_assert_int_eq(states[2].code, 0);
  ck_assert(states[2].result && !states[2].result->data.load.failed && states[2].result->data.load.count == 1);
  memset(output, 0, sizeof(output));
  status = loon_talon_read_async(context, client, uri, 0, output, size, version, &size, talon_complete,
                                 (uintptr_t)&states[3], &operations[3]);
  ck_assert_int_eq(status.err_code, 0);
  loon_ffi_free_result(&status);
  ck_assert_int_eq(completion_wait(&states[3]), 1);
  ck_assert_int_eq(states[3].code, 0);
  ck_assert(memcmp(output, payload, size) == 0);
  status =
      loon_talon_batch_load_async(context, client, batch, 3, talon_complete, (uintptr_t)&states[4], &operations[4]);
  ck_assert_int_eq(status.err_code, 0);
  loon_ffi_free_result(&status);
  ck_assert_int_eq(completion_wait(&states[4]), 1);
  ck_assert(states[4].code != 0);
  ck_assert(states[4].result && states[4].result->data.load.failed && states[4].result->data.load.count == 1);
  ck_assert_int_eq(states[4].result->data.load.entries.failures[0].index, 1);
  ck_assert(states[4].result->data.load.entries.failures[0].error &&
            states[4].result->data.load.entries.failures[0].error[0]);
  loon_async_context_destroy(context);
  test_executor_stop(&executor);
  for (int i = 0; i < 5; ++i) {
    loon_async_release(&operations[i]);
    completion_destroy(&states[i]);
  }
  loon_talon_client_destroy(client);
  status = loon_filesystem_delete_file(fs, path, (uint32_t)strlen(path));
  loon_ffi_free_result(&status);
  loon_filesystem_destroy(fs);
}

void run_talon_suite(void) {
  RUN_TEST(test_talon_header_and_empty_results);
  RUN_TEST(test_talon_invalid_inputs);
  RUN_TEST(test_talon_valid_properties_have_no_new_size_caps);
  RUN_TEST(test_talon_async_admission_and_lifetime);
  RUN_TEST(test_talon_native_wait_does_not_hold_caller_worker);
  RUN_TEST(test_talon_real_raw_operations);
}
