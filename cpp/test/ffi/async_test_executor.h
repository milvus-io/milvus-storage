// Copyright 2026 Zilliz. Licensed under the Apache License, Version 2.0.
// Test caller's bounded executor. Not part of the storage library.
#pragma once
#include "milvus-storage/ffi_c.h"
#include <pthread.h>
#include <stdlib.h>

typedef struct {
  LoonAsyncTask run;
  void* data;
} TestTask;
typedef struct {
  pthread_mutex_t mutex;
  pthread_cond_t ready;
  pthread_t threads[4];
  TestTask queue[2048];
  unsigned head, count, workers;
  int stopped, reject;
  int accepts_remaining;
} TestExecutor;
static void* test_executor_worker(void* context) {
  TestExecutor* pool = (TestExecutor*)context;
  pthread_mutex_lock(&pool->mutex);
  for (;;) {
    while (!pool->count && !pool->stopped) pthread_cond_wait(&pool->ready, &pool->mutex);
    if (!pool->count && pool->stopped)
      break;
    TestTask task = pool->queue[pool->head];
    pool->head = (pool->head + 1) % 2048;
    --pool->count;
    pthread_mutex_unlock(&pool->mutex);
    task.run(task.data);
    pthread_mutex_lock(&pool->mutex);
  }
  pthread_mutex_unlock(&pool->mutex);
  return NULL;
}
static int32_t test_executor_submit(void* context, LoonAsyncTask task, void* data) {
  TestExecutor* pool = (TestExecutor*)context;
  pthread_mutex_lock(&pool->mutex);
  if (pool->stopped || pool->reject || pool->count == 2048 || pool->accepts_remaining == 0) {
    pthread_mutex_unlock(&pool->mutex);
    return 1;
  }
  if (pool->accepts_remaining > 0)
    --pool->accepts_remaining;
  pool->queue[(pool->head + pool->count) % 2048] = (TestTask){task, data};
  ++pool->count;
  pthread_cond_signal(&pool->ready);
  pthread_mutex_unlock(&pool->mutex);
  return 0;
}
static LoonAsyncExecutor test_executor_start(TestExecutor* pool, unsigned workers) {
  pool->head = pool->count = pool->stopped = pool->reject = 0;
  pool->accepts_remaining = -1;
  pool->workers = workers;
  pthread_mutex_init(&pool->mutex, NULL);
  pthread_cond_init(&pool->ready, NULL);
  for (unsigned i = 0; i < workers; ++i) {
    int error = pthread_create(&pool->threads[i], NULL, test_executor_worker, pool);
    if (error)
      abort();
  }
  LoonAsyncExecutor result = {sizeof(LoonAsyncExecutor), 0, pool, test_executor_submit};
  return result;
}
static void test_executor_stop(TestExecutor* pool) {
  pthread_mutex_lock(&pool->mutex);
  pool->stopped = 1;
  pthread_cond_broadcast(&pool->ready);
  pthread_mutex_unlock(&pool->mutex);
  for (unsigned i = 0; i < pool->workers; ++i) pthread_join(pool->threads[i], NULL);
  pthread_cond_destroy(&pool->ready);
  pthread_mutex_destroy(&pool->mutex);
}
