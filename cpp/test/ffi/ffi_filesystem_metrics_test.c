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

#include <stdint.h>
#include <stdlib.h>

#include "milvus-storage/ffi_filesystem_metrics_c.h"
#include "test_runner.h"

static void test_bucket_bounds_are_monotonic(void) {
  int32_t n = 0;
  const int64_t* lat = loon_fs_latency_bucket_bounds_us(&n);
  ck_assert_int_eq(n, LOON_LATENCY_BUCKETS);
  for (int i = 1; i < n; ++i) {
    ck_assert_int_gt(lat[i], lat[i - 1]);
  }

  int32_t sn = 0;
  const int64_t* sz = loon_fs_size_bucket_bounds_bytes(&sn);
  ck_assert_int_eq(sn, LOON_SIZE_BUCKETS);
  for (int i = 1; i < sn; ++i) {
    ck_assert_int_gt(sz[i], sz[i - 1]);
  }
}

static void test_snapshot_struct_is_zeroable(void) {
  LoonFilesystemMetricsSnapshot s = {0};
  ck_assert_int_eq(s.ops[0].latency_count, 0);
  ck_assert_int_eq(s.transfers[0].bytes_total, 0);
  ck_assert_int_eq(s.in_flight, 0);
}

static void test_get_metrics_rejects_null(void) {
  LoonFilesystemMetricsSnapshot snap = {0};
  // rejects a null handle with LOON_INVALID_ARGS (err_code != 0)
  LoonFFIResult r = loon_filesystem_get_metrics(NULL, &snap);
  ck_assert(r.err_code != 0);
  if (r.message) {
    free(r.message);
  }
}

void run_metrics_snapshot_suite(void) {
  RUN_TEST(test_bucket_bounds_are_monotonic);
  RUN_TEST(test_snapshot_struct_is_zeroable);
  RUN_TEST(test_get_metrics_rejects_null);
}
