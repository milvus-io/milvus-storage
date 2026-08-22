// Copyright 2025 Zilliz
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

// The consumer example from docs/error-codes.md, compiled.
//
// Documentation that is only read cannot be wrong loudly. This file is the
// example itself -- if the shape in the docs stops compiling, or a category is
// added without the docs learning about it, this target fails.

#include "test_runner.h"

#include <stdio.h>

#include "milvus-storage/ffi_c.h"

// Verbatim from docs/error-codes.md ("How to consume it"). Keep the two in
// step: this function exists to be the same code, not merely similar code.
static const char* classify_like_the_docs(LoonFFIResult* result) {
  if (loon_ffi_is_success(result)) {
    loon_ffi_free_result(result);
    return "success";
  }

  const char* verdict;
  switch (loon_ffi_error_category(result->err_code)) {
    case loon_error_category_retryable:
      verdict = "transient hint; caller decides";
      break;
    case loon_error_category_conflict:
      verdict = "business handles conflict";
      break;
    case loon_error_category_user:
      verdict = "return to caller";
      break;
    case loon_error_category_data_format:
      verdict = "report data format failure";
      break;
    case loon_error_category_system:
      verdict = "report system failure";
      break;
    default:
      verdict = "unknown: treat as system";
      break;
  }

  // The docs example frees on every path, and so does this: the message is
  // strdup'd, so a retry loop that classifies and loops without freeing leaks
  // once per attempt -- once per attempt of exactly the failures it retries.
  loon_ffi_free_result(result);
  return verdict;
}

// The MINIMUM from docs/error-codes.md: one question, three answers. A consumer
// that implements only this is correct, not partially correct. Kept verbatim
// beside the full switch above so the claim "the short form is safe" is
// something CI can falsify rather than something the docs assert.
static const char* handle_minimally(LoonFFIResult* result) {
  const char* verdict;
  if (loon_ffi_is_success(result)) {
    verdict = "use it";
  } else if (loon_ffi_is_retryable_errcode(result->err_code)) {
    verdict = "retry";
  } else if (loon_ffi_error_category(result->err_code) == loon_error_category_user) {
    verdict = "return to caller";
  } else {
    verdict = "report, do not retry";
  }
  loon_ffi_free_result(result);
  return verdict;
}

static void test_minimal_form_defaults_to_the_conservative_answer(void) {
  // Everything that is not positively retryable and not the caller's input has
  // to land in the same conservative branch -- including categories the short
  // form never names, and codes it has never seen. If a future category slips
  // into "retry" here, a generic caller starts replaying something nobody said
  // was safe to replay.
  struct {
    int code;
    const char* expected;
  } cases[] = {
      {loon_errcode_success, "use it"},
      {loon_errcode_transient_throttling, "retry"},
      {loon_errcode_user_invalid_argument, "return to caller"},
      {loon_errcode_source_invalid, "report, do not retry"},
      {loon_errcode_storage_conflict, "report, do not retry"},
      {loon_errcode_txn_exhausted_retry, "report, do not retry"},
      {loon_errcode_data_corrupted, "report, do not retry"},
      {loon_errcode_storage_access_denied, "report, do not retry"},
      {9999, "report, do not retry"},
  };

  for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); i++) {
    LoonFFIResult rc = {cases[i].code, NULL};
    ck_assert_str_eq(handle_minimally(&rc), cases[i].expected);
  }
}

static void test_retryable_and_category_cannot_disagree(void) {
  // The short form asks two different functions and trusts them to agree. They
  // are generated from one table, and this is what says so out loud: for EVERY
  // exported code, is_retryable is exactly "category == retryable".
  const int codes[] = {
#define MILVUS_STORAGE_ERRCODE_VALUE(name, code, symbol, category, s3_code) loon_errcode_##symbol,
      LOON_INTERNAL_ERROR_CODE_LIST(MILVUS_STORAGE_ERRCODE_VALUE)
          LOON_EXTEND_STATUS_CODE_LIST(MILVUS_STORAGE_ERRCODE_VALUE)
#undef MILVUS_STORAGE_ERRCODE_VALUE
              loon_errcode_success,
      9999,
  };

  for (size_t i = 0; i < sizeof(codes) / sizeof(codes[0]); i++) {
    int retryable = loon_ffi_is_retryable_errcode(codes[i]) != 0;
    int is_retryable_category = loon_ffi_error_category(codes[i]) == loon_error_category_retryable;
    ck_assert_int_eq(retryable, is_retryable_category);
  }
}

static void test_doc_example_handles_success(void) {
  // The bug this pins: loon_ffi_error_category(LOON_SUCCESS) is UNKNOWN,
  // because the function answers "which kind of failure" and success is not
  // one. Without the is_success check first, every successful call lands in the
  // final branch and gets alerted on as a permanent failure.
  LoonFFIResult ok = {loon_errcode_success, NULL};
  ck_assert_str_eq(classify_like_the_docs(&ok), "success");
  ck_assert_int_eq(loon_ffi_error_category(loon_errcode_success), loon_error_category_unknown);
}

static void test_doc_example_covers_every_category(void) {
  // One representative per category, so a category added without a branch here
  // -- and therefore without one in the docs -- shows up as "unknown".
  struct {
    int code;
    const char* expected;
  } cases[] = {
      {loon_errcode_transient_network, "transient hint; caller decides"},
      {loon_errcode_storage_conflict, "business handles conflict"},
      {loon_errcode_user_invalid_argument, "return to caller"},
      {loon_errcode_vortex_data_format, "report data format failure"},
      {loon_errcode_file_not_found, "report system failure"},
  };

  for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); i++) {
    LoonFFIResult rc = {cases[i].code, NULL};
    ck_assert_str_eq(classify_like_the_docs(&rc), cases[i].expected);
  }

  // And a code from the future degrades to the safe answer rather than
  // matching some branch by accident.
  LoonFFIResult unseen = {9999, NULL};
  ck_assert_str_eq(classify_like_the_docs(&unseen), "unknown: treat as system");
}

void run_doc_example_suite(void) {
  RUN_TEST(test_doc_example_handles_success);
  RUN_TEST(test_doc_example_covers_every_category);
  RUN_TEST(test_minimal_form_defaults_to_the_conservative_answer);
  RUN_TEST(test_retryable_and_category_cannot_disagree);
}
