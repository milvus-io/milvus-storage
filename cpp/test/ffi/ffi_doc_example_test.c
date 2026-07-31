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
// It shipped as a `switch` over loon_error_category_*, which does not compile
// in C at all: those symbols are `extern const int` (ffi_c.h), and C requires
// case labels to be integer constant expressions. Anyone following the
// documentation got seven errors from their compiler.
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

  const char* verdict = NULL;
  int category = loon_ffi_error_category(result->err_code);
  if (category == loon_error_category_transient) {
    verdict = "retry";
  } else if (category == loon_error_category_conflict) {
    verdict = "re-read, rebase, re-submit";
  } else if (category == loon_error_category_missing) {
    verdict = "re-read metadata, then decide";
  } else if (category == loon_error_category_user) {
    verdict = "return to caller";
  } else if (category == loon_error_category_config) {
    verdict = "alert an operator";
  } else if (category == loon_error_category_corrupted) {
    verdict = "quarantine and re-fetch";
  } else if (category == loon_error_category_permanent) {
    verdict = "alert a developer";
  } else {
    verdict = "unknown: treat as permanent";
  }

  // The docs example frees on every path, and so does this: the message is
  // strdup'd, so a retry loop that classifies and loops without freeing leaks
  // once per attempt -- once per attempt of exactly the failures it retries.
  loon_ffi_free_result(result);
  return verdict;
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
      {loon_errcode_transient_network, "retry"},
      {loon_errcode_aws_conflict, "re-read, rebase, re-submit"},
      {loon_errcode_file_not_found, "re-read metadata, then decide"},
      {loon_errcode_source_invalid, "return to caller"},
      {loon_errcode_storage_config_invalid, "alert an operator"},
      {loon_errcode_vortex_file_corrupted, "quarantine and re-fetch"},
      {loon_errcode_logical, "alert a developer"},
  };

  for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); i++) {
    LoonFFIResult rc = {cases[i].code, NULL};
    ck_assert_str_eq(classify_like_the_docs(&rc), cases[i].expected);
  }

  // And a code from the future degrades to the safe answer rather than
  // matching some branch by accident.
  LoonFFIResult unseen = {9999, NULL};
  ck_assert_str_eq(classify_like_the_docs(&unseen), "unknown: treat as permanent");
}

void run_doc_example_suite(void) {
  RUN_TEST(test_doc_example_handles_success);
  RUN_TEST(test_doc_example_covers_every_category);
}
