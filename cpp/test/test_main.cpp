// Copyright 2026 Zilliz
// SPDX-License-Identifier: Apache-2.0
#include <gtest/gtest.h>
#include <folly/init/Init.h>

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  // Futures with timeouts and fiber/executor tests use Folly singletons.
  // Complete their registration before any test starts asynchronous work.
  folly::Init folly_init(&argc, &argv, false);
  return RUN_ALL_TESTS();
}
