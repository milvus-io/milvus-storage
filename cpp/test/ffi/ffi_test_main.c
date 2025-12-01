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
#include "test_runner.h"
#include <stdlib.h>
#include <stdio.h>

int global_tests_run = 0;
int global_tests_failed = 0;

void run_properties_suite(void);
void run_writer_suite(void);
void run_reader_suite(void);
void run_manifest_suite(void);
void run_external_suite(void);

int main(void) {
  run_manifest_suite();
  run_properties_suite();
  run_writer_suite();
  run_reader_suite();
  run_external_suite();

  close_filesystems();

  printf("\nRan %d tests, %d failed.\n", global_tests_run, global_tests_failed);
  return (global_tests_failed == 0) ? 0 : 1;
}
