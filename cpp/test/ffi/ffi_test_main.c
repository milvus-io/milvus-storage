#include "milvus-storage/ffi_c.h"
#include <check.h>
#include <stdlib.h>
#include <stdio.h>

Suite* make_properties_suite(void);
Suite* make_writer_suite(void);
Suite* make_reader_suite(void);
Suite* make_manifest_suite(void);

Suite* make_master_suite() {
  Suite* s;
  s = suite_create("FFI test suites");
  return s;
}

int main(void) {
  int failed;
  SRunner* sr;

  setenv("CK_FORK", "NO", 1);
  setenv("CK_DEFAULT_TIMEOUT", "0", 1);
  setenv("CK_VERBOSITY", "verbose", 1);

  sr = srunner_create(make_master_suite());
  srunner_add_suite(sr, make_properties_suite());
  srunner_add_suite(sr, make_writer_suite());
  srunner_add_suite(sr, make_reader_suite());
  srunner_add_suite(sr, make_manifest_suite());
  srunner_set_fork_status(sr, CK_NOFORK);

  srunner_run_all(sr, CK_NORMAL);
  failed = srunner_ntests_failed(sr);
  srunner_free(sr);
  return (failed == 0) ? 0 : 1;
}