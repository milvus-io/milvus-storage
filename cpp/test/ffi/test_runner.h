#ifndef TEST_RUNNER_H
#define TEST_RUNNER_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>

extern int global_tests_run;
extern int global_tests_failed;

#define ASSERT_MSG(cond, fmt, ...)                                                  \
  do {                                                                              \
    if (!(cond)) {                                                                  \
      fprintf(stderr, "FAIL: %s:%d: " fmt "\n", __FILE__, __LINE__, ##__VA_ARGS__); \
      global_tests_failed++;                                                        \
      return;                                                                       \
    }                                                                               \
  } while (0)

#define ASSERT_TRUE(cond) ASSERT_MSG(cond, "%s", #cond)

// Mapping libcheck macros
#define ck_assert(cond) ASSERT_TRUE(cond)
#define ck_assert_msg(cond, fmt, ...) ASSERT_MSG(cond, fmt, ##__VA_ARGS__)

#define ck_assert_int_eq(a, b)                    \
  do {                                            \
    long long _a = (long long)(a);                \
    long long _b = (long long)(b);                \
    ASSERT_MSG(_a == _b, "%lld != %lld", _a, _b); \
  } while (0)

#define ck_assert_int_gt(a, b)                   \
  do {                                           \
    long long _a = (long long)(a);               \
    long long _b = (long long)(b);               \
    ASSERT_MSG(_a > _b, "%lld <= %lld", _a, _b); \
  } while (0)

#define ck_assert_str_eq(a, b)                               \
  do {                                                       \
    const char* _a = (const char*)(a);                       \
    const char* _b = (const char*)(b);                       \
    ASSERT_MSG(strcmp(_a, _b) == 0, "'%s' != '%s'", _a, _b); \
  } while (0)

// Helper to run a test function
#define RUN_TEST(func)                             \
  do {                                             \
    fprintf(stdout, "[ RUN      ] %s\n", #func);   \
    int _prev_failed = global_tests_failed;        \
    func();                                        \
    if (global_tests_failed == _prev_failed) {     \
      fprintf(stdout, "[       OK ] %s\n", #func); \
    } else {                                       \
      fprintf(stdout, "[  FAILED  ] %s\n", #func); \
    }                                              \
    global_tests_run++;                            \
  } while (0)

#endif  // TEST_RUNNER_H
