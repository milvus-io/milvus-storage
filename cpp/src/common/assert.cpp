#include "assert.h"

#include <cstdio>
#include <stdexcept>

void DuckDBAssertInternal(bool condition, const char *condition_name,
                          const char *file, int linenr) {
  if (condition) {
    return;
  }
  throw std::invalid_argument("received negative value");
}