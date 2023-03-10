#pragma once
void DuckDBAssertInternal(bool condition, const char *condition_name,
                          const char *file, int linenr);

#define D_ASSERT(condition) \
  duckdb::DuckDBAssertInternal(bool(condition), #condition, __FILE__, __LINE__)
