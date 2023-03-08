#pragma once
#include <set>
#include <string>
#include <vector>

#include "../filter/filter.h"

struct WriteOption {
  int64_t max_record_per_file;
};

struct ReadOption {
  std::vector<Filter *> filters;
  std::vector<std::string> columns;
  int limit;
  int version;
};

struct SpaceOption {
  std::string primary_column;
  std::string version_column;
  std::string vector_column;
};