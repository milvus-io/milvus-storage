#pragma once

class Filter {
 public:
  virtual bool CheckStatistics();
  virtual bool Apply();
};