#include "status.h"
#include <cstdio>
#include <string>
namespace milvus_storage {
std::string
Status::ToString() const {
  char tmp[30];
  std::string res;
  switch (code_) {
    case kOk:
      return "OK";
      break;
    case kArrowError:
      res = "ArrowError: ";
      break;
    default:
      std::sprintf(tmp, "Unknown code(%d): ", code_);
      res = tmp;
      break;
  }
  res.append(msg_);
  return res;
}

}  // namespace milvus_storage