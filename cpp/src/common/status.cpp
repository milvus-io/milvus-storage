#include "status.h"
#include <cstdio>
#include <string>
namespace milvus_storage {
Status::Status(const Status& s) : code_(s.code_), msg_(s.msg_) {}

Status& Status::operator=(const Status& s) {
  code_ = s.code_;
  msg_ = s.msg_;
  return *this;
}

std::string Status::ToString() const {
  char tmp[30];
  std::string res;
  switch (code_) {
    case kOk:
      return "OK";
      break;
    case kArrowError:
      res = "ArrowError: ";
      break;
    case kInvalidArgument:
      res = "InvalidArgument: ";
      break;
    case kInternalStateError:
      res = "InternalStateError: ";
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