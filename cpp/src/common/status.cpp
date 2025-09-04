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

#include "milvus-storage/common/status.h"
#include <cstdio>
#include <string>
namespace milvus_storage {

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
    case kFileNotFound:
      res = "FileNotFound: ";
    default:
      std::sprintf(tmp, "Unknown code(%d): ", code_);
      res = tmp;
      break;
  }
  res.append(msg_);
  return res;
}

}  // namespace milvus_storage
