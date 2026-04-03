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

#pragma once

#include <string>
#include <typeinfo>
#include <sys/types.h>
#include <unistd.h>
#include "glog/logging.h"
#include <fmt/format.h>

#if __GLIBC__ == 2 && __GLIBC_MINOR__ < 30
#include <sys/syscall.h>
#define gettid() syscall(SYS_gettid)
#endif

// GLOG has no debug and trace level, using VLOG to implement it.
#define GLOG_DEBUG 5
#define GLOG_TRACE 6

/////////////////////////////////////////////////////////////////////////////////////////////////
#define STORAGE_MODULE_NAME "STORAGE"
#define STORAGE_MODULE_CLASS_FUNCTION \
  fmt::format("[{}][{}::{}][{}] ", STORAGE_MODULE_NAME, typeid(*this).name(), __FUNCTION__, GetThreadName())
#define STORAGE_MODULE_FUNCTION fmt::format("[{}][{}][{}] ", STORAGE_MODULE_NAME, __FUNCTION__, GetThreadName())

#define LOG_STORAGE_TRACE_ VLOG(GLOG_TRACE) << STORAGE_MODULE_FUNCTION
#define LOG_STORAGE_DEBUG_ VLOG(GLOG_DEBUG) << STORAGE_MODULE_FUNCTION
#define LOG_STORAGE_INFO_ LOG(INFO) << STORAGE_MODULE_FUNCTION
#define LOG_STORAGE_WARNING_ LOG(WARNING) << STORAGE_MODULE_FUNCTION
#define LOG_STORAGE_ERROR_ LOG(ERROR) << STORAGE_MODULE_FUNCTION
#define LOG_STORAGE_FATAL_ LOG(FATAL) << STORAGE_MODULE_FUNCTION

/////////////////////////////////////////////////////////////////////////////////////////////////

std::string GetThreadName();
