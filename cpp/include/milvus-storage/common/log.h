

#pragma once

#include <string>
#include <sys/types.h>
#include <unistd.h>
#include "glog/logging.h"

// namespace milvus {

#if __GLIBC__ == 2 && __GLIBC_MINOR__ < 30
#include <sys/syscall.h>
#define gettid() syscall(SYS_gettid)
#endif

// Log message format: %timestamp | %request_id | %level | %collection_name | %client_id | %client_tag | %client_ipport
// | %thread_id | %thread_start_timestamp | %command_tag | %module | %error_code | %message

#define VAR_REQUEST_ID (context->request_id())
#define VAR_COLLECTION_NAME (context->collection_name())
#define VAR_CLIENT_ID ("")
#define VAR_CLIENT_TAG (context->client_tag())
#define VAR_CLIENT_IPPORT (context->client_ipport())
#define VAR_THREAD_ID (gettid())
#define VAR_THREAD_START_TIMESTAMP (get_thread_start_timestamp())
#define VAR_COMMAND_TAG (context->command_tag())

/////////////////////////////////////////////////////////////////////////////////////////////////
#define STORAGE_MODULE_NAME "STORAGE"
#define STORAGE_MODULE_CLASS_FUNCTION \
  LogOut("[%s][%s::%s][%s] ", STORAGE_MODULE_NAME, (typeid(*this).name()), __FUNCTION__, GetThreadName().c_str())
#define STORAGE_MODULE_FUNCTION LogOut("[%s][%s][%s] ", STORAGE_MODULE_NAME, __FUNCTION__, GetThreadName().c_str())

#define LOG_STORAGE_TRACE_ DLOG(INFO) << STORAGE_MODULE_FUNCTION
#define LOG_STORAGE_DEBUG_ DLOG(INFO) << STORAGE_MODULE_FUNCTION
#define LOG_STORAGE_INFO_ LOG(INFO) << STORAGE_MODULE_FUNCTION
#define LOG_STORAGE_WARNING_ LOG(WARNING) << STORAGE_MODULE_FUNCTION
#define LOG_STORAGE_ERROR_ LOG(ERROR) << STORAGE_MODULE_FUNCTION
#define LOG_STORAGE_FATAL_ LOG(FATAL) << STORAGE_MODULE_FUNCTION

/////////////////////////////////////////////////////////////////////////////////////////////////

std::string LogOut(const char* pattern, ...);

void SetThreadName(const std::string_view name);

std::string GetThreadName();

int64_t get_thread_start_timestamp();

// }  // namespace milvus
