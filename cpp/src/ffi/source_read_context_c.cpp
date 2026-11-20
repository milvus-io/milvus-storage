// Copyright 2026 Zilliz
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

#include "milvus-storage/ffi_source_read_context_c.h"

#include <limits>
#include <mutex>
#include <unordered_map>

#include "milvus-storage/ffi_internal/result.h"
#include "milvus-storage/source_read_context.h"

namespace milvus_storage {
namespace {

// Independently releasable handles never reuse an old integer. This prevents a
// stale handle from resolving to a later operation's authorization context.
struct ContextHandles {
  std::mutex mutex;
  uintptr_t next = 1;
  std::unordered_map<LoonSourceReadContextHandle, SourceReadContextPtr> values;
};

ContextHandles& Handles() {
  static ContextHandles handles;
  return handles;
}

LoonSourceReadContextHandle RegisterContext(SourceReadContextPtr context) {
  auto& handles = Handles();
  std::lock_guard lock(handles.mutex);
  if (handles.next == std::numeric_limits<uintptr_t>::max())
    return 0;
  const auto handle = handles.next++;
  handles.values.emplace(handle, std::move(context));
  return handle;
}

bool ValidBuffer(const char* buffer, size_t length, size_t maximum) {
  return length <= maximum && (length == 0 || buffer != nullptr);
}

class CallbackProvider final : public SourceCredentialProvider {
  public:
  explicit CallbackProvider(LoonSourceCredentialCallbacks callbacks) : callbacks_(callbacks) {}

  void AdoptUserdata() { owns_userdata_ = true; }

  ~CallbackProvider() override {
    if (owns_userdata_) {
      try {
        callbacks_.destroy_userdata(callbacks_.userdata);
      } catch (...) {
        // Destructors cannot propagate a callback's contract violation. Never
        // log callback exception text, which may contain secrets.
      }
    }
  }

  SourceCredentialResponse Fetch(const SourceCredentialRequest& request) const override {
    const LoonSourceCredentialRequest ffi_request{1,
                                                  sizeof(LoonSourceCredentialRequest),
                                                  request.resource_uri.data(),
                                                  request.resource_uri.size(),
                                                  request.scope_root_uri.data(),
                                                  request.scope_root_uri.size(),
                                                  request.operation,
                                                  request.request_deadline_unix_ms,
                                                  request.operation_deadline_unix_ms};
    LoonSourceCredentialResult result{};
    result.abi_version = 1;
    result.struct_size = sizeof(result);
    result.status = LOON_SOURCE_CREDENTIAL_INVALID_RESPONSE;
    result.credential_kind = LOON_SOURCE_CREDENTIAL_KIND_UNSPECIFIED;
    SourceCredentialResponse copied;
    try {
      callbacks_.fetch(callbacks_.userdata, &ffi_request, &result);
      if (result.abi_version == 1 && result.struct_size >= sizeof(result)) {
        copied.status = result.status;
        if (result.status == LOON_SOURCE_CREDENTIAL_OK) {
          if (!ValidBuffer(result.account_name, result.account_name_length, 1024) ||
              !ValidBuffer(result.scope_root_uri, result.scope_root_uri_length, 65536) ||
              !ValidBuffer(result.sas_token, result.sas_token_length, 65536)) {
            copied.status = LOON_SOURCE_CREDENTIAL_INVALID_RESPONSE;
          } else {
            copied.kind = result.credential_kind;
            if (result.account_name_length)
              copied.account_name.assign(result.account_name, result.account_name_length);
            if (result.scope_root_uri_length)
              copied.scope_root_uri.assign(result.scope_root_uri, result.scope_root_uri_length);
            if (result.sas_token_length)
              copied.sas_token.assign(result.sas_token, result.sas_token_length);
            copied.expires_at_unix_ms = result.expires_at_unix_ms;
          }
        }
      }
    } catch (...) {
      copied = {};
    }
    // Exactly once, including a throwing fetch and invalid output. The result
    // structure is stack-owned; release_result owns only its referenced buffers.
    try {
      callbacks_.release_result(callbacks_.userdata, &result);
    } catch (...) {
      copied = {};
    }
    return copied;
  }

  private:
  const LoonSourceCredentialCallbacks callbacks_;
  bool owns_userdata_ = false;
};

}  // namespace

SourceReadContextPtr GetSourceReadContext(LoonSourceReadContextHandle handle) {
  auto& handles = Handles();
  std::lock_guard lock(handles.mutex);
  const auto found = handles.values.find(handle);
  return found == handles.values.end() ? nullptr : found->second;
}

}  // namespace milvus_storage

LoonFFIResult loon_source_read_context_create(const LoonSourceReadContextOptions* options,
                                              const LoonSourceCredentialCallbacks* callbacks,
                                              LoonSourceReadContextHandle* out_handle) {
  if (out_handle)
    *out_handle = 0;
  if (!options || !callbacks || !out_handle || options->abi_version != 1 || callbacks->abi_version != 1 ||
      options->struct_size < sizeof(*options) || callbacks->struct_size < sizeof(*callbacks) ||
      !options->account_name || !options->allowed_root_uris || options->allowed_root_count == 0 || !callbacks->fetch ||
      !callbacks->release_result || !callbacks->destroy_userdata) {
    RETURN_ERROR(LOON_INVALID_ARGS, "Invalid source-read context arguments");
  }
  try {
    milvus_storage::SourceReadContextOptions copied;
    copied.account_name = options->account_name;
    copied.operation_deadline_unix_ms = options->operation_deadline_unix_ms;
    copied.request_timeout_ms = options->request_timeout_ms;
    for (size_t i = 0; i < options->allowed_root_count; ++i) {
      if (!options->allowed_root_uris[i])
        RETURN_ERROR(LOON_INVALID_ARGS, "Missing source-read root");
      copied.allowed_root_uris.emplace_back(options->allowed_root_uris[i]);
    }
    auto provider = std::make_shared<milvus_storage::CallbackProvider>(*callbacks);
    auto context = milvus_storage::SourceReadContext::Create(std::move(copied), provider);
    if (!context.ok()) {
      RETURN_ERROR(context.status().IsCapacityError() ? LOON_MEMORY_ERROR : LOON_INVALID_ARGS,
                   "Unable to create source-read context");
    }
    const auto handle = milvus_storage::RegisterContext(*context);
    if (!handle)
      RETURN_ERROR(LOON_MEMORY_ERROR, "Source-read context handles exhausted");
    provider->AdoptUserdata();
    *out_handle = handle;
    RETURN_SUCCESS();
  } catch (const std::bad_alloc&) {
    RETURN_ERROR(LOON_MEMORY_ERROR, "Unable to allocate source-read context");
  } catch (...) {
    RETURN_ERROR(LOON_GOT_EXCEPTION, "Unable to create source-read context");
  }
}

LoonFFIResult loon_source_read_context_retain(LoonSourceReadContextHandle handle,
                                              LoonSourceReadContextHandle* out_handle) {
  if (out_handle)
    *out_handle = 0;
  if (!out_handle)
    RETURN_ERROR(LOON_INVALID_ARGS, "Missing retained context output");
  try {
    auto context = milvus_storage::GetSourceReadContext(handle);
    if (!context)
      RETURN_ERROR(LOON_INVALID_ARGS, "Invalid source-read context handle");
    const auto retained = milvus_storage::RegisterContext(std::move(context));
    if (!retained)
      RETURN_ERROR(LOON_MEMORY_ERROR, "Source-read context handles exhausted");
    *out_handle = retained;
    RETURN_SUCCESS();
  } catch (const std::bad_alloc&) {
    RETURN_ERROR(LOON_MEMORY_ERROR, "Unable to retain source-read context");
  } catch (...) {
    RETURN_ERROR(LOON_GOT_EXCEPTION, "Unable to retain source-read context");
  }
}

void loon_source_read_context_cancel(LoonSourceReadContextHandle handle) {
  if (auto context = milvus_storage::GetSourceReadContext(handle))
    context->Cancel();
}

void loon_source_read_context_release(LoonSourceReadContextHandle handle) {
  milvus_storage::SourceReadContextPtr released;
  auto& handles = milvus_storage::Handles();
  {
    std::lock_guard lock(handles.mutex);
    auto found = handles.values.find(handle);
    if (found != handles.values.end()) {
      released = std::move(found->second);
      handles.values.erase(found);
    }
  }
  // The provider may re-enter context APIs from userdata destruction. Never run
  // that callback under the registry mutex.
}
