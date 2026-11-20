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

#include "milvus-storage/source_read_context.h"
#include "milvus-storage/ffi_internal/ffi_error_code.h"

#include <atomic>
#include <future>
#include <gtest/gtest.h>

namespace milvus_storage {
namespace {
constexpr auto kRoot = "https://acct.blob.core.windows.net/tables/team/table";
constexpr auto kFile = "https://acct.blob.core.windows.net/tables/team/table/data";
struct CallbackState {
  std::atomic<int> fetched{0};
  std::atomic<int> released{0};
  std::atomic<int> destroyed{0};
  std::promise<void>* started = nullptr;
  std::shared_future<void> finish;
};
LoonSourceCredentialCallbacks Callbacks(CallbackState* state) {
  return {1,
          sizeof(LoonSourceCredentialCallbacks),
          state,
          [](void* data, const LoonSourceCredentialRequest* request, LoonSourceCredentialResult* result) {
            auto& state = *static_cast<CallbackState*>(data);
            ++state.fetched;
            EXPECT_EQ(request->abi_version, 1u);
            EXPECT_EQ(std::string(request->scope_root_uri, request->scope_root_uri_length), kRoot);
            if (state.started) {
              state.started->set_value();
              state.finish.wait();
            }
            result->status = LOON_SOURCE_CREDENTIAL_DENIED;
          },
          [](void* data, LoonSourceCredentialResult*) { ++static_cast<CallbackState*>(data)->released; },
          [](void* data) { ++static_cast<CallbackState*>(data)->destroyed; }};
}
LoonSourceReadContextHandle Create(CallbackState* state) {
  const char* roots[] = {kRoot};
  const LoonSourceReadContextOptions options{1, sizeof(LoonSourceReadContextOptions), "acct", roots, 1, 0, 3000};
  const auto callbacks = Callbacks(state);
  LoonSourceReadContextHandle handle = 0;
  auto result = loon_source_read_context_create(&options, &callbacks, &handle);
  EXPECT_EQ(result.err_code, LOON_SUCCESS);
  loon_ffi_free_result(&result);
  return handle;
}
TEST(SourceReadContextC, RetainedHandlesAndNativeOwnersHaveIndependentLifetime) {
  CallbackState state;
  const auto first = Create(&state);
  ASSERT_NE(first, 0u);
  auto native = GetSourceReadContext(first);
  LoonSourceReadContextHandle second = 0;
  auto result = loon_source_read_context_retain(first, &second);
  ASSERT_EQ(result.err_code, LOON_SUCCESS);
  loon_ffi_free_result(&result);
  EXPECT_NE(first, second);
  EXPECT_EQ(native, GetSourceReadContext(second));
  loon_source_read_context_release(first);
  EXPECT_FALSE(GetSourceReadContext(first));
  EXPECT_EQ(state.destroyed, 0);
  loon_source_read_context_cancel(second);
  EXPECT_EQ(native->Fetch(kFile, LOON_SOURCE_READ).status, LOON_SOURCE_CREDENTIAL_CANCELLED);
  EXPECT_EQ(state.fetched, 0);
  loon_source_read_context_release(second);
  EXPECT_EQ(state.destroyed, 0);
  native.reset();
  EXPECT_EQ(state.destroyed, 1);
  loon_source_read_context_release(first);
  loon_source_read_context_release(second);
  EXPECT_EQ(state.destroyed, 1);
}
TEST(SourceReadContextC, FailedCreateDoesNotTakeUserdataOwnership) {
  CallbackState state;
  const char* roots[] = {kRoot};
  const LoonSourceReadContextOptions options{1, sizeof(LoonSourceReadContextOptions), "", roots, 1, 0, 3000};
  const auto callbacks = Callbacks(&state);
  LoonSourceReadContextHandle handle = 42;
  auto result = loon_source_read_context_create(&options, &callbacks, &handle);
  EXPECT_NE(result.err_code, LOON_SUCCESS);
  EXPECT_EQ(handle, 0u);
  EXPECT_EQ(state.destroyed, 0);
  loon_ffi_free_result(&result);
}
TEST(SourceReadContextC, ReleaseDuringCallbackRetainsUserdataUntilCallbackDrains) {
  CallbackState state;
  std::promise<void> started;
  std::promise<void> finish;
  state.started = &started;
  state.finish = finish.get_future().share();
  const auto handle = Create(&state);
  auto native = GetSourceReadContext(handle);
  auto fetching =
      std::async(std::launch::async, [context = native] { return context->Fetch(kFile, LOON_SOURCE_READ); });
  started.get_future().wait();
  loon_source_read_context_cancel(handle);
  loon_source_read_context_release(handle);
  EXPECT_FALSE(GetSourceReadContext(handle));
  EXPECT_EQ(state.destroyed, 0);
  EXPECT_EQ(state.released, 0);
  finish.set_value();
  EXPECT_EQ(fetching.get().status, LOON_SOURCE_CREDENTIAL_CANCELLED);
  EXPECT_EQ(state.fetched, 1);
  EXPECT_EQ(state.released, 1);
  native.reset();
  EXPECT_EQ(state.destroyed, 1);
}
}  // namespace
}  // namespace milvus_storage
