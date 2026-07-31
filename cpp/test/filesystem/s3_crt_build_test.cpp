// Copyright 2024 Zilliz
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

#ifdef WITH_CRT

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <future>
#include <mutex>
#include <string>
#include <thread>
#include <type_traits>
#include <vector>

#include <aws/s3-crt/S3CrtClient.h>
#include <aws/s3-crt/S3CrtClientConfiguration.h>
#include <folly/executors/InlineExecutor.h>

#include "milvus-storage/filesystem/async_random_access_file.h"
#include "milvus-storage/filesystem/fs.h"
#include "milvus-storage/filesystem/s3/s3_crt_client.h"
#include "milvus-storage/filesystem/s3/s3_filesystem.h"
#include "milvus-storage/filesystem/s3/s3_global.h"
#include "milvus-storage/format/parquet/folly_arrow_executor.h"
#include "test_env.h"

namespace milvus_storage::test {

namespace {

arrow::Status EnsureS3InitializedForTest() {
  auto status = EnsureS3Initialized();
  if (!status.ok()) {
    return status;
  }
  static std::once_flag finalize_flag;
  std::call_once(finalize_flag, [] { std::atexit([] { (void)EnsureS3Finalized(); }); });
  return arrow::Status::OK();
}

std::shared_ptr<Aws::S3Crt::S3CrtClient> MakeTestS3CrtClient() {
  // The client is never dereferenced. The alias lets lifecycle tests exercise
  // holder/finalizer behavior without constructing a native CRT client.
  auto storage = std::make_shared<char>();
  return {storage, reinterpret_cast<Aws::S3Crt::S3CrtClient*>(storage.get())};
}

bool WaitUntilAcquireRejected(const std::shared_ptr<S3CrtClientHolder>& holder) {
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
  while (std::chrono::steady_clock::now() < deadline) {
    if (!holder->Acquire().ok()) {
      return true;
    }
    std::this_thread::yield();
  }
  return false;
}

constexpr std::size_t kConcurrentOperations = 100;

}  // namespace

TEST(S3CrtBuildSupportTest, HeadersAndStaticClientSymbolsAreAvailable) {
  static_assert(std::is_class_v<Aws::S3Crt::S3CrtClient>);
  static_assert(std::is_default_constructible_v<Aws::S3Crt::S3CrtClientConfiguration>);
  static_assert(!std::is_copy_constructible_v<S3CrtClientLease>);
  static_assert(std::is_move_constructible_v<S3CrtClientLease>);

  const char* service_name = Aws::S3Crt::S3CrtClient::GetServiceName();
  ASSERT_NE(service_name, nullptr);
  EXPECT_FALSE(std::string(service_name).empty());
}

TEST(S3CrtClientFinalizerTest, LeaseIsReentrantAndClientDestructionStaysOnHolderThread) {
  auto finalizer = std::make_shared<S3CrtClientFinalizer>();

  std::promise<std::thread::id> client_destroyed_promise;
  auto client_destroyed = client_destroyed_promise.get_future();
  auto storage = std::shared_ptr<char>(new char, [&client_destroyed_promise](char* ptr) {
    client_destroyed_promise.set_value(std::this_thread::get_id());
    delete ptr;
  });
  auto* client_ptr = reinterpret_cast<Aws::S3Crt::S3CrtClient*>(storage.get());
  std::shared_ptr<Aws::S3Crt::S3CrtClient> client(storage, client_ptr);
  storage.reset();

  ASSERT_AND_ASSIGN(auto holder, finalizer->AddClient(std::move(client), nullptr));
  ASSERT_AND_ASSIGN(auto first_lease, holder->Acquire());
  ASSERT_AND_ASSIGN(auto second_lease, holder->Acquire());

  std::promise<void> holder_destruction_started_promise;
  auto holder_destruction_started = holder_destruction_started_promise.get_future();
  auto holder_destroyed =
      std::async(std::launch::async,
                 [holder = std::move(holder), &holder_destruction_started_promise]() mutable -> std::thread::id {
                   auto thread_id = std::this_thread::get_id();
                   holder_destruction_started_promise.set_value();
                   holder.reset();
                   return thread_id;
                 });

  holder_destruction_started.wait();
  EXPECT_EQ(holder_destroyed.wait_for(std::chrono::milliseconds(250)), std::future_status::timeout);

  auto leases_released = std::async(
      std::launch::async,
      [first_lease = std::move(first_lease), second_lease = std::move(second_lease)]() mutable -> std::thread::id {
        return std::this_thread::get_id();
      });
  const auto lease_release_thread = leases_released.get();

  ASSERT_EQ(holder_destroyed.wait_for(std::chrono::seconds(5)), std::future_status::ready);
  const auto holder_destruction_thread = holder_destroyed.get();
  ASSERT_EQ(client_destroyed.wait_for(std::chrono::seconds(5)), std::future_status::ready);
  const auto client_destruction_thread = client_destroyed.get();

  EXPECT_EQ(client_destruction_thread, holder_destruction_thread);
  EXPECT_NE(client_destruction_thread, lease_release_thread);
}

TEST(S3CrtClientFinalizerTest, LeaseMoveReleasesEachOperationOnce) {
  auto finalizer = std::make_shared<S3CrtClientFinalizer>();
  auto client = MakeTestS3CrtClient();
  std::weak_ptr<Aws::S3Crt::S3CrtClient> weak_client = client;
  ASSERT_AND_ASSIGN(auto holder, finalizer->AddClient(std::move(client), nullptr));
  ASSERT_AND_ASSIGN(auto first, holder->Acquire());
  ASSERT_AND_ASSIGN(auto second, holder->Acquire());

  auto moved = std::move(first);
  auto finalized = std::async(std::launch::async, [finalizer] { finalizer->Finalize(); });
  ASSERT_TRUE(WaitUntilAcquireRejected(holder));

  moved = std::move(second);
  EXPECT_EQ(finalized.wait_for(std::chrono::milliseconds(250)), std::future_status::timeout);

  moved = S3CrtClientLease{};
  ASSERT_EQ(finalized.wait_for(std::chrono::seconds(5)), std::future_status::ready);
  finalized.get();
  EXPECT_TRUE(weak_client.expired());
}

TEST(S3CrtClientFinalizerTest, FinalizationWaitsForEveryActiveOperation) {
  auto finalizer = std::make_shared<S3CrtClientFinalizer>();
  auto client = MakeTestS3CrtClient();
  std::weak_ptr<Aws::S3Crt::S3CrtClient> weak_client = client;
  ASSERT_AND_ASSIGN(auto holder, finalizer->AddClient(std::move(client), nullptr));
  ASSERT_AND_ASSIGN(auto last_lease, holder->Acquire());

  std::vector<S3CrtClientLease> client_leases;
  client_leases.reserve(kConcurrentOperations);
  for (std::size_t i = 0; i < kConcurrentOperations; ++i) {
    ASSERT_AND_ASSIGN(auto client_lease, holder->Acquire());
    client_leases.emplace_back(std::move(client_lease));
  }

  auto finalized = std::async(std::launch::async, [finalizer] { finalizer->Finalize(); });
  if (!WaitUntilAcquireRejected(holder)) {
    last_lease = S3CrtClientLease{};
    client_leases.clear();
    finalized.wait();
    FAIL() << "Finalize did not close new client operations";
  }

  std::promise<void> release_promise;
  auto release_signal = release_promise.get_future().share();
  std::vector<std::future<void>> release_futures;
  release_futures.reserve(kConcurrentOperations);
  for (auto& client_lease : client_leases) {
    release_futures.emplace_back(
        std::async(std::launch::async, [release_signal, lease = std::move(client_lease)]() mutable {
          release_signal.wait();
          lease = S3CrtClientLease{};
        }));
  }
  release_promise.set_value();
  for (auto& release_future : release_futures) {
    release_future.get();
  }

  EXPECT_EQ(finalized.wait_for(std::chrono::seconds(0)), std::future_status::timeout);
  last_lease = S3CrtClientLease{};

  ASSERT_EQ(finalized.wait_for(std::chrono::seconds(5)), std::future_status::ready);
  finalized.get();
  EXPECT_TRUE(weak_client.expired());
  EXPECT_FALSE(holder->Acquire().ok());
  EXPECT_FALSE(finalizer->AddClient(MakeTestS3CrtClient(), nullptr).ok());
}

TEST(S3CrtClientFinalizerTest, InlineContinuationDoesNotDeadlockWithConcurrentFinalization) {
  auto finalizer = std::make_shared<S3CrtClientFinalizer>();
  ASSERT_AND_ASSIGN(auto holder, finalizer->AddClient(MakeTestS3CrtClient(), nullptr));
  ASSERT_AND_ASSIGN(auto client_lease, holder->Acquire());

  auto executor =
      parquet::MakeFollyArrowExecutor(folly::getKeepAliveToken(folly::InlineExecutor::instance()), /*capacity=*/1);
  std::atomic<std::size_t> continuations_ran = 0;
  std::vector<arrow::Future<int64_t>> sources;
  std::vector<arrow::Future<int64_t>> continuations;
  sources.reserve(kConcurrentOperations);
  continuations.reserve(kConcurrentOperations);
  for (std::size_t i = 0; i < kConcurrentOperations; ++i) {
    auto source = arrow::Future<int64_t>::Make();
    continuations.emplace_back(
        executor->TransferAlways(source).Then([holder, &continuations_ran](int64_t value) -> arrow::Result<int64_t> {
          ++continuations_ran;
          auto nested_lease = holder->Acquire();
          if (!nested_lease.ok()) {
            return nested_lease.status();
          }
          return value;
        }));
    sources.emplace_back(std::move(source));
  }

  auto finalized = std::async(std::launch::async, [finalizer] { finalizer->Finalize(); });
  const bool acquire_rejected = WaitUntilAcquireRejected(holder);
  if (!acquire_rejected) {
    client_lease = S3CrtClientLease{};
    finalized.wait();
  }
  ASSERT_TRUE(acquire_rejected);
  EXPECT_EQ(finalized.wait_for(std::chrono::seconds(0)), std::future_status::timeout);

  std::promise<void> finish_promise;
  auto finish_signal = finish_promise.get_future().share();
  std::vector<std::future<void>> finish_futures;
  finish_futures.reserve(kConcurrentOperations);
  for (auto& source : sources) {
    finish_futures.emplace_back(std::async(std::launch::async, [finish_signal, source = &source] {
      finish_signal.wait();
      source->MarkFinished(1);
    }));
  }
  finish_promise.set_value();

  bool all_finished = true;
  for (auto& finish_future : finish_futures) {
    if (finish_future.wait_for(std::chrono::seconds(5)) != std::future_status::ready) {
      all_finished = false;
      break;
    }
  }
  if (!all_finished) {
    client_lease = S3CrtClientLease{};
    finalized.wait();
    for (auto& finish_future : finish_futures) {
      finish_future.wait();
    }
  }
  ASSERT_TRUE(all_finished);
  for (auto& finish_future : finish_futures) {
    finish_future.get();
  }

  EXPECT_EQ(continuations_ran.load(), kConcurrentOperations);
  for (auto& continuation : continuations) {
    auto continuation_result = continuation.result();
    EXPECT_FALSE(continuation_result.ok());
    EXPECT_NE(continuation_result.status().ToString().find("S3 subsystem is finalized"), std::string::npos);
  }
  EXPECT_EQ(finalized.wait_for(std::chrono::seconds(0)), std::future_status::timeout);

  client_lease = S3CrtClientLease{};
  ASSERT_EQ(finalized.wait_for(std::chrono::seconds(5)), std::future_status::ready);
  finalized.get();
}

TEST(S3CrtClientFinalizerTest, FinalizeWaitsForClientDestructorAlreadyInProgress) {
  auto finalizer = std::make_shared<S3CrtClientFinalizer>();

  std::promise<void> client_destructor_entered_promise;
  auto client_destructor_entered = client_destructor_entered_promise.get_future();
  std::promise<void> release_client_destructor_promise;
  auto release_client_destructor = release_client_destructor_promise.get_future().share();
  auto storage = std::shared_ptr<char>(new char, [&](char* ptr) {
    client_destructor_entered_promise.set_value();
    release_client_destructor.wait();
    delete ptr;
  });
  auto* client_ptr = reinterpret_cast<Aws::S3Crt::S3CrtClient*>(storage.get());
  std::shared_ptr<Aws::S3Crt::S3CrtClient> client(storage, client_ptr);
  storage.reset();

  ASSERT_AND_ASSIGN(auto holder, finalizer->AddClient(std::move(client), nullptr));
  auto holder_destroyed = std::async(std::launch::async, [holder = std::move(holder)]() mutable { holder.reset(); });

  if (client_destructor_entered.wait_for(std::chrono::seconds(5)) != std::future_status::ready) {
    release_client_destructor_promise.set_value();
    holder_destroyed.wait();
    FAIL() << "Client destructor did not start";
  }

  auto finalized = std::async(std::launch::async, [finalizer] { finalizer->Finalize(); });
  EXPECT_EQ(finalized.wait_for(std::chrono::milliseconds(250)), std::future_status::timeout);

  release_client_destructor_promise.set_value();
  ASSERT_EQ(holder_destroyed.wait_for(std::chrono::seconds(5)), std::future_status::ready);
  holder_destroyed.get();
  ASSERT_EQ(finalized.wait_for(std::chrono::seconds(5)), std::future_status::ready);
  finalized.get();
}

TEST(S3CrtBuildSupportTest, OpenInputFileUsesCrtBackedAsyncFileWhenCrtEnabled) {
  if (!IsCloudEnv()) {
    GTEST_SKIP() << "CRT OpenInputFile smoke test skipped in non-cloud environment";
  }
  auto provider = GetEnvVar(ENV_VAR_CLOUD_PROVIDER);
  if (provider.ok() && provider.ValueOrDie() == kCloudProviderGCP) {
    GTEST_SKIP() << "CRT OpenInputFile smoke test does not run for GCP provider";
  }

  api::Properties properties;
  ASSERT_STATUS_OK(InitTestProperties(properties));
  ASSERT_AND_ASSIGN(auto fs, GetFileSystem(properties));

  const std::string base_path = GetTestBasePath("s3-crt-open-input-file-smoke");
  ASSERT_STATUS_OK(DeleteTestDir(fs, base_path));
  ASSERT_STATUS_OK(CreateTestDir(fs, base_path));

  const std::string object_path = base_path + "/crt-input-file.txt";
  const std::string data = "abcdefghi";
  ASSERT_AND_ASSIGN(auto output_stream, fs->OpenOutputStream(object_path));
  ASSERT_STATUS_OK(output_stream->Write(data.data(), static_cast<int64_t>(data.size())));
  ASSERT_STATUS_OK(output_stream->Close());

  ASSERT_AND_ASSIGN(auto input_file, fs->OpenInputFile(object_path));
  ASSERT_NE(dynamic_cast<milvus_storage::NonBlockingReadAtFile*>(input_file.get()), nullptr);

  auto async_result = input_file->ReadAsync({}, 2, 4).result();
  ASSERT_STATUS_OK(async_result.status());
  ASSERT_EQ(async_result.ValueOrDie()->ToString(), "cdef");

  ASSERT_AND_ASSIGN(auto sync_buffer, input_file->ReadAt(0, 3));
  ASSERT_EQ(sync_buffer->ToString(), "abc");
  ASSERT_STATUS_OK(input_file->Close());

  ASSERT_STATUS_OK(DeleteTestDir(fs, base_path));
}

TEST(S3CrtBuildSupportTest, OpenInputFileUsesCrtBackedAsyncFileForNonGcpProvider) {
  ASSERT_STATUS_OK(EnsureS3InitializedForTest());

  for (const auto* cloud_provider : {"", kCloudProviderAWS}) {
    SCOPED_TRACE(::testing::Message() << "cloud_provider=" << cloud_provider);

    auto options = S3Options::FromAccessKey("ak", "sk");
    options.cloud_provider = cloud_provider;

    ASSERT_AND_ASSIGN(auto fs, S3FileSystem::Make(options));
    ASSERT_AND_ASSIGN(auto input_file, fs->OpenInputFile("bucket/path/object.txt"));

    EXPECT_NE(dynamic_cast<milvus_storage::NonBlockingReadAtFile*>(input_file.get()), nullptr);
  }
}

TEST(S3CrtBuildSupportTest, OpenInputFileFallsBackToSdkFileForGcpProvider) {
  ASSERT_STATUS_OK(EnsureS3InitializedForTest());

  auto options = S3Options::FromAccessKey("ak", "sk");
  options.cloud_provider = kCloudProviderGCP;
  options.endpoint_override = "storage.googleapis.com";

  ASSERT_AND_ASSIGN(auto fs, S3FileSystem::Make(options));
  ASSERT_AND_ASSIGN(auto input_file, fs->OpenInputFile("bucket/path/object.txt"));

  EXPECT_EQ(dynamic_cast<milvus_storage::NonBlockingReadAtFile*>(input_file.get()), nullptr);
}

}  // namespace milvus_storage::test

#endif  // WITH_CRT
