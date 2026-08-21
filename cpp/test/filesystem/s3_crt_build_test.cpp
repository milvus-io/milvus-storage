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

#include <boost/asio/ip/tcp.hpp>
#include <boost/beast/core.hpp>
#include <boost/beast/http.hpp>
#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <future>
#include <iostream>
#include <mutex>
#include <string>
#include <thread>
#include <type_traits>
#include <vector>

#if !defined(_WIN32)
#include <unistd.h>
#endif

#include <aws/s3-crt/S3CrtClient.h>
#include <aws/s3-crt/S3CrtClientConfiguration.h>
#include <folly/executors/ManualExecutor.h>

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

bool WaitUntilConstructionRejected(const std::shared_ptr<S3CrtClientFinalizer>& finalizer) {
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
  while (std::chrono::steady_clock::now() < deadline) {
    bool factory_called = false;
    auto result = finalizer->AddClient(
        [&factory_called]() -> arrow::Result<std::shared_ptr<Aws::S3Crt::S3CrtClient>> {
          factory_called = true;
          return MakeTestS3CrtClient();
        },
        nullptr);
    if (!result.ok() && !factory_called) {
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

TEST(S3CrtClientFinalizerTest, RejectsSuccessfulNullClientConstruction) {
  auto finalizer = std::make_shared<S3CrtClientFinalizer>();
  auto result = finalizer->AddClient(
      []() -> arrow::Result<std::shared_ptr<Aws::S3Crt::S3CrtClient>> {
        return std::shared_ptr<Aws::S3Crt::S3CrtClient>(nullptr, [](Aws::S3Crt::S3CrtClient*) {});
      },
      nullptr);

  ASSERT_FALSE(result.ok());
  EXPECT_TRUE(result.status().IsInvalid()) << result.status().ToString();
}

TEST(S3CrtClientFinalizerTest, RejectsClientWithAnotherSharedOwner) {
  auto finalizer = std::make_shared<S3CrtClientFinalizer>();
  auto retained_client = MakeTestS3CrtClient();
  auto result = finalizer->AddClient([retained_client] { return retained_client; }, nullptr);

  ASSERT_FALSE(result.ok());
  EXPECT_TRUE(result.status().IsInvalid()) << result.status().ToString();
}

TEST(S3CrtClientFinalizerTest, FinalizationWaitsForClientConstructionToRegister) {
  auto finalizer = std::make_shared<S3CrtClientFinalizer>();
  std::promise<void> construction_entered_promise;
  auto construction_entered = construction_entered_promise.get_future();
  std::promise<void> release_construction_promise;
  auto release_construction = release_construction_promise.get_future().share();
  std::promise<std::weak_ptr<Aws::S3Crt::S3CrtClient>> client_created_promise;
  auto client_created = client_created_promise.get_future();
  auto constructing = std::async(
      std::launch::async, [finalizer, &construction_entered_promise, release_construction, &client_created_promise] {
        return finalizer->AddClient(
            [&construction_entered_promise, release_construction,
             &client_created_promise]() -> arrow::Result<std::shared_ptr<Aws::S3Crt::S3CrtClient>> {
              construction_entered_promise.set_value();
              release_construction.wait();
              auto client = MakeTestS3CrtClient();
              client_created_promise.set_value(std::weak_ptr<Aws::S3Crt::S3CrtClient>(client));
              return client;
            },
            nullptr);
      });
  if (construction_entered.wait_for(std::chrono::seconds(5)) != std::future_status::ready) {
    release_construction_promise.set_value();
    constructing.wait();
    FAIL() << "Client construction did not start";
  }

  auto finalized = std::async(std::launch::async, [finalizer] { finalizer->Finalize(); });

  if (!WaitUntilConstructionRejected(finalizer)) {
    release_construction_promise.set_value();
    constructing.wait();
    finalized.wait();
    FAIL() << "Finalize did not reject new client construction";
  }
  EXPECT_EQ(finalized.wait_for(std::chrono::milliseconds(250)), std::future_status::timeout);

  release_construction_promise.set_value();
  ASSERT_EQ(constructing.wait_for(std::chrono::seconds(5)), std::future_status::ready);
  ASSERT_AND_ASSIGN(auto holder, constructing.get());
  const auto weak_client = client_created.get();
  ASSERT_EQ(finalized.wait_for(std::chrono::seconds(5)), std::future_status::ready);
  finalized.get();

  EXPECT_TRUE(weak_client.expired());
  EXPECT_FALSE(holder->Acquire().ok());
}

TEST(S3CrtClientFinalizerTest, FinalizationWaitsForEveryFailedClientConstruction) {
  auto finalizer = std::make_shared<S3CrtClientFinalizer>();
  auto start_failed_construction = [finalizer](std::promise<void>* entered, std::shared_future<void> release) {
    return std::async(std::launch::async, [finalizer, entered, release] {
      return finalizer->AddClient(
          [entered, release]() -> arrow::Result<std::shared_ptr<Aws::S3Crt::S3CrtClient>> {
            entered->set_value();
            release.wait();
            return arrow::Status::IOError("Injected CRT client construction failure");
          },
          nullptr);
    });
  };

  std::promise<void> first_entered_promise;
  auto first_entered = first_entered_promise.get_future();
  std::promise<void> first_release_promise;
  auto first = start_failed_construction(&first_entered_promise, first_release_promise.get_future().share());
  std::promise<void> second_entered_promise;
  auto second_entered = second_entered_promise.get_future();
  std::promise<void> second_release_promise;
  auto second = start_failed_construction(&second_entered_promise, second_release_promise.get_future().share());

  if (first_entered.wait_for(std::chrono::seconds(5)) != std::future_status::ready ||
      second_entered.wait_for(std::chrono::seconds(5)) != std::future_status::ready) {
    first_release_promise.set_value();
    second_release_promise.set_value();
    first.wait();
    second.wait();
    FAIL() << "Client construction did not start";
  }
  auto finalized = std::async(std::launch::async, [finalizer] { finalizer->Finalize(); });

  if (!WaitUntilConstructionRejected(finalizer)) {
    first_release_promise.set_value();
    second_release_promise.set_value();
    first.wait();
    second.wait();
    finalized.wait();
    FAIL() << "Finalize did not reject new client construction";
  }
  EXPECT_EQ(finalized.wait_for(std::chrono::milliseconds(250)), std::future_status::timeout);

  first_release_promise.set_value();
  ASSERT_EQ(first.wait_for(std::chrono::seconds(5)), std::future_status::ready);
  EXPECT_FALSE(first.get().ok());
  EXPECT_EQ(finalized.wait_for(std::chrono::milliseconds(250)), std::future_status::timeout);

  second_release_promise.set_value();
  ASSERT_EQ(second.wait_for(std::chrono::seconds(5)), std::future_status::ready);
  EXPECT_FALSE(second.get().ok());
  ASSERT_EQ(finalized.wait_for(std::chrono::seconds(5)), std::future_status::ready);
  finalized.get();
}

TEST(S3CrtClientFinalizerTest, FinalizationWaitsForClientFactoryCleanup) {
  struct BlockingFactoryCleanup {
    BlockingFactoryCleanup(std::promise<void>* entered, std::shared_future<void> release)
        : entered(entered), release(std::move(release)) {}
    BlockingFactoryCleanup(const BlockingFactoryCleanup& other) : entered(other.entered), release(other.release) {}
    BlockingFactoryCleanup(BlockingFactoryCleanup&& other) noexcept
        : entered(other.entered), release(std::move(other.release)), armed(other.armed) {
      other.armed = false;
    }
    ~BlockingFactoryCleanup() {
      if (armed) {
        entered->set_value();
        release.wait();
      }
    }

    arrow::Result<std::shared_ptr<Aws::S3Crt::S3CrtClient>> operator()() {
      armed = true;
      return MakeTestS3CrtClient();
    }

    std::promise<void>* entered;
    std::shared_future<void> release;
    bool armed = false;
  };

  auto finalizer = std::make_shared<S3CrtClientFinalizer>();
  std::promise<void> factory_cleanup_entered_promise;
  auto factory_cleanup_entered = factory_cleanup_entered_promise.get_future();
  std::promise<void> release_factory_cleanup_promise;
  auto release_factory_cleanup = release_factory_cleanup_promise.get_future().share();
  S3CrtClientFinalizer::ClientFactory factory(
      BlockingFactoryCleanup(&factory_cleanup_entered_promise, release_factory_cleanup));

  auto adding = std::async(std::launch::async, [finalizer, factory = std::move(factory)]() mutable {
    return finalizer->AddClient(std::move(factory), nullptr);
  });
  if (factory_cleanup_entered.wait_for(std::chrono::seconds(5)) != std::future_status::ready) {
    release_factory_cleanup_promise.set_value();
    adding.wait();
    FAIL() << "Client factory cleanup did not start";
  }

  auto finalized = std::async(std::launch::async, [finalizer] { finalizer->Finalize(); });
  EXPECT_EQ(finalized.wait_for(std::chrono::milliseconds(250)), std::future_status::timeout);

  release_factory_cleanup_promise.set_value();
  ASSERT_EQ(adding.wait_for(std::chrono::seconds(5)), std::future_status::ready);
  ASSERT_AND_ASSIGN(auto holder, adding.get());
  ASSERT_EQ(finalized.wait_for(std::chrono::seconds(5)), std::future_status::ready);
  finalized.get();
  EXPECT_FALSE(holder->Acquire().ok());
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

  ASSERT_AND_ASSIGN(
      auto holder, finalizer->AddClient([client = std::move(client)]() mutable { return std::move(client); }, nullptr));
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
  ASSERT_AND_ASSIGN(
      auto holder, finalizer->AddClient([client = std::move(client)]() mutable { return std::move(client); }, nullptr));
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
  ASSERT_AND_ASSIGN(
      auto holder, finalizer->AddClient([client = std::move(client)]() mutable { return std::move(client); }, nullptr));
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
  bool factory_called = false;
  EXPECT_FALSE(finalizer
                   ->AddClient(
                       [&factory_called]() -> arrow::Result<std::shared_ptr<Aws::S3Crt::S3CrtClient>> {
                         factory_called = true;
                         return MakeTestS3CrtClient();
                       },
                       nullptr)
                   .ok());
  EXPECT_FALSE(factory_called);
}

TEST(S3CrtClientFinalizerTest, InlineContinuationDoesNotDeadlockWithConcurrentFinalization) {
  auto finalizer = std::make_shared<S3CrtClientFinalizer>();
  ASSERT_AND_ASSIGN(auto holder, finalizer->AddClient([] { return MakeTestS3CrtClient(); }, nullptr));
  ASSERT_AND_ASSIGN(auto client_lease, holder->Acquire());

  std::atomic<std::size_t> continuations_ran = 0;
  std::vector<arrow::Future<int64_t>> sources;
  std::vector<arrow::Future<int64_t>> continuations;
  sources.reserve(kConcurrentOperations);
  continuations.reserve(kConcurrentOperations);
  for (std::size_t i = 0; i < kConcurrentOperations; ++i) {
    auto source = arrow::Future<int64_t>::Make();
    continuations.emplace_back(source.Then([holder, &continuations_ran](int64_t value) -> arrow::Result<int64_t> {
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

  ASSERT_AND_ASSIGN(
      auto holder, finalizer->AddClient([client = std::move(client)]() mutable { return std::move(client); }, nullptr));
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
  auto* async_file = dynamic_cast<milvus_storage::NonBlockingRandomAccessFile*>(input_file.get());
  ASSERT_NE(async_file, nullptr);

  auto size_result = async_file->GetSizeAsync().result();
  ASSERT_STATUS_OK(size_result.status());
  ASSERT_EQ(size_result.ValueOrDie(), static_cast<int64_t>(data.size()));

  auto async_result = input_file->ReadAsync({}, 2, 4).result();
  ASSERT_STATUS_OK(async_result.status());
  ASSERT_EQ(async_result.ValueOrDie()->ToString(), "cdef");

  ASSERT_AND_ASSIGN(auto sync_buffer, input_file->ReadAt(0, 3));
  ASSERT_EQ(sync_buffer->ToString(), "abc");
  ASSERT_STATUS_OK(input_file->Close());

  ASSERT_STATUS_OK(DeleteTestDir(fs, base_path));
}

TEST(S3CrtBuildSupportTest, InFlightNativeReadCompletesDuringFinalizeS3) {
#if defined(_WIN32)
  GTEST_SKIP() << "Test requires POSIX process and socket APIs.";
#else
  const auto original_death_test_style = GTEST_FLAG_GET(death_test_style);
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  const auto run_child = []() -> int {
    class BlockingRangeGetServer final {
      using Tcp = boost::asio::ip::tcp;

   public:
      ~BlockingRangeGetServer() { Stop(); }

      bool Start() {
        boost::system::error_code error;
        acceptor_.open(Tcp::v4(), error);
        if (RecordError(error)) {
          return false;
        }
        acceptor_.set_option(Tcp::acceptor::reuse_address(true), error);
        if (RecordError(error)) {
          return false;
        }
        acceptor_.bind(Tcp::endpoint(boost::asio::ip::address_v4::loopback(), 0), error);
        if (RecordError(error)) {
          return false;
        }
        acceptor_.listen(1, error);
        if (RecordError(error)) {
          return false;
        }
        port_ = acceptor_.local_endpoint(error).port();
        if (RecordError(error)) {
          return false;
        }
        worker_ = std::thread([this] { Serve(); });
        return true;
      }

      uint16_t port() const { return port_; }

      bool WaitForRequest(std::chrono::milliseconds timeout) {
        std::unique_lock lock(mutex_);
        cv_.wait_for(lock, timeout, [this] { return request_received_ || stopped_ || !error_.empty(); });
        return request_received_;
      }

      void ReleaseResponse() {
        {
          std::lock_guard lock(mutex_);
          response_released_ = true;
        }
        cv_.notify_all();
      }

      std::string error() const {
        std::lock_guard lock(mutex_);
        return error_;
      }

      void Stop() {
        std::shared_ptr<Tcp::socket> socket;
        {
          std::lock_guard lock(mutex_);
          stopped_ = true;
          response_released_ = true;
          socket = socket_;
        }
        cv_.notify_all();

        boost::system::error_code error;
        acceptor_.close(error);
        if (socket) {
          socket->cancel(error);
          socket->shutdown(Tcp::socket::shutdown_both, error);
          socket->close(error);
        }
        if (worker_.joinable()) {
          worker_.join();
        }
      }

   private:
      bool RecordError(const boost::system::error_code& error) {
        if (!error) {
          return false;
        }
        SetError(error.message());
        return true;
      }

      void SetError(std::string error) {
        {
          std::lock_guard lock(mutex_);
          if (error_.empty()) {
            error_ = std::move(error);
          }
        }
        cv_.notify_all();
      }

      void Serve() {
        auto socket = std::make_shared<Tcp::socket>(io_context_);
        {
          std::lock_guard lock(mutex_);
          socket_ = socket;
        }
        boost::system::error_code error;
        acceptor_.accept(*socket, error);
        if (error) {
          std::lock_guard lock(mutex_);
          if (!stopped_) {
            error_ = error.message();
            cv_.notify_all();
          }
          return;
        }

        boost::beast::flat_buffer buffer;
        boost::beast::http::request<boost::beast::http::empty_body> request;
        boost::beast::http::read(*socket, buffer, request, error);
        if (RecordError(error)) {
          return;
        }
        if (request.method() != boost::beast::http::verb::get ||
            request[boost::beast::http::field::range] != "bytes=2-5") {
          SetError("Unexpected CRT range GET");
          return;
        }

        {
          std::lock_guard lock(mutex_);
          request_received_ = true;
        }
        cv_.notify_all();

        {
          std::unique_lock lock(mutex_);
          cv_.wait(lock, [this] { return response_released_ || stopped_; });
          if (stopped_) {
            return;
          }
        }

        boost::beast::http::response<boost::beast::http::string_body> response{
            boost::beast::http::status::partial_content, request.version()};
        response.set(boost::beast::http::field::content_range, "bytes 2-5/9");
        response.set(boost::beast::http::field::accept_ranges, "bytes");
        response.set(boost::beast::http::field::etag, "\"s3-crt-finalize-test\"");
        response.keep_alive(false);
        response.body() = "cdef";
        response.prepare_payload();
        boost::beast::http::write(*socket, response, error);
        RecordError(error);
      }

      boost::asio::io_context io_context_;
      Tcp::acceptor acceptor_{io_context_};
      std::shared_ptr<Tcp::socket> socket_;
      uint16_t port_ = 0;
      std::thread worker_;
      mutable std::mutex mutex_;
      std::condition_variable cv_;
      std::string error_;
      bool request_received_ = false;
      bool response_released_ = false;
      bool stopped_ = false;
    };

    auto fail = [](const std::string& message) {
      std::cerr << message << std::endl;
      return 1;
    };

    BlockingRangeGetServer server;
    if (!server.Start()) {
      return fail("Failed to start the blocking S3 server: " + server.error());
    }
    auto initialize_status = EnsureS3Initialized();
    if (!initialize_status.ok()) {
      return fail(initialize_status.ToString());
    }

    auto options = S3Options::FromAccessKey("ak", "sk");
    options.cloud_provider = kCloudProviderAWS;
    options.region = "us-east-1";
    options.scheme = "http";
    options.endpoint_override = "127.0.0.1:" + std::to_string(server.port());
    options.connect_timeout = 5;
    options.request_timeout = 5;
    options.retry_strategy = S3RetryStrategy::GetAwsDefaultRetryStrategy(0);
    options.use_crt_async_reads = true;

    auto fs_result = S3FileSystem::Make(options);
    if (!fs_result.ok()) {
      return fail(fs_result.status().ToString());
    }
    auto fs = std::move(fs_result).ValueOrDie();
    arrow::fs::FileInfo file_info("test-bucket/path/object.txt", arrow::fs::FileType::File);
    file_info.set_size(9);
    auto input_result = fs->OpenInputFile(file_info);
    if (!input_result.ok()) {
      return fail(input_result.status().ToString());
    }
    auto input = std::move(input_result).ValueOrDie();
    if (dynamic_cast<NonBlockingRandomAccessFile*>(input.get()) == nullptr) {
      return fail("OpenInputFile did not select the CRT async read path");
    }

    auto read = input->ReadAsync({}, 2, 4);
    if (!server.WaitForRequest(std::chrono::seconds(5))) {
      return fail("Timed out waiting for a real CRT range GET: " + server.error());
    }

    auto finalized = std::async(std::launch::async, [] { return FinalizeS3(); });
    if (!WaitUntilConstructionRejected(GetCrtClientFinalizer())) {
      server.ReleaseResponse();
      (void)finalized.get();
      return fail("CRT client finalization did not start");
    }
    if (finalized.wait_for(std::chrono::milliseconds(100)) != std::future_status::timeout) {
      server.ReleaseResponse();
      auto finalize_status = finalized.get();
      return fail("FinalizeS3 did not wait for the in-flight CRT request: " + finalize_status.ToString());
    }

    server.ReleaseResponse();
    auto read_result = read.result();
    if (!read_result.ok()) {
      (void)finalized.get();
      return fail(read_result.status().ToString());
    }
    if (read_result.ValueOrDie()->ToString() != "cdef") {
      (void)finalized.get();
      return fail("CRT range read returned unexpected data");
    }
    auto finalize_status = finalized.get();
    if (!finalize_status.ok()) {
      return fail(finalize_status.ToString());
    }
    server.Stop();
    return 0;
  };
  EXPECT_EXIT((::alarm(20), ::_exit(run_child())), ::testing::ExitedWithCode(0), "");
  GTEST_FLAG_SET(death_test_style, original_death_test_style);
#endif
}

TEST(S3CrtBuildSupportTest, OpenInputFileUsesCrtBackedAsyncFileForNonGcpProvider) {
  ASSERT_STATUS_OK(EnsureS3InitializedForTest());

  for (const auto* cloud_provider : {"", kCloudProviderAWS}) {
    SCOPED_TRACE(::testing::Message() << "cloud_provider=" << cloud_provider);

    auto options = S3Options::FromAccessKey("ak", "sk");
    options.cloud_provider = cloud_provider;

    ASSERT_AND_ASSIGN(auto fs, S3FileSystem::Make(options));
    ASSERT_AND_ASSIGN(auto input_file, fs->OpenInputFile("bucket/path/object.txt"));

    EXPECT_NE(dynamic_cast<milvus_storage::NonBlockingRandomAccessFile*>(input_file.get()), nullptr);
  }
}

TEST(S3CrtBuildSupportTest, ZeroLengthAsyncReadsDoNotScheduleIoExecutor) {
  ASSERT_STATUS_OK(EnsureS3InitializedForTest());

  folly::ManualExecutor io_executor;
  ASSERT_AND_ASSIGN(auto arrow_executor,
                    parquet::MakeFollyArrowExecutor(folly::getKeepAliveToken(io_executor), /*capacity=*/1));
  arrow::io::IOContext io_context(arrow_executor.get());

  auto options = S3Options::FromAccessKey("ak", "sk");
  options.cloud_provider = kCloudProviderAWS;
  options.region = "us-east-1";
  options.scheme = "http";
  options.endpoint_override = "127.0.0.1:1";
  options.connect_timeout = 0.1;
  options.request_timeout = 0.1;
  options.retry_strategy = S3RetryStrategy::GetAwsDefaultRetryStrategy(0);

  ASSERT_AND_ASSIGN(auto fs, S3FileSystem::Make(options, io_context));
  ASSERT_EQ(fs->io_context().executor(), arrow_executor.get());

  arrow::fs::FileInfo file_info("bucket/path/object.txt", arrow::fs::FileType::File);
  file_info.set_size(1);
  ASSERT_AND_ASSIGN(auto input_file, fs->OpenInputFile(file_info));
  auto* async_file = dynamic_cast<milvus_storage::NonBlockingRandomAccessFile*>(input_file.get());
  ASSERT_NE(async_file, nullptr);

  io_executor.drain();

  uint8_t out = 0;
  auto read_into_future = async_file->ReadAtAsyncInto(0, 0, &out);
  EXPECT_TRUE(read_into_future.is_finished());
  ASSERT_AND_ASSIGN(auto bytes_read, read_into_future.result());
  EXPECT_EQ(bytes_read, 0);
  EXPECT_EQ(io_executor.drain(), 0);

  auto read_future = input_file->ReadAsync(io_context, 0, 0);
  EXPECT_TRUE(read_future.is_finished());
  ASSERT_AND_ASSIGN(auto buffer, read_future.result());
  EXPECT_EQ(buffer->size(), 0);
  EXPECT_EQ(io_executor.drain(), 0);
  ASSERT_STATUS_OK(input_file->Close());
}

TEST(S3CrtBuildSupportTest, OpenInputFileFallsBackToSdkFileForGcpProvider) {
  ASSERT_STATUS_OK(EnsureS3InitializedForTest());

  auto options = S3Options::FromAccessKey("ak", "sk");
  options.cloud_provider = kCloudProviderGCP;
  options.endpoint_override = "storage.googleapis.com";

  ASSERT_AND_ASSIGN(auto fs, S3FileSystem::Make(options));
  ASSERT_AND_ASSIGN(auto input_file, fs->OpenInputFile("bucket/path/object.txt"));

  EXPECT_EQ(dynamic_cast<milvus_storage::NonBlockingRandomAccessFile*>(input_file.get()), nullptr);
}

}  // namespace milvus_storage::test

#endif  // WITH_CRT
