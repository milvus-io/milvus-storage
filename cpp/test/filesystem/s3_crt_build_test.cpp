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
#include <utility>
#include <vector>

#include <arrow/util/key_value_metadata.h>

#if !defined(_WIN32)
#include <unistd.h>
#endif

#include <aws/s3-crt/S3CrtClient.h>
#include <aws/s3-crt/S3CrtClientConfiguration.h>
#include <aws/core/Globals.h>
#include <aws/crt/io/Bootstrap.h>
#include <aws/crt/auth/Credentials.h>
#include <aws/s3/s3_client.h>
#include <folly/executors/ManualExecutor.h>

#include "milvus-storage/common/extend_status.h"
#include "milvus-storage/filesystem/async_random_access_file.h"
#include "milvus-storage/filesystem/fs.h"
#include "milvus-storage/filesystem/s3/s3_crt_client.h"
#include "milvus-storage/filesystem/s3/s3_filesystem.h"
#include "milvus-storage/filesystem/s3/s3_global.h"
#ifdef WITH_TALON
#include "milvus-storage/filesystem/talon/talon_file_system_producer.h"
#endif
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

arrow::Result<aws_s3_client*> MakeTestNativeClient(std::function<void()> on_shutdown) {
  aws_s3_client_config config{};
  config.region = aws_byte_cursor_from_c_str("us-east-1");
  config.client_bootstrap = Aws::GetDefaultClientBootstrap()->GetUnderlyingHandle();
  config.tls_mode = AWS_MR_TLS_DISABLED;
  auto credentials = Aws::Crt::Auth::CredentialsProvider::CreateCredentialsProviderAnonymous();
  aws_signing_config_aws signing{};
  aws_s3_init_default_signing_config(&signing, config.region, credentials->GetUnderlyingHandle());
  config.signing_config = &signing;
  auto done = std::make_unique<std::function<void()>>(std::move(on_shutdown));
  config.shutdown_callback_user_data = done.get();
  config.shutdown_callback = [](void* user) {
    std::unique_ptr<std::function<void()>> callback(static_cast<std::function<void()>*>(user));
    (*callback)();
  };
  auto* client = aws_s3_client_new(aws_default_allocator(), &config);
  if (!client)
    return arrow::Status::IOError("Cannot construct native test client: ", aws_error_str(aws_last_error()));
  done.release();
  return client;
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

TEST(S3CrtClientFinalizerTest, NativeHolderDestructionDoesNotWaitForItsOwnLease) {
  ASSERT_STATUS_OK(EnsureS3InitializedForTest());
  auto finalizer = std::make_shared<S3CrtClientFinalizer>();
  std::promise<void> shutdown_entered_promise;
  auto shutdown_entered = shutdown_entered_promise.get_future();
  std::promise<void> allow_shutdown_promise;
  auto allow_shutdown = allow_shutdown_promise.get_future().share();
  ASSERT_AND_ASSIGN(auto holder, finalizer->AddNativeClient([&](std::function<void()> done) {
    return MakeTestNativeClient([&, done = std::move(done)] {
      shutdown_entered_promise.set_value();
      allow_shutdown.wait();
      done();
    });
  }));
  ASSERT_AND_ASSIGN(auto lease, holder->Acquire());
  ASSERT_NE(lease.native_client(), nullptr);
  EXPECT_EQ(lease.operator->(), nullptr);
  // This is what an inline native completion does when dropping the last
  // transport owner. Its request lease must not make holder destruction wait.
  std::weak_ptr<S3CrtClientHolder> weak_holder = holder;
  auto dropped = std::async(std::launch::async, [holder = std::move(holder)]() mutable { holder.reset(); });
  EXPECT_EQ(dropped.wait_for(std::chrono::seconds(1)), std::future_status::ready);
  EXPECT_TRUE(weak_holder.expired());
  auto finalized = std::async(std::launch::async, [finalizer] { finalizer->Finalize(); });
  EXPECT_EQ(shutdown_entered.wait_for(std::chrono::milliseconds(50)), std::future_status::timeout);
  EXPECT_EQ(finalized.wait_for(std::chrono::milliseconds(50)), std::future_status::timeout);
  // Move assignment must retire the operation once, even after its holder dies.
  auto moved = std::move(lease);
  lease = S3CrtClientLease{};
  moved = S3CrtClientLease{};
  EXPECT_EQ(shutdown_entered.wait_for(std::chrono::seconds(5)), std::future_status::ready);
  EXPECT_EQ(finalized.wait_for(std::chrono::milliseconds(50)), std::future_status::timeout);
  allow_shutdown_promise.set_value();
  EXPECT_EQ(finalized.wait_for(std::chrono::seconds(5)), std::future_status::ready);
  finalized.get();
  dropped.get();
}

TEST(S3CrtClientFinalizerTest, NativeConstructionAndShutdownUseTheSharedBarrier) {
  ASSERT_STATUS_OK(EnsureS3InitializedForTest());
  auto finalizer = std::make_shared<S3CrtClientFinalizer>();
  std::promise<void> factory_entered_promise;
  auto factory_entered = factory_entered_promise.get_future();
  std::promise<void> release_factory_promise;
  auto release_factory = release_factory_promise.get_future().share();
  std::atomic<bool> native_destroyed{false};
  auto constructed = std::async(std::launch::async, [&] {
    return finalizer->AddNativeClient([&](std::function<void()> done) {
      factory_entered_promise.set_value();
      release_factory.wait();
      return MakeTestNativeClient([&, done = std::move(done)] {
        native_destroyed.store(true);
        done();
      });
    });
  });
  factory_entered.wait();
  auto finalized = std::async(std::launch::async, [finalizer] { finalizer->Finalize(); });
  EXPECT_TRUE(WaitUntilConstructionRejected(finalizer));
  EXPECT_EQ(finalized.wait_for(std::chrono::milliseconds(50)), std::future_status::timeout);
  release_factory_promise.set_value();
  ASSERT_AND_ASSIGN(auto holder, constructed.get());
  EXPECT_FALSE(holder->Acquire().ok());
  EXPECT_EQ(finalized.wait_for(std::chrono::seconds(5)), std::future_status::ready);
  finalized.get();
  EXPECT_TRUE(native_destroyed.load());
  bool called = false;
  auto rejected = finalizer->AddNativeClient([&](std::function<void()> done) {
    called = true;
    return MakeTestNativeClient(std::move(done));
  });
  EXPECT_FALSE(rejected.ok());
  EXPECT_FALSE(called);
}

TEST(S3CrtClientFinalizerTest, RejectsSuccessfulNullNativeConstruction) {
  auto finalizer = std::make_shared<S3CrtClientFinalizer>();
  auto result = finalizer->AddNativeClient(
      [](std::function<void()>) -> arrow::Result<aws_s3_client*> { return static_cast<aws_s3_client*>(nullptr); });
  EXPECT_FALSE(result.ok());
  finalizer->Finalize();
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
        response.set(boost::beast::http::field::etag, "\"8aa99b1f439ff71293e95357bac6fd94\"");
        response.keep_alive(false);
        response.body() = "cdef";
        response.prepare_payload();
        boost::beast::http::write(*socket, response, error);
        if (RecordError(error)) {
          return;
        }

        // keep_alive(false) advertises that the server will close the connection,
        // but Beast does not close the socket after write(). Honor the mock response
        // contract without depending on server destruction.
        socket->shutdown(Tcp::socket::shutdown_send, error);
        socket->close(error);
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

TEST(S3CrtBuildSupportTest, OpenInputFileRejectsUnsupportedNativeTransport) {
  ASSERT_STATUS_OK(EnsureS3InitializedForTest());
  auto options = S3Options::FromAccessKey("ak", "sk");
  options.cloud_provider = kCloudProviderAWS;
  options.region = "us-east-1";
  options.scheme = "http";
  options.endpoint_override = "127.0.0.1:1";
  options.use_crt_async_reads = true;
  // Custom retry providers have no native adapter. Opening must reject the
  // unsupported configuration even when a caller supplies the file size.
  options.retry_strategy = S3RetryStrategy::GetAwsDefaultRetryStrategy(0);
  ASSERT_AND_ASSIGN(auto fs, S3FileSystem::Make(options));
  const std::string path = "bucket/path/object.txt";
  arrow::fs::FileInfo info(path, arrow::fs::FileType::File);
  info.set_size(9);
  EXPECT_TRUE(fs->OpenInputFile(path).status().IsNotImplemented());
  EXPECT_TRUE(fs->OpenInputFile(info).status().IsNotImplemented());
  EXPECT_TRUE(fs->OpenInputFileAsync(path).status().IsNotImplemented());
  EXPECT_TRUE(fs->OpenInputFileAsync(info).status().IsNotImplemented());
  EXPECT_TRUE(fs->GetFileInfoAsync(std::vector<std::string>{path}).status().IsNotImplemented());
  arrow::fs::FileSelector selector;
  selector.base_dir = "bucket";
  EXPECT_TRUE(fs->GetFileInfoGenerator(selector)().status().IsNotImplemented());

  options.use_crt_async_reads = false;
  ASSERT_AND_ASSIGN(auto sdk_fs, S3FileSystem::Make(options));
  ASSERT_AND_ASSIGN(auto input, sdk_fs->OpenInputFile(info));
  EXPECT_EQ(dynamic_cast<NonBlockingRandomAccessFile*>(input.get()), nullptr);
  ASSERT_STATUS_OK(input->Close());
  EXPECT_TRUE(sdk_fs->OpenInputFileAsync(path).status().IsNotImplemented());
  EXPECT_TRUE(sdk_fs->OpenInputFileAsync(info).status().IsNotImplemented());
}

struct S3CrtMetadataTestParam {
  boost::beast::http::status response_status;
  bool close_before_completion = false;
  bool size_first = false;
  bool use_talon = false;
};

class S3CrtMetadataTest : public ::testing::TestWithParam<S3CrtMetadataTestParam> {};

TEST_P(S3CrtMetadataTest, AsyncHeadReturnsBeforeResponse) {
#if defined(_WIN32)
  GTEST_SKIP() << "Test requires POSIX process APIs.";
#else
  const auto original_death_test_style = GTEST_FLAG_GET(death_test_style);
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  const auto run_child = [param = GetParam()]() -> int {
    namespace http = boost::beast::http;
    using Tcp = boost::asio::ip::tcp;
    auto fail = [](const std::string& message) {
      std::cerr << message << std::endl;
      return 1;
    };

    const auto initialized = EnsureS3InitializedForTest();
    if (!initialized.ok()) {
      return fail(initialized.ToString());
    }
    boost::asio::io_context server_context;
    Tcp::acceptor acceptor(server_context, {boost::asio::ip::address_v4::loopback(), 0});
    auto options = S3Options::FromAccessKey("ak", "sk");
    options.cloud_provider = kCloudProviderAWS;
    options.region = "us-east-1";
    options.scheme = "http";
    options.endpoint_override = "127.0.0.1:" + std::to_string(acceptor.local_endpoint().port());
    options.connect_timeout = 2;
    options.request_timeout = 5;
    options.use_crt_async_reads = true;

    auto fs_result = S3FileSystem::Make(options);
    if (!fs_result.ok()) {
      return fail(fs_result.status().ToString());
    }
    ArrowFileSystemPtr fs = std::move(fs_result).ValueOrDie();
#ifdef WITH_TALON
    if (param.use_talon) {
      ArrowFileSystemConfig config;
      config.storage_type = "remote";
      config.cloud_provider = kCloudProviderAWS;
      config.bucket_name = "test-bucket";
      config.talon_enabled = true;
      config.talon_coordinator = "127.0.0.1:1";
      auto talon_fs = TalonFileSystemProducer(config, fs).Make();
      if (!talon_fs.ok()) {
        return fail(talon_fs.status().ToString());
      }
      fs = std::move(talon_fs).ValueOrDie();
    }
#endif
    auto input_result = fs->OpenInputFile("test-bucket/path/object.txt");
    if (!input_result.ok()) {
      return fail(input_result.status().ToString());
    }
    auto input = std::move(input_result).ValueOrDie();

    // Hold a real HEAD response until the async method has returned. Running
    // submission separately lets this test release the response on a regression.
    arrow::Future<std::shared_ptr<const arrow::KeyValueMetadata>> metadata_future;
    arrow::Future<int64_t> size_future;
    auto submitted = std::async(std::launch::async, [input, &metadata_future, &size_future, param] {
      if (param.size_first) {
        size_future = dynamic_cast<NonBlockingRandomAccessFile*>(input.get())->GetSizeAsync();
      } else {
        metadata_future = input->ReadMetadataAsync({});
      }
    });
    Tcp::socket socket(server_context);
    acceptor.accept(socket);
    boost::beast::flat_buffer buffer;
    http::request<http::empty_body> request;
    http::read(socket, buffer, request);
    const bool returned_early = submitted.wait_for(std::chrono::seconds(1)) == std::future_status::ready;
    if (returned_early) {
      submitted.get();
      if (param.size_first ? size_future.is_finished() : metadata_future.is_finished()) {
        return fail("HEAD future completed before receiving the response");
      }
      if (param.close_before_completion) {
        if (!input->Close().ok()) {
          return fail("Failed to close input with a pending metadata request");
        }
        input.reset();
      }
    }

    http::response<http::empty_body> response{param.response_status, request.version()};
    response.content_length(param.response_status == http::status::ok ? 9 : 0);
    response.set(http::field::etag, "\"etag-v1\"");
    response.set(http::field::content_type, "application/octet-stream");
    response.set("x-amz-meta-source", "metadata-test");
    response.keep_alive(false);
    http::write(socket, response);
    socket.close();
    // Any unexpected second HEAD now fails instead of silently hiding a cache miss.
    acceptor.close();
    if (!returned_early) {
      (void)submitted.get();
      return fail("Async HEAD blocked until the response was released");
    }
    // The native HTTP adapter uses an absolute-form request target.
    const std::string expected_target = "http://" + options.endpoint_override + "/test-bucket/path/object.txt";
    if (request.method() != http::verb::head || request.target() != expected_target) {
      return fail("Unexpected HEAD request: " + std::string(request.method_string()) + " " +
                  std::string(request.target()));
    }
    if (param.size_first) {
      if (!size_future.Wait(5) || !size_future.result().ok() || size_future.result().ValueOrDie() != 9) {
        return fail("Async size did not resolve the HEAD response");
      }
      metadata_future = input->ReadMetadataAsync({});
      if (!metadata_future.is_finished()) {
        return fail("Async size did not cache HEAD metadata");
      }
    }
    if (!metadata_future.Wait(5)) {
      return fail("Metadata future did not complete");
    }
    const auto& result = metadata_future.result();
    if (param.response_status == http::status::ok) {
      if (!result.ok()) {
        return fail(result.status().ToString());
      }
      const auto metadata = result.ValueOrDie();
      if (metadata == nullptr || metadata->Get("ETag").ValueOrDie() != "\"etag-v1\"" ||
          metadata->Get("Content-Length").ValueOrDie() != "9" ||
          metadata->Get("Content-Type").ValueOrDie() != "application/octet-stream" ||
          metadata->Get("source").ValueOrDie() != "metadata-test") {
        return fail("HEAD metadata was not preserved");
      }
      if (param.close_before_completion) {
        return 0;
      }
      auto cached = input->ReadMetadataAsync({});
      if (!cached.is_finished() || !cached.result().ok() || cached.result().ValueOrDie() != metadata ||
          !input->ReadMetadata().ok() || input->GetSize().ValueOrDie() != 9) {
        return fail("HEAD metadata and size were not cached");
      }
      auto* async_input = dynamic_cast<NonBlockingRandomAccessFile*>(input.get());
      if (async_input == nullptr || async_input->GetSizeAsync().result().ValueOrDie() != 9) {
        return fail("Async size did not reuse the HEAD response");
      }
    } else {
      const auto message = result.status().ToString();
      if (!result.status().IsIOError()) {
        return fail("HEAD failure did not return an I/O error: " + message);
      }
      if (param.response_status == http::status::not_found) {
        if (message.find("test-bucket/path/object.txt") == std::string::npos) {
          return fail("Missing-object error lost its path: " + message);
        }
      } else {
        const auto detail = ExtendStatusDetail::UnwrapStatus(result.status());
        if (detail == nullptr || detail->code() != ExtendStatusCode::AwsErrorAccessDenied ||
            message.find("HeadObject") == std::string::npos || message.find("HTTP 403") == std::string::npos) {
          return fail("HEAD error lost its AWS details: " + message);
        }
      }
    }
    if (!input->Close().ok() || !input->ReadMetadataAsync({}).result().status().IsInvalid()) {
      return fail("Metadata read on a closed file did not fail");
    }
    return 0;
  };
  EXPECT_EXIT((::alarm(20), ::_exit(run_child())), ::testing::ExitedWithCode(0), "");
  GTEST_FLAG_SET(death_test_style, original_death_test_style);
#endif
}

INSTANTIATE_TEST_SUITE_P(S3Crt,
                         S3CrtMetadataTest,
                         ::testing::Values(S3CrtMetadataTestParam{boost::beast::http::status::ok},
                                           S3CrtMetadataTestParam{boost::beast::http::status::ok, true},
                                           S3CrtMetadataTestParam{boost::beast::http::status::ok, false, true},
                                           S3CrtMetadataTestParam{boost::beast::http::status::not_found},
                                           S3CrtMetadataTestParam{boost::beast::http::status::forbidden}));

#ifdef WITH_TALON
INSTANTIATE_TEST_SUITE_P(TalonCrt,
                         S3CrtMetadataTest,
                         ::testing::Values(S3CrtMetadataTestParam{boost::beast::http::status::ok, false, true, true}));
#endif

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
