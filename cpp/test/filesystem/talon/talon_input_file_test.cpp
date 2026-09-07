// Copyright 2025 Zilliz
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

// Integration coverage for the Talon-backed input file. It exercises the real
// Talon C SDK, so it only compiles when WITH_TALON is set and only runs when a
// coordinator is provided; otherwise it skips, matching how the other
// remote-storage tests behave without their backing service. There is no fake
// Talon here on purpose — the SDK is an upstream dependency, not something
// milvus-storage reimplements.
#ifdef WITH_TALON

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include <arrow/buffer.h>
#include <arrow/io/interfaces.h>
#include <arrow/testing/gtest_util.h>

#include "milvus-storage/filesystem/async_random_access_file.h"
#include "milvus-storage/filesystem/talon/talon_client.h"
#include "milvus-storage/filesystem/talon/talon_input_file.h"

namespace milvus_storage::talon {
namespace {

// Configuration comes from the environment so the test binds to whatever Talon
// deployment CI (or a developer) points it at:
//   TALON_COORDINATOR_ADDR - control address of a running coordinator (required)
//   TALON_TEST_URI         - an object URI the deployment can serve (required)
struct TalonEnv {
  std::string coordinator;
  std::string uri;
  bool available = false;
};

TalonEnv ReadEnv() {
  TalonEnv env;
  const char* addr = std::getenv("TALON_COORDINATOR_ADDR");
  const char* uri = std::getenv("TALON_TEST_URI");
  if (addr != nullptr && addr[0] != '\0' && uri != nullptr && uri[0] != '\0') {
    env.coordinator = addr;
    env.uri = uri;
    env.available = true;
  }
  return env;
}

std::shared_ptr<TalonClient> MakeClientOrSkip(const TalonEnv& env) {
  auto result = TalonClient::Make(env.coordinator);
  if (!result.ok()) {
    ADD_FAILURE() << "failed to create Talon client: " << result.status().ToString();
    return nullptr;
  }
  return result.ValueOrDie();
}

class TalonInputFileTest : public ::testing::Test {
  protected:
  void SetUp() override {
    env_ = ReadEnv();
    if (!env_.available) {
      GTEST_SKIP() << "set TALON_COORDINATOR_ADDR and TALON_TEST_URI to run Talon integration tests";
    }
  }

  TalonEnv env_;
};

TEST_F(TalonInputFileTest, GetSizeResolvesObjectLength) {
  auto client = MakeClientOrSkip(env_);
  ASSERT_NE(client, nullptr);

  TalonInputFile file(client, env_.uri);
  ASSERT_OK_AND_ASSIGN(int64_t size, file.GetSize());
  EXPECT_GE(size, 0);
}

TEST_F(TalonInputFileTest, ReadAtMatchesSequentialRead) {
  auto client = MakeClientOrSkip(env_);
  ASSERT_NE(client, nullptr);

  TalonInputFile file(client, env_.uri);
  ASSERT_OK_AND_ASSIGN(int64_t size, file.GetSize());
  if (size == 0) {
    GTEST_SKIP() << "TALON_TEST_URI is empty; nothing to read";
  }

  const int64_t want = std::min<int64_t>(size, 4096);
  ASSERT_OK_AND_ASSIGN(auto via_read_at, file.ReadAt(0, want));
  EXPECT_EQ(via_read_at->size(), want);

  ASSERT_OK(file.Seek(0));
  ASSERT_OK_AND_ASSIGN(int64_t pos, file.Tell());
  EXPECT_EQ(pos, 0);
  ASSERT_OK_AND_ASSIGN(auto via_read, file.Read(want));
  EXPECT_EQ(via_read->size(), want);
  EXPECT_TRUE(via_read_at->Equals(*via_read));

  ASSERT_OK_AND_ASSIGN(pos, file.Tell());
  EXPECT_EQ(pos, want);
}

TEST_F(TalonInputFileTest, AsyncReadIntoFillsCallerBuffer) {
  auto client = MakeClientOrSkip(env_);
  ASSERT_NE(client, nullptr);

  auto file = std::make_shared<TalonInputFile>(client, env_.uri);
  ASSERT_OK_AND_ASSIGN(int64_t size, file->GetSize());
  if (size == 0) {
    GTEST_SKIP() << "TALON_TEST_URI is empty; nothing to read";
  }

  const int64_t want = std::min<int64_t>(size, 1024);
  std::vector<uint8_t> buffer(static_cast<size_t>(want), 0);
  // Hold the Future in a named local: Future::result() returns a reference into
  // the future's shared state, so the future must outlive every use of that
  // reference. Calling .result() on a temporary (…AsyncInto(...).result()) frees
  // the state at the end of the statement while ASSERT_OK_AND_ASSIGN still reads
  // it on the next line — a use-after-free.
  auto read_future = file->ReadAtAsyncInto(0, want, buffer.data());
  ASSERT_OK_AND_ASSIGN(int64_t read, read_future.result());
  EXPECT_EQ(read, want);

  ASSERT_OK_AND_ASSIGN(auto expected, file->ReadAt(0, want));
  EXPECT_EQ(0, std::memcmp(buffer.data(), expected->data(), static_cast<size_t>(want)));
}

TEST_F(TalonInputFileTest, ShortReadPastEofReturnsAvailableBytes) {
  auto client = MakeClientOrSkip(env_);
  ASSERT_NE(client, nullptr);

  TalonInputFile file(client, env_.uri);
  ASSERT_OK_AND_ASSIGN(int64_t size, file.GetSize());

  ASSERT_OK_AND_ASSIGN(auto buffer, file.ReadAt(size, 1024));
  EXPECT_EQ(buffer->size(), 0);
}

// The whole point of implementing arrow::io::RandomAccessFile plus
// NonBlockingReadAtFile: any Arrow-based format reader (Parquet, Lance, IPC, ...)
// can consume this handle, and the ones that special-case async reads can
// recover the fast path via a dynamic_cast.
TEST_F(TalonInputFileTest, ExposesArrowAndAsyncInterfaces) {
  auto client = MakeClientOrSkip(env_);
  ASSERT_NE(client, nullptr);

  ASSERT_OK_AND_ASSIGN(auto opened, OpenTalonInputFile(client, env_.uri));
  std::shared_ptr<arrow::io::RandomAccessFile> as_arrow = opened;
  EXPECT_NE(as_arrow, nullptr);
  EXPECT_NE(dynamic_cast<NonBlockingReadAtFile*>(opened.get()), nullptr);

  ASSERT_OK(opened->Close());
  EXPECT_TRUE(opened->closed());
}

TEST_F(TalonInputFileTest, ReadOnClosedFileFails) {
  auto client = MakeClientOrSkip(env_);
  ASSERT_NE(client, nullptr);

  TalonInputFile file(client, env_.uri);
  ASSERT_OK(file.Close());

  uint8_t byte = 0;
  auto status = file.ReadAtAsyncInto(0, 1, &byte).result().status();
  EXPECT_TRUE(status.IsInvalid()) << status.ToString();
}

}  // namespace
}  // namespace milvus_storage::talon

#endif  // WITH_TALON
