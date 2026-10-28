// Copyright 2026 Zilliz
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//     http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <gtest/gtest.h>

#include <atomic>
#include <filesystem>
#include <fstream>
#include <thread>
#include <sys/stat.h>

#include "milvus-storage/filesystem/local_durability.h"
#include <unistd.h>

#include "milvus-storage/common/fiu_local.h"
#include "milvus-storage/filesystem/local_fs_producer.h"
#include "milvus-storage/filesystem/upload_conditional.h"
#include "milvus-storage/manifest.h"

namespace milvus_storage {
namespace {

class LocalDurabilityTest : public ::testing::Test {
  protected:
  void SetUp() override {
    char pattern[] = "/tmp/milvus-local-durability-XXXXXX";
    auto directory = ::mkdtemp(pattern);
    ASSERT_NE(directory, nullptr);
    root_ = directory;
    config_.root_path = root_;
    auto fs = LocalFileSystemProducer(config_).Make();
    ASSERT_TRUE(fs.ok()) << fs.status().ToString();
    fs_ = fs.ValueOrDie();
    ASSERT_TRUE(fs_->CreateDir("objects", true).ok());
    FIU_INIT();
  }

  void TearDown() override { std::filesystem::remove_all(root_); }

  std::string Read(const std::string& name) {
    std::ifstream input(root_ + "/" + name, std::ios::binary);
    return std::string(std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>());
  }

  ArrowFileSystemConfig config_;
  ArrowFileSystemPtr fs_;
  std::string root_;
};

TEST_F(LocalDurabilityTest, WritesDirectlyAndSynchronizesAtCompletion) {
  auto output = fs_->OpenOutputStream("objects/data").ValueOrDie();
  ASSERT_TRUE(output->Write("complete", 8).ok());
  ASSERT_TRUE(output->Flush().ok());
  ASSERT_EQ(Read("objects/data"), "complete");
  ASSERT_TRUE(output->Close().ok());
  ASSERT_TRUE(output->Close().ok());

  auto replacement = fs_->OpenOutputStream("objects/data").ValueOrDie();
  ASSERT_TRUE(replacement->Write("new", 3).ok());
  ASSERT_TRUE(replacement->Close().ok());
  ASSERT_EQ(Read("objects/data"), "new");
}

TEST_F(LocalDurabilityTest, AbortDoesNotAcknowledgePartialOutput) {
  auto output = fs_->OpenOutputStream("objects/partial").ValueOrDie();
  ASSERT_TRUE(output->Write("partial", 7).ok());
  ASSERT_TRUE(output->Abort().ok());
  ASSERT_FALSE(output->Close().ok());
  // The caller discards failed output; no temporary file/publication protocol.
  ASSERT_EQ(Read("objects/partial"), "partial");
}

TEST_F(LocalDurabilityTest, ExistingConditionalCreateRejectsConcurrentWriter) {
  auto conditional = std::dynamic_pointer_cast<UploadConditional>(fs_);
  ASSERT_NE(conditional, nullptr);
  arrow::Status a, b;
  std::atomic<bool> start{false};
  auto write = [&](const char* data, arrow::Status& status) {
    while (!start.load()) std::this_thread::yield();
    auto output = conditional->OpenConditionalOutputStream("objects/manifest-1.avro", nullptr);
    status = output.status();
    if (status.ok()) {
      status = output.ValueOrDie()->Write(data, 1);
      if (status.ok())
        status = output.ValueOrDie()->Close();
    }
  };
  std::thread one([&] { write("a", a); });
  std::thread two([&] { write("b", b); });
  start.store(true);
  one.join();
  two.join();
  ASSERT_NE(a.ok(), b.ok());
  ASSERT_EQ(Read("objects/manifest-1.avro"), a.ok() ? "a" : "b");
}

TEST_F(LocalDurabilityTest, CopyAppendAndMoveRoundTrip) {
  auto initial = fs_->OpenOutputStream("objects/source").ValueOrDie();
  ASSERT_TRUE(initial->Write("data", 4).ok());
  ASSERT_TRUE(initial->Close().ok());
  ASSERT_TRUE(fs_->CopyFile("objects/source", "objects/copy").ok());
  auto append = fs_->OpenAppendStream("objects/copy").ValueOrDie();
  ASSERT_TRUE(append->Write("tail", 4).ok());
  ASSERT_TRUE(append->Close().ok());
  ASSERT_TRUE(fs_->Move("objects/copy", "objects/moved").ok());
  ASSERT_EQ(Read("objects/moved"), "datatail");
  ASSERT_EQ(Read("objects/source"), "data");
}

TEST_F(LocalDurabilityTest, ManifestVersionsRoundTripAndNeverOverwrite) {
  api::Manifest manifest;
  ASSERT_TRUE(api::Manifest::WriteTo(fs_, "objects/manifest-1.avro", manifest).ok());
  ASSERT_TRUE(api::Manifest::ReadFrom(fs_, "objects/manifest-1.avro").ok());
  ASSERT_TRUE(api::Manifest::WriteTo(fs_, "objects/manifest-1.avro", manifest).IsAlreadyExists());
}

TEST_F(LocalDurabilityTest, SymlinkThenDotDotSelectsPhysicalRoot) {
  std::filesystem::create_directories(root_ + "/actual/child");
  std::filesystem::create_directories(root_ + "/actual/dataset");
  std::filesystem::create_directory_symlink(root_ + "/actual/child", root_ + "/link");
  // Deliberately do not create the lexically normalized root_/dataset.
  config_.root_path = root_ + "/link/../dataset";
  auto result = LocalFileSystemProducer(config_).Make();
  ASSERT_TRUE(result.ok()) << result.status();
  auto output = result.ValueOrDie()->OpenOutputStream("data").ValueOrDie();
  ASSERT_TRUE(output->Write("data", 4).ok());
  ASSERT_TRUE(output->Close().ok());
  ASSERT_EQ(Read("actual/dataset/data"), "data");
  ASSERT_TRUE(SyncLocalDirectoryTree(config_.root_path, root_ + "/actual").ok());
}

TEST_F(LocalDurabilityTest, SearchOnlyAncestorOutsideDataRoot) {
  if (::geteuid() == 0)
    GTEST_SKIP() << "root bypasses directory read permissions";
  std::filesystem::create_directories(root_ + "/search/data");
  auto ancestor = root_ + "/search";
  ASSERT_EQ(::chmod(ancestor.c_str(), 0111), 0);
  config_.root_path = ancestor + "/data";
  auto result = LocalFileSystemProducer(config_).Make();
  // Restore even if an assertion later fails.
  struct Restore {
    std::string path;
    ~Restore() { ::chmod(path.c_str(), 0700); }
  } restore{ancestor};
  ASSERT_TRUE(result.ok()) << result.status();
  auto fs = result.ValueOrDie();
  ASSERT_TRUE(fs->CreateDir("new/parents", true).ok());
  auto output = fs->OpenOutputStream("new/parents/data").ValueOrDie();
  ASSERT_TRUE(output->Write("data", 4).ok());
  ASSERT_TRUE(output->Close().ok());
  ASSERT_TRUE(fs->CopyFile("new/parents/data", "copy").ok());
  ASSERT_TRUE(fs->Move("copy", "moved").ok());

  // Milvus uses '/' as its key namespace but the same physical data boundary.
  config_.root_path = "/";
  config_.local_durability_root = ancestor + "/data";
  result = LocalFileSystemProducer(config_).Make();
  ASSERT_TRUE(result.ok()) << result.status();
  output = result.ValueOrDie()->OpenOutputStream(ancestor + "/data/absolute").ValueOrDie();
  ASSERT_TRUE(output->Write("data", 4).ok());
  ASSERT_TRUE(output->Close().ok());
}

TEST_F(LocalDurabilityTest, BoundaryRejectsOutsideDirectoriesAndSeparatesCache) {
  auto other = config_;
  other.local_durability_root = root_ + "/objects";
  ASSERT_NE(config_.GetCacheKey(), other.GetCacheKey());
  ASSERT_TRUE(SyncLocalDirectoryTree(root_, root_ + "/objects").IsInvalid());
}

#ifdef BUILD_WITH_FIU
TEST_F(LocalDurabilityTest, OpeningProvisionedRootDoesNotSynchronizeReadPaths) {
  auto output = fs_->OpenOutputStream("objects/readable").ValueOrDie();
  ASSERT_TRUE(output->Write("data", 4).ok());
  ASSERT_TRUE(output->Close().ok());
  ScopedFiuFault fault(FIUKEY_LOCAL_DIRECTORY_SYNC_FAIL, false);
  auto reopened = LocalFileSystemProducer(config_).Make();
  ASSERT_TRUE(reopened.ok()) << reopened.status();
  auto input = reopened.ValueOrDie()->OpenInputFile("objects/readable");
  ASSERT_TRUE(input.ok()) << input.status();
  ASSERT_TRUE(input.ValueOrDie()->Close().ok());
  output = reopened.ValueOrDie()->OpenOutputStream("objects/written").ValueOrDie();
  ASSERT_TRUE(output->Write("data", 4).ok());
  ASSERT_FALSE(output->Close().ok());
}

TEST_F(LocalDurabilityTest, CloseFailureCannotReturnSuccess) {
  auto output = fs_->OpenOutputStream("objects/unclosed").ValueOrDie();
  ASSERT_TRUE(output->Write("data", 4).ok());
  ScopedFiuFault fault(FIUKEY_LOCAL_OUTPUT_CLOSE_FAIL, false);
  ASSERT_FALSE(output->Close().ok());
  ASSERT_FALSE(output->Close().ok());
}

TEST_F(LocalDurabilityTest, FileSyncFailureIsStickyAndOnlyHappensOnClose) {
  auto output = fs_->OpenOutputStream("objects/data").ValueOrDie();
  {
    ScopedFiuFault fault(FIUKEY_LOCAL_FILE_SYNC_FAIL, false);
    for (int i = 0; i < 64; ++i) ASSERT_TRUE(output->Write("chunk", 5).ok());
    ASSERT_TRUE(output->Flush().ok());
    ASSERT_FALSE(output->Close().ok());
  }
  ASSERT_FALSE(output->Close().ok());
  auto retry = fs_->OpenOutputStream("objects/data").ValueOrDie();
  ASSERT_TRUE(retry->Write("retry", 5).ok());
  ASSERT_TRUE(retry->Close().ok());
  ASSERT_EQ(Read("objects/data"), "retry");
}

TEST_F(LocalDurabilityTest, DirectorySyncFailureCannotReturnSuccess) {
  ASSERT_TRUE(fs_->CreateDir("new/parents", true).ok());
  auto output = fs_->OpenOutputStream("new/parents/data").ValueOrDie();
  ASSERT_TRUE(output->Write("complete", 8).ok());
  {
    ScopedFiuFault fault(FIUKEY_LOCAL_DIRECTORY_SYNC_FAIL, false);
    ASSERT_FALSE(output->Close().ok());
    ASSERT_FALSE(output->Close().ok());
  }
  auto retry = fs_->OpenOutputStream("new/parents/data").ValueOrDie();
  ASSERT_TRUE(retry->Write("retry", 5).ok());
  ASSERT_TRUE(retry->Close().ok());
}

TEST_F(LocalDurabilityTest, CopyMoveAndCreatePropagateSyncFailure) {
  auto output = fs_->OpenOutputStream("objects/source").ValueOrDie();
  ASSERT_TRUE(output->Write("complete", 8).ok());
  ASSERT_TRUE(output->Close().ok());
  {
    ScopedFiuFault fault(FIUKEY_LOCAL_FILE_SYNC_FAIL, false);
    ASSERT_FALSE(fs_->CopyFile("objects/source", "objects/copy").ok());
    ASSERT_FALSE(fs_->Move("objects/source", "objects/moved").ok());
  }
  ScopedFiuFault fault(FIUKEY_LOCAL_DIRECTORY_SYNC_FAIL, false);
  ASSERT_FALSE(fs_->CreateDir("unsynced/parent", true).ok());
}

TEST_F(LocalDurabilityTest, FailedCreateRetryAndRecreatedDirectoryStillSynchronize) {
  {
    ScopedFiuFault fault(FIUKEY_LOCAL_DIRECTORY_SYNC_FAIL, false);
    ASSERT_FALSE(fs_->CreateDir("retry/nested", true).ok());
  }
  ASSERT_TRUE(fs_->CreateDir("retry/nested", true).ok());
  ASSERT_TRUE(fs_->DeleteDir("retry").ok());
  {
    ScopedFiuFault fault(FIUKEY_LOCAL_DIRECTORY_SYNC_FAIL, false);
    ASSERT_FALSE(fs_->CreateDir("retry/nested", true).ok());
  }
  ASSERT_TRUE(fs_->CreateDir("retry/nested", true).ok());
}

TEST_F(LocalDurabilityTest, ManifestSyncFailureCannotReturnCommitSuccess) {
  api::Manifest manifest;
  ScopedFiuFault fault(FIUKEY_LOCAL_FILE_SYNC_FAIL, false);
  ASSERT_FALSE(api::Manifest::WriteTo(fs_, "objects/manifest-1.avro", manifest).ok());
}
#endif

}  // namespace
}  // namespace milvus_storage
