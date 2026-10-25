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
#include <unistd.h>

#include "milvus-storage/common/fiu_local.h"
#include "milvus-storage/filesystem/local_durability.h"
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

  size_t Temporaries() {
    size_t count = 0;
    for (const auto& entry : std::filesystem::recursive_directory_iterator(root_)) {
      if (entry.path().filename().string().find(".milvus-tmp-") == 0)
        ++count;
    }
    return count;
  }

  ArrowFileSystemConfig config_;
  ArrowFileSystemPtr fs_;
  std::string root_;
};

TEST_F(LocalDurabilityTest, ReplacementIsInvisibleUntilComplete) {
  auto initial = fs_->OpenOutputStream("objects/manifest").ValueOrDie();
  ASSERT_TRUE(initial->Write("old", 3).ok());
  ASSERT_TRUE(initial->Close().ok());
  ASSERT_EQ(::chmod((root_ + "/objects/manifest").c_str(), 0640), 0);
  auto replacement = fs_->OpenOutputStream("objects/manifest").ValueOrDie();
  ASSERT_TRUE(replacement->Write("new", 3).ok());
  ASSERT_TRUE(replacement->Flush().ok());
  ASSERT_EQ(Read("objects/manifest"), "old");
  ASSERT_TRUE(replacement->Close().ok());
  ASSERT_TRUE(replacement->Close().ok());
  ASSERT_EQ(Read("objects/manifest"), "new");
  struct stat info;
  ASSERT_EQ(::stat((root_ + "/objects/manifest").c_str(), &info), 0);
  ASSERT_EQ(info.st_mode & 0777, 0640);
  ASSERT_EQ(Temporaries(), 0);
}

TEST_F(LocalDurabilityTest, AbandonAndAbortDoNotPublish) {
  {
    auto abandoned = fs_->OpenOutputStream("objects/abandoned").ValueOrDie();
    ASSERT_TRUE(abandoned->Write("partial", 7).ok());
  }
  auto aborted = fs_->OpenOutputStream("objects/aborted").ValueOrDie();
  ASSERT_TRUE(aborted->Write("partial", 7).ok());
  ASSERT_TRUE(aborted->Abort().ok());
  ASSERT_FALSE(aborted->Close().ok());
  ASSERT_FALSE(std::filesystem::exists(root_ + "/objects/abandoned"));
  ASSERT_FALSE(std::filesystem::exists(root_ + "/objects/aborted"));
  ASSERT_EQ(Temporaries(), 0);
}

TEST_F(LocalDurabilityTest, ConcurrentManifestPublicationDoesNotReplaceWinner) {
  auto conditional = std::dynamic_pointer_cast<UploadConditional>(fs_);
  ASSERT_NE(conditional, nullptr);
  // Both writers open before either publishes. The object must not become
  // visible as an empty/partial manifest while either stream is being filled.
  auto first = conditional->OpenConditionalOutputStream("objects/manifest-1.avro", nullptr);
  auto second = conditional->OpenConditionalOutputStream("objects/manifest-1.avro", nullptr);
  ASSERT_TRUE(first.ok());
  ASSERT_TRUE(second.ok());
  ASSERT_TRUE(first.ValueOrDie()->Write("first", 5).ok());
  ASSERT_TRUE(second.ValueOrDie()->Write("second", 6).ok());
  ASSERT_FALSE(std::filesystem::exists(root_ + "/objects/manifest-1.avro"));
  arrow::Status a, b;
  std::atomic<bool> start{false};
  std::thread one([&] {
    while (!start.load()) std::this_thread::yield();
    a = first.ValueOrDie()->Close();
  });
  std::thread two([&] {
    while (!start.load()) std::this_thread::yield();
    b = second.ValueOrDie()->Close();
  });
  start.store(true);
  one.join();
  two.join();
  ASSERT_NE(a.ok(), b.ok());
  ASSERT_EQ(Read("objects/manifest-1.avro"), a.ok() ? "first" : "second");
  ASSERT_FALSE(conditional->OpenConditionalOutputStream("objects/manifest-1.avro", nullptr).ok());
  ASSERT_EQ(Temporaries(), 0);
}

TEST_F(LocalDurabilityTest, CopyAndAppendCompleteObjects) {
  auto initial = fs_->OpenOutputStream("objects/source").ValueOrDie();
  ASSERT_TRUE(initial->Write("data", 4).ok());
  ASSERT_TRUE(initial->Close().ok());
  ASSERT_TRUE(fs_->CopyFile("objects/source", "objects/copy").ok());
  auto append = fs_->OpenAppendStream("objects/copy").ValueOrDie();
  ASSERT_TRUE(append->Write("tail", 4).ok());
  ASSERT_TRUE(append->Close().ok());
  ASSERT_EQ(Read("objects/copy"), "datatail");
  ASSERT_EQ(Read("objects/source"), "data");
}

TEST_F(LocalDurabilityTest, ManifestVersionsRoundTripAndNeverOverwrite) {
  api::Manifest manifest;
  ASSERT_TRUE(api::Manifest::WriteTo(fs_, "objects/manifest-1.avro", manifest).ok());
  ASSERT_TRUE(api::Manifest::ReadFrom(fs_, "objects/manifest-1.avro").ok());
  ASSERT_TRUE(api::Manifest::WriteTo(fs_, "objects/manifest-1.avro", manifest).IsAlreadyExists());
}

#ifdef BUILD_WITH_FIU
TEST_F(LocalDurabilityTest, CloseFailureDoesNotPublishObject) {
  auto output = fs_->OpenOutputStream("objects/unclosed").ValueOrDie();
  ASSERT_TRUE(output->Write("data", 4).ok());
  ScopedFiuFault fault(FIUKEY_LOCAL_OUTPUT_CLOSE_FAIL, false);
  ASSERT_FALSE(output->Close().ok());
  ASSERT_FALSE(output->Close().ok());
  ASSERT_FALSE(std::filesystem::exists(root_ + "/objects/unclosed"));
  ASSERT_EQ(Temporaries(), 0);
}

TEST_F(LocalDurabilityTest, FileSyncFailurePreservesOldObjectAndIsSticky) {
  auto initial = fs_->OpenOutputStream("objects/manifest").ValueOrDie();
  ASSERT_TRUE(initial->Write("old", 3).ok());
  ASSERT_TRUE(initial->Close().ok());
  auto replacement = fs_->OpenOutputStream("objects/manifest").ValueOrDie();
  ScopedFiuFault fault(FIUKEY_LOCAL_FILE_SYNC_FAIL, false);
  // No synchronization per row, buffer chunk, or Flush.
  for (int i = 0; i < 64; ++i) ASSERT_TRUE(replacement->Write("chunk", 5).ok());
  ASSERT_TRUE(replacement->Flush().ok());
  ASSERT_FALSE(replacement->Close().ok());
  ASSERT_FALSE(replacement->Close().ok());
  ASSERT_EQ(Read("objects/manifest"), "old");
  ASSERT_EQ(Temporaries(), 0);
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
  ASSERT_EQ(Read("new/parents/data"), "complete");
  auto retry = fs_->OpenOutputStream("new/parents/data").ValueOrDie();
  ASSERT_TRUE(retry->Write("retry", 5).ok());
  ASSERT_TRUE(retry->Close().ok());
  ASSERT_EQ(Read("new/parents/data"), "retry");
}

TEST_F(LocalDurabilityTest, ManifestSyncFailureCannotCommit) {
  api::Manifest manifest;
  ScopedFiuFault fault(FIUKEY_LOCAL_FILE_SYNC_FAIL, false);
  ASSERT_FALSE(api::Manifest::WriteTo(fs_, "objects/manifest-1.avro", manifest).ok());
  ASSERT_FALSE(std::filesystem::exists(root_ + "/objects/manifest-1.avro"));
}
#endif

}  // namespace
}  // namespace milvus_storage
