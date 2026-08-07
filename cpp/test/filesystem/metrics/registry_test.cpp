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

#include <gtest/gtest.h>
#include <chrono>
#include <thread>

#include "milvus-storage/filesystem/observable.h"

namespace milvus_storage::test {

TEST(RegistryTest, ScopedOpRecordsLatencyStatusAndInFlight) {
  FilesystemMetrics m;
  {
    auto op = m.StartOp(OpType::Head);
    EXPECT_EQ(m.GetSnapshot().in_flight, 1);  // in-flight while scope alive
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
  }
  auto s = m.GetSnapshot();
  EXPECT_EQ(s.in_flight, 0);  // decremented
  auto& head = s.ops[static_cast<int>(OpType::Head)];
  EXPECT_EQ(head.count_by_status[static_cast<int>(OpStatus::Ok)], 1);
  EXPECT_EQ(head.latency_count, 1);
  EXPECT_GT(head.latency_sum_us, 0);
}

TEST(RegistryTest, FailAttributesStatusToOp) {
  FilesystemMetrics m;
  {
    auto op = m.StartOp(OpType::DeleteFile);
    op.Fail(OpStatus::NotFound);
  }
  auto s = m.GetSnapshot();
  auto& del = s.ops[static_cast<int>(OpType::DeleteFile)];
  EXPECT_EQ(del.count_by_status[static_cast<int>(OpStatus::NotFound)], 1);
  EXPECT_EQ(del.count_by_status[static_cast<int>(OpStatus::Ok)], 0);
}

TEST(RegistryTest, TransferRecordsBytesAndSizeHistogram) {
  FilesystemMetrics m;
  {
    auto op = m.StartTransfer(OpType::Read);
    op.RecordBytes(4096);
  }
  auto s = m.GetSnapshot();
  auto& t = s.transfers[TransferIndex(OpType::Read)];
  EXPECT_EQ(t.bytes_total, 4096);
  EXPECT_EQ(t.size_count, 1);
  EXPECT_EQ(t.size_sum_bytes, 4096);
  EXPECT_EQ(m.TransferBytes(OpType::Read), 4096);
  EXPECT_EQ(m.OpCount(OpType::Read, OpStatus::Ok), 1);
}

TEST(RegistryTest, CancelRecordsNothingButBalancesInFlight) {
  FilesystemMetrics m;
  {
    auto op = m.StartTransfer(OpType::Read);
    op.Cancel();
  }
  auto s = m.GetSnapshot();
  EXPECT_EQ(s.in_flight, 0);
  EXPECT_EQ(s.ops[static_cast<int>(OpType::Read)].latency_count, 0);
  EXPECT_EQ(m.OpCount(OpType::Read, OpStatus::Ok), 0);
}

TEST(RegistryTest, RecordRetryCountsPerOp) {
  FilesystemMetrics m;
  {
    auto op = m.StartTransfer(OpType::Write);
    op.RecordRetry();
    op.RecordRetry(2);
  }
  auto s = m.GetSnapshot();
  EXPECT_EQ(s.ops[static_cast<int>(OpType::Write)].retry_count, 3);
}

TEST(RegistryTest, QueryAccessorsReflectRegistry) {
  FilesystemMetrics m;
  { auto op = m.StartOp(OpType::CreateDir); }
  {
    auto op = m.StartOp(OpType::DeleteFile);
    op.Fail(OpStatus::Timeout);
  }
  m.IncrementMultipartCreated();
  m.IncrementMultipartFinished();

  EXPECT_EQ(m.OpCount(OpType::CreateDir), 1);
  EXPECT_EQ(m.OpCount(OpType::DeleteFile), 1);  // counted regardless of outcome
  EXPECT_EQ(m.FailedCount(), 1);                // one non-Ok status
  EXPECT_EQ(m.MultipartCreated(), 1);
  EXPECT_EQ(m.MultipartFinished(), 1);
}

TEST(RegistryTest, ResetClearsRegistry) {
  FilesystemMetrics m;
  { auto op = m.StartOp(OpType::List); }
  m.Reset();
  EXPECT_EQ(m.GetSnapshot().ops[static_cast<int>(OpType::List)].latency_count, 0);
}

}  // namespace milvus_storage::test
