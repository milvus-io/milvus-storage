// Copyright 2023 Zilliz
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

#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <arrow/result.h>
#include <arrow/status.h>

namespace arrow {
class Buffer;
namespace fs {
class FileSystem;
}  // namespace fs
namespace io {
class RandomAccessFile;
}  // namespace io
}  // namespace arrow

namespace milvus_storage::vortex {

struct ByteRange {
  uint64_t offset = 0;
  uint64_t length = 0;
};

struct RowRange {
  uint64_t start = 0;
  uint64_t end = 0;
};

enum class VortexPhysicalGranularity : uint16_t {
  kUnknown = 0,
  kFlat = 1,
  kRowGroup = 2,
};

struct VortexCellMeta {
  uint64_t cell_id = 0;
  VortexPhysicalGranularity granularity = VortexPhysicalGranularity::kUnknown;
  uint64_t physical_unit_index = 0;
  uint64_t row_offset = 0;
  uint64_t row_count = 0;
  std::vector<uint64_t> flat_segment_ids;
  std::vector<ByteRange> flat_segment_ranges;
  uint64_t memory_bytes = 0;
  uint64_t storage_bytes = 0;
};

using VortexCellMetas = std::vector<VortexCellMeta>;
using VortexCellMetasPtr = std::shared_ptr<const VortexCellMetas>;

class VortexRangeFile {
  public:
  virtual ~VortexRangeFile() = default;

  VortexRangeFile(const VortexRangeFile&) = delete;
  VortexRangeFile& operator=(const VortexRangeFile&) = delete;

  VortexRangeFile() = default;

  virtual void Resize(uint64_t size) = 0;
  virtual uint64_t Size() const = 0;
  virtual arrow::Status WriteAt(const uint64_t& offset, const std::shared_ptr<arrow::Buffer>& data) = 0;
  virtual arrow::Result<int64_t> ReadAt(int64_t position, int64_t nbytes, void* out) const = 0;
  virtual arrow::Result<std::shared_ptr<arrow::Buffer>> ReadAt(int64_t position, int64_t nbytes) const = 0;
  virtual void Punch(uint64_t offset, uint64_t length) = 0;
};

// The concrete sparse/range file is owned by the caller-provided filesystem.
// Storage only depends on this provider contract when VortexFooterReader or
// VortexTranslater needs to read, fill, or punch byte ranges.
class VortexRangeFileProvider {
  public:
  virtual ~VortexRangeFileProvider() = default;

  virtual arrow::Result<std::shared_ptr<VortexRangeFile>> GetVortexRangeFile(const std::string& path) const = 0;
};

arrow::Result<std::shared_ptr<VortexRangeFile>> GetVortexRangeFile(const std::shared_ptr<arrow::fs::FileSystem>& fs,
                                                                   const std::string& path);

arrow::Status FillVortexRangeFile(const std::shared_ptr<arrow::io::RandomAccessFile>& source_file,
                                  const std::shared_ptr<VortexRangeFile>& range_file,
                                  uint64_t offset,
                                  uint64_t length);

std::vector<ByteRange> MergeByteRanges(std::vector<ByteRange> ranges);

}  // namespace milvus_storage::vortex
