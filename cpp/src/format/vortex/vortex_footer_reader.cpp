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

#include "milvus-storage/common/extend_status.h"
#include "milvus-storage/format/vortex/vortex_footer_reader.h"

#include "milvus-storage/format/vortex/vortex_footer_internal.h"

#include <algorithm>
#include <cerrno>
#include <limits>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>

#include <arrow/buffer.h>
#include <arrow/c/bridge.h>
#include <arrow/filesystem/filesystem.h>
#include <arrow/io/interfaces.h>
#include <arrow/type.h>
#include <arrow/util/io_util.h>
#include <fmt/format.h>

#include "milvus-storage/common/log.h"
#include "milvus-storage/filesystem/ffi/filesystem_internal.h"
#include "milvus-storage/filesystem/util_internal.h"
#include "milvus-storage/format/vortex/vortex_field_layout_internal.h"
#include "milvus-storage/format/vortex/vortex_types.h"
#include "vortex_bridge.h"

namespace milvus_storage::vortex {

arrow::Result<std::shared_ptr<VortexRangeFile>> GetVortexRangeFile(const std::shared_ptr<arrow::fs::FileSystem>& fs,
                                                                   const std::string& path) {
  if (!fs) {
    return MakeExtendErrorMsg(ExtendStatusCode::InternalInvariantViolated, "GetVortexRangeFile requires a non-null filesystem");
  }
  auto provider = std::dynamic_pointer_cast<VortexRangeFileProvider>(fs);
  if (!provider) {
    return arrow::Status::Invalid(fmt::format("filesystem {} does not provide a vortex range file", fs->type_name()));
  }
  return provider->GetVortexRangeFile(path);
}

arrow::Status FillVortexRangeFile(const std::shared_ptr<arrow::io::RandomAccessFile>& source_file,
                                  const std::shared_ptr<VortexRangeFile>& range_file,
                                  uint64_t offset,
                                  uint64_t length) {
  if (length == 0) {
    return arrow::Status::OK();
  }
  if (!source_file) {
    return MakeExtendErrorMsg(ExtendStatusCode::InternalInvariantViolated, "FillVortexRangeFile requires a non-null source file");
  }
  if (!range_file) {
    return MakeExtendErrorMsg(ExtendStatusCode::InternalInvariantViolated, "FillVortexRangeFile requires a non-null range file");
  }
  if (length > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) ||
      offset > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
    return arrow::Status::Invalid(
        fmt::format("Vortex sparse range exceeds Arrow IO limits, offset={}, length={}", offset, length));
  }
  ARROW_ASSIGN_OR_RAISE(auto data, source_file->ReadAt(static_cast<int64_t>(offset), static_cast<int64_t>(length)));
  if (!data || static_cast<uint64_t>(data->size()) != length) {
    return arrow::Status::IOError(
        fmt::format("Short read while filling vortex sparse file, offset={}, expect={}, got={}", offset, length,
                    data ? data->size() : 0));
  }
  return range_file->WriteAt(offset, data);
}

std::vector<ByteRange> MergeByteRanges(std::vector<ByteRange> ranges) {
  ranges.erase(std::remove_if(ranges.begin(), ranges.end(), [](const auto& range) { return range.length == 0; }),
               ranges.end());
  std::sort(ranges.begin(), ranges.end(), [](const auto& lhs, const auto& rhs) { return lhs.offset < rhs.offset; });

  std::vector<ByteRange> merged;
  merged.reserve(ranges.size());
  for (const auto& range : ranges) {
    if (merged.empty()) {
      merged.emplace_back(range);
      continue;
    }

    auto& back = merged.back();
    if (range.offset > std::numeric_limits<uint64_t>::max() - range.length ||
        back.offset > std::numeric_limits<uint64_t>::max() - back.length) {
      merged.emplace_back(range);
      continue;
    }

    const auto range_end = range.offset + range.length;
    const auto back_end = back.offset + back.length;
    if (range.offset <= back_end) {
      back.length = std::max(back_end, range_end) - back.offset;
    } else {
      merged.emplace_back(range);
    }
  }
  return merged;
}

namespace {

// Used only when the caller does not know the footer size. This is the single
// speculative tail read; failure is returned without expanding the range.
constexpr uint64_t kDefaultFooterReadSize = 2 * 1024 * 1024;

static arrow::Result<uint64_t> ResolveInitialFooterBodySize(uint64_t file_size,
                                                            uint64_t footer_size,
                                                            uint64_t eof_size) {
  if (file_size == 0) {
    return 0;
  }

  if (footer_size != 0) {
    if (file_size < eof_size || footer_size > file_size - eof_size) {
      // A recorded footer size that cannot fit inside the file it describes is
      // the metadata contradicting itself -- nothing was read, so this says
      // nothing about the object's bytes. Left as a bare Invalid it reached
      // segcore as a generic 2044, and the size reconciliation on the way out
      // of Open() does not catch it either: that compares file_size, which is
      // correct here. DataCorrupted points at the thing that is wrong.
      return MakeExtendErrorMsg(
          ExtendStatusCode::DataCorrupted,
          fmt::format("Recorded vortex footer size cannot fit in the file it describes: file_size={}, footer_size={}, "
                      "eof_size={}. The object was never read, so this is the metadata that needs attention.",
                      file_size, footer_size, eof_size));
    }
    return footer_size;
  }

  const auto target_tail_size = std::min<uint64_t>(file_size, kDefaultFooterReadSize);
  return target_tail_size > eof_size ? target_tail_size - eof_size : 0;
}

static uint64_t ResolveTailReadSize(uint64_t file_size, uint64_t footer_body_size, uint64_t eof_size) {
  if (footer_body_size > std::numeric_limits<uint64_t>::max() - eof_size) {
    return file_size;
  }
  return std::min<uint64_t>(file_size, footer_body_size + eof_size);
}

}  // namespace

namespace internal {

arrow::Result<ByteRange> FlatSegmentByteRangeFromBytes(const std::vector<uint64_t>& bytes, uint64_t flat_segment_id) {
  if (bytes.size() != 2) {
    return arrow::Status::Invalid("Vortex bridge returned ", bytes.size(), " values for flat segment ", flat_segment_id,
                                  ", expected 2");
  }
  return ByteRange{bytes[0], bytes[1]};
}

arrow::Status ValidateVortexFieldLayoutEncoding(const std::vector<uint64_t>& raw_offsets,
                                                uint64_t rows,
                                                const std::string& field_name) {
  auto data_format_error = [&field_name](auto&&... message) {
    return MakeExtendErrorMsg(ExtendStatusCode::VortexDataFormat, std::forward<decltype(message)>(message)...,
                              " for field ", field_name);
  };

  if (raw_offsets.size() < 2) {
    return data_format_error("Vortex field layout units are malformed (", raw_offsets.size(),
                             " offsets, need at least 2)");
  }

  const auto granularity = raw_offsets[0];
  if (granularity != static_cast<uint64_t>(VortexPhysicalGranularity::kFlat) &&
      granularity != static_cast<uint64_t>(VortexPhysicalGranularity::kRowGroup)) {
    return data_format_error("Invalid vortex physical unit granularity ", granularity);
  }

  const auto total_units = raw_offsets[1];
  if (total_units == 0) {
    if (rows != 0) {
      return data_format_error("Non-empty vortex file has no physical units");
    }
    if (raw_offsets.size() != 2) {
      return data_format_error("Trailing words after an empty vortex field layout (", raw_offsets.size() - 2,
                               " extra)");
    }
    return arrow::Status::OK();
  }

  // Every unit consumes at least four words. Reject an impossible persisted
  // count before entering a potentially enormous loop.
  const auto available_words = raw_offsets.size() - 2;
  if (total_units > available_words / 4) {
    return data_format_error("Truncated vortex field layout units: declares ", total_units, " units in ",
                             available_words, " words");
  }

  size_t cursor = 2;
  for (uint64_t i = 0; i < total_units; ++i) {
    if (raw_offsets.size() - cursor < 4) {
      return data_format_error("Truncated vortex field layout unit ", i);
    }
    const auto row_offset = raw_offsets[cursor + 1];
    const auto row_count = raw_offsets[cursor + 2];
    if (row_offset > rows || row_count > rows - row_offset) {
      return data_format_error("Vortex field layout unit ", i, " has row range [", row_offset, ", ", row_offset, " + ",
                               row_count, ") outside file row count ", rows);
    }
    const auto num_segments = raw_offsets[cursor + 3];
    cursor += 4;
    if (num_segments > raw_offsets.size() - cursor) {
      return data_format_error("Truncated vortex segment list in unit ", i, ": declares ", num_segments,
                               " segments with ", raw_offsets.size() - cursor, " words remaining");
    }
    cursor += static_cast<size_t>(num_segments);
  }

  if (cursor != raw_offsets.size()) {
    return data_format_error("Trailing words after vortex field layout units (", raw_offsets.size() - cursor,
                             " extra)");
  }
  return arrow::Status::OK();
}

arrow::Status ClassifyVortexLayoutSegmentLookupFailure(const arrow::Status& failure, uint64_t flat_segment_id) {
  const auto context = fmt::format("Vortex layout names a segment the file does not have: id={}", flat_segment_id);
  if (ExtendStatusDetail::UnwrapStatus(failure) != nullptr || failure.IsUnknownError() ||
      arrow::internal::ErrnoFromStatus(failure) == ENOENT) {
    return MakeVortexErrorStatus(context, failure);
  }
  return MakeExtendErrorMsg(ExtendStatusCode::VortexDataFormat, context, " (", failure.message(), ")");
}

}  // namespace internal

namespace {

struct ParsedFieldUnit {
  uint64_t physical_unit_index = 0;
  uint64_t row_offset = 0;
  uint64_t row_count = 0;
  std::vector<VortexFlatSegmentRange> flat_segments;
};

struct ParsedFieldUnits {
  VortexPhysicalGranularity granularity = VortexPhysicalGranularity::kUnknown;
  std::vector<ParsedFieldUnit> units;
};

// Classified here rather than in the bridge, because THIS is the layer that
// knows where the segment id came from. Down in segment_bytes an out-of-range
// id is indistinguishable from a caller passing a bad number -- an API misuse,
// not a data-format failure. The ids reaching this function were read out of
// the file's own layout, so a range the file cannot satisfy is a format error.
static arrow::Result<ByteRange> ReadFlatSegmentByteRange(const VortexFile& vxfile, uint64_t flat_segment_id) {
  auto bytes_result = vxfile.SegmentBytes(flat_segment_id);
  if (!bytes_result.ok()) {
    return internal::ClassifyVortexLayoutSegmentLookupFailure(bytes_result.status(), flat_segment_id);
  }
  return internal::FlatSegmentByteRangeFromBytes(std::move(bytes_result).ValueOrDie(), flat_segment_id);
}

static arrow::Status LoadFlatSegments(const std::shared_ptr<arrow::io::RandomAccessFile>& input_file,
                                      const std::shared_ptr<VortexRangeFile>& range_file,
                                      const VortexFile& vxfile,
                                      const std::vector<uint64_t>& flat_segment_ids,
                                      const std::string& path,
                                      const std::string& purpose) {
  for (auto flat_segment_id : flat_segment_ids) {
    ARROW_ASSIGN_OR_RAISE(auto byte_range, ReadFlatSegmentByteRange(vxfile, flat_segment_id));
    ARROW_RETURN_NOT_OK(FillVortexRangeFile(input_file, range_file, byte_range.offset, byte_range.length));
  }
  LOG_STORAGE_INFO_ << fmt::format("[VortexFooterReader] loaded {} {} flat segments for {}", flat_segment_ids.size(),
                                   purpose, path);
  return arrow::Status::OK();
}

static arrow::Result<ParsedFieldUnits> ParseFieldUnits(const VortexFile& vxfile,
                                                       const std::shared_ptr<arrow::Schema>& file_schema,
                                                       uint64_t rows,
                                                       const std::string& field_name) {
  // A requested field name is a caller input. Check it before asking the Rust
  // bridge for persisted layout units: the bridge deliberately represents an
  // absent field as an empty layout, which is also the representation of a
  // malformed non-empty persisted layout. Only the schema can distinguish them.
  if (file_schema && !file_schema->GetFieldByName(field_name)) {
    return arrow::Status::KeyError(fmt::format("Vortex field not found: {}", field_name));
  }

  auto raw_offsets_result = vxfile.FieldLayoutUnits(field_name);
  if (!raw_offsets_result.ok()) {
    return MakeVortexErrorStatus(fmt::format("Failed to get vortex field layout units for {}", field_name),
                                 raw_offsets_result.status());
  }
  auto raw_offsets = std::move(raw_offsets_result).ValueOrDie();

  ARROW_RETURN_NOT_OK(internal::ValidateVortexFieldLayoutEncoding(raw_offsets, rows, field_name));

  const auto granularity_value = raw_offsets[0];
  VortexPhysicalGranularity granularity = VortexPhysicalGranularity::kUnknown;
  if (granularity_value == static_cast<uint64_t>(VortexPhysicalGranularity::kFlat)) {
    granularity = VortexPhysicalGranularity::kFlat;
  } else if (granularity_value == static_cast<uint64_t>(VortexPhysicalGranularity::kRowGroup)) {
    granularity = VortexPhysicalGranularity::kRowGroup;
  }

  const auto total_units = raw_offsets[1];
  if (total_units == 0) {
    return ParsedFieldUnits{granularity, {}};
  }

  std::vector<ParsedFieldUnit> units;
  units.reserve(total_units);

  size_t cursor = 2;
  for (uint64_t i = 0; i < total_units; ++i) {
    ParsedFieldUnit unit;
    unit.physical_unit_index = raw_offsets[cursor++];
    unit.row_offset = raw_offsets[cursor++];
    unit.row_count = raw_offsets[cursor++];
    const auto num_segments = raw_offsets[cursor++];

    unit.flat_segments.reserve(num_segments);
    for (uint64_t j = 0; j < num_segments; ++j) {
      const auto flat_segment_id = raw_offsets[cursor++];
      ARROW_ASSIGN_OR_RAISE(auto byte_range, ReadFlatSegmentByteRange(vxfile, flat_segment_id));
      unit.flat_segments.emplace_back(VortexFlatSegmentRange{flat_segment_id, byte_range});
    }
    units.emplace_back(std::move(unit));
  }
  return ParsedFieldUnits{granularity, std::move(units)};
}

}  // namespace

struct VortexFooterReader::Impl {
  arrow::Status ResolveFileSize(const std::shared_ptr<arrow::fs::FileSystem>& fs);
  arrow::Status CheckFileLongEnough() const;
  arrow::Status PrepareRangeFile();
  arrow::Status OpenSparseVortexFile();
  arrow::Status LoadFooter(const std::shared_ptr<arrow::io::RandomAccessFile>& input_file);
  arrow::Status LoadZoneMaps(const std::shared_ptr<arrow::io::RandomAccessFile>& input_file);
  arrow::Status LoadFileSchema();

  std::shared_ptr<FileSystemWrapper> fs_holder;
  std::shared_ptr<arrow::fs::FileSystem> sparse_fs;
  std::shared_ptr<VortexRangeFile> range_file;
  std::string path;
  std::string sparse_path;
  uint64_t file_size = 0;
  uint64_t footer_size = 0;
  uint64_t rows = 0;
  bool zonemap_loaded = false;
  std::shared_ptr<arrow::Schema> file_schema;
  std::unique_ptr<VortexFile> vxfile;
  mutable std::mutex field_layout_cache_mutex;
  std::unordered_map<std::string, VortexFieldLayout> field_layout_cache;
};

arrow::Status VortexFooterReader::Impl::ResolveFileSize(const std::shared_ptr<arrow::fs::FileSystem>& fs) {
  if (file_size != 0) {
    return arrow::Status::OK();
  }

  ARROW_ASSIGN_OR_RAISE(auto info, fs->GetFileInfo(path));
  if (info.type() == arrow::fs::FileType::NotFound) {
    return arrow::fs::internal::PathNotFound(path);
  }
  if (!info.IsFile()) {
    return arrow::Status::Invalid(fmt::format("Vortex file is not a regular file: {}", path));
  }
  if (info.size() < 0) {
    return arrow::Status::Invalid(fmt::format("Vortex file size is unavailable: {}", path));
  }
  file_size = static_cast<uint64_t>(info.size());
  return arrow::Status::OK();
}

arrow::Status VortexFooterReader::Impl::CheckFileLongEnough() const {
  // A vortex file cannot be shorter than its EOF trailer. Checked here rather
  // than left to the Rust side: this is the zero-length object a failed or
  // aborted write leaves behind -- a common malformed object shape -- and
  // vortex reports it as "Initial read must be at least EOF_SIZE". The intent
  // is unambiguous here and is not at the generic bridge boundary.
  //
  if (file_size < VortexEofSize()) {
    return MakeExtendErrorMsg(ExtendStatusCode::VortexDataFormat,
                              "Vortex file is too short to hold its trailer: ", path, " (", file_size,
                              " bytes, minimum ", VortexEofSize(), ")");
  }
  return arrow::Status::OK();
}

arrow::Status VortexFooterReader::Impl::PrepareRangeFile() {
  if (!sparse_fs) {
    return MakeExtendErrorMsg(ExtendStatusCode::InternalInvariantViolated, "VortexFooterReader requires a non-null range filesystem");
  }
  if (sparse_path.empty()) {
    return MakeExtendErrorMsg(ExtendStatusCode::InternalInvariantViolated, "VortexFooterReader requires a non-empty sparse path");
  }

  if (!range_file) {
    ARROW_ASSIGN_OR_RAISE(range_file, GetVortexRangeFile(sparse_fs, sparse_path));
  }
  if (!range_file) {
    return MakeExtendErrorMsg(ExtendStatusCode::InternalInvariantViolated, "VortexFooterReader requires a non-null range file");
  }
  range_file->Resize(file_size);
  fs_holder = std::make_shared<FileSystemWrapper>(sparse_fs);
  return arrow::Status::OK();
}

arrow::Status VortexFooterReader::Impl::OpenSparseVortexFile() {
  auto result =
      VortexFile::OpenUnique(reinterpret_cast<uint8_t*>(fs_holder.get()), sparse_path, file_size, footer_size);
  if (!result.ok()) {
    return MakeVortexErrorStatus(fmt::format("Failed to open vortex file {}", path), result.status());
  }
  vxfile = std::move(result).ValueOrDie();
  return arrow::Status::OK();
}

namespace internal {

arrow::Status CheckVortexFooterRange(const std::vector<uint64_t>& footer_range,
                                     uint64_t file_size,
                                     const std::string& path) {
  // Three ways the descriptor can contradict the file it describes: it is not a
  // {offset, length} pair at all, the offset starts past the end, or the length
  // runs off the end from that offset. The last is written as
  // `length > file_size - offset` rather than `offset + length > file_size`
  // because both are uint64 and the sum wraps.
  if (footer_range.size() != 2 || footer_range[0] > file_size || footer_range[1] > file_size - footer_range[0]) {
    return MakeExtendErrorMsg(ExtendStatusCode::VortexDataFormat,
                              fmt::format("Invalid vortex footer byte range for {}", path));
  }
  return arrow::Status::OK();
}

}  // namespace internal

arrow::Status VortexFooterReader::Impl::LoadFooter(const std::shared_ptr<arrow::io::RandomAccessFile>& input_file) {
  const auto eof_size = VortexEofSize();
  const auto supplied_footer_size = footer_size;
  ARROW_ASSIGN_OR_RAISE(const auto footer_body_size,
                        ResolveInitialFooterBodySize(file_size, supplied_footer_size, eof_size));
  const auto tail_read_size = ResolveTailReadSize(file_size, footer_body_size, eof_size);
  ARROW_RETURN_NOT_OK(FillVortexRangeFile(input_file, range_file, file_size - tail_read_size, tail_read_size));

  auto open_result =
      VortexFile::OpenUnique(reinterpret_cast<uint8_t*>(fs_holder.get()), sparse_path, file_size, footer_body_size);
  if (!open_result.ok()) {
    return MakeVortexErrorStatus(fmt::format("Failed to open vortex file {}", path), open_result.status());
  }
  vxfile = std::move(open_result).ValueOrDie();

  ARROW_ASSIGN_OR_RAISE(auto footer_range, vxfile->FooterByteRange(file_size));
  ARROW_RETURN_NOT_OK(internal::CheckVortexFooterRange(footer_range, file_size, path));
  const auto actual_footer_size = footer_range[1] > eof_size ? footer_range[1] - eof_size : 0;
  if (actual_footer_size > footer_body_size) {
    if (supplied_footer_size != 0) {
      return MakeExtendErrorMsg(ExtendStatusCode::DataCorrupted,
                                "Recorded vortex footer size is smaller than the actual footer");
    }
    return arrow::Status::IOError("Vortex footer exceeds the single-read window");
  }
  footer_size = actual_footer_size;
  rows = vxfile->RowCount();
  return arrow::Status::OK();
}

arrow::Status VortexFooterReader::Impl::LoadZoneMaps(const std::shared_ptr<arrow::io::RandomAccessFile>& input_file) {
  if (zonemap_loaded) {
    return arrow::Status::OK();
  }
  if (!vxfile) {
    return arrow::Status::Invalid("VortexFooterReader requires an opened footer before loading zonemaps");
  }

  auto zone_segment_ids = vxfile->ZoneMapSegmentIds();
  if (!zone_segment_ids.ok()) {
    return MakeVortexErrorStatus(fmt::format("Failed to load vortex zonemap segments {}", path),
                                 zone_segment_ids.status());
  }
  ARROW_RETURN_NOT_OK(
      LoadFlatSegments(input_file, range_file, *vxfile, zone_segment_ids.ValueOrDie(), path, "zonemap"));
  zonemap_loaded = true;
  return arrow::Status::OK();
}

arrow::Status VortexFooterReader::Impl::LoadFileSchema() {
  ArrowSchema c_schema;
  ARROW_RETURN_NOT_OK(
      MakeVortexErrorStatus(fmt::format("Failed to get vortex file schema {}", path), vxfile->GetFileSchema(c_schema)));
  ARROW_ASSIGN_OR_RAISE(file_schema, arrow::ImportSchema(&c_schema));
  return arrow::Status::OK();
}

VortexFooterReader::VortexFooterReader(std::shared_ptr<arrow::fs::FileSystem> sparse_fs,
                                       std::string sparse_path,
                                       std::string path,
                                       uint64_t file_size,
                                       uint64_t footer_size)
    : impl_(std::make_unique<Impl>()) {
  impl_->sparse_fs = std::move(sparse_fs);
  impl_->sparse_path = std::move(sparse_path);
  impl_->path = std::move(path);
  impl_->file_size = file_size;
  impl_->footer_size = footer_size;
}

VortexFooterReader::~VortexFooterReader() = default;

arrow::Status VortexFooterReader::Open(const std::shared_ptr<arrow::fs::FileSystem>& fs, bool load_zonemap) {
  if (impl_->vxfile) {
    return MakeExtendErrorMsg(ExtendStatusCode::InternalInvariantViolated,
                              "VortexFooterReader is already opened; create a new reader for another read mode");
  }
  if (!fs) {
    return MakeExtendErrorMsg(ExtendStatusCode::InternalInvariantViolated, "VortexFooterReader::Open requires a non-null filesystem");
  }

  ARROW_RETURN_NOT_OK(impl_->ResolveFileSize(fs));

  ARROW_RETURN_NOT_OK(impl_->CheckFileLongEnough());
  ARROW_RETURN_NOT_OK(impl_->PrepareRangeFile());
  ARROW_ASSIGN_OR_RAISE(auto input_file, fs->OpenInputFile(impl_->path));

  ARROW_RETURN_NOT_OK(impl_->LoadFooter(input_file));
  if (load_zonemap) {
    ARROW_RETURN_NOT_OK(impl_->LoadZoneMaps(input_file));
    // Vortex caches segments covered by its footer tail read. Reopen after
    // zonemap bytes are materialized so pruning cannot read sparse zero-fill
    // data from that internal cache.
    ARROW_RETURN_NOT_OK(impl_->OpenSparseVortexFile());
  }
  ARROW_RETURN_NOT_OK(impl_->LoadFileSchema());
  return arrow::Status::OK();
}

bool VortexFooterReader::opened() const { return impl_->vxfile != nullptr; }

uint64_t VortexFooterReader::rows() const { return impl_->rows; }

std::shared_ptr<arrow::Schema> VortexFooterReader::file_schema() const { return impl_->file_schema; }

const std::string& VortexFooterReader::path() const { return impl_->path; }

uint64_t VortexFooterReader::file_size() const { return impl_->file_size; }

uint64_t VortexFooterReader::footer_size() const { return impl_->footer_size; }

arrow::Result<VortexFieldLayout> VortexFooterReader::GetFieldLayout(const std::string& field_name) const {
  if (!impl_->vxfile) {
    return MakeExtendErrorMsg(ExtendStatusCode::InternalInvariantViolated, "VortexFooterReader is not opened");
  }
  {
    std::lock_guard<std::mutex> lock(impl_->field_layout_cache_mutex);
    auto cached = impl_->field_layout_cache.find(field_name);
    if (cached != impl_->field_layout_cache.end()) {
      return cached->second;
    }
  }

  ARROW_ASSIGN_OR_RAISE(auto parsed, ParseFieldUnits(*impl_->vxfile, impl_->file_schema, impl_->rows, field_name));

  VortexFieldLayout layout;
  layout.granularity = parsed.granularity;
  if (layout.granularity == VortexPhysicalGranularity::kFlat) {
    layout.flats.reserve(parsed.units.size());
    for (auto& unit : parsed.units) {
      layout.flats.emplace_back(VortexFlatUnit{
          .flat_id = unit.physical_unit_index,
          .row_offset = unit.row_offset,
          .row_count = unit.row_count,
          .flat_segments = std::move(unit.flat_segments),
      });
    }
  } else if (layout.granularity == VortexPhysicalGranularity::kRowGroup) {
    layout.row_groups.reserve(parsed.units.size());
    for (auto& unit : parsed.units) {
      layout.row_groups.emplace_back(VortexRowGroupUnit{
          .row_group_id = unit.physical_unit_index,
          .row_offset = unit.row_offset,
          .row_count = unit.row_count,
          .flat_segments = std::move(unit.flat_segments),
      });
    }
  } else {
    return arrow::Status::Invalid(fmt::format("Unsupported vortex field granularity for {}", field_name));
  }
  std::lock_guard<std::mutex> lock(impl_->field_layout_cache_mutex);
  auto [it, inserted] = impl_->field_layout_cache.emplace(field_name, std::move(layout));
  (void)inserted;
  return it->second;
}

arrow::Result<std::vector<uint64_t>> VortexFooterReader::PruneRowGroups(
    const std::string& predicate, const std::vector<uint64_t>& candidate_row_group_ids) const {
  if (!impl_->vxfile) {
    return MakeExtendErrorMsg(ExtendStatusCode::InternalInvariantViolated, "VortexFooterReader is not opened");
  }
  if (predicate.empty() || candidate_row_group_ids.empty()) {
    return candidate_row_group_ids;
  }
  if (!impl_->zonemap_loaded) {
    LOG_STORAGE_INFO_ << fmt::format("[VortexFooterReader] zonemap is not loaded for {}, skip row-group pruning",
                                     impl_->path);
    return candidate_row_group_ids;
  }

  auto result = impl_->vxfile->PruneRowGroups(predicate, candidate_row_group_ids);
  if (!result.ok()) {
    return MakeVortexErrorStatus("Failed to prune vortex row groups", result.status());
  }
  return std::move(result).ValueOrDie();
}

}  // namespace milvus_storage::vortex
