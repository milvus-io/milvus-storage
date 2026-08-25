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

#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <arrow/filesystem/filesystem.h>
#include <arrow/result.h>
#include <arrow/status.h>

#include "milvus-storage/format/vortex/vortex_types.h"

namespace milvus_storage::vortex::internal {

/// \brief Check a footer byte range against the file it came from.
///
/// Extracted from VortexFooterReader::Impl::LoadFooter so the rule can be
/// tested. It could not be reached from a test where it lived: the range and
/// the bound are both derived from the same `file_size`, and the range is only
/// consulted after `VortexFile::OpenUnique` has already accepted that same
/// `file_size` -- so tampering with it in either direction moves the failure
/// earlier (inflate and the tail read runs past EOF into sparse zeroes, deflate
/// and it lands mid-file; either way no EOF trailer parses and OpenUnique
/// rejects the file first). Reaching this guard end-to-end needs a hand-built
/// vortex file whose EOF trailer parses but whose footer descriptor lies, and
/// there is no such fixture in the tree.
///
/// So this tests the rule, not the path to it. Stated plainly because the two
/// are not the same thing, and a green test here does not prove a malformed
/// Vortex file in production reaches this code.
///
/// \param footer_range as reported by the file's own footer descriptor:
///        exactly two elements, {offset, length}
/// \param file_size the size the file actually has
/// \param path used only for the message
/// \return OK when the range fits inside the file; a VortexDataFormat status
///         otherwise.
arrow::Status CheckVortexFooterRange(const std::vector<uint64_t>& footer_range,
                                     uint64_t file_size,
                                     const std::string& path);

/// Convert the bridge's {offset, length} result to a byte range. An empty
/// segment is legal; a result with any other arity is an internal bridge error.
arrow::Result<ByteRange> FlatSegmentByteRangeFromBytes(const std::vector<uint64_t>& bytes, uint64_t flat_segment_id);

/// \brief Validate the compact field-layout encoding read from a Vortex file.
///
/// The vector is persisted layout data, not a caller argument. It contains
/// `{granularity, unit_count}` followed by `unit_count` records of
/// `{unit_id, row_offset, row_count, segment_count, segment_ids...}`. Any
/// unsupported granularity, impossible unit count, truncation, or trailing word
/// means the encoded layout is invalid and therefore returns VortexDataFormat.
arrow::Status ValidateVortexFieldLayoutEncoding(const std::vector<uint64_t>& raw_offsets,
                                                uint64_t rows,
                                                const std::string& field_name);

/// Classify a segment lookup whose id came from the persisted Vortex layout.
/// Ordinary lookup failures mean the layout contradicts the file and are
/// DataFormat; an already-classified failure or caught panic keeps its original
/// classification instead of being relabelled as DataFormat.
arrow::Status ClassifyVortexLayoutSegmentLookupFailure(const arrow::Status& failure, uint64_t flat_segment_id);

}  // namespace milvus_storage::vortex::internal
