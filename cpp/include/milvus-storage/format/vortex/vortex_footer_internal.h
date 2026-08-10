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
/// are not the same thing, and a green test here does not prove a corrupt
/// vortex file in production reaches this code.
///
/// \param footer_range as reported by the file's own footer descriptor:
///        exactly two elements, {offset, length}
/// \param file_size the size the file actually has
/// \param path used only for the message
/// \return OK when the range fits inside the file; a VortexFileCorrupted status
///         otherwise. Corrupted rather than a bare Invalid because the bytes
///         were parsed and found to contradict the file -- and because an
///         untagged Invalid now lands on a generic StorageError, the coarse
///         fallback having stopped guessing at corruption.
arrow::Status CheckVortexFooterRange(const std::vector<uint64_t>& footer_range,
                                     uint64_t file_size,
                                     const std::string& path);

/// \brief Turn the two-element {offset, length} a vortex flat segment reports
/// into a ByteRange, rejecting only what is actually wrong.
///
/// A zero LENGTH is legal and must not be called corruption. Three independent
/// places already say so: vortex's own SegmentSpec carries a plain `u32 length`
/// with no non-zero constraint and its writer computes it as a sum of buffer
/// lengths with no guard (vortex-file/src/segments/writer.rs), `MergeByteRanges`
/// in the reader drops empty ranges as an ordinary case, and
/// `FillVortexRangeFile` short-circuits length == 0 as a no-op. This function
/// used to be the one dissenter, and it dissented with the most expensive verdict
/// available -- VortexFileCorrupted sends an operator to quarantine and rebuild a
/// file whose bytes were never examined. It also made the other two sites
/// unreachable: this is the only producer feeding them.
///
/// A wrong-sized vector IS an error, but it is OUR error, not the file's: the
/// Rust bridge's segment_bytes always returns exactly two elements, so anything
/// else is a broken contract on our side of the FFI. It gets a plain Invalid,
/// which the coarse fallback files as a generic internal storage failure, rather
/// than an accusation against the data.
///
/// Extracted here for the same reason CheckVortexFooterRange was: reaching it
/// end-to-end needs a vortex file the current writer does not emit.
arrow::Result<ByteRange> FlatSegmentByteRangeFromBytes(const std::vector<uint64_t>& bytes, uint64_t flat_segment_id);

/// \brief Decide whether a caller-supplied file size was ever true, before a
/// final verdict about the file's bytes is allowed to stand.
///
/// Every vortex read is anchored at the file size. When the caller supplied it
/// (from a manifest) and the operation failed with either an unclassified error
/// or a corruption claim, one stat settles whether the anchor itself was wrong:
/// on a mismatch the failure becomes ManifestCorrupted -- the bytes were never
/// judged, the metadata was -- and on a match (or when the stat cannot answer)
/// the original status stands. Failures already classified as anything other
/// than corruption (retryable, missing, config) pass through untouched, as does
/// ENOENT; an OK status or a supplied_size of 0 (the "stat it yourself"
/// sentinel) short-circuits, so the healthy path never pays for this.
///
/// Shared by VortexFooterReader::Open and VortexFormatReader's sync/async opens
/// so that every reader anchored on a manifest size gets the same correction.
arrow::Status ReconcileSuppliedVortexSize(const std::shared_ptr<arrow::fs::FileSystem>& fs,
                                          const std::string& path,
                                          uint64_t supplied_size,
                                          arrow::Status status);

}  // namespace milvus_storage::vortex::internal
