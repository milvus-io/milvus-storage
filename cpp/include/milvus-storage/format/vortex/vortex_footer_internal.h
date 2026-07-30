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
#include <string>
#include <vector>

#include <arrow/status.h>

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
/// \return OK when the range fits inside the file; a PackedFileCorrupted status
///         otherwise. Corrupted rather than a bare Invalid because the bytes
///         were parsed and found to contradict the file -- and because an
///         untagged Invalid now lands on a generic StorageError, the coarse
///         fallback having stopped guessing at corruption.
arrow::Status CheckVortexFooterRange(const std::vector<uint64_t>& footer_range,
                                     uint64_t file_size,
                                     const std::string& path);

}  // namespace milvus_storage::vortex::internal
