// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#include "vortex_bridge.h"

#include <charconv>
#include <optional>
#include <string>
#include <string_view>

#include "milvus-storage/common/extend_status.h"

namespace milvus_storage::vortex {
namespace {

constexpr std::string_view kVortexFfiErrCodeMarker = "__LOON_VORTEX_FFI_ERRCODE__=";

std::string StripBridgeMarker(std::string_view error, size_t marker_pos, size_t code_end) {
  auto message_start = code_end;
  if (message_start < error.size() && error[message_start] == ';') {
    ++message_start;
  }
  if (message_start < error.size() && error[message_start] == ' ') {
    ++message_start;
  }

  std::string message;
  message.reserve(error.size());
  message.append(error.substr(0, marker_pos));
  message.append(error.substr(message_start));
  if (message.empty()) {
    return "Unknown Vortex error";
  }
  return message;
}

struct ParsedVortexBridgeError {
  std::string message;
  std::optional<int> ffi_err_code;
};

ParsedVortexBridgeError ParseVortexBridgeError(std::string_view error) {
  auto marker_pos = error.find(kVortexFfiErrCodeMarker);
  if (marker_pos == std::string_view::npos) {
    return {std::string(error), std::nullopt};
  }

  auto code_start = marker_pos + kVortexFfiErrCodeMarker.size();
  auto code_end = code_start;
  while (code_end < error.size() && error[code_end] >= '0' && error[code_end] <= '9') {
    ++code_end;
  }
  if (code_end == code_start) {
    return {std::string(error), std::nullopt};
  }

  int ffi_err_code = 0;
  auto parse_result = std::from_chars(error.data() + code_start, error.data() + code_end, ffi_err_code);
  if (parse_result.ec != std::errc()) {
    return {std::string(error), std::nullopt};
  }

  return {StripBridgeMarker(error, marker_pos, code_end), ffi_err_code};
}

std::string JoinContextAndMessage(std::string_view context, std::string_view message) {
  if (context.empty()) {
    return std::string(message);
  }
  if (message.empty()) {
    return std::string(context);
  }
  std::string result;
  result.reserve(context.size() + 2 + message.size());
  result.append(context);
  result.append(": ");
  result.append(message);
  return result;
}

arrow::Status MakeExtendErrorWithContext(std::string_view context, const arrow::Status& status) {
  auto detail = ExtendStatusDetail::UnwrapStatus(status);
  auto full_message = JoinContextAndMessage(context, status.message());
  return MakeExtendError(detail->code(), full_message, full_message);
}

template <typename T, typename Fn>
arrow::Result<T> CatchRustResult(Fn&& fn) {
  try {
    return fn();
  } catch (const rust::cxxbridge1::Error& e) {
    return MakeVortexBridgeErrorStatus(e.what());
  }
}

template <typename Fn>
arrow::Status CatchRustStatus(Fn&& fn) {
  try {
    fn();
    return arrow::Status::OK();
  } catch (const rust::cxxbridge1::Error& e) {
    return MakeVortexBridgeErrorStatus(e.what());
  }
}

}  // namespace

arrow::Status MakeVortexBridgeErrorStatus(std::string_view message) {
  auto parsed = ParseVortexBridgeError(message);
  if (parsed.ffi_err_code.has_value()) {
    if (auto code = ExtendStatusCodeFromInt(*parsed.ffi_err_code); code.has_value()) {
      return MakeExtendError(*code, parsed.message, parsed.message);
    }
  }
  return arrow::Status::IOError(parsed.message);
}

arrow::Status MakeVortexErrorStatus(std::string_view context, std::string_view message) {
  return MakeVortexErrorStatus(context, MakeVortexBridgeErrorStatus(message));
}

arrow::Status MakeVortexErrorStatus(std::string_view context, const arrow::Status& status) {
  if (status.ok()) {
    return arrow::Status::OK();
  }
  if (ExtendStatusDetail::UnwrapStatus(status)) {
    return MakeExtendErrorWithContext(context, status);
  }
  auto message = status.message();
  auto parsed_status = MakeVortexBridgeErrorStatus(message);
  if (ExtendStatusDetail::UnwrapStatus(parsed_status)) {
    return MakeExtendErrorWithContext(context, parsed_status);
  }
  return arrow::Status::IOError(JoinContextAndMessage(context, parsed_status.message()));
}

uint64_t VortexEofSize() { return ffi::vortex_eof_size(); }

namespace dtype {

DType null() { return DType(ffi::dtype_null()); }

DType bool_(bool nullable) { return DType(ffi::dtype_bool(nullable)); }

DType primitive(PType ptype, bool nullable) {
  return DType(ffi::dtype_primitive(static_cast<ffi::PType>(ptype), nullable));
}

DType int8(bool nullable) { return primitive(PType::I8, nullable); }

DType int16(bool nullable) { return primitive(PType::I16, nullable); }

DType int32(bool nullable) { return primitive(PType::I32, nullable); }

DType int64(bool nullable) { return primitive(PType::I64, nullable); }

DType uint8(bool nullable) { return primitive(PType::U8, nullable); }

DType uint16(bool nullable) { return primitive(PType::U16, nullable); }

DType uint32(bool nullable) { return primitive(PType::U32, nullable); }

DType uint64(bool nullable) { return primitive(PType::U64, nullable); }

DType float16(bool nullable) { return primitive(PType::F16, nullable); }

DType float32(bool nullable) { return primitive(PType::F32, nullable); }

DType float64(bool nullable) { return primitive(PType::F64, nullable); }

DType decimal(uint8_t precision, int8_t scale, bool nullable) {
  return DType(ffi::dtype_decimal(precision, scale, nullable));
}

DType utf8(bool nullable) { return DType(ffi::dtype_utf8(nullable)); }

DType binary(bool nullable) { return DType(ffi::dtype_binary(nullable)); }

arrow::Result<DType> from_arrow(struct ArrowSchema& schema, bool non_nullable) {
  return CatchRustResult<DType>(
      [&]() { return DType(ffi::from_arrow(reinterpret_cast<uint8_t*>(&schema), non_nullable)); });
}
// Methods
std::string DType::ToString() const {
  auto rust_str = impl_->to_string();
  return {rust_str.data(), rust_str.length()};
}
}  // namespace dtype

namespace expr {

Expr literal(Scalar scalar) { return Expr(ffi::literal(std::move(scalar).IntoImpl())); }

Expr root() { return Expr(ffi::root()); }

Expr column(std::string_view name) { return Expr(ffi::column(rust::String(name.data(), name.length()))); }

Expr get_item(std::string_view field, Expr child) {
  return Expr(ffi::get_item(rust::String(field.data(), field.length()), std::move(child).IntoImpl()));
}

Expr not_(Expr child) { return Expr(ffi::not_(std::move(child).IntoImpl())); }

Expr is_null(Expr child) { return Expr(ffi::is_null(std::move(child).IntoImpl())); }

// Macro to define binary operator functions
#define DEFINE_BINARY_OP(name) \
  Expr name(Expr lhs, Expr rhs) { return Expr(ffi::name(std::move(lhs).IntoImpl(), std::move(rhs).IntoImpl())); }

DEFINE_BINARY_OP(eq)
DEFINE_BINARY_OP(not_eq_)
DEFINE_BINARY_OP(gt)
DEFINE_BINARY_OP(gt_eq)
DEFINE_BINARY_OP(lt)
DEFINE_BINARY_OP(lt_eq)
DEFINE_BINARY_OP(and_)
DEFINE_BINARY_OP(or_)
DEFINE_BINARY_OP(checked_add)

#undef DEFINE_BINARY_OP

Expr select(const std::vector<std::string_view>& fields, Expr child) {
  ::rust::Vec<::rust::String> rs_fields;
  for (auto f : fields) {
    rs_fields.emplace_back(f.data(), f.length());
  }
  return Expr(ffi::select(rs_fields, std::move(child).IntoImpl()));
}

arrow::Result<std::optional<Expr>> parse_predicate(const std::string& predicate,
                                                   const std::vector<PredicateColumn>& schema) {
  rust::Vec<rust::String> names;
  rust::Vec<uint8_t> tags;
  for (const auto& c : schema) {
    names.push_back(rust::String(c.name));
    tags.push_back(c.type_tag);
  }
  ARROW_ASSIGN_OR_RAISE(auto parsed, CatchRustResult<rust::Box<ffi::ParsedPredicate>>([&]() {
                          return ffi::parse_predicate_string(rust::Str(predicate.data(), predicate.length()),
                                                             std::move(names), std::move(tags));
                        }));
  if (!parsed->has_filter()) {
    return std::nullopt;
  }
  return Expr(parsed->take_filter());
}

}  // namespace expr

namespace scalar {

Scalar bool_(bool value) { return Scalar(ffi::bool_scalar_new(value)); }

Scalar int8(int8_t value) { return Scalar(ffi::i8_scalar_new(value)); }

Scalar int16(int16_t value) { return Scalar(ffi::i16_scalar_new(value)); }

Scalar int32(int32_t value) { return Scalar(ffi::i32_scalar_new(value)); }

Scalar int64(int64_t value) { return Scalar(ffi::i64_scalar_new(value)); }

Scalar uint8(uint8_t value) { return Scalar(ffi::u8_scalar_new(value)); }

Scalar uint16(uint16_t value) { return Scalar(ffi::u16_scalar_new(value)); }

Scalar uint32(uint32_t value) { return Scalar(ffi::u32_scalar_new(value)); }

Scalar uint64(uint64_t value) { return Scalar(ffi::u64_scalar_new(value)); }

Scalar float32(float value) { return Scalar(ffi::f32_scalar_new(value)); }

Scalar float64(double value) { return Scalar(ffi::f64_scalar_new(value)); }

Scalar string(std::string_view value) {
  return Scalar(ffi::string_scalar_new(rust::Str(value.data(), value.length())));
}

Scalar binary(const uint8_t* data, size_t length) {
  return Scalar(ffi::binary_scalar_new(rust::Slice<const uint8_t>(data, length)));
}

arrow::Result<Scalar> cast(Scalar scalar, DType dtype) {
  return CatchRustResult<Scalar>(
      [&]() { return Scalar(std::move(scalar).IntoImpl()->cast_scalar(*std::move(dtype).GetImpl())); });
}

}  // namespace scalar

arrow::Result<VortexWriter> VortexWriter::Open(
    uint8_t* fs_rawptr, const std::string& path, bool enable_stats, uint32_t format_version, uint64_t row_group_size) {
  return CatchRustResult<VortexWriter>([&]() {
    return VortexWriter(ffi::open_writer(fs_rawptr, rust::Str(path.data(), path.length()), enable_stats, format_version,
                                         row_group_size));
  });
}

arrow::Status VortexWriter::Write(ArrowSchema& in_schema, ArrowArray& in_array) {
  return CatchRustStatus(
      [&]() { impl_->write(reinterpret_cast<uint8_t*>(&in_schema), reinterpret_cast<uint8_t*>(&in_array)); });
}

arrow::Status VortexWriter::Flush() {
  return CatchRustStatus([&]() { impl_->flush(); });
}

arrow::Result<ffi::VortexWriteSummary> VortexWriter::Close() {
  return CatchRustResult<ffi::VortexWriteSummary>([&]() { return impl_->close(); });
}

arrow::Result<VortexFile> VortexFile::Open(uint8_t* fs_rawptr,
                                           const std::string& path,
                                           uint64_t file_size,
                                           uint64_t footer_size) {
  return CatchRustResult<VortexFile>([&]() {
    return VortexFile(ffi::open_file(fs_rawptr, rust::Str(path.data(), path.length()), file_size, footer_size));
  });
}

arrow::Result<std::unique_ptr<VortexFile>> VortexFile::OpenUnique(uint8_t* fs_rawptr,
                                                                  const std::string& path,
                                                                  uint64_t file_size,
                                                                  uint64_t footer_size) {
  return CatchRustResult<std::unique_ptr<VortexFile>>([&]() {
    return std::unique_ptr<VortexFile>(
        new VortexFile(ffi::open_file(fs_rawptr, rust::Str(path.data(), path.length()), file_size, footer_size)));
  });
}

uint64_t VortexFile::RowCount() const { return impl_->row_count(); }

arrow::Status VortexFile::GetFileSchema(ArrowSchema& out_schema) const {
  return CatchRustStatus([&]() { impl_->get_schema(reinterpret_cast<uint8_t*>(&out_schema)); });
}

arrow::Result<ScanBuilder> VortexFile::CreateScanBuilder(ffi::CoalescingWindow coalescing_window) const {
  return CatchRustResult<ScanBuilder>([&]() { return ScanBuilder(impl_->scan_builder(coalescing_window)); });
}

arrow::Result<ScanBuilder> VortexFile::CreateScanBuilderWithSchema(ArrowSchema& in_schema) const {
  return CatchRustResult<ScanBuilder>(
      [&]() { return ScanBuilder(impl_->scan_builder_with_schema(reinterpret_cast<uint8_t*>(&in_schema))); });
}

arrow::Result<std::vector<uint64_t>> VortexFile::Splits() const {
  return CatchRustResult<std::vector<uint64_t>>([&]() {
    ::rust::Vec<::rust::u64> rs_splits = impl_->splits();

    return std::vector<uint64_t>(rs_splits.begin(), rs_splits.end());
  });
}

std::vector<uint64_t> VortexFile::GetUncompressedSizes() const {
  ::rust::Vec<::rust::u64> rs_sizes = impl_->uncompressed_sizes();

  return {rs_sizes.begin(), rs_sizes.end()};
}

std::string VortexFile::RootLayoutEncoding() const {
  auto rust_str = impl_->root_layout_encoding();
  return {rust_str.data(), rust_str.length()};
}

arrow::Result<uint64_t> VortexFile::RowGroupZoneMapCount() const {
  return CatchRustResult<uint64_t>([&]() { return impl_->row_group_zone_map_count(); });
}

arrow::Result<bool> VortexFile::RowGroupZoneMapDataBeforeZones() const {
  return CatchRustResult<bool>([&]() { return impl_->row_group_zone_map_data_before_zones(); });
}

arrow::Result<std::vector<uint64_t>> VortexFile::ZoneMapSegmentIds() const {
  return CatchRustResult<std::vector<uint64_t>>([&]() {
    ::rust::Vec<::rust::u64> rs = impl_->zone_map_segment_ids();
    return std::vector<uint64_t>(rs.begin(), rs.end());
  });
}

arrow::Result<std::vector<uint64_t>> VortexFile::FooterByteRange(uint64_t file_size) const {
  return CatchRustResult<std::vector<uint64_t>>([&]() {
    ::rust::Vec<::rust::u64> rs = impl_->footer_byte_range(file_size);
    return std::vector<uint64_t>(rs.begin(), rs.end());
  });
}

arrow::Result<std::vector<uint64_t>> VortexFile::SegmentBytes(uint64_t flat_segment_id) const {
  return CatchRustResult<std::vector<uint64_t>>([&]() {
    ::rust::Vec<::rust::u64> rs = impl_->segment_bytes(flat_segment_id);
    return std::vector<uint64_t>(rs.begin(), rs.end());
  });
}

arrow::Result<std::vector<uint64_t>> VortexFile::FieldLayoutUnits(const std::string& field_name) const {
  return CatchRustResult<std::vector<uint64_t>>([&]() {
    ::rust::Vec<::rust::u64> rs_units = impl_->field_layout_units(::rust::Str(field_name.data(), field_name.size()));
    return std::vector<uint64_t>(rs_units.begin(), rs_units.end());
  });
}

arrow::Result<std::vector<uint64_t>> VortexFile::PruneRowGroups(
    const std::string& predicate, const std::vector<uint64_t>& candidate_row_group_ids) const {
  return CatchRustResult<std::vector<uint64_t>>([&]() {
    ::rust::Vec<::rust::u64> rs_ids = impl_->prune_row_groups(
        ::rust::Str(predicate.data(), predicate.size()),
        rust::Slice<const uint64_t>(candidate_row_group_ids.data(), candidate_row_group_ids.size()));
    return std::vector<uint64_t>(rs_ids.begin(), rs_ids.end());
  });
}

ScanBuilder& ScanBuilder::WithFilter(expr::Expr&& expr) & {
  impl_->with_filter(std::move(expr).IntoImpl());
  return *this;
}
ScanBuilder& ScanBuilder::WithFilter(const expr::Expr& expr) & {
  impl_->with_filter_ref(expr.Impl());
  return *this;
}
ScanBuilder&& ScanBuilder::WithFilter(expr::Expr&& expr) && {
  impl_->with_filter(std::move(expr).IntoImpl());
  return std::move(*this);
}
ScanBuilder&& ScanBuilder::WithFilter(const expr::Expr& expr) && {
  impl_->with_filter_ref(expr.Impl());
  return std::move(*this);
}

ScanBuilder& ScanBuilder::WithProjection(expr::Expr&& expr) & {
  impl_->with_projection(std::move(expr).IntoImpl());
  return *this;
}
ScanBuilder& ScanBuilder::WithProjection(const expr::Expr& expr) & {
  impl_->with_projection_ref(expr.Impl());
  return *this;
}
ScanBuilder&& ScanBuilder::WithProjection(expr::Expr&& expr) && {
  impl_->with_projection(std::move(expr).IntoImpl());
  return std::move(*this);
}
ScanBuilder&& ScanBuilder::WithProjection(const expr::Expr& expr) && {
  impl_->with_projection_ref(expr.Impl());
  return std::move(*this);
}

ScanBuilder& ScanBuilder::WithRowIndicesProjection(const std::string& field_name) & {
  impl_->with_row_indices_projection(rust::Str(field_name.data(), field_name.length()));
  return *this;
}

ScanBuilder&& ScanBuilder::WithRowIndicesProjection(const std::string& field_name) && {
  impl_->with_row_indices_projection(rust::Str(field_name.data(), field_name.length()));
  return std::move(*this);
}

ScanBuilder& ScanBuilder::WithSplitRowIndices(bool split_row_indices) & {
  impl_->with_split_row_indices(split_row_indices);
  return *this;
}
ScanBuilder&& ScanBuilder::WithSplitRowIndices(bool split_row_indices) && {
  impl_->with_split_row_indices(split_row_indices);
  return std::move(*this);
}

ScanBuilder& ScanBuilder::WithRowRange(uint64_t row_range_start, uint64_t row_range_end) & {
  impl_->with_row_range(row_range_start, row_range_end);
  return *this;
}
ScanBuilder&& ScanBuilder::WithRowRange(uint64_t row_range_start, uint64_t row_range_end) && {
  impl_->with_row_range(row_range_start, row_range_end);
  return std::move(*this);
}

ScanBuilder& ScanBuilder::WithRowRanges(const uint64_t* starts, const uint64_t* ends, std::size_t size) & {
  impl_->with_row_ranges(rust::Slice<const uint64_t>(starts, size), rust::Slice<const uint64_t>(ends, size));
  return *this;
}

ScanBuilder&& ScanBuilder::WithRowRanges(const uint64_t* starts, const uint64_t* ends, std::size_t size) && {
  impl_->with_row_ranges(rust::Slice<const uint64_t>(starts, size), rust::Slice<const uint64_t>(ends, size));
  return std::move(*this);
}

ScanBuilder& ScanBuilder::WithLimit(uint64_t limit) & {
  impl_->with_limit(limit);
  return *this;
}

ScanBuilder&& ScanBuilder::WithLimit(uint64_t limit) && {
  impl_->with_limit(limit);
  return std::move(*this);
}

ScanBuilder& ScanBuilder::WithIncludeByIndex(const uint64_t* indices, std::size_t size) & {
  impl_->with_include_by_index(rust::Slice<const uint64_t>(indices, size));
  return *this;
}

ScanBuilder&& ScanBuilder::WithIncludeByIndex(const uint64_t* indices, std::size_t size) && {
  impl_->with_include_by_index(rust::Slice<const uint64_t>(indices, size));
  return std::move(*this);
}

arrow::Status ScanBuilder::WithOutputSchema(ArrowSchema& output_schema) & {
  return CatchRustStatus([&]() { impl_->with_output_schema(reinterpret_cast<uint8_t*>(&output_schema)); });
}

arrow::Status ScanBuilder::WithOutputSchema(ArrowSchema& output_schema) && {
  return CatchRustStatus([&]() { impl_->with_output_schema(reinterpret_cast<uint8_t*>(&output_schema)); });
}

arrow::Result<ArrowArrayStream> ScanBuilder::IntoStream() && {
  return CatchRustResult<ArrowArrayStream>([&]() {
    ArrowArrayStream stream;
    ffi::scan_builder_into_stream(std::move(impl_), reinterpret_cast<uint8_t*>(&stream));
    return stream;
  });
}

}  // namespace milvus_storage::vortex
