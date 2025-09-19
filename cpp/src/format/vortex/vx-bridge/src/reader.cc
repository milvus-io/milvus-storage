// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors


#include "reader_ffi.hpp"

namespace milvus_storage::vortex {
namespace dtype {

DType null() {
    return DType(ffi::dtype_null());
}

DType bool_(bool nullable) {
    return DType(ffi::dtype_bool(nullable));
}

DType primitive(PType ptype, bool nullable) {
    return DType(ffi::dtype_primitive(static_cast<ffi::PType>(ptype), nullable));
}

DType int8(bool nullable) {
    return primitive(PType::I8, nullable);
}

DType int16(bool nullable) {
    return primitive(PType::I16, nullable);
}

DType int32(bool nullable) {
    return primitive(PType::I32, nullable);
}

DType int64(bool nullable) {
    return primitive(PType::I64, nullable);
}

DType uint8(bool nullable) {
    return primitive(PType::U8, nullable);
}

DType uint16(bool nullable) {
    return primitive(PType::U16, nullable);
}

DType uint32(bool nullable) {
    return primitive(PType::U32, nullable);
}

DType uint64(bool nullable) {
    return primitive(PType::U64, nullable);
}

DType float16(bool nullable) {
    return primitive(PType::F16, nullable);
}

DType float32(bool nullable) {
    return primitive(PType::F32, nullable);
}

DType float64(bool nullable) {
    return primitive(PType::F64, nullable);
}

DType decimal(uint8_t precision, int8_t scale, bool nullable) {
    return DType(ffi::dtype_decimal(precision, scale, nullable));
}

DType utf8(bool nullable) {
    return DType(ffi::dtype_utf8(nullable));
}

DType binary(bool nullable) {
    return DType(ffi::dtype_binary(nullable));
}

DType from_arrow(struct ArrowSchema &schema, bool non_nullable) {
    try {
        return DType(ffi::from_arrow(reinterpret_cast<uint8_t *>(&schema), non_nullable));
    } catch (const rust::cxxbridge1::Error &e) {
        throw VortexException(e.what());
    }
}
// Methods
std::string DType::ToString() const {
    auto rust_str = impl_->to_string();
    return std::string(rust_str.data(), rust_str.length());
}
} // namespace dtype

namespace expr {

Expr literal(Scalar scalar) {
    return Expr(ffi::literal(std::move(scalar).IntoImpl()));
}

Expr root() {
    return Expr(ffi::root());
}

Expr column(std::string_view name) {
    return Expr(ffi::column(rust::String(name.data(), name.length())));
}

Expr get_item(std::string_view field, Expr child) {
    return Expr(ffi::get_item(rust::String(field.data(), field.length()), std::move(child).IntoImpl()));
}

Expr not_(Expr child) {
    return Expr(ffi::not_(std::move(child).IntoImpl()));
}

Expr is_null(Expr child) {
    return Expr(ffi::is_null(std::move(child).IntoImpl()));
}

// Macro to define binary operator functions
#define DEFINE_BINARY_OP(name)                                                                               \
    Expr name(Expr lhs, Expr rhs) {                                                                          \
        return Expr(ffi::name(std::move(lhs).IntoImpl(), std::move(rhs).IntoImpl()));                        \
    }

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

Expr select(const std::vector<std::string_view> &fields, Expr child) {
    ::rust::Vec<::rust::String> rs_fields;
    for (auto f : fields) {
        rs_fields.emplace_back(f.data(), f.length());
    }
    return Expr(ffi::select(rs_fields, std::move(child).IntoImpl()));
}

} // namespace expr

namespace scalar {

Scalar bool_(bool value) {
    return Scalar(ffi::bool_scalar_new(value));
}

Scalar int8(int8_t value) {
    return Scalar(ffi::i8_scalar_new(value));
}

Scalar int16(int16_t value) {
    return Scalar(ffi::i16_scalar_new(value));
}

Scalar int32(int32_t value) {
    return Scalar(ffi::i32_scalar_new(value));
}

Scalar int64(int64_t value) {
    return Scalar(ffi::i64_scalar_new(value));
}

Scalar uint8(uint8_t value) {
    return Scalar(ffi::u8_scalar_new(value));
}

Scalar uint16(uint16_t value) {
    return Scalar(ffi::u16_scalar_new(value));
}

Scalar uint32(uint32_t value) {
    return Scalar(ffi::u32_scalar_new(value));
}

Scalar uint64(uint64_t value) {
    return Scalar(ffi::u64_scalar_new(value));
}

Scalar float32(float value) {
    return Scalar(ffi::f32_scalar_new(value));
}

Scalar float64(double value) {
    return Scalar(ffi::f64_scalar_new(value));
}

Scalar string(std::string_view value) {
    return Scalar(ffi::string_scalar_new(rust::Str(value.data(), value.length())));
}

Scalar binary(const uint8_t *data, size_t length) {
    return Scalar(ffi::binary_scalar_new(rust::Slice<const uint8_t>(data, length)));
}

Scalar cast(Scalar scalar, DType dtype) {
    return Scalar(std::move(scalar).IntoImpl()->cast_scalar(*std::move(dtype).GetImpl()));
}

} // namespace scalar

ObjectStoreWrapper2 ObjectStoreWrapper2::OpenObjectStore(const std::string &ostype,
            const std::string &endpoint,
            const std::string &access_key_id,
            const std::string &secret_access_key,
            const std::string &region,
            const std::string &bucket_name)
{
    try {
        return ObjectStoreWrapper2(ffi::open_object_store(ostype, endpoint, access_key_id, 
            secret_access_key, region, bucket_name));
    } catch (const rust::cxxbridge1::Error &e) {
        throw VortexException(e.what());
    }
}

VortexFile VortexFile::Open(const ObjectStoreWrapper2 &obs, const std::string &path) {
    try {
        return VortexFile(ffi::open_file(obs.impl_, path));
    } catch (const rust::cxxbridge1::Error &e) {
        throw VortexException(e.what());
    }
}

uint64_t VortexFile::RowCount() const {
    return impl_->row_count();
}

ScanBuilder VortexFile::CreateScanBuilder() const {
    return ScanBuilder(impl_->scan_builder());
}

ScanBuilder &ScanBuilder::WithFilter(expr::Expr &&expr) & {
    impl_->with_filter(std::move(expr).IntoImpl());
    return *this;
}
ScanBuilder &ScanBuilder::WithFilter(const expr::Expr &expr) & {
    impl_->with_filter_ref(expr.Impl());
    return *this;
}
ScanBuilder &&ScanBuilder::WithFilter(expr::Expr &&expr) && {
    impl_->with_filter(std::move(expr).IntoImpl());
    return std::move(*this);
}
ScanBuilder &&ScanBuilder::WithFilter(const expr::Expr &expr) && {
    impl_->with_filter_ref(expr.Impl());
    return std::move(*this);
}

ScanBuilder &ScanBuilder::WithProjection(expr::Expr &&expr) & {
    impl_->with_projection(std::move(expr).IntoImpl());
    return *this;
}
ScanBuilder &ScanBuilder::WithProjection(const expr::Expr &expr) & {
    impl_->with_projection_ref(expr.Impl());
    return *this;
}
ScanBuilder &&ScanBuilder::WithProjection(expr::Expr &&expr) && {
    impl_->with_projection(std::move(expr).IntoImpl());
    return std::move(*this);
}
ScanBuilder &&ScanBuilder::WithProjection(const expr::Expr &expr) && {
    impl_->with_projection_ref(expr.Impl());
    return std::move(*this);
}

ScanBuilder &ScanBuilder::WithRowRange(uint64_t row_range_start, uint64_t row_range_end) & {
    impl_->with_row_range(row_range_start, row_range_end);
    return *this;
}
ScanBuilder &&ScanBuilder::WithRowRange(uint64_t row_range_start, uint64_t row_range_end) && {
    impl_->with_row_range(row_range_start, row_range_end);
    return std::move(*this);
}

ScanBuilder &ScanBuilder::WithLimit(uint64_t limit) & {
    impl_->with_limit(limit);
    return *this;
}

ScanBuilder &&ScanBuilder::WithLimit(uint64_t limit) && {
    impl_->with_limit(limit);
    return std::move(*this);
}

ScanBuilder &ScanBuilder::WithIncludeByIndex(const uint64_t *indices, std::size_t size) & {
    impl_->with_include_by_index(rust::Slice<const uint64_t>(indices, size));
    return *this;
}

ScanBuilder &&ScanBuilder::WithIncludeByIndex(const uint64_t *indices, std::size_t size) && {
    impl_->with_include_by_index(rust::Slice<const uint64_t>(indices, size));
    return std::move(*this);
}

ScanBuilder &ScanBuilder::WithOutputSchema(ArrowSchema &output_schema) & {
    try {
        impl_->with_output_schema(reinterpret_cast<uint8_t *>(&output_schema));
    } catch (const rust::cxxbridge1::Error &e) {
        throw VortexException(e.what());
    }
    return *this;
}

ScanBuilder &&ScanBuilder::WithOutputSchema(ArrowSchema &output_schema) && {
    try {
        impl_->with_output_schema(reinterpret_cast<uint8_t *>(&output_schema));
    } catch (const rust::cxxbridge1::Error &e) {
        throw VortexException(e.what());
    }
    return std::move(*this);
}

ArrowArrayStream ScanBuilder::IntoStream() && {
    try {
        ArrowArrayStream stream;
        ffi::scan_builder_into_stream(std::move(impl_), reinterpret_cast<uint8_t *>(&stream));
        return stream;
    } catch (const rust::cxxbridge1::Error &e) {
        throw VortexException(e.what());
    }
}

StreamDriver ScanBuilder::IntoStreamDriver() && {
    try {
        rust::Box<ffi::ThreadsafeCloneableReader> reader =
            ffi::scan_builder_into_threadsafe_cloneable_reader(std::move(impl_));
        return StreamDriver(std::move(reader));
    } catch (const rust::cxxbridge1::Error &e) {
        throw VortexException(e.what());
    }
}

ArrowArrayStream StreamDriver::CreateArrayStream() const {
    ArrowArrayStream stream;
    impl_->clone_a_stream(reinterpret_cast<uint8_t *>(&stream));
    return stream;
}

} // namespace milvus_storage::vortex
