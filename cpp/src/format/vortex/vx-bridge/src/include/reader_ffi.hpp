// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#pragma once
#include <string>
#include <string_view>
#include <cstdint>
#include <type_traits>
#include <vector>
#include <stdexcept>
#include <arrow/c/abi.h>

#include "rust/cxx.h"
#include "vx-bridge/lib.h"

namespace milvus_storage::vortex {

class VortexException : public std::runtime_error {
public:
    explicit VortexException(const std::string &message) : std::runtime_error(message) {
    }
};

enum class PType : uint8_t {
    U8 = 0,
    U16,
    U32,
    U64,
    I8,
    I16,
    I32,
    I64,
    F16,
    F32,
    F64,
};

namespace dtype {
    class DType {
    public:
        DType() = delete;
        explicit DType(rust::Box<ffi::DType> impl) : impl_(std::move(impl)) {
        }
        DType(DType &&other) noexcept = default;
        DType &operator=(DType &&other) = default;
        ~DType() = default;

        DType(const DType &) = delete;
        DType &operator=(const DType &) = delete;

        std::string ToString() const;

        const rust::Box<ffi::DType> &GetImpl() {
            return impl_;
        }

    private:
        rust::Box<ffi::DType> impl_;
    };

    // Factory functions
    DType null();
    DType bool_(bool nullable = false);
    DType primitive(PType ptype, bool nullable = false);
    DType int8(bool nullable = false);
    DType int16(bool nullable = false);
    DType int32(bool nullable = false);
    DType int64(bool nullable = false);
    DType uint8(bool nullable = false);
    DType uint16(bool nullable = false);
    DType uint32(bool nullable = false);
    DType uint64(bool nullable = false);
    DType float16(bool nullable = false);
    DType float32(bool nullable = false);
    DType float64(bool nullable = false);
    DType decimal(uint8_t precision = 10, int8_t scale = 0, bool nullable = false);
    DType utf8(bool nullable = false);
    DType binary(bool nullable = false);
    /// TODO: Other DTypes are only supported by creating from Arrow for now.
    DType from_arrow(struct ArrowSchema &schema, bool non_nullable = false);
} // namespace dtype

namespace scalar {
using dtype::DType;
class Scalar {
public:
    Scalar() = delete;
    explicit Scalar(rust::Box<ffi::Scalar> impl) : impl_(std::move(impl)) {
    }
    Scalar(Scalar &&other) noexcept = default;
    Scalar &operator=(Scalar &&other) noexcept = default;
    ~Scalar() = default;

    Scalar(const Scalar &) = delete;
    Scalar &operator=(const Scalar &) = delete;

    rust::Box<ffi::Scalar> IntoImpl() && {
        return std::move(impl_);
    }

private:
    rust::Box<ffi::Scalar> impl_;
};

// Factory functions for creating scalar values
Scalar bool_(bool value);
Scalar int8(int8_t value);
Scalar int16(int16_t value);
Scalar int32(int32_t value);
Scalar int64(int64_t value);
Scalar uint8(uint8_t value);
Scalar uint16(uint16_t value);
Scalar uint32(uint32_t value);
Scalar uint64(uint64_t value);
Scalar float32(float value);
Scalar float64(double value);
Scalar string(std::string_view value);
Scalar binary(const uint8_t *data, size_t length);
/// TODO: Other Scalars are only supported by casting for now.
Scalar cast(Scalar scalar, DType dtype);
}

namespace expr {
using scalar::Scalar;
class Expr {
public:
    Expr() = delete;
    explicit Expr(rust::Box<ffi::Expr> impl) : impl_(std::move(impl)) {
    }
    Expr(Expr &&other) noexcept = default;
    Expr &operator=(Expr &&other) noexcept = default;
    ~Expr() = default;

    Expr(const Expr &) = delete;
    Expr &operator=(const Expr &) = delete;

    rust::Box<ffi::Expr> IntoImpl() && {
        return std::move(impl_);
    }

    const ffi::Expr &Impl() const & {
        return *impl_;
    }

private:
    rust::Box<ffi::Expr> impl_;
};

Expr literal(Scalar scalar);
Expr root();
Expr column(std::string_view name);
Expr get_item(std::string_view field, Expr expr);
Expr not_(Expr expr);
Expr is_null(Expr expr);
Expr eq(Expr lhs, Expr rhs);
Expr not_eq_(Expr lhs, Expr rhs);
Expr gt(Expr lhs, Expr rhs);
Expr gt_eq(Expr lhs, Expr rhs);
Expr lt(Expr lhs, Expr rhs);
Expr lt_eq(Expr lhs, Expr rhs);
Expr and_(Expr lhs, Expr rhs);
Expr or_(Expr lhs, Expr rhs);
Expr checked_add(Expr lhs, Expr rhs);
Expr select(const std::vector<std::string_view> &fields, Expr child);
} // namespace expr


/// The StreamDriver internally holds a `RecordBatchIteratorAdapter` from the Rust side, which is thread-safe
/// and cloneable. The `RecordBatchIteratorAdapter` internally holds a `WorkStealingArrayIterator`.
class StreamDriver {
public:
    StreamDriver(StreamDriver &&other) noexcept = default;
    StreamDriver &operator=(StreamDriver &&other) noexcept = default;
    ~StreamDriver() = default;

    StreamDriver(const StreamDriver &) = delete;
    StreamDriver &operator=(const StreamDriver &) = delete;

    /// Create a stream of record batches.
    ///
    /// This function is thread-safe and can be called from multiple threads to create one stream per
    /// thread to make progress on the same StreamDriver that is built from a ScanBuilder concurrently.
    ///
    /// Within each thread, the record batches will be emitted in the original order they are within
    /// the scan. Between threads, the order is not guaranteed.
    ///
    /// Example: If the scan contains batches [b0, b1, b2, b3, b4, b5] and two threads call this
    /// function respectively to make progress on their own stream, Thread 1 might receive [b0,
    /// b2, b4] and Thread 2 might receive [b1, b3, b5]. Each thread maintains order within its
    /// subset, but overall ordering between threads is not guaranteed (e.g., Thread 2 could emit b1
    /// before Thread 1 emits b0).
    ArrowArrayStream CreateArrayStream() const;

private:
    friend class ScanBuilder;

    explicit StreamDriver(rust::Box<ffi::ThreadsafeCloneableReader> impl) : impl_(std::move(impl)) {
    }

    rust::Box<ffi::ThreadsafeCloneableReader> impl_;
};

class ScanBuilder;
class VortexFile;

class ObjectStoreWrapper2 {
public:
  ObjectStoreWrapper2(ObjectStoreWrapper2 &&other) noexcept = default;
  ObjectStoreWrapper2 &operator=(ObjectStoreWrapper2 &&other) noexcept = default;
  ~ObjectStoreWrapper2() = default;

  ObjectStoreWrapper2(const ObjectStoreWrapper2 &) = delete;
  ObjectStoreWrapper2 &operator=(const ObjectStoreWrapper2 &) = delete;

  static ObjectStoreWrapper2 OpenObjectStore(const std::string &ostype,
            const std::string &endpoint,
            const std::string &access_key_id,
            const std::string &secret_access_key,
            const std::string &region,
            const std::string &bucket_name);

private:
  friend class VortexFile;
  explicit ObjectStoreWrapper2(rust::Box<ffi::ObjectStoreWrapper2> impl) : impl_(std::move(impl)) {
  }

  rust::Box<ffi::ObjectStoreWrapper2> impl_;
};

class VortexFile {
public:
    static VortexFile Open(const ObjectStoreWrapper2 &obs, const std::string &path);

    VortexFile(VortexFile &&other) noexcept = default;
    VortexFile &operator=(VortexFile &&other) noexcept = default;
    ~VortexFile() = default;

    VortexFile(const VortexFile &) = delete;
    VortexFile &operator=(const VortexFile &) = delete;

    /// Get the number of rows in the file.
    uint64_t RowCount() const;

    /// Create a scan builder for the file.
    /// The scan builder can be used to scan the file.
    ScanBuilder CreateScanBuilder() const;

private:
    explicit VortexFile(rust::Box<ffi::VortexFile> impl) : impl_(std::move(impl)) {
    }

    rust::Box<ffi::VortexFile> impl_;
};


class ScanBuilder {
public:
    ScanBuilder(ScanBuilder &&other) noexcept = default;
    ScanBuilder &operator=(ScanBuilder &&other) noexcept {
        if (this != &other) {
            impl_ = std::move(other.impl_);
        }
        return *this;
    }
    ~ScanBuilder() = default;

    ScanBuilder(const ScanBuilder &) = delete;
    ScanBuilder &operator=(const ScanBuilder &) = delete;

    /// Only include rows that match the filter expressions.
    ScanBuilder &WithFilter(expr::Expr &&expr) &;
    ScanBuilder &WithFilter(const expr::Expr &expr) &;
    ScanBuilder &&WithFilter(expr::Expr &&expr) &&;
    ScanBuilder &&WithFilter(const expr::Expr &expr) &&;

    /// Only include columns that match the projection expressions.
    ScanBuilder &WithProjection(expr::Expr &&expr) &;
    ScanBuilder &WithProjection(const expr::Expr &expr) &;
    ScanBuilder &&WithProjection(expr::Expr &&expr) &&;
    ScanBuilder &&WithProjection(const expr::Expr &expr) &&;

    /// Only include rows in the range [row_range_start, row_range_end).
    ScanBuilder &WithRowRange(uint64_t row_range_start, uint64_t row_range_end) &;
    ScanBuilder &&WithRowRange(uint64_t row_range_start, uint64_t row_range_end) &&;

    /// Only include rows with the given indices.
    ScanBuilder &WithIncludeByIndex(const uint64_t *indices, std::size_t size) &;
    ScanBuilder &&WithIncludeByIndex(const uint64_t *indices, std::size_t size) &&;

    /// Set the limit on the number of rows to scan out.
    ScanBuilder &WithLimit(uint64_t limit) &;
    ScanBuilder &&WithLimit(uint64_t limit) &&;

    /// Set the output schema on the scan builder.
    /// TODO: currently if pass in this option, the schema needs to be the schema after adding projection.
    ScanBuilder &WithOutputSchema(ArrowSchema &output_schema) &;
    ScanBuilder &&WithOutputSchema(ArrowSchema &output_schema) &&;

    /// Take ownership and consume the scan builder to a stream of record batches.
    ArrowArrayStream IntoStream() &&;

    /// Take ownership and consume the scan builder to a stream driver.
    /// Under the hood, this function calls `ScanBuilder::into_record_batch_reader` and holds a
    /// `WorkStealingArrayIterator` in StreamDriver.
    StreamDriver IntoStreamDriver() &&;

private:
    friend class VortexFile;

    explicit ScanBuilder(rust::Box<ffi::VortexScanBuilder> impl) : impl_(std::move(impl)) {
    }

    rust::Box<ffi::VortexScanBuilder> impl_;
};

} // namespace milvus_storage::vortex