
#pragma once

#include <atomic>
#include <cstdint>
#include <exception>
#include <functional>
#include <memory>
#include <new>
#include <stdexcept>
#include <string_view>
#include <type_traits>
#include <utility>

#include "arrow/filesystem/filesystem.h"
#include "arrow/io/interfaces.h"
#include "arrow/status.h"
#include "arrow/util/thread_pool.h"
#include "arrow/util/uri.h"
#include "arrow/util/visibility.h"

#include "milvus-storage/common/extend_status.h"
#include "milvus-storage/common/writer_status.h"

namespace arrow {
using util::Uri;
namespace fs {
namespace internal {

template <typename OutputType, typename InputType>
inline OutputType checked_cast(InputType&& value) {
  static_assert(
      std::is_class<typename std::remove_pointer<typename std::remove_reference<InputType>::type>::type>::value,
      "checked_cast input type must be a class");
  static_assert(
      std::is_class<typename std::remove_pointer<typename std::remove_reference<OutputType>::type>::type>::value,
      "checked_cast output type must be a class");
#ifdef NDEBUG
  return static_cast<OutputType>(value);
#else
  return dynamic_cast<OutputType>(value);
#endif
}

ARROW_EXPORT
TimePoint CurrentTimePoint();

ARROW_EXPORT
arrow::Status CopyStream(const std::shared_ptr<io::InputStream>& src,
                         const std::shared_ptr<io::OutputStream>& dest,
                         int64_t chunk_size,
                         const io::IOContext& io_context);

ARROW_EXPORT
arrow::Status PathNotFound(std::string_view path);

ARROW_EXPORT
arrow::Status IsADir(std::string_view path);

ARROW_EXPORT
arrow::Status NotADir(std::string_view path);

ARROW_EXPORT
arrow::Status NotEmpty(std::string_view path);

ARROW_EXPORT
arrow::Status NotAFile(std::string_view path);

ARROW_EXPORT
arrow::Status InvalidDeleteDirContents(std::string_view path);

/// \brief Parse the string as a URI
/// \param uri_string the string to parse
///
/// This is the same as Uri::Parse except it tolerates Windows
/// file URIs that contain backslash instead of /
arrow::Result<Uri> ParseFileSystemUri(const std::string& uri_string);

/// \brief check if the string is a local absolute path
ARROW_EXPORT
bool DetectAbsolutePath(const std::string& s);

/// \brief describes how to handle the authority (host) component of the URI
enum class AuthorityHandlingBehavior {
  // Return an invalid status if the authority is non-empty
  kDisallow = 0,
  // Prepend the authority to the path (e.g. authority/some/path)
  kPrepend = 1,
  // Convert to a Windows style network path (e.g. //authority/some/path)
  kWindows = 2,
  // Ignore the authority and just use the path
  kIgnore = 3
};

/// \brief check to see if uri_string matches one of the supported schemes and return the
/// path component
/// \param uri_string a uri or local path to test and convert
/// \param supported_schemes the set of URI schemes that should be accepted
/// \param accept_local_paths if true, allow an absolute path
/// \return the path portion of the URI
arrow::Result<std::string> PathFromUriHelper(const std::string& uri_string,
                                             std::vector<std::string> supported_schemes,
                                             bool accept_local_paths,
                                             AuthorityHandlingBehavior authority_handling);

/// \brief Return files matching the glob pattern on the filesystem
///
/// Globbing starts from the root of the filesystem.
ARROW_EXPORT
arrow::Result<FileInfoVector> GlobFiles(const std::shared_ptr<FileSystem>& filesystem, const std::string& glob);

extern FileSystemGlobalOptions global_options;

ARROW_EXPORT
arrow::Status PathNotFound(std::string_view path);

}  // namespace internal
}  // namespace fs

namespace io {
namespace internal {

/// Record a background I/O failure on the writer and return the writer's
/// terminal status.
///
/// Recording allocates. If that allocation fails the exception propagates and
/// the process dies, by design: the alternative -- swallowing it so the caller
/// can carry on -- is what used to leave the aggregate future unfinished with
/// its publish slot already consumed, hanging every waiter forever.
inline arrow::Status ObserveBackgroundIOStatus(milvus_storage::WriterStatus* writer_status,
                                               const arrow::Status& status) {
  if (status.ok()) {
    return arrow::Status::OK();
  }
  writer_status->ObserveFailure(status);
  return writer_status->Check();
}

template <typename... SubmitArgs>
auto SubmitIO(arrow::io::IOContext io_context, SubmitArgs&&... submit_args)
    -> decltype(std::declval<::arrow::internal::Executor*>()->Submit(submit_args...)) {
  arrow::internal::TaskHints hints;
  hints.external_id = io_context.external_id();
  return io_context.executor()->Submit(hints, io_context.stop_token(), std::forward<SubmitArgs>(submit_args)...);
}

/// Submit a background I/O task and invoke `completion` exactly once for every
/// recoverable outcome: executor rejection, a returned task failure, or an
/// exception before the task can report its status. Allocation failure is
/// fail-stop once the caller has registered the operation, because recovery
/// could strand its completion slot. Pre-execution cancellation is handled by
/// the executor stop callback; every other recoverable path settles here.
template <typename Task, typename Completion>
arrow::Status SubmitIOWithCompletion(arrow::io::IOContext io_context, Task&& task, Completion&& completion) noexcept {
  using CompletionType = std::decay_t<Completion>;
  static_assert(std::is_copy_constructible_v<CompletionType>, "The completion callback must be copyable");

  struct CompletionState {
    explicit CompletionState(const CompletionType& callback) : callback(callback) {}

    void Complete(const arrow::Status& status) noexcept {
      if (!completed.exchange(true, std::memory_order_acq_rel)) {
        callback(status);
      }
    }

    std::atomic<bool> completed{false};
    CompletionType callback;
  };

  std::shared_ptr<CompletionState> completion_state;
  try {
    // Copy the named callback only after make_shared has allocated storage, so
    // allocation/construction failure leaves the caller's callback usable.
    completion_state = std::make_shared<CompletionState>(completion);
  } catch (const std::bad_alloc&) {
    // Callers register the background operation before entering this helper.
    // Returning or unwinding here would leave that completion slot unresolved.
    std::terminate();
  } catch (const std::exception& e) {
    auto status = milvus_storage::MakeExtendErrorMsg(milvus_storage::ExtendStatusCode::InternalInvariantViolated,
                                                     "Failed to prepare background I/O completion: ", e.what());
    completion(status);
    return status;
  } catch (...) {
    auto status = milvus_storage::MakeExtendErrorMsg(
        milvus_storage::ExtendStatusCode::InternalInvariantViolated,
        "Failed to prepare background I/O completion due to a non-standard exception");
    completion(status);
    return status;
  }

  try {
    auto deferred = [task = std::forward<Task>(task), completion_state]() mutable noexcept {
      arrow::Status task_status;
      try {
        task_status = task();
      } catch (const std::bad_alloc&) {
        // A waiter already owns this task's completion slot. There is no safe
        // recoverable status to publish while allocation itself is failing.
        std::terminate();
      } catch (const std::exception& e) {
        task_status = milvus_storage::MakeExtendErrorMsg(milvus_storage::ExtendStatusCode::InternalInvariantViolated,
                                                         "Background I/O task threw an exception: ", e.what());
      } catch (...) {
        task_status = milvus_storage::MakeExtendErrorMsg(milvus_storage::ExtendStatusCode::InternalInvariantViolated,
                                                         "Background I/O task threw a non-standard exception");
      }
      completion_state->Complete(task_status);
    };

    arrow::internal::TaskHints hints;
    hints.external_id = io_context.external_id();
    auto stop_callback = [completion_state](const arrow::Status& status) { completion_state->Complete(status); };
    auto submit_status =
        io_context.executor()->Spawn(hints, std::move(deferred), io_context.stop_token(), std::move(stop_callback));
    if (!submit_status.ok()) {
      completion_state->Complete(submit_status);
      return submit_status;
    }
  } catch (const std::bad_alloc&) {
    // Executor submission may have accepted the task before throwing. Failing
    // stop is the only verdict that cannot strand or double-set completion.
    std::terminate();
  } catch (const std::exception& e) {
    auto status = milvus_storage::MakeExtendErrorMsg(milvus_storage::ExtendStatusCode::InternalInvariantViolated,
                                                     "Failed to submit background I/O: ", e.what());
    completion_state->Complete(status);
    return status;
  } catch (...) {
    auto status = milvus_storage::MakeExtendErrorMsg(milvus_storage::ExtendStatusCode::InternalInvariantViolated,
                                                     "Failed to submit background I/O due to a non-standard exception");
    completion_state->Complete(status);
    return status;
  }

  return arrow::Status::OK();
}

}  // namespace internal
}  // namespace io

}  // namespace arrow
