#include <arrow/io/interfaces.h>
#include <arrow/status.h>
#include <arrow/util/thread_pool.h>
#include <stdexcept>
#include "arrow/util/logging.h"

namespace milvus_storage {

template <typename... SubmitArgs>
auto SubmitIO(arrow::io::IOContext io_context, SubmitArgs&&... submit_args)
    -> decltype(std::declval<::arrow::internal::Executor*>()->Submit(submit_args...)) {
  arrow::internal::TaskHints hints;
  hints.external_id = io_context.external_id();
  return io_context.executor()->Submit(hints, io_context.stop_token(), std::forward<SubmitArgs>(submit_args)...);
};

inline void CloseFromDestructor(arrow::io::FileInterface* file) {
  arrow::Status st = file->Close();
  if (!st.ok()) {
    auto file_type = typeid(*file).name();
    std::stringstream ss;
    ss << "When destroying file of type " << file_type << ": " << st.message();
    ARROW_LOG(ERROR) << st.WithMessage(ss.str());
    throw std::runtime_error(ss.str());
  }
}

}  // namespace milvus_storage