#include "common/fs_util.h"
#include <arrow/filesystem/localfs.h>
#include <arrow/filesystem/hdfs.h>
#include <arrow/filesystem/s3fs.h>
#include <arrow/util/uri.h>
#include <cstdlib>
#include "common/log.h"
#include "common/macro.h"
namespace milvus_storage {

Result<std::shared_ptr<arrow::fs::FileSystem>> BuildFileSystem(const std::string& uri, std::string* out_path) {
  arrow::internal::Uri uri_parser;
  RETURN_ARROW_NOT_OK(uri_parser.Parse(uri));
  auto schema = uri_parser.scheme();
  if (schema == "file") {
    if (out_path == nullptr) {
      return Status::InvalidArgument("out_path should not be nullptr if schema is file");
    }
    ASSIGN_OR_RETURN_ARROW_NOT_OK(auto option, arrow::fs::LocalFileSystemOptions::FromUri(uri_parser, out_path));
    return std::shared_ptr<arrow::fs::FileSystem>(new arrow::fs::LocalFileSystem(option));
  }

  // if (schema == "hdfs") {
  //   ASSIGN_OR_RETURN_ARROW_NOT_OK(auto option, arrow::fs::HdfsOptions::FromUri(uri_parser));
  //   ASSIGN_OR_RETURN_ARROW_NOT_OK(auto fs, arrow::fs::HadoopFileSystem::Make(option));
  //   return std::shared_ptr<arrow::fs::FileSystem>(fs);
  // }

  if (schema == "s3") {
    if (!arrow::fs::IsS3Initialized()) {
      RETURN_ARROW_NOT_OK(arrow::fs::EnsureS3Initialized());
      std::atexit([]() {
        auto status = arrow::fs::EnsureS3Finalized();
        if (!status.ok()) {
          LOG_STORAGE_WARNING_ << "Failed to finalize S3: " << status.message();
        }
      });
    }
    ASSIGN_OR_RETURN_ARROW_NOT_OK(auto option, arrow::fs::S3Options::FromUri(uri_parser, out_path));
    ASSIGN_OR_RETURN_ARROW_NOT_OK(auto fs, arrow::fs::S3FileSystem::Make(option));

    return std::shared_ptr<arrow::fs::FileSystem>(fs);
  }

  return Status::InvalidArgument("Unsupported schema: " + schema);
}
/**
 * Uri Convert to Path
 */
std::string UriToPath(const std::string& uri) {
  arrow::internal::Uri uri_parser;
  auto status = uri_parser.Parse(uri);

  if (status.ok()) {
    return uri_parser.path();
  } else {
    return std::string("");
  }
}
};  // namespace milvus_storage