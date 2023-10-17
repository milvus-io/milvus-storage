#include "common/fs_util.h"
#include <arrow/filesystem/localfs.h>
#include <arrow/filesystem/hdfs.h>
#include <arrow/filesystem/s3fs.h>
#include <arrow/util/uri.h>
#include "common/macro.h"
namespace milvus_storage {

Result<std::shared_ptr<arrow::fs::FileSystem>> BuildFileSystem(const std::string& uri) {
  arrow::internal::Uri uri_parser;
  RETURN_ARROW_NOT_OK(uri_parser.Parse(uri));
  auto schema = uri_parser.scheme();
  if (schema == "file") {
    auto output_path = uri_parser.path();
    ASSIGN_OR_RETURN_ARROW_NOT_OK(auto option, arrow::fs::LocalFileSystemOptions::FromUri(uri_parser, &output_path));
    return std::shared_ptr<arrow::fs::FileSystem>(new arrow::fs::LocalFileSystem(option));
  }

  // if (schema == "hdfs") {
  //   ASSIGN_OR_RETURN_ARROW_NOT_OK(auto option, arrow::fs::HdfsOptions::FromUri(uri_parser));
  //   ASSIGN_OR_RETURN_ARROW_NOT_OK(auto fs, arrow::fs::HadoopFileSystem::Make(option));
  //   return std::shared_ptr<arrow::fs::FileSystem>(fs);
  // }

  if (schema == "s3") {
    RETURN_ARROW_NOT_OK(arrow::fs::InitializeS3(arrow::fs::S3GlobalOptions{}));
    ASSIGN_OR_RETURN_ARROW_NOT_OK(auto option, arrow::fs::S3Options::FromUri(uri_parser));
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