#include "fs_util.h"
#include <arrow/filesystem/localfs.h>
#include <arrow/filesystem/hdfs.h>
#include <arrow/filesystem/s3fs.h>
#include <arrow/result.h>
#include <arrow/util/uri.h>
#include <parquet/exception.h>
#include "common/exception.h"
namespace milvus_storage {

std::shared_ptr<arrow::fs::FileSystem>
BuildFileSystem(const std::string& uri) {
  arrow::internal::Uri uri_parser;
  PARQUET_THROW_NOT_OK(uri_parser.Parse(uri));
  auto schema = uri_parser.scheme();
  if (schema == "file") {
    PARQUET_ASSIGN_OR_THROW(auto option, arrow::fs::LocalFileSystemOptions::FromUri(uri_parser, nullptr));
    return std::shared_ptr<arrow::fs::FileSystem>(new arrow::fs::LocalFileSystem(option));
  }

  if (schema == "hdfs") {
    PARQUET_ASSIGN_OR_THROW(auto option, arrow::fs::HdfsOptions::FromUri(uri_parser));
    PARQUET_ASSIGN_OR_THROW(auto fs, arrow::fs::HadoopFileSystem::Make(option));
    return std::shared_ptr<arrow::fs::FileSystem>(fs);
  }

  if (schema == "s3") {
    PARQUET_ASSIGN_OR_THROW(auto option, arrow::fs::S3Options::FromUri(uri_parser));
    PARQUET_ASSIGN_OR_THROW(auto fs, arrow::fs::S3FileSystem::Make(option));
    return std::shared_ptr<arrow::fs::FileSystem>(fs);
  }

  throw StorageException("unsupported schema: " + schema);
}
};  // namespace milvus_storage