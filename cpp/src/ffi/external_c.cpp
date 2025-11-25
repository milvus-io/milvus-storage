#include "milvus-storage/ffi_c.h"

#include <cstring>

#include <parquet/arrow/reader.h>
#include <arrow/c/abi.h>
#include <arrow/c/bridge.h>
#include <arrow/type_fwd.h>

#include "milvus-storage/ffi_internal/result.h"
#include "milvus-storage/filesystem/fs.h"
#include "milvus-storage/properties.h"

FFIResult external_get_file_info(const char* format,
                                 const char* file_path,
                                 const Properties* properties,
                                 uint64_t* out_num_of_rows,
                                 [[maybe_unused]] struct ArrowSchema* out_schema) {
  if (!format || !file_path || !properties || !out_num_of_rows) {
    RETURN_ERROR(LOON_INVALID_ARGS,
                 "Invalid arguments: format, file_path, properties, and out_num_of_rows must not be null");
  }

  milvus_storage::ArrowFileSystemConfig fs_config;
  milvus_storage::api::Properties properties_map;

  auto opt = milvus_storage::api::ConvertFFIProperties(properties_map, properties);
  if (opt != std::nullopt) {
    RETURN_ERROR(LOON_INVALID_PROPERTIES, "Failed to parse properties [", opt->c_str(), "]");
  }

  if (strcmp(format, "parquet") != 0) {
    RETURN_ERROR(LOON_INVALID_ARGS, "Invalid format: ", format);
  }

  // Configure filesystem from properties
  auto status = milvus_storage::ArrowFileSystemConfig::create_file_system_config(properties_map, fs_config);
  if (!status.ok()) {
    RETURN_ERROR(LOON_ARROW_ERROR, status.ToString());
  }

  auto fs_res = milvus_storage::CreateArrowFileSystem(fs_config);
  if (!fs_res.ok()) {
    RETURN_ERROR(LOON_ARROW_ERROR, fs_res.status().ToString());
  }
  auto fs = fs_res.ValueOrDie();

  // Get file info
  auto file_info_res = fs->GetFileInfo(file_path);
  if (!file_info_res.ok()) {
    RETURN_ERROR(LOON_ARROW_ERROR, file_info_res.status().ToString());
  }
  auto file_info = file_info_res.ValueOrDie();

  if (file_info.type() == arrow::fs::FileType::NotFound) {
    RETURN_ERROR(LOON_INVALID_ARGS, "File not found: ", file_path);
  }

  if (file_info.type() != arrow::fs::FileType::File) {
    RETURN_ERROR(LOON_INVALID_ARGS, "Path is not a file: ", file_path);
  }

  // Open and read file metadata
  auto file_reader_res = fs->OpenInputFile(file_info.path());
  if (!file_reader_res.ok()) {
    RETURN_ERROR(LOON_ARROW_ERROR, file_reader_res.status().ToString());
  }
  auto file_reader = file_reader_res.ValueOrDie();

  auto parquet_reader = parquet::ParquetFileReader::Open(file_reader);
  *out_num_of_rows = parquet_reader->metadata()->num_rows();

  auto close_status = file_reader->Close();
  if (!close_status.ok()) {
    RETURN_ERROR(LOON_ARROW_ERROR, close_status.ToString());
  }

  RETURN_SUCCESS();
}
