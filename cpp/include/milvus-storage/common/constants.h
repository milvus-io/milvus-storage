

#pragma once
#include <string>

namespace milvus_storage {

const int kReadBatchSize = 1024;

const std::string kManifestTempFileSuffix = ".manifest.tmp";
const std::string kManifestFileSuffix = ".manifest";
const std::string kManifestsDir = "versions";
const std::string kScalarDataDir = "scalar";
const std::string kVectorDataDir = "vector";
const std::string kDeleteDataDir = "delete";
const std::string kBlobDir = "blobs";
const std::string kParquetDataFileSuffix = ".parquet";
const std::string kOffsetFieldName = "__offset";

const std::string ARROW_FIELD_ID_KEY = "PARQUET:field_id";

const std::string GROUP_DELIMITER = ";";
const std::string COLUMN_DELIMITER = ",";
const std::string GROUP_FIELD_ID_LIST_META_KEY = "group_field_id_list";
const std::string STORAGE_VERSION_KEY = "storage_version";
constexpr char ROW_GROUP_META_KEY[] = "row_group_metadata";

const int64_t DEFAULT_ARROW_FILESYSTEM_S3_REQUEST_TIMEOUT_SEC = 10;

}  // namespace milvus_storage
