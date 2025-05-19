

#pragma once

#include <memory>
#include <vector>

namespace milvus_storage {

class ColumnGroup;

using ColumnGroupVector = std::vector<std::shared_ptr<ColumnGroup>>;
class RowGroupSizeVector;
class FieldIDList;
class GroupFieldIDList;

struct ColumnOffset;

using FieldID = int64_t;

}  // namespace milvus_storage