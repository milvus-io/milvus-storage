#pragma once

#include <arrow/type.h>
#include <memory>
#include "storage/options.h"
std::shared_ptr<arrow::Schema>
BuildScalarSchema(arrow::Schema* schema, SpaceOption* options);

std::shared_ptr<arrow::Schema>
BuildVectorSchema(arrow::Schema* schema, SpaceOption* options);

std::shared_ptr<arrow::Schema>
BuildDeleteSchema(arrow::Schema* schema, SpaceOption* options);

bool
ValidateSchema(arrow::Schema* schema);