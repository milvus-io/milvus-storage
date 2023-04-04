#pragma once
#include <arrow/filesystem/filesystem.h>
#include <string>

std::shared_ptr<arrow::fs::FileSystem>
BuildFileSystem(const std::string& uri);