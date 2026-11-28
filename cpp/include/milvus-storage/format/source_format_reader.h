// Copyright 2026 Zilliz
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include "milvus-storage/format/format_reader.h"
#include "milvus-storage/common/arrow_util.h"
#include <folly/CancellationToken.h>
#include "milvus-storage/source_read_context.h"
#include "milvus-storage/format/format.h"
#include "milvus-storage/format/format_reader_cache.h"
#include "milvus-storage/filesystem/azure/source_sas_token_policy.h"

namespace milvus_storage {
// Keeps authority on cloned readers and returned streams, including operations
// satisfied entirely by format-level caches. Only wrap a source-aware reader;
// this guard cannot retrofit authority onto a legacy reader's I/O.
std::shared_ptr<FormatReader> GuardSourceFormatReader(std::shared_ptr<FormatReader> reader,
                                                      SourceReadContextPtr context);
// Source streams can return already-buffered batches without filesystem I/O.
std::shared_ptr<arrow::RecordBatchReader> GuardSourceBatchReader(std::shared_ptr<arrow::RecordBatchReader> reader,
                                                                 SourceReadContextPtr context);

// A source cache is bound to one context at construction. The loader must
// resolve source I/O explicitly; the legacy trait loader is never a fallback.
template <typename ReaderT>
arrow::Result<std::shared_ptr<FormatReader>> OpenSourceFormatReader(
    const MetadataCache& cache,
    const std::string& format,
    const api::ColumnGroupFile& file,
    const std::shared_ptr<const api::Properties>& properties,
    const std::shared_ptr<arrow::Schema>& schema,
    const std::vector<std::string>& columns,
    const KeyRetriever& key_retriever,
    const std::string& predicate) {
  const auto context = cache.source_context();
  ARROW_RETURN_NOT_OK(fs::SourceCredentialArrowStatus(context ? context->CheckState() : LOON_SOURCE_CREDENTIAL_DENIED));
  if (!cache.enabled()) {
    ARROW_ASSIGN_OR_RAISE(auto factory, Format::get(format));
    ARROW_ASSIGN_OR_RAISE(
        auto reader, factory->create_reader_with_context(schema, file, properties, columns, key_retriever, context));
    ARROW_RETURN_NOT_OK(reader->set_predicate(predicate));
    return reader;
  }
  if constexpr (requires {
                  ReaderT::MetaTrait::load_metadata_with_context(file, properties, key_retriever, context);
                }) {
    ARROW_ASSIGN_OR_RAISE(auto metadata, cache.get<ReaderT>()->get_or_open(ReaderT::MetaTrait::cache_key(file), [=] {
      return ReaderT::MetaTrait::load_metadata_with_context(file, properties, key_retriever, context);
    }));
    ARROW_RETURN_NOT_OK(fs::SourceCredentialArrowStatus(context->CheckState()));
    ARROW_ASSIGN_OR_RAISE(auto reader,
                          ReaderT::MetaTrait::create_from_metadata(metadata, file, schema, columns, predicate));
    ARROW_RETURN_NOT_OK(fs::SourceCredentialArrowStatus(context->CheckState()));
    return GuardSourceFormatReader(std::move(reader), context);
  }
  return arrow::Status::NotImplemented("Format metadata does not support source-read contexts");
}

template <typename ReaderT>
folly::SemiFuture<arrow::Result<std::shared_ptr<FormatReader>>> OpenSourceFormatReaderAsync(
    MetadataCache cache,
    std::string format,
    api::ColumnGroupFile file,
    std::shared_ptr<const api::Properties> properties,
    std::shared_ptr<arrow::Schema> schema,
    std::vector<std::string> columns,
    KeyRetriever key_retriever,
    std::string predicate,
    folly::CancellationToken cancellation) {
  using Result = arrow::Result<std::shared_ptr<FormatReader>>;
  // Retain every input until consumption and preserve native async loaders.
  // Formats with a blocking-only source loader defer it onto the caller's executor.
  return folly::makeSemiFuture().deferValue([cache, format, file, properties, schema, columns, key_retriever, predicate,
                                             cancellation](folly::Unit) -> folly::SemiFuture<Result> {
    FOLLY_ARROW_RETURN_IF_CANCELLED(cancellation);
    auto context = cache.source_context();
    auto status = fs::SourceCredentialArrowStatus(context ? context->CheckState() : LOON_SOURCE_CREDENTIAL_DENIED);
    if (!status.ok())
      return folly::makeSemiFuture(Result(status));
    if constexpr (requires {
                    ReaderT::MetaTrait::load_metadata_with_context(file, properties, key_retriever, context);
                  }) {
      auto load = [file, properties, key_retriever, context] {
        if constexpr (requires {
                        ReaderT::MetaTrait::load_metadata_with_context_async(file, properties, key_retriever, context);
                      }) {
          return ReaderT::MetaTrait::load_metadata_with_context_async(file, properties, key_retriever, context);
        } else {
          return folly::makeSemiFuture().deferValue([=](folly::Unit) {
            return ReaderT::MetaTrait::load_metadata_with_context(file, properties, key_retriever, context);
          });
        }
      };
      auto metadata =
          cache.enabled() ? cache.get<ReaderT>()->get_or_open_async(ReaderT::MetaTrait::cache_key(file), load) : load();
      return std::move(metadata).deferValue(
          [file, schema, columns, predicate, context,
           cancellation](arrow::Result<typename ReaderT::MetaTrait::MetadataPtr> result) -> folly::SemiFuture<Result> {
            FOLLY_ARROW_RETURN_NOT_OK(fs::SourceCredentialArrowStatus(context->CheckState()));
            FOLLY_ARROW_RETURN_IF_CANCELLED(cancellation);
            FOLLY_ARROW_ASSIGN_OR_RAISE(auto metadata, std::move(result));
            auto opened = [&] {
              if constexpr (requires {
                              ReaderT::MetaTrait::create_from_metadata_async(metadata, file, schema, columns,
                                                                             predicate);
                            }) {
                return ReaderT::MetaTrait::create_from_metadata_async(metadata, file, schema, columns, predicate);
              } else {
                return folly::makeSemiFuture(
                    ReaderT::MetaTrait::create_from_metadata(metadata, file, schema, columns, predicate));
              }
            }();
            return std::move(opened).deferValue(
                [context, cancellation](arrow::Result<std::shared_ptr<ReaderT>> result) -> Result {
                  ARROW_RETURN_NOT_OK(fs::SourceCredentialArrowStatus(context->CheckState()));
                  ARROW_RETURN_IF_CANCELLED(cancellation);
                  ARROW_ASSIGN_OR_RAISE(auto reader, std::move(result));
                  return GuardSourceFormatReader(std::move(reader), context);
                });
          });
    }
    if (!cache.enabled()) {
      return folly::makeSemiFuture(
          OpenSourceFormatReader<ReaderT>(cache, format, file, properties, schema, columns, key_retriever, predicate));
    }
    return folly::makeSemiFuture(
        Result(arrow::Status::NotImplemented("Format metadata does not support source-read contexts")));
  });
}
}  // namespace milvus_storage
