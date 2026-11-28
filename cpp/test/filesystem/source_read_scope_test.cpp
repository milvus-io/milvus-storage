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

#include "milvus-storage/filesystem/source_read_scope.h"

#include <gtest/gtest.h>
#include <arrow/status.h>
#include <arrow/util/uri.h>

namespace milvus_storage {
namespace {

TEST(SourceReadScope, EquivalentAzureEndpointsPreserveAccountCloudAndPath) {
  const auto dfs = AzureSourceResource::Parse("abfss://tables@acct.dfs.core.windows.net/team/table/", "acct");
  const auto blob = AzureSourceResource::Parse("https://acct.blob.core.windows.net/tables/team/table", "acct");
  const auto https_dfs = AzureSourceResource::Parse("https://ACCT.DFS.CORE.WINDOWS.NET/tables/team/table", "acct");
  ASSERT_TRUE(dfs);
  ASSERT_TRUE(blob);
  ASSERT_TRUE(https_dfs);
  EXPECT_EQ(*dfs, *blob);
  EXPECT_EQ(*dfs, *https_dfs);
  EXPECT_EQ(dfs->CanonicalUri(), "https://acct.blob.core.windows.net/tables/team/table");
}

TEST(SourceReadScope, RootMatchingUsesComponentsAndExactAuthority) {
  const auto root = AzureSourceResource::Parse("abfss://tables@acct.dfs.core.windows.net/team/table", "acct");
  ASSERT_TRUE(root);
  for (const auto* uri : {
           "https://acct.blob.core.windows.net/tables/team/table",
           "https://acct.blob.core.windows.net/tables/team/table/data/file.parquet",
           "abfss://tables@acct.dfs.core.windows.net/team/table/metadata/snapshot.json",
       }) {
    const auto resource = AzureSourceResource::Parse(uri, "acct");
    ASSERT_TRUE(resource) << uri;
    EXPECT_TRUE(root->Contains(*resource)) << uri;
  }
  for (const auto* uri : {
           "https://acct.blob.core.windows.net/tables/team/table-other/data",
           "https://acct.blob.core.windows.net/tables/team/Table/data",
           "https://acct.blob.core.windows.net/tables/team",
           "https://acct.blob.core.windows.net/other/team/table/data",
           "https://acct.blob.core.windows.net.attacker.example/tables/team/table/data",
           "https://acct.blob.core.usgovcloudapi.net/tables/team/table/data",
       }) {
    const auto resource = AzureSourceResource::Parse(uri, "acct");
    ASSERT_TRUE(resource) << uri;
    EXPECT_FALSE(root->Contains(*resource)) << uri;
  }
  EXPECT_FALSE(AzureSourceResource::Parse("https://other.blob.core.windows.net/tables/team/table", "acct"));
}

TEST(SourceReadScope, RejectsTraversalAmbiguityAndCredentialBearingUris) {
  for (const auto* uri : {
           "https://acct.blob.core.windows.net/tables/team/../private",
           "https://acct.blob.core.windows.net/tables/team/%2e%2e/private",
           "https://acct.blob.core.windows.net/tables/team/.%2E/private",
           "https://acct.blob.core.windows.net/tables/team/./file",
           "https://acct.blob.core.windows.net/tables/team%2Fprivate/file",
           "https://acct.blob.core.windows.net/tables/team%5cprivate/file",
           "https://acct.blob.core.windows.net/tables/team\\private/file",
           "https://acct.blob.core.windows.net/tables/team//file",
           "https://acct.blob.core.windows.net/tables/%00/file",
           "https://acct.blob.core.windows.net/tables/%0a/file",
           "https://acct.blob.core.windows.net/tables/%/file",
           "https://acct.blob.core.windows.net/tables/%xy/file",
           "https://acct.blob.core.windows.net/tables/team?sig=secret",
           "https://acct.blob.core.windows.net/tables/team#fragment",
           "https://user:secret@acct.blob.core.windows.net/tables/team",
           "abfss://tables:secret@acct.dfs.core.windows.net/team",
           "abfss://tables@user@acct.dfs.core.windows.net/team",
           "abfss://tables@acct.blob.core.windows.net/team",
           "http://acct.blob.core.windows.net/tables/team",
           "abfs://tables@acct.dfs.core.windows.net/team",
           "az://tables/team",
           "tables/team",
       }) {
    EXPECT_FALSE(AzureSourceResource::Parse(uri, "acct")) << uri;
  }
}

TEST(SourceReadScope, CanonicalizationDecodesExactlyOnceAndRoundTrips) {
  const auto root =
      AzureSourceResource::Parse("https://acct.blob.core.windows.net/tables/team%20data/%E4%B8%AD", "acct");
  ASSERT_TRUE(root);
  EXPECT_EQ(root->path, (std::vector<std::string>{"team data", "中"}));
  const auto canonical = root->CanonicalUri();
  const auto round_trip = AzureSourceResource::Parse(canonical, "acct");
  ASSERT_TRUE(round_trip);
  EXPECT_EQ(*round_trip, *root);

  const auto literal_percent =
      AzureSourceResource::Parse("https://acct.blob.core.windows.net/tables/%252e%252e/file", "acct");
  ASSERT_TRUE(literal_percent);
  EXPECT_EQ(literal_percent->path.front(), "%2e%2e");
  EXPECT_EQ(literal_percent->CanonicalUri(), "https://acct.blob.core.windows.net/tables/%252e%252e/file");
  const auto literal_question =
      AzureSourceResource::Parse("https://acct.blob.core.windows.net/tables/name%3Fpart", "acct");
  ASSERT_TRUE(literal_question);
  EXPECT_EQ(literal_question->CanonicalUri(), "https://acct.blob.core.windows.net/tables/name%3Fpart");
}

TEST(SourceReadScope, ExplicitRootsDoNotAuthorizeTheirCommonParent) {
  const auto first = AzureSourceResource::Parse("https://acct.blob.core.windows.net/tables/team/first", "acct");
  const auto second = AzureSourceResource::Parse("https://acct.blob.core.windows.net/tables/team/second", "acct");
  const auto nested = AzureSourceResource::Parse("https://acct.blob.core.windows.net/tables/team/first/nested", "acct");
  const auto parent = AzureSourceResource::Parse("https://acct.blob.core.windows.net/tables/team", "acct");
  const auto nested_file =
      AzureSourceResource::Parse("abfss://tables@acct.dfs.core.windows.net/team/first/nested/file", "acct");
  ASSERT_TRUE(first);
  ASSERT_TRUE(second);
  ASSERT_TRUE(nested);
  ASSERT_TRUE(parent);
  ASSERT_TRUE(nested_file);
  const std::vector<AzureSourceResource> roots{*first, *second, *nested};
  EXPECT_EQ(FindSourceRoot(roots, *first), 0);
  EXPECT_EQ(FindSourceRoot(roots, *second), 1);
  EXPECT_EQ(FindSourceRoot(roots, *nested_file), 2);
  EXPECT_FALSE(FindSourceRoot(roots, *parent));
  EXPECT_FALSE(FindSourceRoot({}, *first));
}

TEST(SourceReadScope, StorageAliasesUseOnlyTheSelectedAccountEndpointAndContainer) {
  const auto binding = AzureSourceStorageBinding::Create("acct", "core.windows.net", "tables");
  ASSERT_TRUE(binding);
  const auto expected = AzureSourceResource::Parse("https://acct.blob.core.windows.net/tables/team/table/data", "acct");
  ASSERT_TRUE(expected);
  for (const auto* uri : {
           "azure://core.windows.net/tables/team/table/data",
           "az://core.windows.net/tables/team/table/data",
           "az://tables/team/table/data",
           "https://acct.blob.core.windows.net/tables/team/table/data",
           "abfss://tables@acct.dfs.core.windows.net/team/table/data",
       }) {
    const auto resolved = binding->ResolveUri(uri);
    ASSERT_TRUE(resolved) << uri;
    EXPECT_EQ(*resolved, *expected) << uri;
  }
  for (const auto* uri : {
           "azure://core.windows.net.attacker.example/tables/team/table/data",
           "azure://core.usgovcloudapi.net/tables/team/table/data",
           "azure://core.windows.net/other/team/table/data",
           "azure://tables/team/table/data",
           "az://other/team/table/data",
           "az://user@tables/team/table/data",
           "https://other.blob.core.windows.net/tables/team/table/data",
           "https://acct.blob.core.usgovcloudapi.net/tables/team/table/data",
           "abfss://tables@other.dfs.core.windows.net/team/table/data",
           "abfss://tables/team/table/data",
           "az://tables/team/%2e%2e/private",
           "az://tables/team%2Fprivate/file",
           "az://tables/team/table/data?sig=secret",
           "az://tables/team/table/data#fragment",
           "team/table/data",
       }) {
    EXPECT_FALSE(binding->ResolveUri(uri)) << uri;
  }
}

TEST(SourceReadScope, AliasResolutionDoesNotGrantAccessToSiblingRoots) {
  const auto binding = AzureSourceStorageBinding::Create("acct", "core.windows.net", "tables");
  const auto root = AzureSourceResource::Parse("abfss://tables@acct.dfs.core.windows.net/team/table", "acct");
  ASSERT_TRUE(binding);
  ASSERT_TRUE(root);
  const auto sibling = binding->ResolveUri("azure://core.windows.net/tables/team/table-other/data");
  const auto parent = binding->ResolveUri("az://tables/team");
  const auto data = binding->ResolvePath("team/table/data");
  ASSERT_TRUE(sibling);
  ASSERT_TRUE(parent);
  ASSERT_TRUE(data);
  EXPECT_FALSE(FindSourceRoot({*root}, *sibling));
  EXPECT_FALSE(FindSourceRoot({*root}, *parent));
  EXPECT_EQ(FindSourceRoot({*root}, *data), 0);
}

TEST(SourceReadScope, RelativeFilesystemKeysAreDecodedPathsNotUris) {
  const auto binding = AzureSourceStorageBinding::Create("acct", "CORE.WINDOWS.NET", "tables");
  ASSERT_TRUE(binding);
  const auto literal = binding->ResolvePath("team/%2e%2e/name?part#fragment");
  ASSERT_TRUE(literal);
  EXPECT_EQ(literal->CanonicalUri(),
            "https://acct.blob.core.windows.net/tables/team/%252e%252e/name%3Fpart%23fragment");
  const auto encoded = binding->ResolveUri("az://tables/team/%252e%252e/name%3Fpart%23fragment");
  ASSERT_TRUE(encoded);
  EXPECT_EQ(*literal, *encoded);
  for (const auto* path : {"/team/file", "team/../file", "team/./file", "team//file", "team\\file"}) {
    EXPECT_FALSE(binding->ResolvePath(path)) << path;
  }
  EXPECT_FALSE(binding->ResolvePath(std::string("team/\0file", 10)));
}

TEST(SourceReadScope, StorageBindingRejectsIncompleteOrAmbiguousConfiguration) {
  EXPECT_FALSE(AzureSourceStorageBinding::Create("", "core.windows.net", "tables"));
  EXPECT_FALSE(AzureSourceStorageBinding::Create("Acct", "core.windows.net", "tables"));
  EXPECT_FALSE(AzureSourceStorageBinding::Create("acct.other", "core.windows.net", "tables"));
  EXPECT_FALSE(AzureSourceStorageBinding::Create("acct", "", "tables"));
  EXPECT_FALSE(AzureSourceStorageBinding::Create("acct", "https://core.windows.net", "tables"));
  EXPECT_FALSE(AzureSourceStorageBinding::Create("acct", "core.windows.net:443", "tables"));
  EXPECT_FALSE(AzureSourceStorageBinding::Create("acct", "core.windows.net/path", "tables"));
  EXPECT_FALSE(AzureSourceStorageBinding::Create("acct", "core.windows.net", "tables/path"));
}

TEST(SourceReadScope, ChecksOriginalUriBeforeArrowDecodesFilesystemKeys) {
  const auto binding = AzureSourceStorageBinding::Create("acct", "core.windows.net", "tables");
  ASSERT_TRUE(binding);
  const std::string uri = "azure://core.windows.net/tables/team%20data/%252e%252e/file";
  const auto resource = binding->ResolveUri(uri);
  ASSERT_TRUE(resource);
  arrow::util::Uri parsed;
  ASSERT_TRUE(parsed.Parse(uri).ok());
  // StorageUri::Parse gets its key from Arrow's decoded path. The path API
  // must not decode that key a second time and turn a literal name into '..'.
  EXPECT_EQ(parsed.path(), "/tables/team data/%2e%2e/file");
  const auto filesystem_resource = binding->ResolvePath(parsed.path().substr(8));
  ASSERT_TRUE(filesystem_resource);
  EXPECT_EQ(*filesystem_resource, *resource);

  const std::string ambiguous = "azure://core.windows.net/tables/team%2Fprivate/file";
  ASSERT_TRUE(parsed.Parse(ambiguous).ok());
  EXPECT_EQ(parsed.path(), "/tables/team/private/file");
  // Once normalized, the original encoded separator is lost. Authorize the
  // original URI before deriving the relative key passed to FileSystemProxy.
  EXPECT_FALSE(binding->ResolveUri(ambiguous));
}

TEST(SourceReadScope, HttpListsAuthorizeQueryDirectoryInsteadOfContainerPath) {
  const auto root = AzureSourceResource::Parse("https://acct.blob.core.windows.net/tables/team/table", "acct");
  ASSERT_TRUE(root);
  for (const auto* uri : {
           "https://acct.blob.core.windows.net/tables?comp=list&restype=container&prefix=team/table/&delimiter=/",
           "https://acct.blob.core.windows.net/tables?comp=list&restype=container&prefix=team%2Ftable%2F&marker=a%2Bb",
           "https://acct.dfs.core.windows.net/tables?resource=filesystem&directory=team/table&recursive=true",
           "https://acct.dfs.core.windows.net/"
           "tables?resource=filesystem&directory=team%2Ftable&recursive=false&maxresults=10",
       }) {
    const auto request = AzureSourceHttpRequest::Parse("GET", uri, "acct");
    ASSERT_TRUE(request) << uri;
    EXPECT_TRUE(request->is_list);
    EXPECT_EQ(request->resource, *root);
    EXPECT_EQ(FindSourceRoot({*root}, request->resource), 0);
  }
  // A well-formed sibling request is classifiable but still unauthorized.
  const auto sibling = AzureSourceHttpRequest::Parse(
      "GET", "https://acct.blob.core.windows.net/tables?comp=list&restype=container&prefix=team/table-other/", "acct");
  ASSERT_TRUE(sibling);
  EXPECT_FALSE(FindSourceRoot({*root}, sibling->resource));
}

TEST(SourceReadScope, HttpRequestRejectsWritesRootProbesCredentialsAndAmbiguousListPrefixes) {
  const std::string object = "https://acct.blob.core.windows.net/tables/team/table/data";
  for (const auto* method : {"PUT", "POST", "DELETE", "PATCH", "OPTIONS"}) {
    EXPECT_FALSE(AzureSourceHttpRequest::Parse(method, object, "acct"));
  }
  EXPECT_FALSE(AzureSourceHttpRequest::Parse(
      "HEAD", "https://acct.dfs.core.windows.net/tables/?action=getAccessControl", "acct"));
  EXPECT_FALSE(AzureSourceHttpRequest::Parse("GET", object + "?sig=secret", "acct"));
  for (const auto* query : {
           "comp=list&restype=container&prefix=team/table",
           "comp=list&restype=container&prefix=team/table/&prefix=private/",
           "comp=list&restype=container&prefix=team/table/&%70refix=private/",
           "comp=list&restype=container&prefix=team/table/&sig=secret",
           "comp=list&restype=container&prefix=team/table/&directory=private",
           "comp=list&restype=container&prefix=team/table/&include=deleted",
           "comp=list&restype=container&prefix=team/table/&delimiter=x",
           "comp=list&restype=container&prefix=team/table/&",
           "comp=list&restype=container&prefix=team/table/%2e%2e/",
           "comp=list&restype=container&prefix=team%5Ctable/",
           "comp=list&restype=container&prefix=team+table/",
           "comp=list&restype=container&prefix=team/%00/",
           "comp=list&restype=container&prefix=team/%xy/",
           "comp=list&restype=container&prefix=team//table/",
           "resource=filesystem&directory=team/table&recursive=true",
       }) {
    EXPECT_FALSE(
        AzureSourceHttpRequest::Parse("GET", "https://acct.blob.core.windows.net/tables?" + std::string(query), "acct"))
        << query;
  }
  EXPECT_FALSE(AzureSourceHttpRequest::Parse(
      "GET", "https://acct.dfs.core.windows.net/tables?comp=list&restype=container&prefix=team/table/", "acct"));
}

TEST(SourceReadScope, HttpRequestsKeepAccountCloudAndDecodeDirectoryExactlyOnce) {
  const auto request =
      AzureSourceHttpRequest::Parse("GET",
                                    "https://acct.dfs.core.usgovcloudapi.net/tables?resource=filesystem&directory=team/"
                                    "%252e%252e/a%2Bb%20c&recursive=true",
                                    "acct");
  ASSERT_TRUE(request);
  EXPECT_EQ(request->resource.CanonicalUri(),
            "https://acct.blob.core.usgovcloudapi.net/tables/team/%252e%252e/a%2Bb%20c");
  const auto different_cloud = AzureSourceResource::Parse("https://acct.blob.core.windows.net/tables/team", "acct");
  ASSERT_TRUE(different_cloud);
  EXPECT_FALSE(FindSourceRoot({*different_cloud}, request->resource));
  EXPECT_FALSE(AzureSourceHttpRequest::Parse(
      "GET", "https://other.dfs.core.windows.net/tables?resource=filesystem&directory=team&recursive=true", "acct"));
  for (const auto* method : {"GET", "HEAD"}) {
    const auto read =
        AzureSourceHttpRequest::Parse(method, "https://acct.blob.core.windows.net/tables/team/a%20b", "acct");
    ASSERT_TRUE(read);
    EXPECT_FALSE(read->is_list);
    EXPECT_EQ(read->resource.path, (std::vector<std::string>{"team", "a b"}));
  }
}

}  // namespace
}  // namespace milvus_storage
