#!/usr/bin/env python3
# Copyright 2026 Zilliz
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Classifier regressions; no compiler, third-party packages, or network."""

import pathlib
import unittest

from error_handling_sdk_exceptions import AZURE_PATH, count_throw_lines


class SdkExceptionAllowanceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = pathlib.Path(__file__).resolve().parents[2]
        cls.source = (root / AZURE_PATH).read_text()

    def count(self, source, path=AZURE_PATH):
        return count_throw_lines(path, source)

    def test_actual_two_reviewed_sites(self):
        self.assertEqual(self.count(self.source), 0)

    def test_moved_file_is_not_allowed(self):
        self.assertEqual(self.count(self.source, "cpp/src/other.cpp"), 2)

    def test_moved_member_is_not_allowed(self):
        self.assertEqual(self.count(self.source.replace(
            "RefreshingAzureCredential::GetToken", "OtherCredential::GetToken")), 1)
        self.assertEqual(self.count(self.source.replace(
            "class AzureStorageCredential final", "class OtherCredential final")), 1)

    def test_moved_namespace_is_not_allowed(self):
        self.assertEqual(self.count(self.source.replace(
            "namespace milvus_storage::fs", "namespace other")), 2)
        self.assertEqual(self.count(self.source.replace(
            "namespace milvus_storage::fs", "namespace other /* namespace milvus_storage::fs */")), 2)

    def test_nested_callable_cannot_inherit_allowance(self):
        for statement in (
                "throw Azure::Storage::StorageException(error.what());",
                'throw Azure::Core::Credentials::AuthenticationException('
                '"Azure credential refresh failed; no usable cached token");'):
            for nested in (
                    "std::thread([&] { " + statement + " }).detach(); return {};",
                    "std::thread([&] <% " + statement + " %>).detach(); return {};",
                    "struct Local { void Run() { " + statement + " } };",
                    "if (condition) { " + statement + " }"):
                with self.subTest(nested=nested):
                    self.assertEqual(self.count(self.source.replace(statement, nested)), 1)

    def test_moved_outside_reviewed_catch_is_not_allowed(self):
        self.assertEqual(self.count(self.source.replace(
            "catch (const Azure::Core::Credentials::AuthenticationException& error)",
            "catch (const std::exception& error)")), 1)

    def test_extra_throw_or_rethrow_is_not_allowed(self):
        for addition in ("throw;", 'throw std::runtime_error("error");',
                         'throw Azure::Storage::StorageException("error");'):
            with self.subTest(addition=addition):
                self.assertEqual(self.count(self.source + "\n" + addition), 1)

    def test_same_line_addition_cannot_hide(self):
        statement = "throw Azure::Storage::StorageException(error.what());"
        self.assertEqual(self.count(self.source.replace(
            statement, statement + " throw;")), 1)
        self.assertEqual(self.count(self.source.replace(
            statement, statement + ' Log("throw");')), 1)

    def test_duplicate_same_type_is_not_allowed(self):
        for statement in (
                "throw Azure::Storage::StorageException(error.what());",
                'throw Azure::Core::Credentials::AuthenticationException('
                '"Azure credential refresh failed; no usable cached token");'):
            with self.subTest(statement=statement):
                self.assertGreater(self.count(self.source.replace(
                    statement, statement + " " + statement)), 0)
                self.assertEqual(self.count(self.source.replace(
                    statement, statement + "\n" + statement)), 2)

    def test_wrong_type_or_expression_is_not_allowed(self):
        self.assertEqual(self.count(self.source.replace(
            "throw Azure::Storage::StorageException(error.what());", "throw;")), 1)
        self.assertEqual(self.count(self.source.replace(
            "throw Azure::Storage::StorageException(error.what());",
            'throw std::runtime_error("error");')), 1)
        self.assertEqual(self.count(self.source.replace(
            "StorageException(error.what())", 'StorageException("different")')), 1)

    def test_comments_cannot_authorize_a_site(self):
        # A commented allowed expression does not authorize the real rethrow.
        statement = "throw Azure::Storage::StorageException(error.what());"
        self.assertEqual(self.count(self.source.replace(
            statement, "/* " + statement + " */ throw;")), 1)
        # A fake declaration in comments cannot authorize an unrelated method.
        changed = self.source.replace(
            "RefreshingAzureCredential::GetToken",
            "OtherCredential::GetToken /* RefreshingAzureCredential::GetToken */")
        self.assertEqual(self.count(changed), 1)
        self.assertEqual(self.count(self.source + "\n// throw;\n/* throw; */"), 0)

    def test_literals_cannot_fake_function_or_class_scopes(self):
        self.assertEqual(self.count('const char* s = R"x(' + self.source + ')x";'), 2)

    def test_unrelated_files_keep_existing_line_count(self):
        source = 'throw one; throw two;\nthrow;\n// throw;\nLog("throw");'
        self.assertEqual(self.count(source, "cpp/src/other.cpp"), 3)


if __name__ == "__main__":
    unittest.main()
