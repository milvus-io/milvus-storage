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

"""Count throw lines, recognizing only the two reviewed Azure SDK boundaries.

Input is the ratchet's GNU-preprocessed source. The small lexer also excludes
comments in unit-test inputs and keeps strings opaque for scope matching. This
is deliberately not a general exception annotation or file exclusion facility.
See error_handling_sdk_exceptions.md for the contract and containment evidence.
"""

import re
import sys
from collections import Counter


AZURE_PATH = "cpp/src/filesystem/azure/refreshing_token_credential.cpp"
_LEX = re.compile(
    r'(?P<raw>(?:u8|u|U|L)?R"(?P<delimiter>[^ ()\\\t\r\n]{0,16})\(.*?\)(?P=delimiter)")'
    r'|(?P<string>(?:u8|u|U|L)?"(?:\\.|[^"\\])*"|(?:u8|u|U|L)?\'(?:\\.|[^\'\\])*\')'
    r"|(?P<comment>//[^\n]*|/\*.*?\*/)"
    r"|(?P<word>[A-Za-z_][A-Za-z_0-9]*)|(?P<space>\s+)|(?P<symbol><%|%>|.)",
    re.DOTALL,
)


def _lex(source):
    tokens, offsets, clean = [], [], []
    for match in _LEX.finditer(source):
        value = match.group()
        if match.lastgroup == "comment":
            clean.append(re.sub(r"[^\n]", " ", value))
        else:
            clean.append(value)
            if match.lastgroup != "space":
                # C++ brace digraphs must not hide a nested callable scope.
                tokens.append({"<%": "{", "%>": "}"}.get(value, value))
                offsets.append(match.start())
    return tokens, offsets, "".join(clean)


def _tokens(source):
    return _lex(source)[0]


def _body_ranges(tokens, signature):
    """Find balanced bodies with this exact declaration, ignoring literals."""
    prefix = _tokens(signature) + ["{"]
    for start in range(len(tokens) - len(prefix) + 1):
        if tokens[start:start + len(prefix)] != prefix:
            continue
        opening = start + len(prefix) - 1
        depth = 1
        for end in range(opening + 1, len(tokens)):
            depth += (tokens[end] == "{") - (tokens[end] == "}")
            if depth == 0:
                yield opening, end
                break


_CONTEXT = (
    "const Azure::Core::Credentials::TokenRequestContext& request, "
    "const Azure::Core::Context& context"
)
_CLASS = "class AzureStorageCredential final : public Azure::Core::Credentials::TokenCredential"
_STORAGE_METHOD = f"AccessToken GetToken({_CONTEXT}) const override"
_RAW_METHOD = f"AccessToken RefreshingAzureCredential::GetToken({_CONTEXT}) const"
_STORAGE_CATCH = "catch (const Azure::Core::Credentials::AuthenticationException& error)"
_STORAGE_THROW = "throw Azure::Storage::StorageException(error.what());"
_RAW_THROW = (
    'throw Azure::Core::Credentials::AuthenticationException('
    '"Azure credential refresh failed; no usable cached token");'
)


def count_throw_lines(path, source):
    tokens, offsets, clean = _lex(source)
    occurrences = list(re.finditer(r"\bthrow\b", clean))
    lines = Counter(clean.count("\n", 0, item.start()) for item in occurrences)
    if path != AZURE_PATH:
        return len(lines)

    def openings(signature):
        return {begin for begin, _ in _body_ranges(tokens, signature)}

    namespace = openings("namespace milvus_storage::fs")
    # Bind the full lexical scope, not just an outer method's body range: a
    # nested lambda/local class could execute outside the reviewed catch path.
    storage_scopes = [namespace, openings("namespace"), openings(_CLASS),
                      openings(_STORAGE_METHOD), openings(_STORAGE_CATCH)]
    raw_scopes = [namespace, openings(_RAW_METHOD)]
    parents, stack = {}, []
    for index, token in enumerate(tokens):
        if token == "throw":
            parents[index] = tuple(stack)
        elif token == "{":
            stack.append(index)
        elif token == "}" and stack:
            stack.pop()

    permitted = Counter()
    for statement, scopes in ((_STORAGE_THROW, storage_scopes), (_RAW_THROW, raw_scopes)):
        expected = _tokens(statement)
        matches = [i for i, enclosing in parents.items()
                   if len(enclosing) == len(scopes)
                   and all(parent in scope for parent, scope in zip(enclosing, scopes))
                   and tokens[i:i + len(expected)] == expected]
        # A duplicate invalidates this allowance entirely, even on one line.
        if len(matches) == 1:
            permitted[clean.count("\n", 0, offsets[matches[0]])] += 1

    # Preserve the old per-line count. Never erase an additional throw/rethrow
    # or even a literal containing "throw" that shares an allowed site's line.
    return sum(count != permitted[line] for line, count in lines.items())


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: error_handling_sdk_exceptions.py <repo-relative-path>")
    print(count_throw_lines(sys.argv[1], sys.stdin.read()))
