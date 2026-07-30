#!/usr/bin/env python3
# Copyright 2025 Zilliz
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Guards the error taxonomy against the two ways it rots.

1. **Documentation drift.** `docs/error-codes.md` restates every code's
   category, retriability, S3 name and segcore code. Those four columns are
   transcribed from the source, so they drift silently -- and did: the table
   still described three categories after the code moved on. Here they are
   re-derived from the X-macro table and the `ToSegcoreErrorCode` switch and
   compared. `--fix` rewrites them.

2. **Dead codes.** A code that no layer ever emits is worse than no code: it
   makes the taxonomy look more complete than it is, and it passes every
   self-consistency test because those only compare the tables to each other.
   Two codes shipped in exactly that state. Here every ExtendStatusCode must
   have at least one producer in `cpp/src` outside the files that define and
   translate the taxonomy itself.

Run with no arguments to check, `--fix` to rewrite the doc table.
"""

from __future__ import annotations

import argparse
import collections
import re
import subprocess
import sys
from pathlib import Path

# repo root: cpp/scripts/ -> cpp/ -> root
ROOT = Path(__file__).resolve().parents[2]
HEADER = ROOT / "cpp/include/milvus-storage/ffi_internal/ffi_error_code.h"
MAPPING = ROOT / "cpp/src/common/extend_status.cpp"
DOC = ROOT / "docs/error-codes.md"
SRC = ROOT / "cpp/src"

# Files that mention every code by construction: the table itself, the
# translation switches, and the FFI constant exports. A mention here is not a
# producer.
NON_PRODUCER = {
    "cpp/src/common/extend_status.cpp",
    "cpp/src/ffi/result_c.cpp",
    "cpp/include/milvus-storage/ffi_internal/ffi_error_code.h",
    "cpp/include/milvus-storage/ffi_c.h",
}

RETRYABLE_CATEGORIES = {"TRANSIENT", "CONFLICT"}

# milvus::ErrorCode numeric values, from milvus-common's include/common/EasyAssert.h.
# Duplicated here because that header lives in another repo and arrives via conan;
# the C++ test asserts the symbols, this asserts the numbers the doc prints.
SEGCORE_VALUES = {
    "ConfigInvalid": 2006,
    "BucketInvalid": 2016,
    "ObjectNotExist": 2017,
    "DataFormatBroken": 2024,
    "InvalidParameter": 2042,
    "StorageError": 2044,
    "StorageTransientError": 2045,
}

# Codes that legitimately have no C++ producer, with the reason. Anything added
# here needs a justification that survives review; the default answer to "no
# producer" is to wire one up or delete the code.
PRODUCERLESS_ALLOWED: dict[str, str] = {}


def parse_header() -> list[dict]:
    """Pull (name, category, s3) out of the two X-macro lists."""
    text = HEADER.read_text()
    # Join line continuations so a wrapped X(...) entry is one line.
    flat = re.sub(r"\\\s*\n\s*", " ", text)
    # Strip block comments; they contain prose that looks like arguments.
    flat = re.sub(r"/\*.*?\*/", " ", flat, flags=re.S)

    codes = []
    # The two lists differ in their first column: internal codes lead with a
    # quoted display name, ExtendStatusCode entries lead with the enum name.
    pattern = re.compile(
        r'X\(\s*(?:"(?P<display>[^"]*)"|(?P<enum>[A-Za-z_][A-Za-z0-9_]*))\s*,\s*'
        r"(?P<value>[A-Z0-9_]+|\d+)\s*,\s*"  # numeric code or LOON_* macro
        r"(?P<symbol>[a-z0-9_]+)\s*,\s*"  # exported loon_errcode_* symbol
        r"LOON_ERROR_CATEGORY_(?P<category>[A-Z]+)\s*,\s*"
        r'"(?P<s3>[^"]*)"\s*\)'
    )
    for block_name in ("LOON_INTERNAL_ERROR_CODE_LIST", "LOON_EXTEND_STATUS_CODE_LIST"):
        start = flat.index(f"#define {block_name}(X)")
        end = flat.index("#define ", start + 10) if "#define " in flat[start + 10 :] else len(flat)
        for m in pattern.finditer(flat[start:end]):
            codes.append(
                {
                    "name": m.group("enum") or m.group("display"),
                    "raw_value": m.group("value"),
                    "symbol": m.group("symbol"),
                    "category": m.group("category"),
                    "s3": m.group("s3"),
                    # Internal codes have no enum: the doc row and every producer
                    # refer to them by the LOON_* macro instead.
                    "key": m.group("enum") or m.group("value"),
                    "list": block_name,
                }
            )
    if not codes:
        sys.exit("check_error_table: parsed zero codes -- the X-macro format changed")
    return codes


def resolve_values(codes: list[dict]) -> None:
    """Resolve `LOON_FOO` macro names to their integer values."""
    text = HEADER.read_text()
    defines = dict(re.findall(r"#define\s+(LOON_[A-Z0-9_]+)\s+(-?\d+)\s*$", text, re.M))
    for c in codes:
        if c["raw_value"].isdigit() or c["raw_value"].lstrip("-").isdigit():
            c["value"] = int(c["raw_value"])
        elif c["raw_value"] in defines:
            c["value"] = int(defines[c["raw_value"]])
        else:
            sys.exit(f"check_error_table: cannot resolve {c['raw_value']} for {c['name']}")


def parse_segcore_mapping() -> dict[str, str]:
    """Read `ToSegcoreErrorCode`: which milvus::ErrorCode each code becomes.

    Fall-through `case` labels share the `return` below them, which is exactly
    the shape a naive line-by-line reader gets wrong.
    """
    text = MAPPING.read_text()
    start = text.index("milvus::ErrorCode ToSegcoreErrorCode(")
    body = text[start : text.index("\n}\n", start)]
    body = re.sub(r"//[^\n]*", "", body)

    mapping: dict[str, str] = {}
    pending: list[str] = []
    for line in body.splitlines():
        line = line.strip()
        case = re.match(r"case ExtendStatusCode::([A-Za-z0-9_]+):", line)
        if case:
            pending.append(case.group(1))
            continue
        ret = re.match(r"return milvus::([A-Za-z0-9_]+);", line)
        if ret and pending:
            for name in pending:
                mapping[name] = ret.group(1)
            pending = []
    return mapping


def find_producers(codes: list[dict]) -> dict[str, list[str]]:
    """Locate every site that emits each code, ignoring the taxonomy's own files."""
    keys = [c["key"] for c in codes]
    try:
        proc = subprocess.run(
            # Producers live in cpp/src and in header-only classifiers such as
            # filesystem/s3/s3_internal.h, so both trees are in scope. Internal
            # codes are emitted as RETURN_ERROR(LOON_X, ...), ExtendStatusCodes
            # as MakeExtendError(ExtendStatusCode::X, ...) -- one grep, both.
            ["git", "grep", "-n", "-E", "ExtendStatusCode::|LOON_[A-Z0-9_]+", "--", "cpp/src", "cpp/include"],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        sys.exit("check_error_table: git not available")
    if proc.returncode not in (0, 1):
        sys.exit(f"check_error_table: git grep failed: {proc.stderr.strip()}")

    producers: dict[str, list[str]] = {k: [] for k in keys}
    for line in proc.stdout.splitlines():
        parts = line.split(":", 2)
        if len(parts) < 3:
            continue
        path, lineno, rest = parts
        if path in NON_PRODUCER:
            continue
        # A `case` label is a consumer of the code, not a producer of it. Without
        # this the check passes on any code that is merely translated somewhere.
        if re.match(r"\s*case\s+(ExtendStatusCode::|LOON_)", rest):
            continue
        for m in re.finditer(r"(?:ExtendStatusCode::([A-Za-z0-9_]+)|\b(LOON_[A-Z0-9_]+)\b)", rest):
            key = m.group(1) or m.group(2)
            if key in producers:
                producers[key].append(f"{path}:{lineno}")
    return producers


def doc_rows() -> dict[str, tuple[str, str]]:
    """Existing table rows, keyed by code name -> (full line, leading match)."""
    rows = {}
    for line in DOC.read_text().splitlines():
        m = re.match(r"\|\s*`([A-Za-z0-9_]+)`\s*\|", line)
        if m:
            rows[m.group(1)] = line
    return rows


def render_cells(c: dict, segcore: dict[str, str]) -> tuple[str, str, str, str]:
    cat = c["category"].capitalize()
    retryable = c["category"] in RETRYABLE_CATEGORIES
    # Bold the surprising half so a reader scanning the table sees the
    # exceptions rather than having to compare every row.
    cat_cell = f"**{cat}**" if retryable or c["category"] == "USER" else cat
    retry_cell = "**yes**" if retryable else "no"
    s3_cell = f"`{c['s3']}`" if c["s3"] else "—"
    seg = segcore.get(c["name"])
    seg_cell = f"`{seg}`" if seg else "?"
    return cat_cell, retry_cell, s3_cell, seg_cell


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fix", action="store_true", help="rewrite the doc table in place")
    args = ap.parse_args()

    codes = parse_header()
    resolve_values(codes)
    segcore = parse_segcore_mapping()
    producers = find_producers(codes)
    rows = doc_rows()

    problems: list[str] = []

    # --- 0. no duplicated exported symbol, in any of the four hand-kept lists ---
    #
    # Merging two codes is a DELETE plus a KEEP. Doing it as a rename instead
    # produces a verbatim duplicate line, which is a compile error in the C
    # definitions and silent drift everywhere else -- that is exactly how this
    # check came to exist. The four lists cannot be macro-generated (different
    # languages and formats), so they get a gate.
    symbol_lists = {
        "cpp/ffi_exports.map": r"^\s*(loon_[a-z0-9_]+);",
        "cpp/ffi_exports_mac.map": r"^\s*_(loon_[a-z0-9_]+)\s*$",
        # Only the category constants here: the loon_errcode_* declarations are
        # generated from the X-macro table and a regex cannot see them, which is
        # fine -- generated declarations cannot drift. The categories are still
        # hand-written, so they can, and did.
        "cpp/include/milvus-storage/ffi_c.h": r"^FFI_EXPORT extern const int (loon_error_category_[a-z0-9_]+);",
        "python/milvus_storage/_ffi.py": r'^\s*"(loon_[a-z0-9_]+)",\s*$',
    }
    seen_per_list = {}
    for rel, pattern in symbol_lists.items():
        path = ROOT / rel
        if not path.exists():
            problems.append(f"{rel} is missing -- the symbol lists moved without updating this gate")
            continue
        names = re.findall(pattern, path.read_text(), re.M)
        seen_per_list[rel] = set(names)
        for name, n in collections.Counter(names).items():
            if n > 1:
                problems.append(
                    f"{rel} lists {name} {n} times. Merging two codes means deleting one, "
                    f"not renaming it onto the other."
                )

    # The four lists must agree on the TAXONOMY surface. They deliberately do
    # not agree on everything -- the linker maps export every loon_* function
    # too, and the python binding only cdefs what it uses -- so compare just the
    # error codes and categories, which is the part this PR owns and the part a
    # merge can silently unbalance.
    def taxonomy_only(names):
        return {n for n in names if n.startswith(("loon_errcode_", "loon_error_category_"))}

    # ffi_c.h is excluded from the cross-list comparison on purpose: its
    # loon_errcode_* declarations are generated from the X-macro table, so a
    # regex cannot see them -- and generated declarations cannot drift, which is
    # why they were generated. It stays in the duplicate check above, because
    # its category constants are still written by hand and that is exactly where
    # a merge-by-rename left a duplicate.
    if len(seen_per_list) == len(symbol_lists):
        reference = taxonomy_only(seen_per_list["cpp/ffi_exports.map"])
        for rel, names in seen_per_list.items():
            if rel == "cpp/ffi_exports.map":
                continue
            # ffi_c.h only contributes its hand-written category constants, so
            # compare it against that slice rather than the whole surface.
            if rel == "cpp/include/milvus-storage/ffi_c.h":
                reference_here = {n for n in reference if n.startswith("loon_error_category_")}
                missing = reference_here - names
                extra = names - reference_here
                if missing:
                    problems.append(f"{rel} is missing category constants {sorted(missing)}")
                if extra:
                    problems.append(f"{rel} declares category constants {sorted(extra)} that are not exported")
                continue
            mine = taxonomy_only(names)
            missing = reference - mine
            extra = mine - reference
            if missing:
                problems.append(
                    f"{rel} is missing {sorted(missing)}, which ffi_exports.map exports. "
                    f"An exported-but-unbound code is invisible to that consumer."
                )
            if extra:
                problems.append(
                    f"{rel} names {sorted(extra)}, which ffi_exports.map does not export. "
                    f"That fails to link."
                )

    # --- 1. every ExtendStatusCode needs a producer ---
    for c in codes:
        if producers[c["key"]]:
            continue
        if c["key"] in PRODUCERLESS_ALLOWED:
            continue
        problems.append(
            f"{c['key']} ({c['value']}) has no producer: nothing under cpp/src emits it.\n"
            f"    Either wire it up at the layer that detects the condition, or drop the code.\n"
            f"    A code only the translation switch mentions is dead weight that makes the\n"
            f"    taxonomy look more complete than it is."
        )

    # --- 2. every ExtendStatusCode needs a segcore landing ---
    for c in codes:
        if c["list"] == "LOON_EXTEND_STATUS_CODE_LIST" and c["name"] not in segcore:
            problems.append(f"{c['name']} has no case in ToSegcoreErrorCode")

    # --- 3. the doc table must restate the source correctly ---
    doc_text = DOC.read_text()
    fixed = doc_text
    for c in codes:
        line = rows.get(c["key"])
        if line is None:
            problems.append(f"{c['key']} is missing from docs/error-codes.md")
            continue
        cells = [x.strip() for x in line.strip().strip("|").split("|")]
        # The internal-code table has no segcore column: those codes are FFI
        # return values, they never travel as an arrow Status into segcore.
        is_extend = c["list"] == "LOON_EXTEND_STATUS_CODE_LIST"
        expected_cols = 7 if is_extend else 6
        if len(cells) != expected_cols:
            problems.append(f"{c['key']}: doc row has {len(cells)} columns, expected {expected_cols}")
            continue
        cat_cell, retry_cell, s3_cell, seg_cell = render_cells(c, segcore)
        want = list(cells)
        want[1] = str(c["value"])
        want[2], want[3] = cat_cell, retry_cell
        # The doc may list sibling S3 codes the same condition arrives under
        # ("`NoSuchKey` / `NoSuchBucket`"); that is extra information, not drift.
        # Only require that the code the source actually carries is among them.
        want[4] = cells[4] if c["s3"] and f"`{c['s3']}`" in cells[4] else s3_cell
        if is_extend:
            if seg_cell == "?":
                problems.append(f"{c['key']}: no ToSegcoreErrorCode case, cannot check the segcore column")
            else:
                value = SEGCORE_VALUES.get(segcore[c["name"]])
                if value is None:
                    problems.append(
                        f"{c['key']}: ToSegcoreErrorCode returns milvus::{segcore[c['name']]}, "
                        f"which is not in SEGCORE_VALUES -- add it from milvus-common's EasyAssert.h"
                    )
                else:
                    want[5] = f"{value} {seg_cell}"
        if want != cells:
            new_line = "| " + " | ".join(want) + " |"
            if args.fix:
                fixed = fixed.replace(line, new_line, 1)
            else:
                names = ["Code", "Value", "Category", "Retry", "S3", "segcore", "Raised when"]
                if not is_extend:
                    names = ["Code", "Value", "Category", "Retry", "S3", "Raised when"]
                for i, (a, b) in enumerate(zip(cells, want)):
                    if a != b:
                        problems.append(f"{c['key']}: doc says {names[i]}={a!r}, source says {b!r}")

    if args.fix and fixed != doc_text:
        DOC.write_text(fixed)
        print(f"check_error_table: rewrote {DOC.relative_to(ROOT)}")
        problems = [p for p in problems if "doc says" not in p]

    if problems:
        print("check_error_table: FAILED\n", file=sys.stderr)
        for p in problems:
            print(f"  - {p}", file=sys.stderr)
        print(
            "\nThe error taxonomy is the contract milvus retries on. "
            "Fix the source, then re-run with --fix to update the table.",
            file=sys.stderr,
        )
        return 1

    n_ext = sum(1 for c in codes if c["list"] == "LOON_EXTEND_STATUS_CODE_LIST")
    print(f"check_error_table: OK -- {len(codes)} codes ({n_ext} ExtendStatusCode), all with producers and doc rows in sync")
    return 0


if __name__ == "__main__":
    sys.exit(main())
