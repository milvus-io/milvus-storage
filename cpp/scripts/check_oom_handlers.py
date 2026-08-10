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

"""Every catch-all that turns an exception into a status must answer bad_alloc first.

What this gate is FOR, and what it deliberately no longer claims
----------------------------------------------------------------
It keeps an out-of-memory failure legible AS an out-of-memory failure. It does
NOT say an OOM may be retried: LOON_MEMORY_ERROR is Permanent, because the
condition clearing says nothing about whether the request that hit it can be
sent again, and many of the entry points these handlers wrap have already
committed by the time an allocation fails. See ffi_error_code.h.

That is a narrower purpose than the one this gate was written for, and it is
still worth enforcing mechanically. An OOM swallowed by a catch-all becomes
"got exception" -- indistinguishable, in a log or a metric, from a null
dereference or a bad cast. Memory pressure is the failure most likely to arrive
in a burst across many nodes at once, and the one whose fix (give it more
memory) is unrelated to every other cause it would be filed under. Losing the
type there costs an operator the diagnosis at exactly the moment the diagnosis
is easiest to act on.

This gate exists because two softer approaches both failed.

Fixing the sites as they were reported failed: "an OOM was reclassified as a
generic internal error" came back in eleven of thirteen review rounds, at
forty-five separate places, because each fix addressed the instance named and
not the rule.

Writing a unit test that enumerates the transforms also failed, more subtly.
It listed the paths I could remember, so it passed while two async callbacks
kept doing the very thing it was written to forbid -- the same blind spot, one
level up. A list maintained by hand cannot police a codebase it does not know
about.

So the enumeration is mechanical. A generic handler -- `catch (std::exception)`
or `catch (...)` -- that produces a status is a place where the type of the
exception is about to be discarded. std::bad_alloc discarded there becomes an
unattributed internal error, and the information is not recoverable downstream.
So the rule is that such a handler must be preceded by a bad_alloc handler in
the same try block.

The check is textual, which makes it approximate in a specific direction: it
can be silenced, and the allowlist below records where and why. It cannot be
silenced by accident, which is the property that matters.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "cpp/src"

# A handler that discards the exception type but produces no status is not in
# scope: it is not classifying anything. These are the shapes that DO classify.
PRODUCES_STATUS = re.compile(
    r"return\s+(arrow::Status|MakeExtendError|Status::|.*Status\()|"
    r"RETURN_ERROR|RETURN_EXCEPTION|set_\w*callback_exception|promise\.setValue",
)

GENERIC_CATCH = re.compile(r"\bcatch\s*\(\s*(?:const\s+)?(?:std::exception\s*&|\.\.\.)")
# RETURN_EXCEPTION recovers the exception type itself, by rethrowing inside
# FFIExceptionErrorCode -- a handler that uses it has already answered
# bad_alloc, at the one choke point that covers every FFI entry point.
ANSWERS_VIA_MACRO = re.compile(r"\bRETURN_EXCEPTION\b")
BAD_ALLOC_CATCH = re.compile(r"\bcatch\s*\(\s*const\s+std::bad_alloc\s*&")
TRY_START = re.compile(r"\btry\s*\{")

# Handlers that legitimately have nothing to say about bad_alloc, with the
# reason. Anything added here needs to survive review; the default answer to a
# new hit is to add the handler, not the exemption.
ALLOWLIST: dict[str, str] = {
    # This IS the bad_alloc answer -- it recovers the type by rethrowing.
    "cpp/include/milvus-storage/ffi_internal/result.h": "FFIExceptionErrorCode rethrows to detect bad_alloc",
}


def find_violations() -> list[tuple[str, int, str]]:
    violations: list[tuple[str, int, str]] = []
    for path in sorted(SRC.rglob("*.cpp")) + sorted(SRC.rglob("*.cc")):
        rel = str(path.relative_to(ROOT))
        if rel in ALLOWLIST:
            continue
        lines = path.read_text(errors="replace").splitlines()

        # Walk each generic handler back to its try, and forward through its
        # body, to decide whether it classifies and whether bad_alloc was
        # already answered for the same try block.
        for i, line in enumerate(lines):
            if not GENERIC_CATCH.search(line):
                continue

            # Does this handler produce a status? Look at its body, which runs
            # until the brace depth returns to where the handler opened.
            depth = 0
            classifies = False
            answered_by_macro = False
            for j in range(i, min(i + 40, len(lines))):
                depth += lines[j].count("{") - lines[j].count("}")
                if j > i and ANSWERS_VIA_MACRO.search(lines[j]):
                    answered_by_macro = True
                if j > i and PRODUCES_STATUS.search(lines[j]):
                    classifies = True
                if j > i and depth <= 0:
                    break
            if not classifies or answered_by_macro:
                continue

            # Was bad_alloc handled earlier in the same try/catch chain? Scan
            # back to the try that opened it.
            handled = False
            for k in range(i - 1, max(i - 200, -1), -1):
                if BAD_ALLOC_CATCH.search(lines[k]):
                    handled = True
                    break
                if TRY_START.search(lines[k]):
                    break
            if not handled:
                violations.append((rel, i + 1, line.strip()))
    return violations


def main() -> int:
    violations = find_violations()
    if not violations:
        print("check_oom_handlers: OK -- every classifying catch-all answers bad_alloc first")
        return 0

    print("check_oom_handlers: FAILED\n")
    for rel, line_no, text in violations:
        print(f"  {rel}:{line_no}")
        print(f"      {text}")
    print(
        "\nA generic handler that returns a status discards the exception type. "
        "std::bad_alloc discarded there becomes a permanent error, and memory "
        "pressure is the one condition a retry is most likely to resolve.\n"
        "Add `catch (const std::bad_alloc&) { ... arrow::Status::OutOfMemory(...) }` "
        "before the generic handler, or -- with a reason that survives review -- "
        "add the file to ALLOWLIST in this script."
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
