#!/usr/bin/env bash
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

# Error-handling ratchet.
#
# Library code must report failures as arrow::Status / arrow::Result (with an
# ExtendStatusDetail tag where classification matters), not by aborting the
# process or throwing exceptions across the library boundary. Existing
# violations are grandfathered in a checked-in baseline and burned down over
# time; this script keeps the count from ever growing back.
#
# Categories counted (per line, per file):
#   abort : ValueOrDie( / ValueUnsafe( / MoveValueUnsafe(
#           An unguarded call aborts the whole process; even guarded calls are
#           counted so the baseline shrinks monotonically as sites migrate to
#           ARROW_ASSIGN_OR_RAISE.
#   throw : any `throw` statement (raw std exceptions, bridge exception types,
#           rethrows). Ring-1 discipline: the library must not leak exceptions.
#
# Scope: git-tracked *.cpp / *.cc / *.h / *.hpp under cpp/src and cpp/include.
# cpp/test is out of scope; vendored or generated trees are excluded simply by
# not being git-tracked.
#
# The baseline must match EXACTLY:
#   - a count went UP   -> fix the new code (return Status, don't abort/throw)
#   - a count went DOWN -> good; record the burn-down in the same PR:
#                            cpp/scripts/error_handling_ratchet.sh update
#
# Usage:
#   error_handling_ratchet.sh check    # (default) diff against the baseline
#   error_handling_ratchet.sh update   # regenerate the baseline

set -euo pipefail

cd "$(dirname "$0")/../.."  # repo root
BASELINE="cpp/scripts/error_handling_baseline.tsv"
MODE="${1:-check}"

collect() {
  git ls-files -- 'cpp/src' 'cpp/include' \
    | grep -E '\.(cpp|cc|h|hpp)$' \
    | while IFS= read -r f; do
        abort_n=$(grep -cE 'ValueOrDie\(|ValueUnsafe\(|MoveValueUnsafe\(' "$f" || true)
        throw_n=$(grep -cE '\bthrow\b' "$f" || true)
        if [ "$abort_n" -gt 0 ]; then printf 'abort\t%s\t%s\n' "$f" "$abort_n"; fi
        if [ "$throw_n" -gt 0 ]; then printf 'throw\t%s\t%s\n' "$f" "$throw_n"; fi
      done | LC_ALL=C sort
}

summarize() {
  awk -F'\t' '{sum[$1]+=$3} END {for (c in sum) printf "  %s: %d\n", c, sum[c]}' "$1" | LC_ALL=C sort
}

case "$MODE" in
  update)
    collect > "$BASELINE"
    echo "Baseline regenerated: $BASELINE"
    summarize "$BASELINE"
    ;;
  check)
    if [ ! -f "$BASELINE" ]; then
      echo "error: baseline $BASELINE not found; run '$0 update' and commit it" >&2
      exit 1
    fi
    current="$(mktemp)"
    trap 'rm -f "$current"' EXIT
    collect > "$current"
    if diff -u "$BASELINE" "$current" > /dev/null; then
      echo "error-handling ratchet: OK"
      summarize "$BASELINE"
    else
      echo "error-handling ratchet: counts diverged from $BASELINE" >&2
      echo >&2
      diff -u "$BASELINE" "$current" >&2 || true
      echo >&2
      echo "If a count went UP: new abort/throw sites were added to library code." >&2
      echo "  Return arrow::Status/arrow::Result instead (tag with ExtendStatusDetail" >&2
      echo "  where classification matters); do not raise the baseline." >&2
      echo "If a count went DOWN: thanks for the burn-down - record it by running" >&2
      echo "  cpp/scripts/error_handling_ratchet.sh update" >&2
      echo "  and committing the regenerated baseline in this PR." >&2
      exit 1
    fi
    ;;
  *)
    echo "usage: $0 [check|update]" >&2
    exit 2
    ;;
esac
