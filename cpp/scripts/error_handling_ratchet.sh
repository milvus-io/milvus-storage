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
# Known counting limits (kept simple on purpose; the gate is a ratchet, not a
# semantic linter):
#   * matches inside comments/strings are counted too, so a file can mask a
#     new real site by deleting a commented one (and unrelated comment edits
#     require a baseline regen);
#   * grep -c counts matching LINES, not call sites: two sites on one line
#     count once, and a site added to an already-matching line does not trip
#     the gate.
# Migrating to clang-query later can reuse the same baseline flow.
#
# Enforcement is two-layered:
#   1. The committed baseline must EXACTLY match the current tree
#      ("regenerate on any change" keeps the file honest):
#        - count went UP   -> fix the new code (return Status, don't abort/throw)
#        - count went DOWN -> record the burn-down: run
#            cpp/scripts/error_handling_ratchet.sh update
#          and commit the regenerated baseline in the same PR.
#   2. In CI, per-category TOTALS are additionally compared against the PR's
#      BASE branch baseline and must not increase. This closes the
#      self-reference loophole: regenerating a *raised* baseline inside the
#      same PR keeps layer 1 green but fails layer 2 — the ratchet direction
#      is machine-enforced, not review-enforced. (Totals rather than per-file,
#      so moving grandfathered code between files stays neutral.)
#
# Usage:
#   error_handling_ratchet.sh check [base_baseline.tsv]
#   error_handling_ratchet.sh update

set -euo pipefail

cd "$(dirname "$0")/../.."  # repo root
BASELINE="cpp/scripts/error_handling_baseline.tsv"
MODE="${1:-check}"
BASE_BASELINE="${2:-}"

collect() {
  git -c core.quotePath=false ls-files -z -- 'cpp/src' 'cpp/include' \
    | while IFS= read -r -d '' f; do
        case "$f" in
          *.cpp | *.cc | *.h | *.hpp) ;;
          *) continue ;;
        esac
        abort_n=$(grep -cE 'ValueOrDie\(|ValueUnsafe\(|MoveValueUnsafe\(' "$f" || true)
        throw_n=$(grep -cE '\bthrow\b' "$f" || true)
        if [ "$abort_n" -gt 0 ]; then printf 'abort\t%s\t%s\n' "$f" "$abort_n"; fi
        if [ "$throw_n" -gt 0 ]; then printf 'throw\t%s\t%s\n' "$f" "$throw_n"; fi
      done | LC_ALL=C sort
}

summarize() {
  awk -F'\t' '{sum[$1]+=$3} END {for (c in sum) printf "  %s: %d\n", c, sum[c]}' "$1" | LC_ALL=C sort
}

# Compare per-category totals of $2 (current) against $1 (base baseline);
# fail if any category's total increased.
check_totals_against_base() {
  local base_file="$1" current_file="$2"
  local violations
  violations=$(awk -F'\t' '
    FNR == NR { base[$1] += $3; next }
    { cur[$1] += $3 }
    END {
      for (c in cur) if (cur[c] > base[c] + 0)
        printf "  %s: %d (base) -> %d (this PR)\n", c, base[c] + 0, cur[c]
    }' "$base_file" "$current_file")
  if [ -n "$violations" ]; then
    echo "error-handling ratchet: category totals INCREASED versus the base branch:" >&2
    echo "$violations" >&2
    echo >&2
    echo "New abort/throw sites were added to library code. Raising the committed" >&2
    echo "baseline cannot pass this check: totals are compared against the BASE" >&2
    echo "branch's baseline. Return arrow::Status/arrow::Result instead; tag with" >&2
    echo "ExtendStatusDetail where classification matters." >&2
    return 1
  fi
  return 0
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

    # Layer 2 (CI): the ratchet direction, enforced against the base branch.
    if [ -n "$BASE_BASELINE" ] && [ -s "$BASE_BASELINE" ]; then
      check_totals_against_base "$BASE_BASELINE" "$current"
    fi

    # Layer 1: the committed baseline must match the tree exactly.
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
      echo "  where classification matters); raising the baseline will not pass CI," >&2
      echo "  which also compares category totals against the base branch." >&2
      echo "If a count went DOWN: thanks for the burn-down - record it by running" >&2
      echo "  cpp/scripts/error_handling_ratchet.sh update" >&2
      echo "  and committing the regenerated baseline in this PR." >&2
      exit 1
    fi
    ;;
  *)
    echo "usage: $0 [check [base_baseline.tsv]|update]" >&2
    exit 2
    ;;
esac
