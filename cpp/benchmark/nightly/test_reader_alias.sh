#!/usr/bin/env bash
set -euo pipefail

if (( $# < 1 || $# > 2 )); then
  printf 'usage: %s <benchmark-binary> [registration|smoke]\n' "$0" >&2
  exit 2
fi

binary=$1
mode=${2:-registration}
if [[ ! -x $binary ]]; then
  printf 'benchmark binary is not executable: %s\n' "$binary" >&2
  exit 1
fi

temp_dir=$(mktemp -d "${TMPDIR:-/tmp}/reader-benchmark-alias.XXXXXX")
trap 'rm -rf "$temp_dir"' EXIT

all_tests="$temp_dir/all-tests.txt"
expected_file="$temp_dir/expected.txt"
canonical_file="$temp_dir/canonical.txt"
alias_file="$temp_dir/aliases.txt"

"$binary" --benchmark_list_tests=true >"$all_tests"

datasets=(
  SyntheticSmall
  SyntheticMedium
  SyntheticLarge
  ScalarMedium
  RandomVector64MiB
  LowEntropyVector256MiB
  RandomVector2GiB
)
sync_formats=(Parquet Vortex Lance)
sync_operations=(RecordBatchRead ChunkRead Take)
async_formats=(Parquet Vortex)
async_operations=(ChunkRead Take)

if [[ $mode == smoke ]]; then
  canonical_json="$temp_dir/canonical.json"
  canonical_log="$temp_dir/canonical.log"
  alias_json="$temp_dir/aliases.json"
  alias_log="$temp_dir/aliases.log"

  if ! "$binary" \
    --benchmark_filter='^ReaderBenchmark/(Sync|Async)/.*/.*/SyntheticSmall' \
    --benchmark_repetitions=1 \
    --benchmark_min_time=0.01s \
    --benchmark_out="$canonical_json" \
    --benchmark_out_format=json >"$canonical_log" 2>&1; then
    sed -n '1,240p' "$canonical_log" >&2
    exit 1
  fi
  if ! "$binary" \
    --benchmark_filter='^NIGHTLY_CI_TARGET/(Sync|Async)/.*/.*/SyntheticSmall' \
    --benchmark_repetitions=1 \
    --benchmark_min_time=0.01s \
    --benchmark_out="$alias_json" \
    --benchmark_out_format=json >"$alias_log" 2>&1; then
    sed -n '1,240p' "$alias_log" >&2
    exit 1
  fi

  jq -s -e --arg executor_threads "${NIGHTLY_CI_EXECUTOR_THREADS:-1}" '
    def normalized($document; $prefix):
      [$document.benchmarks[]
       | select((.run_type // "iteration") == "iteration")
       | {
           suffix: ((.run_name // .name)
                    | sub("/real_time$"; "")
                    | sub("^" + $prefix; "")),
           label: (.label // ""),
           error_occurred: (.error_occurred // false),
           error_message: (.error_message // ""),
           # Throughput units depend on the Google Benchmark adaptive iteration
           # count, so compare the counter kind rather than its rendered unit.
           counter_keys: ([keys[]
                           | select(. == "rows/s"
                                    or . == "rows_taken"
                                    or . == "executor_threads"
                                    or startswith("throughput("))
                           | if startswith("throughput(") then "throughput" else . end]
                          | unique
                          | sort),
           rows_taken: (.rows_taken // null),
           executor_threads: (.executor_threads // null)
         }]
      | sort_by(.suffix);

    normalized(.[0]; "ReaderBenchmark/") as $canonical
    | normalized(.[1]; "NIGHTLY_CI_TARGET/") as $aliases
    | {
        canonical_count: ($canonical | length),
        alias_count: ($aliases | length),
        identical_stable_results: ($canonical == $aliases),
        take_count: ([$canonical[] | select(.rows_taken == 1000)] | length),
        async_count: ([$canonical[]
                       | select(.executor_threads == ($executor_threads | tonumber))]
                      | length),
        error_count: ([$canonical[], $aliases[]
                       | select(.error_occurred == true
                                or (.error_message | test("cleanup"; "i")))]
                      | length)
      }
    | select(.canonical_count == 13
             and .alias_count == 13
             and .identical_stable_results
             and .take_count == 5
             and .async_count == 4
             and .error_count == 0)
  ' "$canonical_json" "$alias_json"

  if grep -qi 'Reader benchmark cleanup failed' "$canonical_log" "$alias_log"; then
    printf 'Reader benchmark cleanup failed during alias smoke\n' >&2
    exit 1
  fi
  exit 0
fi

if [[ $mode != registration ]]; then
  printf 'unknown mode: %s\n' "$mode" >&2
  exit 2
fi

for dataset in "${datasets[@]}"; do
  for format in "${sync_formats[@]}"; do
    for operation in "${sync_operations[@]}"; do
      printf 'Sync/%s/%s/%s\n' "$operation" "$format" "$dataset"
    done
  done
  for format in "${async_formats[@]}"; do
    for operation in "${async_operations[@]}"; do
      printf 'Async/%s/%s/%s\n' "$operation" "$format" "$dataset"
    done
  done
done | sort >"$expected_file"

awk '/^ReaderBenchmark\// {
  sub(/^ReaderBenchmark\//, "")
  sub(/\/real_time$/, "")
  print
}' "$all_tests" | sort >"$canonical_file"

awk '/^NIGHTLY_CI_TARGET\// {
  sub(/^NIGHTLY_CI_TARGET\//, "")
  sub(/\/real_time$/, "")
  print
}' "$all_tests" | sort >"$alias_file"

assert_exact_matrix() {
  local label=$1 file=$2 count unique_count
  count=$(awk 'END {print NR + 0}' "$file")
  unique_count=$(sort -u "$file" | awk 'END {print NR + 0}')
  if (( count != 91 || unique_count != 91 )); then
    printf '%s registrations: expected 91 unique entries, found %s entries/%s unique\n' \
      "$label" "$count" "$unique_count" >&2
    return 1
  fi
  if grep -q 'Iceberg' "$file"; then
    printf '%s registrations unexpectedly contain Iceberg\n' "$label" >&2
    return 1
  fi
  diff -u "$expected_file" "$file"
}

assert_exact_matrix canonical "$canonical_file"
assert_exact_matrix alias "$alias_file"
diff -u "$canonical_file" "$alias_file"

if grep -q '^Typical/' "$all_tests"; then
  printf 'removed Typical registrations are still present\n' >&2
  exit 1
fi

printf 'reader benchmark registration contract passed: canonical=91 aliases=91\n'
