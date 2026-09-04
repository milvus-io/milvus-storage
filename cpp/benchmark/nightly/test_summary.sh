#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
renderer="$script_dir/summary.jq"
temp_dir="$(mktemp -d "${TMPDIR:-/tmp}/nightly-summary.XXXXXX")"
trap 'rm -rf "$temp_dir"' EXIT

render() {
  local expected_count=${2:-5}
  jq -r \
    --arg command render \
    --arg expected_count "$expected_count" \
    --arg commit 'test-sha' \
    --arg trigger 'test' \
    --arg minio_release 'test-minio' \
    --arg minio_cpu_list '0' \
    --arg benchmark_cpu_list '1,2,3' \
    --arg executor_threads '3' \
    -f "$renderer" "$1"
}

validate() {
  jq -e \
    --arg command validate \
    --arg expected_count "$2" \
    --arg commit 'test-sha' \
    --arg trigger 'test' \
    --arg minio_release 'test-minio' \
    --arg minio_cpu_list '0' \
    --arg benchmark_cpu_list '1,2,3' \
    --arg executor_threads '3' \
    -f "$renderer" "$1" >/dev/null
}

assert_contains() {
  local haystack=$1 needle=$2
  [[ $haystack == *"$needle"* ]] || {
    printf 'expected output to contain %s\n%s\n' "$needle" "$haystack" >&2
    exit 1
  }
}

assert_validate_fails() {
  local fixture=$1 expected_count=$2 diagnostic=$3 stderr
  if stderr="$(validate "$fixture" "$expected_count" 2>&1)"; then
    printf 'expected validation failure for %s\n' "$fixture" >&2
    exit 1
  fi
  assert_contains "$stderr" "$diagnostic"
}

write_fixture() {
  local path=$1 records=$2
  printf '{"benchmarks":%s}\n' "$records" >"$path"
}

median() {
  local name=$1 time=$2 unit=$3 throughput=$4
  printf '{"run_name":"%s","run_type":"aggregate","aggregate_name":"median","real_time":%s,"time_unit":"%s","throughput(MB/s)":%s}' \
    "$name" "$time" "$unit" "$throughput"
}

cv() {
  local name=$1 value=$2
  printf '{"run_name":"%s","run_type":"aggregate","aggregate_name":"cv","real_time":%s,"time_unit":"ms"}' \
    "$name" "$value"
}

sync_parquet='NIGHTLY_CI_TARGET/Sync/RecordBatchRead/Parquet/SyntheticSmall'
sync_vortex='NIGHTLY_CI_TARGET/Sync/RecordBatchRead/Vortex/SyntheticSmall/repeats:4'
sync_lance='NIGHTLY_CI_TARGET/Sync/RecordBatchRead/Lance/SyntheticSmall'
async_parquet='NIGHTLY_CI_TARGET/Async/ChunkRead/Parquet/SyntheticSmall'
async_vortex='NIGHTLY_CI_TARGET/Async/ChunkRead/Vortex/SyntheticSmall'

success_records="[$(median "$sync_parquet" 1000000 ns 100),$(cv "$sync_parquet" 0.02),$(median "$sync_vortex" 500 us 200),$(cv "$sync_vortex" 0.11),$(median "$sync_lance" 2 ms 50),$(cv "$sync_lance" 0.01),$(median "$async_parquet" 1 s 10),$(cv "$async_parquet" 0.03),$(median "$async_vortex" 250 ms 40),$(cv "$async_vortex" 0.04)]"
success_fixture="$temp_dir/success.json"
write_fixture "$success_fixture" "$success_records"

# A missing renderer must make this real invocation fail, proving RED before implementation.
if [[ ! -e $renderer ]]; then
  if render "$success_fixture"; then
    printf 'missing renderer unexpectedly rendered output\n' >&2
    exit 1
  fi
  exit 1
fi

output="$(render "$success_fixture")"
assert_contains "$output" '## C++ Nightly Reader Benchmark'
assert_contains "$output" 'Cases: 5 completed / 5 expected'
assert_contains "$output" '### Sync comparison'
assert_contains "$output" '### Async comparison'
assert_contains "$output" '### Case details'
assert_contains "$output" 'Executor threads: `3`'
assert_contains "$output" '2.000x'
assert_contains "$output" 'WARN'
sync_section=${output#*'### Sync comparison'}
sync_section=${sync_section%%'### Async comparison'*}
assert_contains "$sync_section" '| Dataset | Operation | Parquet | Vortex | Lance |'
async_section=${output#*'### Async comparison'}
async_section=${async_section%%'### Case details'*}
assert_contains "$async_section" '| Dataset | Operation | Parquet | Vortex |'
[[ $async_section != *'Lance'* ]] || {
  printf 'Async comparison must not contain a Lance column\n%s\n' "$async_section" >&2
  exit 1
}
validate "$success_fixture" 5

malformed_fixture="$temp_dir/malformed.json"
write_fixture "$malformed_fixture" "[$(median 'not/a/nightly/target' 1 ms 1),$(cv 'not/a/nightly/target' 0.01)]"

missing_median_fixture="$temp_dir/missing-median.json"
write_fixture "$missing_median_fixture" "[$(cv "$sync_parquet" 0.01)]"

duplicate_median_fixture="$temp_dir/duplicate-median.json"
write_fixture "$duplicate_median_fixture" "[$(median "$sync_parquet" 1 ms 1),$(median "$sync_parquet" 2 ms 1),$(cv "$sync_parquet" 0.01)]"

error_fixture="$temp_dir/error.json"
write_fixture "$error_fixture" "[$(median "$sync_parquet" 1 ms 1),$(cv "$sync_parquet" 0.01),{\"run_name\":\"$sync_parquet\",\"error_occurred\":true,\"error_message\":\"read failed|retry\"}]"

missing_field_fixture="$temp_dir/missing-field.json"
write_fixture "$missing_field_fixture" "[{\"run_name\":\"$sync_parquet\",\"run_type\":\"aggregate\",\"aggregate_name\":\"median\",\"time_unit\":\"ns\"},$(cv "$sync_parquet" 0.01)]"

async_recordbatch_fixture="$temp_dir/async-recordbatch.json"
async_recordbatch='NIGHTLY_CI_TARGET/Async/RecordBatchRead/Parquet/SyntheticSmall'
write_fixture "$async_recordbatch_fixture" "[$(median "$async_recordbatch" 1 ms 1),$(cv "$async_recordbatch" 0.01)]"

async_lance_fixture="$temp_dir/async-lance.json"
async_lance='NIGHTLY_CI_TARGET/Async/ChunkRead/Lance/SyntheticSmall'
write_fixture "$async_lance_fixture" "[$(median "$async_lance" 1 ms 1),$(cv "$async_lance" 0.01)]"

split_raw_suffix_fixture="$temp_dir/split-raw-suffix.json"
split_median='NIGHTLY_CI_TARGET/Sync/RecordBatchRead/Parquet/SyntheticSmall/repeats:4'
split_cv='NIGHTLY_CI_TARGET/Sync/RecordBatchRead/Parquet/SyntheticSmall/repeats:5'
write_fixture "$split_raw_suffix_fixture" "[$(median "$split_median" 1 ms 1),$(cv "$split_cv" 0.01)]"

duplicate_raw_suffix_fixture="$temp_dir/duplicate-raw-suffix.json"
write_fixture "$duplicate_raw_suffix_fixture" "[$(median "$split_median" 1 ms 1),$(cv "$split_median" 0.01),$(median "$split_cv" 2 ms 1),$(cv "$split_cv" 0.02)]"

wrong_run_type_fixture="$temp_dir/wrong-run-type.json"
write_fixture "$wrong_run_type_fixture" "[{\"run_name\":\"$sync_parquet\",\"run_type\":\"iteration\",\"aggregate_name\":\"median\",\"real_time\":1,\"time_unit\":\"ms\"},{\"run_name\":\"$sync_parquet\",\"run_type\":\"iteration\",\"aggregate_name\":\"cv\",\"real_time\":0.01,\"time_unit\":\"ms\"}]"

for fixture in "$malformed_fixture" "$missing_median_fixture" "$duplicate_median_fixture" "$error_fixture" "$missing_field_fixture" "$async_recordbatch_fixture" "$async_lance_fixture" "$split_raw_suffix_fixture" "$duplicate_raw_suffix_fixture" "$wrong_run_type_fixture"; do
  assert_contains "$(render "$fixture")" '### Benchmark errors'
done

error_output="$(render "$error_fixture")"
assert_contains "$error_output" 'read failed\|retry'
[[ $error_output != *'read failed\\|retry'* ]] || {
  printf 'Markdown pipe escape must use one backslash\n' >&2
  exit 1
}

assert_contains "$(render "$missing_median_fixture" 1)" 'Cases: 0 completed / 1 expected'
assert_contains "$(render "$error_fixture" 1)" 'Cases: 0 completed / 1 expected'

assert_validate_fails "$success_fixture" invalid 'expected_count is not a number'
assert_validate_fails "$malformed_fixture" 0 'malformed target name'
assert_validate_fails "$missing_median_fixture" 1 'expected exactly one median'
assert_validate_fails "$duplicate_median_fixture" 1 'expected exactly one median'
assert_validate_fails "$error_fixture" 1 'benchmark error'
assert_validate_fails "$missing_field_fixture" 1 'unsupported or missing time unit'
assert_validate_fails "$async_recordbatch_fixture" 1 'unapproved semantic target: Async/RecordBatchRead/Parquet/SyntheticSmall'
assert_validate_fails "$async_lance_fixture" 1 'unapproved semantic target: Async/ChunkRead/Lance/SyntheticSmall'
assert_validate_fails "$split_raw_suffix_fixture" 1 "expected exactly one real-time CV for raw run $split_median"
assert_validate_fails "$duplicate_raw_suffix_fixture" 1 'multiple raw runs map to semantic target Sync/RecordBatchRead/Parquet/SyntheticSmall'
assert_validate_fails "$wrong_run_type_fixture" 0 'median/CV-shaped record must use run_type aggregate'
assert_validate_fails "$success_fixture" 4 'expected 4 unique semantic targets, found 5'

printf 'nightly summary renderer tests passed\n'
