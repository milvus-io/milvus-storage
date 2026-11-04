#!/usr/bin/env bash

# Linux CI only: reuse the runner runtime and isolate writable test paths.
set -euo pipefail

mode=${1:-parallel}
case "$mode" in probe|parallel|serial) ;; *) echo "Usage: $0 [probe|parallel|serial]" >&2; exit 2 ;; esac
cpp_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
build_dir="$cpp_root/build/${BUILD_TYPE:-Release}"
results_dir="${RUNNER_TEMP:?}/storage-ut/$mode"
backends=(local minio rustfs azurite)
pids=()
runner_uid=$(id -u)
runner_gid=$(id -g)
host_marker=$(mktemp /tmp/storage-ut-probe.XXXXXX)
printf 'host\n' > "$host_marker"
export LD_LIBRARY_PATH="$build_dir:$build_dir/libs${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

cleanup() {
    local active pid i supervisor deadline
    active=" $(jobs -pr | tr '\n' ' ') "
    for i in "${!pids[@]}"; do
        pid=${pids[$i]}
        if [[ "$active" == *" $pid "* ]]; then
            supervisor="$results_dir/${backends[$i]}/supervisor.pid"
            deadline=$((SECONDS + 5))
            while [[ ! -s "$supervisor" ]] && kill -0 "$pid" 2>/dev/null && (( SECONDS < deadline )); do
                sleep 0.05
            done
            # unshare --fork ignores TERM. Killing its supervisor kills the
            # privileged namespace init and all descendants via --kill-child.
            if [[ -s "$supervisor" ]]; then
                sudo -n kill -KILL "$(cat "$supervisor")" 2>/dev/null || true
            else
                sudo -n kill -KILL "$pid" 2>/dev/null || true
            fi
        fi
    done
    wait || true
    rm -f "$host_marker"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

sudo -n true
for binary in "$build_dir/test/milvus_test" "$build_dir/test/Test_FFI"; do
    dependencies=$(ldd "$binary")
    if [[ "$dependencies" == *"not found"* ]]; then
        printf '%s\n%s\n' "$binary" "$dependencies" >&2
        exit 1
    fi
done

start=$SECONDS
status=0
for backend in "${backends[@]}"; do
    result_dir="$results_dir/$backend"
    mkdir -p "$result_dir"/{tmp,home,work,gcov}
    find "$build_dir" -name '*.gcno' -exec cp --parents -t "$result_dir/gcov" {} +
    echo "Starting $mode $backend tests"
    sudo -n setpriv --pdeathsig KILL bash -c '
        echo "$$" > "$1/supervisor.pid"
        exec unshare --mount --pid --fork --kill-child --mount-proc --propagation private bash -euo pipefail -s -- "$@"
    ' -- "$result_dir" "$backend" "$mode" "$cpp_root" "$build_dir" "$runner_uid" "$runner_gid" "$PATH" "$LD_LIBRARY_PATH" "$host_marker" <<'NAMESPACE' > "$result_dir/tests.log" 2>&1 &
result_dir=$1
mount --bind "$result_dir/tmp" /tmp
mount --bind "$4" "$4"
mount -o remount,bind,ro "$4"
cd "$result_dir/work"
# Keep this privileged PID 1 shell alive while the unprivileged test runs:
# changing PID 1's credentials would clear unshare's parent-death signal.
if setpriv --reuid="$6" --regid="$7" --init-groups env -i \
    PATH="$8" HOME="$result_dir/home" LD_LIBRARY_PATH="$9" \
    GCOV_PREFIX="$result_dir/gcov" GCOV_PREFIX_STRIP=0 \
    RESULT_DIR="$result_dir" BACKEND="$2" MODE="$3" CPP_ROOT="$4" BUILD_DIR="$5" \
    HOST_MARKER="${10}" bash -euo pipefail -s <<'TEST'; then
start=$SECONDS
trap 'code=$?; printf "%s\n" "$((SECONDS - start))" > "$RESULT_DIR/duration-seconds"; exit "$code"' EXIT
if [[ "$BACKEND" != local ]]; then
    source "$CPP_ROOT/scripts/${BACKEND}_env.sh"
fi
args=(--gtest_output="xml:$RESULT_DIR/gtest.xml")
if [[ "$MODE" == probe ]]; then
    printf '%s\n' "$BACKEND" > "$HOST_MARKER"
    printf '%s\n' "$BACKEND" > isolation-marker
    touch "$RESULT_DIR/ready"
    deadline=$((SECONDS + 30))
    while [[ ! -f "$RESULT_DIR/../local/ready" || ! -f "$RESULT_DIR/../minio/ready" || ! -f "$RESULT_DIR/../rustfs/ready" || ! -f "$RESULT_DIR/../azurite/ready" ]]; do
        if (( SECONDS >= deadline )); then echo 'Isolation barrier timed out' >&2; exit 1; fi
        sleep 0.1
    done
    [[ $(cat "$HOST_MARKER") == "$BACKEND" ]]
    [[ $(cat isolation-marker) == "$BACKEND" ]]
    if [[ "$BACKEND" == local ]]; then
        args+=(--gtest_filter=LocalFsTest.TestRootPath:LocalFsTest.TestMetricsAfterFileOperations)
    else
        args+=(--gtest_filter=LocalFsTest.TestRootPath:CloudFsTest.WriteAndReadSmallFile:CloudFsTest.DeleteFile)
    fi
fi
"$BUILD_DIR/test/milvus_test" "${args[@]}"
if [[ "$MODE" != probe && ( "$BACKEND" == local || "$BACKEND" == rustfs ) ]]; then
    "$BUILD_DIR/test/Test_FFI"
fi
TEST
    exit 0
else
    exit "$?"
fi
NAMESPACE
    pid=$!
    pids+=("$pid")
    if [[ "$mode" == serial ]]; then
        code=0
        wait "$pid" || code=$?
        printf '%s\n' "$code" > "$result_dir/exit-code"
        if (( code )); then status=1; fi
    fi
done

if [[ "$mode" != serial ]]; then
    for i in "${!pids[@]}"; do
        code=0
        wait "${pids[$i]}" || code=$?
        printf '%s\n' "$code" > "$results_dir/${backends[$i]}/exit-code"
        if (( code )); then status=1; fi
    done
fi
pids=()
elapsed=$((SECONDS - start))
[[ $(cat "$host_marker") == host ]] || status=1

if [[ "$mode" == probe && "$status" == 0 ]]; then
    python3 - "$results_dir" <<'PY'
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

root = Path(sys.argv[1])
for backend, expected in [('local', 2), ('minio', 3), ('rustfs', 3), ('azurite', 3)]:
    directory = root / backend
    cases = ET.parse(directory / 'gtest.xml').findall('.//testcase')
    assert len(cases) == expected, (backend, 'unexpected test count', len(cases))
    assert all(case.get('status') == 'run' and not list(case) for case in cases), (backend, 'failed or skipped probe')
    assert any((directory / 'gcov').rglob('*.gcda')), (backend, 'missing private coverage')
    print(f'{backend}: isolation and {expected} read/write probes passed')
PY
fi

summary="$results_dir/summary.md"
printf '\n%s UT wall time: %s seconds\n\n| Backend | Test seconds | Exit code |\n| --- | ---: | ---: |\n' "$mode" "$elapsed" > "$summary"
for backend in "${backends[@]}"; do
    result_dir="$results_dir/$backend"
    duration=n/a
    if [[ -f "$result_dir/duration-seconds" ]]; then duration=$(cat "$result_dir/duration-seconds"); fi
    printf '| %s | %s | %s |\n' "$backend" "$duration" "$(cat "$result_dir/exit-code")" >> "$summary"
    echo "::group::$mode $backend test log (last 80 lines)"
    tail -n 80 "$result_dir/tests.log"
    echo '::endgroup::'
done
printf '%s\n' "$elapsed" > "$results_dir/wall-seconds"
cat "$summary"
if [[ -n "${GITHUB_STEP_SUMMARY:-}" ]]; then cat "$summary" >> "$GITHUB_STEP_SUMMARY"; fi
exit "$status"
