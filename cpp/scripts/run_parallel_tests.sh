#!/usr/bin/env bash

# Run the four CI backends on one Linux runner. The backend services and the
# milvus-storage-ut image must already be available.
set -euo pipefail

cpp_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
build_dir="$cpp_root/build/${BUILD_TYPE:-Release}"
results_dir="${RUNNER_TEMP:?}/storage-ut"
backends=(local minio rustfs azurite)
containers=()
pids=()

cleanup() {
    if ((${#containers[@]})); then
        docker rm -f "${containers[@]}" >/dev/null 2>&1 || true
    fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

for backend in "${backends[@]}"; do
    result_dir="$results_dir/$backend"
    mkdir -p "$result_dir/work" "$result_dir/gcov"
    # GCOV_PREFIX preserves the absolute build path below this private directory.
    # Put matching notes alongside it so lcov can capture each run independently.
    find "$build_dir" -name '*.gcno' -exec cp --parents -t "$result_dir/gcov" {} +
    container="storage-ut-${GITHUB_RUN_ID:-$$}-${GITHUB_RUN_ATTEMPT:-1}-$backend"
    containers+=("$container")
    echo "Starting $backend tests"
    (
        start=$SECONDS
        if docker run --rm --init --name "$container" --network host \
            --user "$(id -u):$(id -g)" \
            --mount "type=bind,src=$cpp_root,dst=$cpp_root,readonly" \
            --mount "type=bind,src=$result_dir,dst=/results" \
            --workdir /results/work \
            --env HOME=/tmp \
            --env "CPP_ROOT=$cpp_root" \
            --env "BUILD_DIR=$build_dir" \
            --env "LD_LIBRARY_PATH=$build_dir:$build_dir/libs" \
            --env GCOV_PREFIX=/results/gcov --env GCOV_PREFIX_STRIP=0 \
            milvus-storage-ut bash -euo pipefail -c '
                backend=$1
                if [[ "$backend" != local ]]; then
                    source "$CPP_ROOT/scripts/${backend}_env.sh"
                fi
                "$BUILD_DIR/test/milvus_test" --gtest_output=xml:/results/gtest.xml
                # Preserve the existing CI coverage: FFI runs on local and RustFS.
                if [[ "$backend" == local || "$backend" == rustfs ]]; then
                    "$BUILD_DIR/test/Test_FFI"
                fi
            ' -- "$backend"; then
            exit_code=0
        else
            exit_code=$?
        fi
        printf '%s\n' "$exit_code" > "$result_dir/exit-code"
        printf '%s\n' "$((SECONDS - start))" > "$result_dir/duration-seconds"
        exit "$exit_code"
    ) > "$result_dir/tests.log" 2>&1 &
    pids+=("$!")
done

# Wait for every backend, including after a failure, to keep all diagnostics.
status=0
for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
        status=1
    fi
done

summary="$results_dir/summary.md"
printf '| Backend | Seconds | Exit code |\n| --- | ---: | ---: |\n' > "$summary"
for backend in "${backends[@]}"; do
    result_dir="$results_dir/$backend"
    printf '| %s | %s | %s |\n' "$backend" \
        "$(cat "$result_dir/duration-seconds")" "$(cat "$result_dir/exit-code")" >> "$summary"
    echo "::group::$backend test log (last 80 lines)"
    tail -n 80 "$result_dir/tests.log"
    echo '::endgroup::'
done
cat "$summary"
if [[ -n "${GITHUB_STEP_SUMMARY:-}" ]]; then
    cat "$summary" >> "$GITHUB_STEP_SUMMARY"
fi
exit "$status"
