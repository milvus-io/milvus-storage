#!/usr/bin/env bash

set -euo pipefail

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <jni-shared-library>" >&2
    exit 2
fi

JNI_LIBRARY="$1"

if [ ! -f "$JNI_LIBRARY" ]; then
    echo "JNI shared library not found: $JNI_LIBRARY" >&2
    exit 1
fi

if ! command -v ldd >/dev/null 2>&1; then
    echo "ldd is required to verify bundled native dependencies" >&2
    exit 1
fi

# Do not allow a developer shell or CI job to hide missing RUNPATH entries.
ldd_output="$(env -u LD_LIBRARY_PATH -u LD_PRELOAD ldd "$JNI_LIBRARY")"
missing="$(printf '%s\n' "$ldd_output" | sed -n '/not found/p' | sort -u)"

if [ -n "$missing" ]; then
    echo "Bundled JNI library has unresolved dependencies:" >&2
    printf '%s\n' "$missing" >&2
    exit 1
fi

echo "Verified bundled dependency resolution for $JNI_LIBRARY"
