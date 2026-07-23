#!/usr/bin/env bash

set -euo pipefail

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <native-library-directory>" >&2
    exit 2
fi

NATIVE_DIR="$1"

if [ ! -d "$NATIVE_DIR" ]; then
    echo "Native library directory not found: $NATIVE_DIR" >&2
    exit 1
fi

if ! command -v patchelf >/dev/null 2>&1; then
    echo "patchelf is required to make bundled native libraries relocatable" >&2
    exit 1
fi

patched=0
while IFS= read -r -d '' library; do
    # Some packages may ship linker scripts with a .so suffix. patchelf only
    # understands ELF files, so leave non-ELF entries untouched.
    if ! patchelf --print-rpath "$library" >/dev/null 2>&1; then
        echo "Skipping non-ELF shared library: $library"
        continue
    fi

    # NativeLibraryLoader preserves subdirectories while extracting resources.
    # Point a top-level library at $ORIGIN and a nested library back up to the
    # native root so every bundled dependency can resolve its siblings.
    relative_path="${library#"$NATIVE_DIR"/}"
    relative_dir="$(dirname "$relative_path")"
    runpath='$ORIGIN'
    while [ "$relative_dir" != "." ]; do
        runpath="$runpath/.."
        relative_dir="$(dirname "$relative_dir")"
    done

    patchelf --set-rpath "$runpath" "$library"
    patched=$((patched + 1))
done < <(find "$NATIVE_DIR" -type f \( -name '*.so' -o -name '*.so.*' \) -print0)

if [ "$patched" -eq 0 ]; then
    echo "No ELF shared libraries found in $NATIVE_DIR" >&2
    exit 1
fi

echo "Patched RUNPATH for $patched bundled shared libraries in $NATIVE_DIR"
