#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 7 ]]; then
    printf 'Usage: %s <ld> <nm> <objcopy> <ar> <aws-lc-archive> <rust-archive> <output-archive>\n' "$0" >&2
    exit 2
fi
linker=$1
nm_tool=$2
objcopy_tool=$3
ar_tool=$4
aws_archive=$5
rust_archive=$6
output=$7

workdir="$(mktemp -d "$(dirname "$output")/isolate-awslc.XXXXXX")"
trap 'rm -rf "$workdir"' EXIT
symbols="$workdir/awslc-symbols.txt"
"$nm_tool" --just-symbols --extern-only --defined-only "$aws_archive" |
    awk 'NF == 1 && $1 !~ /:$/ { print $1 }' |
    LC_ALL=C sort -u > "$symbols"
grep -qx FIPS_mode "$symbols"
grep -qx awslc_version_string "$symbols"

# The Rust staticlib already bundles libcrypto.a. Resolve its references before
# localizing the definitions, so other Storage objects can resolve against OpenSSL.
"$linker" -r -o "$workdir/rust-bridge.o" --whole-archive "$rust_archive" --no-whole-archive
"$objcopy_tool" --localize-symbols="$symbols" \
    "$workdir/rust-bridge.o" "$workdir/rust-bridge-isolated.o"
"$ar_tool" crs "$workdir/rust-bridge-isolated.a" "$workdir/rust-bridge-isolated.o"
mv -f -- "$workdir/rust-bridge-isolated.a" "$output"
