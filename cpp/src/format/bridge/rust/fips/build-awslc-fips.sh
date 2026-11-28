#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 || -e "$1" ]]; then
    printf 'Usage: %s <new-output-directory> <bindgen>\n' "$0" >&2
    exit 2
fi
output="$(realpath -m "$1")"
bindgen=$2
parent="$(dirname "$output")"
mkdir -p "$parent"
workspace="$(mktemp -d "$parent/awslc-fips-build.XXXXXX")"
trap 'rm -rf "$workspace"' EXIT

commit=1276fc7d9cd0f509000294e9657914c3a23e65d2
source_sha256=33b5443d2213bfca45071b300daeb45decf13e3b46f726890b48d6267968f049
sys_crate=aws-lc-fips-sys-0.14.2
sys_crate_sha256=03367707e92796b190a4207d4d39b0a4271d574503d2969c2b0cfbf5c87658ee

# Optional local cache mirrors must contain the same verified inputs.
if [[ -n "${AWS_LC_FIPS_SOURCE_ARCHIVE:-}" ]]; then
    cp "$AWS_LC_FIPS_SOURCE_ARCHIVE" "$workspace/aws-source.zip"
else
    curl --fail --location --retry 3 "https://codeload.github.com/aws/aws-lc/zip/$commit" \
        --output "$workspace/aws-source.zip"
fi
if [[ -n "${AWS_LC_FIPS_SYS_CRATE_ARCHIVE:-}" ]]; then
    cp "$AWS_LC_FIPS_SYS_CRATE_ARCHIVE" "$workspace/sys.crate"
else
    curl --fail --location --retry 3 "https://static.crates.io/crates/aws-lc-fips-sys/$sys_crate.crate" \
        --output "$workspace/sys.crate"
fi
printf '%s  %s\n' "$source_sha256" "$workspace/aws-source.zip" \
    "$sys_crate_sha256" "$workspace/sys.crate" | sha256sum --check
unzip -q "$workspace/aws-source.zip" -d "$workspace"
tar -xf "$workspace/sys.crate" -C "$workspace" "$sys_crate/include/rust_wrapper.h"

# NIST CMVP #5596 security policy §11.1.
# Use an out-of-tree build and leave the source and flags intact.
source_dir="$workspace/aws-lc-$commit"
build_dir="$workspace/build"
output_dir="$workspace/output"
env -u CFLAGS -u CPPFLAGS -u CXXFLAGS -u LDFLAGS cmake \
    -S "$source_dir" -B "$build_dir" -DFIPS=1 -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build "$build_dir" --parallel "${AWS_LC_FIPS_BUILD_JOBS:-4}"
test "$("$build_dir/tool/bssl" isfips)" = 1
test "$("$build_dir/tool/bssl" version)" = 4.0.0
cmake --install "$build_dir" --prefix "$output_dir"
for library in lib64/libcrypto.a lib/libcrypto.a; do
    if [[ -f "$output_dir/$library" ]]; then
        ln -s "$library" "$output_dir/libcrypto.a"
        break
    fi
done
test -r "$output_dir/libcrypto.a"

test "$("$bindgen" --version)" = 'bindgen 0.72.1'
"$bindgen" \
    --allowlist-file '.*/openssl(/[^/]+)+\.h' \
    --allowlist-file '.*/rust_wrapper\.h' \
    --rustified-enum point_conversion_form_t \
    --default-macro-constant-type signed \
    --with-derive-default --with-derive-partialeq --with-derive-eq \
    --rust-target 1.70 --formatter rustfmt --output "$output_dir/aws-bindings.rs" \
    "$workspace/$sys_crate/include/rust_wrapper.h" -- -I"$output_dir/include"

# Consumers link the exact installed archive without modifying it.
# Package only after the source check, FIPS check and binding generation pass.
mv -T -- "$output_dir" "$output"
printf 'AWS-LC FIPS 4.0.0 installed under %s\n' "$output"
