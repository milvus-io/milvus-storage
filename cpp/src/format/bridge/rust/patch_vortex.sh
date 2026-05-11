#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PATCH_DIR="${1:-${SCRIPT_DIR}/_vortex_patched}"
PATCHES_SRC="${SCRIPT_DIR}/patches"
VORTEX_REPO="https://github.com/vortex-data/vortex.git"
VORTEX_TAG="0.56.0"

# Collect patches sorted by numeric prefix (e.g. 1-foo.patch, 2-bar.patch)
patches=()
for f in "${PATCHES_SRC}"/*.patch; do
    [ -f "$f" ] && patches+=("$f")
done

if [ ${#patches[@]} -eq 0 ]; then
    echo "[patch_vortex] No patches found in ${PATCHES_SRC}, skipping."
    exit 0
fi

# Compute a fingerprint of all patches to detect changes
current_hash=$(cat "${patches[@]}" | shasum -a 256 | cut -d' ' -f1)

if [ -f "${PATCH_DIR}/.patched" ] && [ "$(cat "${PATCH_DIR}/.patched")" = "${current_hash}" ]; then
    echo "[patch_vortex] Already patched (hash match), skipping."
    exit 0
fi

echo "[patch_vortex] Cloning vortex ${VORTEX_TAG} ..."
rm -rf "${PATCH_DIR}"
git clone --depth 1 --branch "${VORTEX_TAG}" "${VORTEX_REPO}" "${PATCH_DIR}"

for patch_file in "${patches[@]}"; do
    patch_name="$(basename "${patch_file}")"
    echo "[patch_vortex] Applying ${patch_name} ..."
    if ! git -C "${PATCH_DIR}" apply "${patch_file}"; then
        echo "[patch_vortex] ERROR: ${patch_name} failed. Diagnostics:" >&2
        git -C "${PATCH_DIR}" apply --check "${patch_file}" 2>&1 || true
        rm -rf "${PATCH_DIR}"
        exit 1
    fi
done

echo "${current_hash}" > "${PATCH_DIR}/.patched"
echo "[patch_vortex] Done. Applied ${#patches[@]} patch(es)."
