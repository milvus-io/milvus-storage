#!/usr/bin/env bash
#
# Create a relocatable distribution tarball for milvus-storage.
#
# Layout:
#   milvus-storage-<version>-<os>-<arch>/
#     bin/loon
#     lib/libmilvus-storage.{so,dylib} + all shared deps
#     lib/pkgconfig/libstorage.pc
#     include/milvus-storage/...
#
# Usage: scripts/make_dist.sh <build-dir>
#   e.g.: scripts/make_dist.sh build/Release

set -euo pipefail

BUILD_DIR="${1:?Usage: $0 <build-dir>}"
VERSION="0.1.0"
OS="$(uname -s | tr '[:upper:]' '[:lower:]')"
ARCH="$(uname -m)"
DIST_NAME="milvus-storage-${VERSION}-${OS}-${ARCH}"
DIST_DIR="${BUILD_DIR}/dist/${DIST_NAME}"
SRC_DIR="$(cd "$(dirname "$0")/.." && pwd)"

echo "=== Creating distribution: ${DIST_NAME} ==="

# Clean and create layout
rm -rf "${DIST_DIR}"
mkdir -p "${DIST_DIR}/bin" "${DIST_DIR}/lib/pkgconfig" "${DIST_DIR}/include"

# 1. Copy loon binary
if [ -f "${BUILD_DIR}/tools/loon" ]; then
  cp "${BUILD_DIR}/tools/loon" "${DIST_DIR}/bin/"
  echo "  bin/loon"
else
  echo "WARNING: loon binary not found at ${BUILD_DIR}/tools/loon"
fi

# 2. Copy main library
if [ "${OS}" = "darwin" ]; then
  LIB_EXT="dylib"
else
  LIB_EXT="so"
fi

for f in "${BUILD_DIR}"/lib/libmilvus-storage*.${LIB_EXT}*; do
  [ -e "$f" ] || continue
  cp -a "$f" "${DIST_DIR}/lib/"
  echo "  lib/$(basename "$f")"
done

# 3. Copy shared deps from Conan (conan imports copies them to libs/)
LIBS_DIR="${BUILD_DIR}/libs"
if [ -d "${LIBS_DIR}" ]; then
  echo "  Collecting shared dependencies from ${LIBS_DIR}..."
  for f in "${LIBS_DIR}"/*.${LIB_EXT}*; do
    [ -e "$f" ] || continue
    cp -a "$f" "${DIST_DIR}/lib/"
  done
else
  echo "WARNING: ${LIBS_DIR} not found. Run 'conan imports' first."
fi

# 4. Copy headers
cp -r "${SRC_DIR}/include/milvus-storage" "${DIST_DIR}/include/"
echo "  include/milvus-storage/"

# 5. Copy pkg-config
if [ -f "${BUILD_DIR}/libstorage.pc" ]; then
  cp "${BUILD_DIR}/libstorage.pc" "${DIST_DIR}/lib/pkgconfig/"
  echo "  lib/pkgconfig/libstorage.pc"
fi

# 6. Fix RPATH on Linux (loon should find libs in ../lib relative to bin/)
if [ "${OS}" = "linux" ] && command -v patchelf &>/dev/null; then
  echo "  Patching RPATH..."
  patchelf --set-rpath '$ORIGIN/../lib' "${DIST_DIR}/bin/loon" 2>/dev/null || true
  # Also patch the main library to find deps in the same directory
  for f in "${DIST_DIR}"/lib/libmilvus-storage*.so*; do
    [ -e "$f" ] || continue
    patchelf --set-rpath '$ORIGIN' "$f" 2>/dev/null || true
  done
fi

# 7. Create tarball
TARBALL="${BUILD_DIR}/dist/${DIST_NAME}.tar.gz"
(cd "${BUILD_DIR}/dist" && tar czf "${DIST_NAME}.tar.gz" "${DIST_NAME}")
echo ""
echo "=== Distribution created ==="
echo "  Directory: ${DIST_DIR}"
echo "  Tarball:   ${TARBALL}"

# Show contents summary
echo ""
echo "Contents:"
echo "  bin/: $(ls "${DIST_DIR}/bin/" 2>/dev/null | wc -l | tr -d ' ') files"
echo "  lib/: $(ls "${DIST_DIR}/lib/"*.${LIB_EXT}* 2>/dev/null | wc -l | tr -d ' ') libraries"
echo "  include/: $(find "${DIST_DIR}/include" -name '*.h' | wc -l | tr -d ' ') headers"
