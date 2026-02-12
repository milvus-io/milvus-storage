# Plan: Bundle Dependency Libraries into JNI Fat JAR

## Context

Currently `package_fat_jar.sh` only copies `libmilvus-storage.so` and `libmilvus-storage-jni.so` into the fat jar, but misses all dependency libraries under `cpp/build/Release/libs/` (Arrow, Boost, AWS SDK, OpenSSL, etc. — approximately 67 .so files). Additionally, the RUNPATH of these two main .so files is hardcoded to the build machine's Conan cache path, making the fat jar unusable on other machines.

`NativeLibraryLoader.java` also only extracts and loads the single JNI .so file, without extracting any dependency libraries.

## Modification Plan

### 1. Modify `package_fat_jar.sh`
**File**: `java/package_fat_jar.sh`

- After copying the main .so and JNI .so, add copying of all .so files under `cpp/build/Release/libs/` (including subdirectories `ossl-modules/` and `engines-3/`) to `native/linux-x86_64/`
- **Must use `cp -L`** (follow symlinks) instead of `cp -P`, because `libs/*.so` are symlinks pointing to versioned files (e.g. `libboost_atomic.so -> libboost_atomic.so.1.82.0`); `cp -P` would create broken symlinks causing sbt assembly to fail to package them
- Use `patchelf` to change the RUNPATH of `libmilvus-storage.so` and `libmilvus-storage-jni.so` to `$ORIGIN`, so they look for dependencies in the same directory

### 2. Modify `NativeLibraryLoader.java`
**File**: `java/src/main/java/io/milvus/storage/NativeLibraryLoader.java`

- Extract **all** .so files under `/native/` from JAR resources to a temporary directory (not just the JNI .so)
- Support two runtime modes: extraction from JAR (production) and extraction from filesystem (development)
- After extraction, `System.load()` the JNI .so (at this point RUNPATH=$ORIGIN can find dependencies in the same directory)

### 3. `build.sbt` — No Changes Needed
- The existing `unmanagedResourceDirectories` configuration already points to the `native` directory; sbt assembly can recursively package all subdirectories and files
- The existing merge strategy already handles `.so` files correctly

## Detailed Changes

### `package_fat_jar.sh` additions (lines 75-101)

```bash
# Copy dependency libraries from libs/
LIBS_DIR="$CPP_BUILD_DIR/Release/libs"
if [ -d "$LIBS_DIR" ]; then
    echo "copy dependency libraries from $LIBS_DIR"
    # Copy .so files (follow symlinks to get real file content, only unversioned .so names)
    # NOTE: Must use cp -L (not cp -P) because libs/*.so are symlinks to versioned files
    # e.g. libboost_atomic.so -> libboost_atomic.so.1.82.0; cp -P creates broken symlinks
    for f in "$LIBS_DIR"/*.so; do
        [ -e "$f" ] || [ -L "$f" ] && cp -L "$f" native/linux-x86_64/
    done
    # Copy OpenSSL modules subdirectories
    for subdir in ossl-modules engines-3; do
        if [ -d "$LIBS_DIR/$subdir" ]; then
            mkdir -p "native/linux-x86_64/$subdir"
            cp -rL "$LIBS_DIR/$subdir"/* "native/linux-x86_64/$subdir/"
        fi
    done
else
    echo "Warning: libs directory not found: $LIBS_DIR"
fi

# Patch RUNPATH to $ORIGIN so libs find each other in the same directory
if command -v patchelf &>/dev/null; then
    echo "patching RUNPATH to \$ORIGIN..."
    patchelf --set-rpath '$ORIGIN' native/linux-x86_64/libmilvus-storage.so
    patchelf --set-rpath '$ORIGIN' native/linux-x86_64/libmilvus-storage-jni.so
else
    echo "Warning: patchelf not found, RUNPATH not patched"
fi
```

### `NativeLibraryLoader.java` rewrite

Core changes:
- `loadFromResources()` → first extract all .so files to a temporary directory, then load the JNI .so
- New `extractAllNativeLibs(tempDir)` → detect jar/file protocol, handle each separately
- New `extractFromJar()` → scan JAR entries, extract by `native/` prefix, strip `native/<platform>/` prefix while preserving subdirectory structure
- New `extractFromFileSystem()` → extract from filesystem in development mode
- New `extractFromDirectory()` → recursively extract .so files from a directory
- New `stripPlatformPrefix()` → extract `libfoo.so` from `native/linux-x86_64/libfoo.so`
- New `isNativeLibrary()` → determine whether a file is a .so/.dylib/.dll

## Known Issues and Fixes

### `cp -P` causes broken symlinks (fixed)
Most `.so` files under `libs/` are symlinks pointing to versioned files in the same directory:
```
libboost_atomic.so -> libboost_atomic.so.1.82.0
libarrow.so -> libarrow.so.1700
libfolly.so -> libfolly.so.0.58.0-dev
```
After copying with `cp -P` (preserve symlinks), the versioned files are missing in the target directory, resulting in broken symlinks. sbt assembly silently skips these files.

**Fix**: Use `cp -L` instead (follow symlinks, copy the actual file contents).

## Verification Steps

1. `cd cpp && make build` — compile the C++ libraries
2. `cd java && bash package_fat_jar.sh` — build the fat jar
3. Verify fat jar contents: `jar tf target/scala-2.13/milvus-storage-jni-fat.jar | grep '\.so' | wc -l` — should contain ~77 .so files
4. Verify RUNPATH: `readelf -d native/linux-x86_64/libmilvus-storage-jni.so | grep RUNPATH` — should show `$ORIGIN`
5. Run fat jar tests in a different environment without Conan cache

## Build Artifacts

| Item | Value |
|------|-------|
| Fat JAR path | `dist/milvus-storage-jni-fat.jar` |
| Fat JAR size | ~126M |
| Total .so files | 77 (69 dependencies + 2 main libraries + 6 OpenSSL modules) |
| RUNPATH | `$ORIGIN` |
