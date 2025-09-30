use std::env;
use std::path::PathBuf;

fn main() {
    // Detect build profile and use appropriate C++ build directory
    let profile = env::var("PROFILE").unwrap_or_else(|_| "debug".to_string());
    let cpp_build_subdir = match profile.as_str() {
        "release" => "Release",
        _ => "Debug", // Default to Debug for debug builds and any other profile
    };
    
    let cpp_build_dir = format!("../cpp/build/{}", cpp_build_subdir);
    println!("cargo:rustc-link-search=native={}", cpp_build_dir);
    println!("cargo:warning=Using C++ build directory: {}", cpp_build_dir);
    
    // Tell cargo to link the milvus-storage library when available
    println!("cargo:rustc-link-lib=dylib=milvus-storage");
    
    // Set runtime library path (rpath) for dynamic linking
    // This is NECESSARY in addition to rustc-link-search because:
    // - rustc-link-search: tells linker where to find libs during COMPILATION
    // - rpath: tells dynamic loader where to find libs during RUNTIME
    let target = env::var("TARGET").unwrap();
    
    // Get the library path (same for both platforms)
    let current_dir = env::current_dir().expect("Failed to get current directory");
    let lib_path = current_dir.join(&cpp_build_dir);
    let canonical_path = lib_path.canonicalize()
        .unwrap_or_else(|_| lib_path);
    println!("cargo:rustc-link-arg=-Wl,-rpath,{}", canonical_path.display());
    
    if target.contains("apple") {
        // macOS: Uses dyld, supports both absolute paths and @rpath
        // We could also use @executable_path/../cpp/build/{Debug|Release} for relative paths
        println!("cargo:rustc-link-lib=dylib=c++");
    } else if target.contains("linux") {
        // Linux: Uses ld.so, supports $ORIGIN for relative paths
        // We could use $ORIGIN/../cpp/build/{Debug|Release} for relative paths
        // Also set runpath (newer alternative to rpath)
        println!("cargo:rustc-link-arg=-Wl,--enable-new-dtags");
        println!("cargo:rustc-link-lib=dylib=stdc++");
    } else {
        // Other Unix-like systems
        println!("cargo:rustc-link-lib=dylib=stdc++");
    }

    // Tell cargo to invalidate the built crate whenever the wrapper changes
    println!("cargo:rerun-if-changed=wrapper.h");
    println!("cargo:rerun-if-changed=../cpp/include/milvus-storage/ffi_c.h");

    // Generate bindings using bindgen
    generate_bindgen_bindings();
}

/// Generate bindings using bindgen from the actual C header file
fn generate_bindgen_bindings() {
    let bindings = bindgen::Builder::default()
        .header("wrapper.h")
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .clang_arg("-I../cpp/include")
        .clang_arg("-x")
        .clang_arg("c++")
        .clang_arg("-std=c++17")
        // Only generate bindings for our specific types and functions from the new FFI interface
        .allowlist_function("reader_.*")
        .allowlist_function("writer_.*")
        .allowlist_function("chunk_reader_.*")
        .allowlist_function("get_.*")
        .allowlist_function("take")
        .allowlist_function("free_.*")
        .allowlist_function("IsSuccess")
        .allowlist_function("GetErrorMessage")
        .allowlist_function("FreeFFIResult")
        .allowlist_function("properties_.*")
        .allowlist_type("FFIResult")
        .allowlist_type("Properties")
        .allowlist_type("ReaderHandle")
        .allowlist_type("WriterHandle")
        .allowlist_type("ChunkReaderHandle")
        // Prevent Copy derive for types with destructors
        .no_copy("Properties")
        .generate()
        .expect("Unable to generate bindings");

    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Couldn't write bindings!");
}
