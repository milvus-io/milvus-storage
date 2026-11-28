function(prepare_storage_rust_fips manifest)
    if (NOT CMAKE_SYSTEM_NAME STREQUAL "Linux" OR CMAKE_CROSSCOMPILING)
        message(FATAL_ERROR "The AWS-LC FIPS build requires a native Linux build")
    endif()
    find_package(Python3 3.8 REQUIRED COMPONENTS Interpreter)

    set(fips_dir "${CMAKE_CURRENT_FUNCTION_LIST_DIR}")
    set(workspace "${CMAKE_CURRENT_BINARY_DIR}/cargo-fips")
    file(GLOB patches CONFIGURE_DEPENDS "${fips_dir}/patches/*.patch")
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
        "${fips_dir}/prepare.py" "${fips_dir}/Cargo.lock"
        "${CMAKE_CURRENT_SOURCE_DIR}/Cargo.toml" "${CMAKE_CURRENT_SOURCE_DIR}/Cargo.lock"
        ${patches}
    )
    execute_process(
        COMMAND "${CMAKE_COMMAND}" -E env "RUSTC=${_CORROSION_RUSTC}"
            "${Python3_EXECUTABLE}" "${fips_dir}/prepare.py" --cargo "${_CORROSION_CARGO}"
            --target "${_CORROSION_RUST_CARGO_TARGET}" --output "${workspace}"
        COMMAND_ERROR_IS_FATAL ANY
    )
    set(${manifest} "${workspace}/Cargo.toml" PARENT_SCOPE)
endfunction()

function(link_storage_rust_fips)
    set(fips_dir "${CMAKE_CURRENT_FUNCTION_LIST_DIR}")
    set(build_dir "${CMAKE_CURRENT_BINARY_DIR}/aws-lc-fips")
    set(builder "${fips_dir}/build-awslc-fips.sh")
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${builder}")
    file(SHA256 "${builder}" module_inputs)
    set(module "${build_dir}/module-${module_inputs}")
    set(bindgen "${build_dir}/bindgen/bin/bindgen")
    set(crypto "${module}/libcrypto.a")
    set(bindings "${module}/aws-bindings.rs")
    add_custom_command(OUTPUT "${bindgen}"
        COMMAND "${CMAKE_COMMAND}" -E env "RUSTC=${_CORROSION_RUSTC}"
            "${_CORROSION_CARGO}" install --locked bindgen-cli --version 0.72.1
            --root "${build_dir}/bindgen" --bin bindgen
        VERBATIM
    )
    add_custom_command(OUTPUT "${crypto}" "${bindings}"
        COMMAND "${CMAKE_COMMAND}" -E rm -rf "${module}"
        COMMAND bash "${builder}" "${module}" "${bindgen}"
        DEPENDS "${bindgen}" "${builder}"
        VERBATIM
    )
    add_custom_target(storage_aws_lc_fips DEPENDS "${crypto}" "${bindings}")
    add_dependencies(cargo-prebuild_rust_bridge storage_aws_lc_fips)
    string(REPLACE "-" "_" rust_target "${_CORROSION_RUST_CARGO_TARGET}")
    corrosion_set_env_vars(rust_bridge
        "AWS_LC_FIPS_SYS_SYSTEM_DIR_${rust_target}=${module}"
        "AWS_LC_FIPS_SYS_SYSTEM_BINDINGS_${rust_target}=${bindings}"
        "AWS_LC_FIPS_SYS_STATIC_${rust_target}=1"
        "AWS_LC_FIPS_SYS_USE_SYSTEM_${rust_target}=1"
    )

    set(isolated "${build_dir}/$<CONFIG>/librust_bridge_isolated.a")
    add_custom_command(OUTPUT "${isolated}"
        COMMAND "${CMAKE_COMMAND}" -E make_directory "${build_dir}/$<CONFIG>"
        COMMAND bash "${fips_dir}/localize-aws-lc.sh"
            "${CMAKE_LINKER}" "${CMAKE_NM}" "${CMAKE_OBJCOPY}" "${CMAKE_AR}"
            "${crypto}" "$<TARGET_FILE:rust_bridge-static>" "${isolated}"
        DEPENDS "$<TARGET_FILE:rust_bridge-static>" rust_bridge-static
            "${crypto}" "${fips_dir}/localize-aws-lc.sh"
        VERBATIM
    )
    add_custom_target(storage_isolate_aws_lc DEPENDS "${isolated}")
    add_dependencies(rust_bridge storage_isolate_aws_lc)

    # Keep the native link dependencies Corrosion sets for the Rust static library.
    get_target_property(link_libraries rust_bridge INTERFACE_LINK_LIBRARIES)
    list(TRANSFORM link_libraries REPLACE "^rust_bridge-static$" "${isolated}")
    set_target_properties(rust_bridge PROPERTIES INTERFACE_LINK_LIBRARIES "${link_libraries}")
    target_link_libraries(rust_bridge INTERFACE "$<TARGET_PROPERTY:rust_bridge-static,INTERFACE_LINK_LIBRARIES>")
    target_link_options(rust_bridge INTERFACE "$<TARGET_PROPERTY:rust_bridge-static,INTERFACE_LINK_OPTIONS>")
    target_link_directories(rust_bridge INTERFACE "$<TARGET_PROPERTY:rust_bridge-static,INTERFACE_LINK_DIRECTORIES>")
endfunction()
