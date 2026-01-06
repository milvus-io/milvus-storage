# CopyDependencies.cmake
# Copies shared library dependencies from Conan to the output directory
#
# Expected variables:
#   TARGET_FILE    - Path to the target library (milvus-storage)
#   JNI_FILE       - Path to the JNI library (optional)
#   OUTPUT_DIR     - Directory to copy dependencies to
#   CONAN_LIB_DIRS - Semicolon-separated list of Conan library directories

if(NOT TARGET_FILE)
    message(FATAL_ERROR "TARGET_FILE not specified")
endif()

if(NOT OUTPUT_DIR)
    message(FATAL_ERROR "OUTPUT_DIR not specified")
endif()

# Ensure output directory exists
file(MAKE_DIRECTORY "${OUTPUT_DIR}")

# Copy target libraries first
message(STATUS "Copying ${TARGET_FILE} to ${OUTPUT_DIR}")
file(COPY "${TARGET_FILE}" DESTINATION "${OUTPUT_DIR}")

if(JNI_FILE AND EXISTS "${JNI_FILE}")
    message(STATUS "Copying ${JNI_FILE} to ${OUTPUT_DIR}")
    file(COPY "${JNI_FILE}" DESTINATION "${OUTPUT_DIR}")
endif()

if(NOT CONAN_LIB_DIRS)
    message(WARNING "CONAN_LIB_DIRS not specified, skipping dependency copy")
    return()
endif()

# Convert CONAN_LIB_DIRS to a list
string(REPLACE ";" ";" CONAN_DIRS "${CONAN_LIB_DIRS}")

# Collect all conan lib directories
set(ALL_CONAN_LIB_DIRS "")
foreach(CONAN_DIR ${CONAN_DIRS})
    file(GLOB_RECURSE CONAN_SO_FILES "${CONAN_DIR}/*.so*")
    foreach(SO_FILE ${CONAN_SO_FILES})
        get_filename_component(SO_DIR "${SO_FILE}" DIRECTORY)
        list(APPEND ALL_CONAN_LIB_DIRS "${SO_DIR}")
    endforeach()
endforeach()
list(REMOVE_DUPLICATES ALL_CONAN_LIB_DIRS)

# Function to copy a library and its symlinks
function(copy_lib_with_symlinks LIB_PATH OUTPUT_DIR)
    if(EXISTS "${LIB_PATH}")
        get_filename_component(LIB_NAME "${LIB_PATH}" NAME)
        get_filename_component(LIB_DIR "${LIB_PATH}" DIRECTORY)
        get_filename_component(LIB_NAME_WE "${LIB_PATH}" NAME_WE)

        # Copy the library itself
        file(COPY "${LIB_PATH}" DESTINATION "${OUTPUT_DIR}")

        # Copy related symlinks
        file(GLOB RELATED_LIBS "${LIB_DIR}/${LIB_NAME_WE}*")
        foreach(RELATED_LIB ${RELATED_LIBS})
            if(EXISTS "${RELATED_LIB}")
                file(COPY "${RELATED_LIB}" DESTINATION "${OUTPUT_DIR}")
            endif()
        endforeach()
    endif()
endfunction()

# Function to find library in conan directories
function(find_lib_in_conan LIB_NAME CONAN_DIRS RESULT_VAR)
    set(${RESULT_VAR} "" PARENT_SCOPE)
    foreach(CONAN_DIR ${CONAN_DIRS})
        if(EXISTS "${CONAN_DIR}/${LIB_NAME}")
            set(${RESULT_VAR} "${CONAN_DIR}/${LIB_NAME}" PARENT_SCOPE)
            return()
        endif()
    endforeach()
endfunction()

execute_process(
    COMMAND ldd "${TARGET_FILE}"
    OUTPUT_VARIABLE LDD_OUTPUT
    ERROR_QUIET
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

# Parse the output and find libraries
string(REPLACE "\n" ";" LDD_LINES "${LDD_OUTPUT}")

foreach(LINE ${LDD_LINES})
    set(LIB_PATH "")
    set(LIB_NAME_TO_FIND "")

    # Check for "not found" libraries
    string(REGEX MATCH "([^ \t]+\\.so[^ \t]*) => not found" NOT_FOUND_MATCH "${LINE}")
    if(NOT_FOUND_MATCH)
        string(REGEX REPLACE ".*\t([^ \t]+\\.so[^ \t]*) => not found.*" "\\1" LIB_NAME_TO_FIND "${LINE}")
    else()
        # Extract path after "=>"
        string(REGEX MATCH "=> ([^ ]+\\.so[^ ]*)" MATCH "${LINE}")
        if(MATCH)
            string(REGEX REPLACE ".*=> ([^ ]+\\.so[^ ]*).*" "\\1" LIB_PATH "${LINE}")
        endif()
    endif()

    # Handle "not found" libraries
    if(LIB_NAME_TO_FIND)
        message(STATUS "Searching for missing library: ${LIB_NAME_TO_FIND}")
        find_lib_in_conan("${LIB_NAME_TO_FIND}" "${ALL_CONAN_LIB_DIRS}" FOUND_PATH)
        if(FOUND_PATH)
            message(STATUS "Found: ${FOUND_PATH}")
            copy_lib_with_symlinks("${FOUND_PATH}" "${OUTPUT_DIR}")
        endif()
    endif()

    # Handle found libraries from conan
    if(LIB_PATH AND EXISTS "${LIB_PATH}")
        foreach(CONAN_DIR ${CONAN_DIRS})
            string(FIND "${LIB_PATH}" "${CONAN_DIR}" POS)
            if(NOT POS EQUAL -1)
                get_filename_component(LIB_NAME "${LIB_PATH}" NAME)
                message(STATUS "Copying dependency: ${LIB_NAME}")
                copy_lib_with_symlinks("${LIB_PATH}" "${OUTPUT_DIR}")
                break()
            endif()
        endforeach()
    endif()
endforeach()

message(STATUS "Dependency copy complete")
