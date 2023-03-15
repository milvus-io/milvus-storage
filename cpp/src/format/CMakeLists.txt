project(format)

aux_source_directory(parquet SRC_LIST)
add_library(${PROJECT_NAME} ${SRC_LIST})

target_include_directories(${PROJECT_NAME}
    PUBLIC  ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(${PROJECT_NAME} PRIVATE storage parquet arrow common)
