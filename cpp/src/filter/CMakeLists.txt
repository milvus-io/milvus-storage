project(filter)

add_library(${PROJECT_NAME} STATIC constant_filter.cpp value.cpp)

target_include_directories(${PROJECT_NAME} PUBLIC ${PROJECT_SOURCE_DIR})
target_link_libraries(${PROJECT_NAME} PRIVATE parquet common)