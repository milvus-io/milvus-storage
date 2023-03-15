project(storage)

add_library(${PROJECT_NAME} default_space.cpp)

target_include_directories(${PROJECT_NAME} PUBLIC  
 $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>)

target_link_libraries(${PROJECT_NAME} PRIVATE arrow common format reader parquet-format)