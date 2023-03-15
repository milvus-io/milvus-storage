cmake_minimum_required(VERSION 3.0.0)
message(STATUS ${CMAKE_INSTALL_PREFIX})

project(milvus-storage VERSION 0.1.0)


set(CMAKE_CXX_STANDARD 20)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)



include(CTest)
enable_testing()

add_subdirectory(thirdparty)

add_subdirectory(src)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
message(STATUS "PROJECT_NAME: ${PROJECT_TARGET_DIR}")
include(CPack)

add_subdirectory(test)
