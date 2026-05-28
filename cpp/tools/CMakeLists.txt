add_executable(loon loon.cpp)

set_target_properties(loon PROPERTIES CXX_STANDARD 17)

if (NOT APPLE)
  target_link_options(loon PRIVATE "LINKER:--disable-new-dtags")
endif()

target_link_libraries(loon PRIVATE milvus-storage)
