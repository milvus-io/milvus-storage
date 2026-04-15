add_executable(loon loon.cpp)

set_target_properties(loon PROPERTIES CXX_STANDARD 17)

if (NOT APPLE)
  target_link_options(loon PRIVATE "LINKER:--disable-new-dtags")
  # Use $ORIGIN/../lib so loon finds libmilvus-storage.so and deps
  # in the standard bin/ + lib/ layout after `make dist`
  set_target_properties(loon PROPERTIES
    BUILD_RPATH "$ORIGIN/../lib"
    INSTALL_RPATH "$ORIGIN/../lib"
  )
endif()

target_link_libraries(loon PRIVATE milvus-storage)

install(TARGETS loon DESTINATION ${CMAKE_INSTALL_PREFIX}/bin)
