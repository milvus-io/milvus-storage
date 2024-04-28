include(FetchContent)

message(STATUS "Fetching protobuf")
FetchContent_Declare(protobuf
  GIT_REPOSITORY https://github.com/protocolbuffers/protobuf.git
  GIT_TAG        v3.21.4
)

set(protobuf_BUILD_TESTS OFF CACHE INTERNAL "")
set(protobuf_BUILD_PROTOC_BINARIES ON CACHE INTERNAL "")
FetchContent_MakeAvailable(protobuf)