include(FetchContent)

message(STATUS "Fetching protobuf")
FetchContent_Declare(protobuf
  GIT_REPOSITORY git@github.com:protocolbuffers/protobuf.git
  GIT_TAG        v3.21.4
)

set(protobuf_BUILD_TESTS OFF)
set(protobuf_BUILD_PROTOC_BINARIES OFF)
FetchContent_MakeAvailable(protobuf)