include(FetchContent)

message(STATUS "Fetching Boost 1.84.0")
FetchContent_Declare(
  Boost
  URL https://github.com/boostorg/boost/releases/download/boost-1.84.0/boost-1.84.0.tar.xz
  URL_MD5 893b5203b862eb9bbd08553e24ff146a
  DOWNLOAD_EXTRACT_TIMESTAMP ON
)

set(BOOST_INCLUDE_LIBRARIES uuid algorithm filesystem)
FetchContent_MakeAvailable(Boost)
