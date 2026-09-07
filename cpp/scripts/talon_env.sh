# Environment for the Talon integration test and benchmark. Source this after
# scripts/setup_talon.sh has brought up the local stack:
#
#   source ./scripts/talon_env.sh
#   ./build/Release/test/milvus_test --gtest_filter='Talon*'
#
# The values match setup_talon.sh's defaults; already-exported values win, so a
# custom topology (different coordinator address or object URI) is honored.
export TALON_COORDINATOR_ADDR="${TALON_COORDINATOR_ADDR:-127.0.0.1:7000}"
export TALON_TEST_URI="${TALON_TEST_URI:-s3://test-bucket/talon-test/obj}"
