.PHONY: build clean test format

TEST_THREADS ?= 4

build:
	conan install . -if build --build=missing
	conan build .

clean:
	rm -rf build

test:
	conan install . -if build --build=missing
	conan build .
	cd build && ctest . -j ${TEST_THREADS} --output-on-failure

format:
	find ./src -type f ! -name "*.pb.h"  -iname *.h -o -iname *.cpp | xargs clang-format -i
