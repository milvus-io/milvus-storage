.PHONY: build clean test format

TEST_THREADS ?= 4
ifneq ($(TIDY_THREADS),)
	TIDY_THREAD_PARAMETER := -j ${TIDY_THREADS}
endif

build:
	conan install . -if build --build=missing
	conan install .
	conan build .
debug:
	conan install . -if build --build=missing -s build_type=Debug
	conan install .
	conan build . -s build_type=Debug
clean:
	rm -rf build

test: build
	cd build/Release && ctest . -j ${TEST_THREADS} --output-on-failure

format:
	find ./src -type f ! -name "*.pb.h"  -iname *.h -o -iname *.cpp | xargs clang-format -i
	find ./test -type f ! -name "*.pb.h"  -iname *.h -o -iname *.cpp | xargs clang-format -i

check-format:
	find ./src -type f ! -name "*.pb.h"  -iname *.h -o -iname *.cpp | xargs clang-format --dry-run --Werror
	find ./test -type f ! -name "*.pb.h"  -iname *.h -o -iname *.cpp | xargs clang-format --dry-run --Werror

check-tidy:
	python3 ./scripts/run-clang-tidy.py -p build/Release


fix-tidy: 
	python3 ./scripts/run-clang-tidy.py -fix -p build/Release
