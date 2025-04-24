.PHONY: build clean test format

TEST_THREADS ?= 4
ifneq ($(TIDY_THREADS),)
	TIDY_THREAD_PARAMETER := -j ${TIDY_THREADS}
endif

use_asan = False
ifeq ($(USE_ASAN), True)
	use_asan = ${USE_ASAN}
endif

build:
	mkdir -p build && cd build && \
	conan install .. --build=missing -s build_type=Release \
	-s compiler.libcxx=libstdc++11 \
	-o with_ut=True -o with_asan=${use_asan} && \
	conan build .. && \
	conan install .. --install-folder . -g make

package: build
	mkdir -p build && cd build && \
	conan export .. milvus-storage/0.1.0@milvus/dev

debug:
	mkdir -p build && cd build && \
	conan install .. --build=missing -s build_type=Debug && \
	conan build ..

clean:
	rm -rf build

test: build
	cd build/Release/test && ./milvus_test

fix-format:
	find ./src -type f ! -name "*.pb.h"  -iname *.h -o -iname *.cpp | xargs clang-format -i
	find ./include -type f ! -name "*.pb.h"  -iname *.h -o -iname *.cpp | xargs clang-format -i
	find ./test -type f ! -name "*.pb.h"  -iname *.h -o -iname *.cpp | xargs clang-format -i
	find ./benchmark -type f ! -name "*.pb.h"  -iname *.h -o -iname *.cpp | xargs clang-format -i

check-format:
	find ./src -type f ! -name "*.pb.h"  -iname *.h -o -iname *.cpp | xargs clang-format --dry-run --Werror
	find ./include -type f ! -name "*.pb.h"  -iname *.h -o -iname *.cpp | xargs clang-format --dry-run --Werror
	find ./test -type f ! -name "*.pb.h"  -iname *.h -o -iname *.cpp | xargs clang-format --dry-run --Werror
	find ./benchmark -type f ! -name "*.pb.h"  -iname *.h -o -iname *.cpp | xargs clang-format --dry-run --Werror

check-tidy:
	python3 ./scripts/run-clang-tidy.py -p build/Release

fix-tidy: 
	python3 ./scripts/run-clang-tidy.py -fix -p build/Release

proto: 
	protoc -I="src/proto" --cpp_out="src/proto" src/proto/*.proto
