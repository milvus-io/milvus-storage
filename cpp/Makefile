.PHONY: build clean test format

TEST_THREADS ?= 4
ifneq ($(TIDY_THREADS),)
	TIDY_THREAD_PARAMETER := -j ${TIDY_THREADS}
endif

build:
	mkdir -p build && cd build && \
	cmake ../.. && \
	make -j 8

debug:
	mkdir -p build && cd build && \
	cmake -DCMAKE_BUILD_TYPE=Debug .. && \
	make -j 8

clean:
	rm -rf build

test: build
	cd build/test && ctest . -j ${TEST_THREADS} --output-on-failure

fix-format:
	find ./src -type f ! -name "*.pb.h"  -iname *.h -o -iname *.cpp | xargs clang-format -i
	find ./include -type f ! -name "*.pb.h"  -iname *.h -o -iname *.cpp | xargs clang-format -i
	find ./test -type f ! -name "*.pb.h"  -iname *.h -o -iname *.cpp | xargs clang-format -i

check-format:
	find ./src -type f ! -name "*.pb.h"  -iname *.h -o -iname *.cpp | xargs clang-format --dry-run --Werror
	find ./include -type f ! -name "*.pb.h"  -iname *.h -o -iname *.cpp | xargs clang-format --dry-run --Werror
	find ./test -type f ! -name "*.pb.h"  -iname *.h -o -iname *.cpp | xargs clang-format --dry-run --Werror

check-tidy:
	python3 ./scripts/run-clang-tidy.py -p build 

fix-tidy: 
	python3 ./scripts/run-clang-tidy.py -fix -p build

proto: 
	protoc -I="src/proto" --cpp_out="src/proto" src/proto/*.proto
