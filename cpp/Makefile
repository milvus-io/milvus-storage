.PHONY: build clean test format

TEST_THREADS ?= 4
ifneq ($(TIDY_THREADS),)
	TIDY_THREAD_PARAMETER := -j ${TIDY_THREADS}
endif

use_asan ?= True
with_ut ?= True  
build_type ?= Release
use_jni ?= False

ifdef USE_ASAN
use_asan = $(USE_ASAN)
endif
ifdef WITH_UT
with_ut = $(WITH_UT)
endif
ifdef BUILD_TYPE
build_type = $(BUILD_TYPE)
endif
ifdef USE_JNI
use_jni = $(USE_JNI)
endif

build:
	# Example building debug mode: BUILD_TYPE=Debug WITH_UT=False make
	@echo "with_ut: ${with_ut}"
	@echo "use_asan: ${use_asan}"
	@echo "build_type: ${build_type}"
	@echo "use_jni: ${use_jni}"
	
	mkdir -p build && cd build && \
	conan install .. --build=missing -s build_type=${build_type} --update \
	-o with_ut=${with_ut} -o with_asan=${use_asan} -o with_jni=${use_jni} && \
	conan build .. 

package: build
	mkdir -p build && cd build && \
	conan export .. milvus-storage/0.1.0@milvus/dev

clean:
	rm -rf build

test: build
	cd build/Release/test && ./milvus_test

# Test cloud storage with all providers or a specific one
# Usage: make test-cloud-storage     # Test all cloud providers
#        make test-cloud-storage aws  # Test AWS only
#        make test-cloud-storage gcp  # Test GCP only
# Available providers: aws, gcp, azure, aliyun, tencent
test-cloud-storage: build
	@if [ -n "$(filter aws gcp azure aliyun tencent,$(MAKECMDGOALS))" ]; then \
		PROVIDER=$(filter aws gcp azure aliyun tencent,$(MAKECMDGOALS)); \
		echo "Testing specific cloud provider: $$PROVIDER"; \
		bash test/packed/run_cloud_test.sh $$PROVIDER; \
	else \
		echo "Testing all cloud providers"; \
		bash test/packed/run_cloud_test.sh; \
	fi

# Individual cloud provider targets
aws: test-cloud-storage
gcp: test-cloud-storage
azure: test-cloud-storage
aliyun: test-cloud-storage
tencent: test-cloud-storage

fix-format:
	find ./src -type f \( ! -name "*.pb.h" \) \( -iname "*.h" -o -iname "*.cpp" \) -exec clang-format -i {} +
	find ./include -type f \( ! -name "*.pb.h" \) \( -iname "*.h" -o -iname "*.cpp" \) -exec clang-format -i {} +
	find ./test -type f \( ! -name "*.pb.h" \) \( -iname "*.c" -o -iname "*.h" -o -iname "*.cpp" \) -exec clang-format -i {} +
	find ./benchmark -type f \( ! -name "*.pb.h" \) \( -iname "*.h" -o -iname "*.cpp" \) -exec clang-format -i {} +

check-format:
	find ./src -type f \( ! -name "*.pb.h" \) \( -iname "*.h" -o -iname "*.cpp" \) -exec clang-format --dry-run --Werror {} +
	find ./include -type f \( ! -name "*.pb.h" \) \( -iname "*.h" -o -iname "*.cpp" \) -exec clang-format --dry-run --Werror {} +
	find ./test -type f \( ! -name "*.pb.h" \) \( -iname "*.c" -o -iname "*.h" -o -iname "*.cpp" \) -print0 | xargs -0 clang-format --dry-run --Werror
	find ./benchmark -type f \( ! -name "*.pb.h" \) \( -iname "*.h" -o -iname "*.cpp" \) -exec clang-format --dry-run --Werror {} +

check-tidy:
	python3 ./scripts/run-clang-tidy.py -p build/Release

fix-tidy: 
	python3 ./scripts/run-clang-tidy.py -fix -p build/Release

proto: 
	protoc -I="src/proto" --cpp_out="src/proto" src/proto/*.proto
