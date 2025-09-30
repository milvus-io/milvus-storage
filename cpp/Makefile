.PHONY: build build-no-jni clean test format

TEST_THREADS ?= 4
ifneq ($(TIDY_THREADS),)
	TIDY_THREAD_PARAMETER := -j ${TIDY_THREADS}
endif

use_asan = False
ifeq ($(USE_ASAN), True)
	use_asan = ${USE_ASAN}
endif

# JDK path for JNI compilation
JAVA_HOME ?= /usr/lib/jvm/default-java
ifneq ($(JDK_PATH),)
	JAVA_HOME = ${JDK_PATH}
endif

build:
	mkdir -p build && cd build && \
	JAVA_HOME=${JAVA_HOME} conan install .. --build=missing -s build_type=Release --update \
	-o with_ut=True -o with_asan=${use_asan} -o with_jni=True && \
	JAVA_HOME=${JAVA_HOME} conan build .. && \
	JAVA_HOME=${JAVA_HOME} conan install .. --install-folder . -g make

package: build
	mkdir -p build && cd build && \
	conan export .. milvus-storage/0.1.0@milvus/dev

debug:
	mkdir -p build && cd build && \
	JAVA_HOME=${JAVA_HOME} conan install .. --build=missing -s build_type=Debug -o with_asan=True -o with_jni=True && \
	JAVA_HOME=${JAVA_HOME} conan build ..

build-no-jni:
	mkdir -p build && cd build && \
	conan install .. --build=missing -s build_type=Release --update \
	-o with_ut=True -o with_asan=${use_asan} -o with_jni=False && \
	conan build .. && \
	conan install .. --install-folder . -g make

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
