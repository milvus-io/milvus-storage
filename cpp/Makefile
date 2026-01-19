.PHONY: build clean test format python-lib

TEST_THREADS ?= 4
ifneq ($(TIDY_THREADS),)
	TIDY_THREAD_PARAMETER := -j ${TIDY_THREADS}
endif

use_asan ?= True
with_ut ?= True
with_benchmark ?= True
build_type ?= Release
use_jni ?= False
use_python_binding ?= False
with_vortex ?= False
with_lance ?= False

ifdef USE_ASAN
use_asan = $(USE_ASAN)
endif
ifdef WITH_UT
with_ut = $(WITH_UT)
endif
ifdef WITH_BENCHMARK
with_benchmark = $(WITH_BENCHMARK)
endif
ifdef BUILD_TYPE
build_type = $(BUILD_TYPE)
endif
ifdef USE_JNI
use_jni = $(USE_JNI)
endif
ifdef USE_PYTHON_BINDING
use_python_binding = $(USE_PYTHON_BINDING)
endif
ifdef WITH_VORTEX
with_vortex = $(WITH_VORTEX)
endif
ifdef WITH_LANCE
with_lance = $(WITH_LANCE)
endif

libcxx_setting =
uname_s := $(shell uname -s)
ifeq ($(uname_s),Darwin)
	libcxx_setting = -s compiler.libcxx=libc++
endif

build: fix-format
	# Example building debug mode: BUILD_TYPE=Debug WITH_UT=False make
	@echo "with_ut: ${with_ut}"
	@echo "with_benchmark: ${with_benchmark}"
	@echo "use_asan: ${use_asan}"
	@echo "build_type: ${build_type}"
	@echo "use_jni: ${use_jni}"
	@echo "use_python_binding: ${use_python_binding}"
	@echo "with_vortex: ${with_vortex}"
	@echo "with_lance: ${with_lance}"

	mkdir -p build && cd build && \
	conan install .. --build=missing ${libcxx_setting} -s build_type=${build_type} --update \
	-o with_ut=${with_ut} -o with_benchmark=${with_benchmark} -o with_asan=${use_asan} \
	-o with_jni=${use_jni} -o with_python_binding=${use_python_binding} \
	-o with_vortex=${with_vortex} -o with_lance=${with_lance} && conan build ..

package: build
	mkdir -p build && cd build && \
	conan export .. milvus-storage/0.1.0@milvus/dev

python-lib:
	@echo "Building library for Python bindings with hidden symbols..."
	mkdir -p build && cd build && \
	conan install .. --build=missing ${libcxx_setting} -s build_type=Release --update \
	-o with_ut=False -o with_benchmark=False -o with_asan=False -o with_jni=False -o with_jemalloc=False -o with_python_binding=True -o with_azure=False -o with_vortex=True -o with_lance=True && \
	conan build .. 

java-lib:
	@echo "Building library for Java/Scala dependencies..."
	mkdir -p build && cd build && \
	conan install .. --build=missing ${libcxx_setting} -s build_type=Release --update \
	-o with_ut=False -o with_benchmark=False -o with_asan=False -o with_jni=True -o with_jemalloc=False -o with_python_binding=False -o with_azure=True -o with_vortex=True -o with_lance=True && \
	conan build .. 

clean:
	rm -rf build

test: build
	cd build/Release/test && ./milvus_test && ./Test_FFI

# Test cloud storage with all providers or a specific one
# Usage: make test-cloud-storage     # Test all cloud providers
#        make test-cloud-storage aws  # Test AWS only
#        make test-cloud-storage gcp  # Test GCP only
# Available providers: aws, gcp, azure, aliyun, tencent
test-cloud-storage: build
	@if [ -n "$(filter aws gcp azure aliyun tencent huawei, $(MAKECMDGOALS))" ]; then \
		PROVIDER=$(filter aws gcp azure aliyun tencent huawei, $(MAKECMDGOALS)); \
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
huawei: test-cloud-storage

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
