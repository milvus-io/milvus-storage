.PHONY: build clean test test-all test-cloud-storage test-talon benchmark-talon \
	python-lib java-lib package \
	fix-format check-format fix-tidy check-tidy \
	check-error-ratchet update-error-ratchet \
	aws gcp azure aliyun tencent huawei

TEST_THREADS ?= 4
ifneq ($(TIDY_THREADS),)
	TIDY_THREAD_PARAMETER := -j ${TIDY_THREADS}
endif

# Build options (override via env: USE_ASAN=False make build)
use_asan := $(or $(USE_ASAN),True)
with_ut := $(or $(WITH_UT),True)
with_fiu := $(or $(WITH_FIU),True)
with_crt := $(or $(WITH_CRT),False)
with_talon := $(or $(WITH_TALON),False)
with_benchmark := $(or $(WITH_BENCHMARK),True)
build_type := $(or $(BUILD_TYPE),Release)
use_jni := $(or $(USE_JNI),False)
use_python_binding := $(or $(USE_PYTHON_BINDING),False)

libcxx_setting = -s:h compiler.cppstd=20 -s:b compiler.cppstd=20
uname_s := $(shell uname -s)
ifeq ($(uname_s),Darwin)
	libcxx_setting = -s:h compiler.libcxx=libc++ -s:h compiler.cppstd=20 -s:b compiler.libcxx=libc++ -s:b compiler.cppstd=20
	# Ensure enough Mach-O header padding so Conan's fix_apple_shared_install_name
	# (install_name_tool) can rewrite rpaths in dependency binaries like openssl.
	export LDFLAGS := $(LDFLAGS) -headerpad_max_install_names
endif

# Common conan flags
CONAN_SETTINGS = ${libcxx_setting} -s:h build_type=${build_type} -s:b build_type=${build_type}
CONAN_BASE = conan install .. --build=arrow --build=missing ${CONAN_SETTINGS}

build:
	# Example building debug mode: BUILD_TYPE=Debug WITH_UT=False make
	# Example building with fault injection: WITH_FIU=True make
	@echo "with_ut: ${with_ut}"
	@echo "with_benchmark: ${with_benchmark}"
	@echo "use_asan: ${use_asan}"
	@echo "build_type: ${build_type}"
	@echo "use_jni: ${use_jni}"
	@echo "use_python_binding: ${use_python_binding}"
	@echo "with_fiu: ${with_fiu}"
	@echo "with_crt: ${with_crt}"
	@echo "with_talon: ${with_talon}"

	mkdir -p build && cd build && \
	${CONAN_BASE} \
	-o "&:with_ut=${with_ut}" -o "&:with_benchmark=${with_benchmark}" -o "&:with_asan=${use_asan}" \
	-o "&:with_jni=${use_jni}" -o "&:with_python_binding=${use_python_binding}" -o "&:with_fiu=${with_fiu}" \
	-o "&:with_crt=${with_crt}" -o "&:with_talon=${with_talon}" && \
	conan build .. ${CONAN_SETTINGS} \
	-o "&:with_ut=${with_ut}" -o "&:with_benchmark=${with_benchmark}" -o "&:with_asan=${use_asan}" \
	-o "&:with_jni=${use_jni}" -o "&:with_python_binding=${use_python_binding}" -o "&:with_fiu=${with_fiu}" \
	-o "&:with_crt=${with_crt}" -o "&:with_talon=${with_talon}"

package: build
	mkdir -p build && cd build && \
	conan export .. --user=milvus --channel=dev

python-lib:
	@echo "Building library for Python bindings with hidden symbols..."
	@echo "with_fiu: ${with_fiu}"
	mkdir -p build && cd build && \
	${CONAN_BASE} \
	-o "&:with_ut=False" -o "&:with_benchmark=False" -o "&:with_asan=False" -o "&:with_jni=False" -o "&:with_jemalloc=False" \
	-o "&:with_python_binding=True" -o "&:with_fiu=${with_fiu}" && \
	conan build .. ${CONAN_SETTINGS} \
	-o "&:with_ut=False" -o "&:with_benchmark=False" -o "&:with_asan=False" -o "&:with_jni=False" -o "&:with_jemalloc=False" \
	-o "&:with_python_binding=True" -o "&:with_fiu=${with_fiu}"

java-lib:
	@echo "Building library for Java/Scala dependencies..."
	mkdir -p build && cd build && \
	${CONAN_BASE} \
	-o "&:with_ut=False" -o "&:with_benchmark=False" -o "&:with_asan=False" -o "&:with_jni=True" -o "&:with_jemalloc=False" \
	-o "&:with_python_binding=False" -o "&:with_fiu=${with_fiu}" && \
	conan build .. ${CONAN_SETTINGS} \
	-o "&:with_ut=False" -o "&:with_benchmark=False" -o "&:with_asan=False" -o "&:with_jni=True" -o "&:with_jemalloc=False" \
	-o "&:with_python_binding=False" -o "&:with_fiu=${with_fiu}"

clean:
	rm -rf build

test: build
	cd build/${build_type}/test && ./milvus_test && ./Test_FFI

test-all: build
	# running milvus_test and Test_FFI without minio
	./build/${build_type}/test/milvus_test && ./build/${build_type}/test/Test_FFI

	# use the sub-shell to split the diffent cloud env
	# running milvus_test with minio
	./scripts/setup_minio.sh
	(source ./scripts/minio_env.sh && ./build/${build_type}/test/milvus_test)
	(source ./scripts/minio_env.sh && ./build/${build_type}/test/Test_FFI)

	# running milvus_test with azurite
	./scripts/setup_azurite.sh
	(source ./scripts/azurite_env.sh && ./build/${build_type}/test/milvus_test)
	(source ./scripts/azurite_env.sh && ./build/${build_type}/test/Test_FFI)
	
# Test cloud storage with all providers or a specific one
# Usage: make test-cloud-storage     # Test all cloud providers
#        make test-cloud-storage aws  # Test AWS only
#        make test-cloud-storage gcp  # Test GCP only
# Available providers: aws, gcp, azure, aliyun, tencent, huawei
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
aws gcp azure aliyun tencent huawei: test-cloud-storage

# Talon read-through cache. These targets are "bare": they build with the Talon C
# SDK and run the test / benchmark against a stack that is ALREADY up. The stack
# (a containerized coordinator + worker backed by MinIO) is owned by
# scripts/setup_talon.sh, which builds the server images from the same pinned
# talon source the C SDK was fetched from. The canonical path is the orchestrator,
# which runs the already-built binary on the stack network (the only place the
# worker is reachable — see the setup_talon.sh header):
#
#   ./scripts/setup_talon.sh up      # bring the stack up (leave running)
#   ./scripts/setup_talon.sh test    # run the Talon gtest against it
#   ./scripts/setup_talon.sh down    # tear it down
#
# or in one self-cleaning shot:  ./scripts/setup_talon.sh e2e
#
# The targets below are the manual build+run equivalent, for a dev shell that is
# itself joined to the stack network; set TALON_COORDINATOR_ADDR to the
# coordinator's on-network address there (talon_env.sh's 127.0.0.1 default only
# fits a host-published stack, which the containerized stack is not).
test-talon:
	$(MAKE) build WITH_TALON=True
	export LD_LIBRARY_PATH="$$LD_LIBRARY_PATH:$(CURDIR)/build/${build_type}:$(CURDIR)/build/${build_type}/libs"; \
	. ./scripts/talon_env.sh; \
	./build/${build_type}/test/milvus_test --gtest_filter='Talon*'

# Built for measurement, not debugging: USE_ASAN=False (ASan perturbs timing and
# the default build turns it on) and WITH_UT=False (the unit-test build injects
# -fprofile-arcs coverage flags that the benchmark would inherit, and drags in
# Test_FFI, whose folly/libaio link is broken in non-ASan builds).
#
# The clean build is required, not just tidy: toggling these options on an existing
# build dir does not reliably reconfigure (conan/CMake keep the previously cached
# WITH_ASAN/WITH_UT, so a build left over from `test-talon` would still be ASan +
# would still build the broken Test_FFI). A fresh configure applies the options from
# scratch and also yields clean, reproducible throughput numbers.
benchmark-talon:
	$(MAKE) clean
	$(MAKE) build WITH_TALON=True USE_ASAN=False WITH_UT=False
	export LD_LIBRARY_PATH="$$LD_LIBRARY_PATH:$(CURDIR)/build/${build_type}:$(CURDIR)/build/${build_type}/libs"; \
	. ./scripts/talon_env.sh; \
	./build/${build_type}/benchmark/benchmark --benchmark_filter='TalonReaderBenchmark'

# Source directories for formatting
FORMAT_DIRS := ./src ./include ./test ./benchmark
FORMAT_FIND = $(foreach dir,$(FORMAT_DIRS),\
	find $(dir) -type f \( ! -name "*.pb.h" \) \( -iname "*.c" -o -iname "*.h" -o -iname "*.cpp" \))

fix-format:
	@set -e; for dir in $(FORMAT_DIRS); do \
		find $$dir -type f \( ! -name "*.pb.h" \) \( -iname "*.c" -o -iname "*.h" -o -iname "*.cpp" \) -exec clang-format -i {} +; \
	done

check-format:
	@set -e; for dir in $(FORMAT_DIRS); do \
		find $$dir -type f \( ! -name "*.pb.h" \) \( -iname "*.c" -o -iname "*.h" -o -iname "*.cpp" \) -exec clang-format --dry-run --Werror {} +; \
	done

check-error-ratchet:
	@bash scripts/error_handling_ratchet.sh check

update-error-ratchet:
	@bash scripts/error_handling_ratchet.sh update

check-tidy:
	python3 ./scripts/run-clang-tidy.py -p build/${build_type}

fix-tidy:
	python3 ./scripts/run-clang-tidy.py -fix -p build/${build_type}
