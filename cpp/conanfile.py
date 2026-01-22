from conan.tools.microsoft import is_msvc, msvc_runtime_flag
from conan.tools.build import check_min_cppstd
from conan.tools.scm import Version
from conan import ConanFile
from conan.tools.cmake import CMake, CMakeDeps, CMakeToolchain, cmake_layout
from conan.errors import ConanInvalidConfiguration
from conans import tools
import os

required_conan_version = ">=1.60.0"


class StorageConan(ConanFile):
    name = "milvus-storage"
    description = "empty"
    topics = ("vector", "cloud", "ann")
    url = "https://github.com/milvus-io/milvus-storage"
    homepage = "https://github.com/milvus-io/milvus-storage"
    license = "Apache-2.0"
    version = "0.1.0"

    settings = "os", "arch", "compiler", "build_type"
    options = {
        "shared": [True, False],
        "fPIC": [True, False],
        "with_asan": [True, False],
        "with_profiler": [True, False],
        "with_ut": [True, False],
        "with_benchmark": [True, False],
        "with_jemalloc": [True, False],
        "with_azure": [True, False],
        "with_jni": [True, False],
        "with_python_binding": [True, False],
        "with_vortex": [True, False],
        "with_lance": [True, False],
    }
    default_options = {
        "shared": True,
        "fPIC": False,
        "with_asan": False,
        "with_profiler": False,
        "with_ut": True,
        "with_benchmark": True,
        "with_azure": True,
        "with_jemalloc": True,
        "with_jni": False,
        "with_python_binding": False,
        "with_vortex": False,
        "with_lance": False,
        "glog:with_gflags": True,
        "glog:shared": True,
        "aws-sdk-cpp:config": True,
        "aws-sdk-cpp:text-to-speech": False,
        "aws-sdk-cpp:transfer": False,
        "arrow:with_s3": True,
        "arrow:filesystem_layer": True,
        "arrow:dataset_modules": True,
        "arrow:parquet": True,
        "arrow:with_re2": True,
        "arrow:with_zstd": True,
        "arrow:with_boost": True,
        "arrow:with_thrift": True,
        "arrow:encryption": True,
        "arrow:with_openssl": True,
        "boost:without_test": True,
        "boost:without_stacktrace": True,
        "fmt:header_only": True,
        "xz_utils:shared": True,
    }
    exports_sources = (
        "src/*",
        "include/*",
        "thirdparty/*",
        "test/*",
        "benchmark/*",
        "CMakeLists.txt",
        "*.cmake",
        "conanfile.py",
        "ffi_exports.map",
    )

    @property
    def _minimum_cpp_standard(self):
        return 17

    @property
    def _minimum_compilers_version(self):
        return {
            "gcc": "8",
            "Visual Studio": "16",
            "clang": "6",
            "apple-clang": "10",
        }

    def config_options(self):
        if self.settings.os == "Windows":
            self.options.rm_safe("fPIC")

    def configure(self):
        if self.options.shared:
            self.options.rm_safe("fPIC")
        if self.settings.arch not in ("x86_64", "x86"):
            del self.options["folly"].use_sse4_2
        self.options["arrow"].with_jemalloc = self.options.with_jemalloc
        self.options["arrow"].with_azure = self.options.with_azure
        if self.options.with_jni and self.settings.os != "Macos":
            self.options["arrow"].shared = True
            self.options["arrow"].acero = True
            self.options["boost"].shared = True
            self.options["protobuf"].shared = True
            self.options["folly"].shared = True
            self.options["aws-sdk-cpp"].shared = True
            self.options["libavrocpp"].shared = True
            self.options["openssl"].shared = True
            self.options["libcurl"].shared = True
            self.options["zlib"].shared = True
            self.options["zstd"].shared = True
            self.options["glog"].shared = True
            self.options["gflags"].shared = True

    def requirements(self):
        self.requires("xz_utils/5.4.0#a6d90890193dc851fa0d470163271c7a") # required by folly
        self.requires("glog/0.6.0#d22ebf9111fed68de86b0fa6bf6f9c3f") # required by folly
        self.requires("zstd/1.5.5#34e9debe03bf0964834a09dfbc31a5dd") # required by folly && arrow
        self.requires("fmt/9.1.0#95259249fb7ef8c6b5674a40b00abba3")  # required by folly
        self.requires("boost/1.82.0#744a17160ebb5838e9115eab4d6d0c06")
        self.requires("arrow/17.0.0@milvus/dev-2.6#7af258a853e20887f9969f713110aac8")
        self.requires("openssl/3.1.2#02594c4c0a6e2b4feb3cd15119993597")
        self.requires("protobuf/3.21.4#fd372371d994b8585742ca42c12337f9")
        self.requires("zlib/1.2.13#df233e6bed99052f285331b9f54d9070")
        self.requires("libcurl/7.86.0#bbc887fae3341b3cb776c601f814df05")
        self.requires("folly/2023.10.30.08@milvus/dev#81d7729cd4013a1b708af3340a3b04d9")
        self.requires("libavrocpp/1.11.3")
        self.requires("google-cloud-cpp/2.5.0@milvus/2.4#c5591ab30b26b53ea6068af6f07128d3")
        self.requires("googleapis/cci.20221108#65604e1b3b9a6b363044da625b201a2a")
        if self.options.with_benchmark:
            self.requires("benchmark/1.7.0")
        if self.options.with_ut:
            self.requires("gtest/1.13.0")
        if self.settings.os == "Macos":
            # Macos M1 cannot use jemalloc and arrow azure fs
            self.options["arrow"].with_azure = False
            self.options["arrow"].with_jemalloc = False
        else:
            self.requires("libunwind/1.7.2")

    def validate(self):
        if self.settings.compiler.get_safe("cppstd"):
            check_min_cppstd(self, self._minimum_cpp_standard)
        min_version = self._minimum_compilers_version.get(str(self.settings.compiler))
        if not min_version:
            self.output.warn(
                "{} recipe lacks information about the {} compiler support.".format(
                    self.name, self.settings.compiler
                )
            )
        else:
            if Version(self.settings.compiler.version) < min_version:
                raise ConanInvalidConfiguration(
                    "{} requires C++{} support. The current compiler {} {} does not support it.".format(
                        self.name,
                        self._minimum_cpp_standard,
                        self.settings.compiler,
                        self.settings.compiler.version,
                    )
                )

    def layout(self):
        cmake_layout(self)

    def generate(self):
        tc = CMakeToolchain(self)
        tc.variables["CMAKE_POSITION_INDEPENDENT_CODE"] = self.options.get_safe(
            "fPIC", True
        )
        # Relocatable shared lib on Macos
        tc.cache_variables["CMAKE_POLICY_DEFAULT_CMP0042"] = "NEW"
        # Honor BUILD_SHARED_LIBS from conan_toolchain (see https://github.com/conan-io/conan/issues/11840)
        tc.cache_variables["CMAKE_POLICY_DEFAULT_CMP0077"] = "NEW"

        cxx_std_flag = tools.cppstd_flag(self.settings)
        cxx_std_value = (
            cxx_std_flag.split("=")[1]
            if cxx_std_flag
            else "c++{}".format(self._minimum_cpp_standard)
        )
        tc.variables["CXX_STD"] = cxx_std_value
        if is_msvc(self):
            tc.variables["MSVC_LANGUAGE_VERSION"] = cxx_std_value
            tc.variables["MSVC_ENABLE_ALL_WARNINGS"] = False
            tc.variables["MSVC_USE_STATIC_RUNTIME"] = "MT" in msvc_runtime_flag(self)
        tc.variables["WITH_ASAN"] = self.options.with_asan
        tc.variables["WITH_PROFILER"] = self.options.with_profiler
        tc.variables["WITH_UT"] = self.options.with_ut
        tc.variables["WITH_BENCHMARK"] = self.options.with_benchmark
        tc.variables["WITH_AZURE_FS"] = self.options.with_azure
        tc.variables["ARROW_WITH_JEMALLOC"] = self.options.with_jemalloc
        tc.variables["WITH_JNI"] = self.options.with_jni
        tc.variables["WITH_PYTHON_BINDING"] = self.options.with_python_binding
        tc.variables["WITH_VORTEX"] = self.options.with_vortex
        tc.variables["WITH_LANCE"] = self.options.with_lance

        # Set JAVA_HOME for JNI compilation
        if self.options.with_jni:
            java_home = os.environ.get("JAVA_HOME")
            if java_home:
                tc.variables["JAVA_HOME"] = java_home
                self.output.info(f"Using JAVA_HOME: {java_home}")
            else:
                self.output.warn("JNI enabled but JAVA_HOME not set")

        tc.generate()

        deps = CMakeDeps(self)
        deps.generate()

    def build(self):
        cmake = CMake(self)
        cmake.configure()
        cmake.build()

    def package(self):
        cmake = CMake(self)
        cmake.install()
        self.copy("*_c.h")
        
    def imports(self):
        dest_dir = "build/{}/libs".format(self.settings.build_type)

        # export all dynamic libs
        self.copy("*.dll", dst=dest_dir, src="bin")
        self.copy("*.so*", dst=dest_dir, src="lib")
        self.copy("*.dylib*", dst=dest_dir, src="lib")

    def package_info(self):
        self.cpp_info.set_property("cmake_file_name", "storage")
        self.cpp_info.set_property("cmake_target_name", "storage::storage")
        self.cpp_info.set_property("pkg_config_name", "libstorage")

        self.cpp_info.components["libstorage"].libs = ["milvus-storage"]

        if self.options.with_ut:
            self.cpp_info.components["libstorage"].requires.append("gtest::gtest")

        self.cpp_info.filenames["cmake_find_package"] = "storage"
        self.cpp_info.filenames["cmake_find_package_multi"] = "storage"
        self.cpp_info.names["cmake_find_package"] = "storage"
        self.cpp_info.names["cmake_find_package_multi"] = "storage"
        self.cpp_info.names["pkg_config"] = "libmilvus-storage"
        self.cpp_info.components["libstorage"].names["cmake_find_package"] = "storage"
        self.cpp_info.components["libstorage"].names[
            "cmake_find_package_multi"
        ] = "storage"

        self.cpp_info.components["libstorage"].set_property(
            "cmake_target_name", "storage::storage"
        )
        self.cpp_info.components["libstorage"].set_property(
            "pkg_config_name", "libstorage"
        )

        if self.options.with_jni:
            self.cpp_info.builddirs = ["build"]
