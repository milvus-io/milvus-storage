required_conan_version = ">=2.0"

from conan.tools.microsoft import is_msvc, msvc_runtime_flag
from conan.tools.build import check_min_cppstd
from conan.tools.scm import Version
from conan import ConanFile
from conan.tools.cmake import CMake, CMakeDeps, CMakeToolchain, cmake_layout
from conan.tools.files import copy
from conan.errors import ConanInvalidConfiguration
import os


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
        "with_fiu": [True, False],
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
        "with_fiu": False,
        "folly/*:shared": True,
        "glog/*:with_gflags": True,
        "glog/*:shared": True,
        "gflags/*:shared": True,
        "openssl/*:shared": True,
        "aws-sdk-cpp/*:config": True,
        "aws-sdk-cpp/*:text-to-speech": False,
        "aws-sdk-cpp/*:transfer": False,
        "arrow/*:with_s3": True,
        "arrow/*:filesystem_layer": True,
        "arrow/*:dataset_modules": True,
        "arrow/*:parquet": True,
        "arrow/*:with_re2": True,
        "arrow/*:with_zstd": True,
        "arrow/*:with_boost": True,
        "arrow/*:with_thrift": True,
        "arrow/*:encryption": True,
        "arrow/*:with_openssl": True,
        "arrow/*:with_snappy": True,
        "arrow/*:with_lz4": True,
        "boost/*:without_test": True,
        "boost/*:without_stacktrace": True,
        "fmt/*:header_only": False,
        # xz_utils must be shared because glog (shared) depends on liblzma.
        # If xz_utils is static, auditwheel bundles glog.so but liblzma symbols
        # are missing, causing "undefined symbol: lzma_index_uncompressed_size".
        "xz_utils/*:shared": True,
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
        self.requires("xz_utils/5.4.5#fc4e36861e0a47ecd4a40a00e6d29ac8")
        self.requires("glog/0.7.1#a306e61d7b8311db8cb148ad62c48030")
        self.requires("zstd/1.5.5#70dc5eb8ea16708fc946fbac884c507e")
        self.requires("fmt/11.0.2#eb98daa559c7c59d591f4720dde4cd5c")
        self.requires("boost/1.83.0#4e8a94ac1b88312af95eded83cd81ca8", force=True)
        self.requires("aws-sdk-cpp/1.11.692@milvus/dev#c309ce91fa572fff68f9f4e36d477a04")
        self.requires("arrow/17.0.0@milvus/dev-2.6#c743ea7a6f2420ba5811b2be3df59892")
        self.requires("openssl/3.3.2#9f9f130d58e7c13e76bb8a559f0a6a8b")
        self.requires("zlib/1.3.1#8045430172a5f8d56ba001b14561b4ea")
        self.requires("libcurl/8.10.1#a3113369c86086b0e84231844e7ed0a9")
        self.requires("folly/2024.08.12.00@milvus/dev#f9b2bdf162c0ec47cb4e5404097b340d")
        self.requires("libavrocpp/1.12.1.1@milvus/dev#cde7bb587a29f6f233bae7e18b71815d")
        self.requires("google-cloud-cpp/2.28.0@milvus/dev#468918b43cec43624531a0340398cf43")
        # Force override transitive deps to align with milvus-common
        self.requires("protobuf/5.27.0@milvus/dev#42f031a96d21c230a6e05bcac4bdd633", force=True)
        self.requires("grpc/1.67.1@milvus/dev#efeaa484b59bffaa579004d5e82ec4fd", force=True, override=True)
        self.requires("abseil/20250127.0#481edcc75deb0efb16500f511f0f0a1c", force=True, override=True)
        self.requires("snappy/1.2.1#b940695c64ccbff63c1aabd4b1eee3f3", force=True, override=True)
        self.requires("lz4/1.9.4#7f0b5851453198536c14354ee30ca9ae", force=True, override=True)
        if self.options.with_benchmark:
            # don't use 1.7.0 which have core when `--help`.
            self.requires("benchmark/1.8.3")
        if self.options.with_ut:
            self.requires("gtest/1.15.0")
        if self.settings.os == "Macos":
            # Macos M1 cannot use jemalloc and arrow azure fs
            self.options["arrow"].with_azure = False
            self.options["arrow"].with_jemalloc = False
        else:
            self.requires("libunwind/1.8.1#748a981ace010b80163a08867b732e71")

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
        # CMake 4.x: use upper-case <PACKAGENAME>_ROOT variables in find_package
        tc.cache_variables["CMAKE_POLICY_DEFAULT_CMP0144"] = "NEW"

        cppstd = self.settings.compiler.get_safe("cppstd")
        if cppstd:
            cxx_std_value = "gnu++{}".format(cppstd[3:]) if cppstd.startswith("gnu") else "c++{}".format(cppstd)
        else:
            cxx_std_value = "c++{}".format(self._minimum_cpp_standard)
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
        tc.variables["WITH_FIU"] = self.options.with_fiu

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

        # Copy all dependency shared libraries into build/<build_type>/libs.
        # Only needed for GitHub CI: the unittest job runs on a separate runner
        # without Conan cache, so it relies on these libs being uploaded as artifacts.
        dest_dir = os.path.join(self.build_folder, "libs")
        os.makedirs(dest_dir, exist_ok=True)
        for dep in self.dependencies.values():
            dep_cpp = dep.cpp_info
            if dep_cpp.libdirs:
                for libdir in dep_cpp.libdirs:
                    copy(self, "*.so*", src=libdir, dst=dest_dir)
                    copy(self, "*.dylib*", src=libdir, dst=dest_dir)
                    copy(self, "*.dll", src=libdir, dst=dest_dir)

    def build(self):
        cmake = CMake(self)
        cmake.configure()
        cmake.build()

    def package(self):
        cmake = CMake(self)
        cmake.install()
        copy(self, "*_c.h", src=self.source_folder, dst=os.path.join(self.package_folder, "include"))

    def package_info(self):
        self.cpp_info.set_property("cmake_file_name", "storage")
        self.cpp_info.set_property("cmake_target_name", "storage::storage")
        self.cpp_info.set_property("pkg_config_name", "libstorage")

        self.cpp_info.components["libstorage"].libs = ["milvus-storage"]

        if self.options.with_ut:
            self.cpp_info.components["libstorage"].requires.append("gtest::gtest")

        self.cpp_info.components["libstorage"].set_property(
            "cmake_target_name", "storage::storage"
        )
        self.cpp_info.components["libstorage"].set_property(
            "pkg_config_name", "libstorage"
        )

        if self.options.with_jni:
            self.cpp_info.builddirs = ["build"]
