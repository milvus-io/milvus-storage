from conan.tools.microsoft import is_msvc, msvc_runtime_flag
from conan.tools.build import check_min_cppstd
from conan.tools.scm import Version
from conan.tools import files
from conan import ConanFile
from conan.tools.cmake import CMake, CMakeDeps, CMakeToolchain, cmake_layout
from conan.errors import ConanInvalidConfiguration
from conans import tools
import os

required_conan_version = ">=1.54.0"


class StorageConan(ConanFile):
    name = "storage"
    description = "empty"
    topics = ("vector", "cloud", "ann")
    url = "https://github.com/milvus-io/milvus-storage"
    homepage = "https://github.com/milvus-io/milvus-storage"
    license = "Apache-2.0"

    settings = "os", "arch", "compiler", "build_type"
    options = {
        "shared": [True, False],
        "fPIC": [True, False],
        "with_asan": [True, False],
        "with_profiler": [True, False],
        "with_ut": [True, False],
    }
    default_options = {
        "shared": True,
        "fPIC": False,
        "with_asan": False,
        "with_profiler": False,
        "with_ut": True,
        # "arrow:with_s3": True,
        # "aws-sdk-cpp:config": True,
        # "aws-sdk-cpp:text-to-speech": False,
        # "aws-sdk-cpp:transfer": False,
        "arrow:filesystem_layer": True,
        "arrow:dataset_modules": True,
        "arrow:parquet": True,
        "arrow:with_re2": True,
        "arrow:with_zstd": True,
        "arrow:with_boost": True,
        "arrow:with_thrift": True,
        "arrow:with_jemalloc": True,
        "boost:without_test": True,
    }

    exports_sources = (
        "src/*",
        "thirdparty/*",
        "test/*",
        "CMakeLists.txt",
        "*.cmake",
        "conanfile.py",
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

    def requirements(self):
        self.requires("boost/1.81.0")
        self.requires("arrow/12.0.0")
        self.requires("protobuf/3.21.4")
        self.requires("glog/0.6.0")
        if self.options.with_ut:
            self.requires("gtest/1.13.0")

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
        files.rmdir(self, os.path.join(self.package_folder, "lib", "cmake"))
        files.rmdir(self, os.path.join(self.package_folder, "lib", "pkgconfig"))

    def package_info(self):
        self.cpp_info.set_property("cmake_file_name", "storage")
        self.cpp_info.set_property("cmake_target_name", "storage::storage")
        self.cpp_info.set_property("pkg_config_name", "libstorage")

        self.cpp_info.components["libstorage"].libs = ["storage"]

        self.cpp_info.components["libstorage"].requires = [
            "boost::uuid",
            "boost::algorithm",
        ]
        if self.options.with_ut:
            self.cpp_info.components["libstorage"].requires.append("gtest::gtest")

        self.cpp_info.filenames["cmake_find_package"] = "storage"
        self.cpp_info.filenames["cmake_find_package_multi"] = "storage"
        self.cpp_info.names["cmake_find_package"] = "storage"
        self.cpp_info.names["cmake_find_package_multi"] = "storage"
        self.cpp_info.names["pkg_config"] = "libstorage"
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
