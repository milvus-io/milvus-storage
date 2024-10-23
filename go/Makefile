include ../cpp/build/conanbuildinfo.mak

MILVUS_STORAGE_ROOT = $(abspath $(CURDIR)/..)
MILVUS_STORAGE_INCLUDE_DIR = $(abspath $(MILVUS_STORAGE_ROOT)/cpp/include)
MILVUS_STORAGE_LD_DIR = $(abspath $(MILVUS_STORAGE_ROOT)/cpp/build/Release)

CFLAGS              += $(CONAN_CFLAGS)
CXXFLAGS            += $(CONAN_CXXFLAGS)
INCLUDE_DIRS        = $(CONAN_INCLUDE_DIRS_ARROW) $(MILVUS_STORAGE_INCLUDE_DIR)
CPPFLAGS            = $(addprefix -I, $(INCLUDE_DIRS))
LDFLAGS             += $(addprefix -L, $(MILVUS_STORAGE_LD_DIR))

.EXPORT_ALL_VARIABLES:
.PHONY: build 

build:
	CGO_CFLAGS="$(CPPFLAGS)" CGO_LDFLAGS="$(LDFLAGS)" go build ./...

test:
	CGO_CFLAGS="$(CPPFLAGS)" CGO_LDFLAGS="$(LDFLAGS) -Wl,-rpath,$(MILVUS_STORAGE_LD_DIR)" go test -timeout 30s ./...
proto:
	mkdir -p proto/manifest_proto
	mkdir -p proto/schema_proto
	protoc -I="proto" --go_out=paths=source_relative:./proto/manifest_proto proto/manifest.proto
	protoc -I="proto" --go_out=paths=source_relative:./proto/schema_proto proto/storage_schema.proto

