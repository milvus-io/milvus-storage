include ../cpp/build/conanbuildinfo.mak

CFLAGS              += $(CONAN_CFLAGS)
CXXFLAGS            += $(CONAN_CXXFLAGS)
INCLUDE_DIRS        = $(CONAN_INCLUDE_DIRS_ARROW) $(CONAN_INCLUDE_DIRS_MILVUS-STORAGE)
CPPFLAGS            += $(addprefix -I, $(INCLUDE_DIRS))
LDFLAGS             += $(addprefix -L, $(CONAN_LIB_DIRS_MILVUS-STORAGE))

.EXPORT_ALL_VARIABLES:
.PHONY: build 

build:
	CGO_CFLAGS="$(CPPFLAGS)" CGO_LDFLAGS="$(LDFLAGS)" go build ./...

test:
	CGO_CFLAGS="$(CPPFLAGS)" CGO_LDFLAGS="$(LDFLAGS)" go test -timeout 30s ./...

proto:
	mkdir -p proto/manifest_proto
	mkdir -p proto/schema_proto
	protoc -I="proto" --go_out=paths=source_relative:./proto/manifest_proto proto/manifest.proto
	protoc -I="proto" --go_out=paths=source_relative:./proto/schema_proto proto/storage_schema.proto

