include ../cpp/build/conanbuildinfo.mak

MILVUS_STORAGE_ROOT = $(abspath $(CURDIR)/..)
MILVUS_STORAGE_INCLUDE_DIR = $(abspath $(MILVUS_STORAGE_ROOT)/go/)
MILVUS_STORAGE_LD_DIR = $(abspath $(MILVUS_STORAGE_ROOT)/cpp/build/Release)

CFLAGS              += $(CONAN_CFLAGS)
CXXFLAGS            += $(CONAN_CXXFLAGS)
INCLUDE_DIRS        = $(CONAN_INCLUDE_DIRS_ARROW) $(MILVUS_STORAGE_INCLUDE_DIR)
CPPFLAGS            += $(addprefix -I, $(INCLUDE_DIRS))
LDFLAGS             += $(addprefix -L, $(MILVUS_STORAGE_LD_DIR)) -Wl,-rpath,$(MILVUS_STORAGE_LD_DIR)
CGO_CFLAGS = "$(shell pkg-config --cflags arrow libstorage)"
CGO_LDFLAGS = "$(shell pkg-config --libs arrow libstorage)"
.EXPORT_ALL_VARIABLES:
.PHONY: build test proto

build:
	@echo "CGO_CFLAGS: $(CPPFLAGS)"
	@echo "CGO_LDFLAGS: $(LDFLAGS)"
	@. $(PWD)/scripts/setenv.sh && \
	CGO_CFLAGS="$(CPPFLAGS)" CGO_LDFLAGS="$(LDFLAGS) -lmilvus-storage" go build ./...

test:
	@. $(PWD)/scripts/setenv.sh && LD_LIBRARY_PATH=$(MILVUS_STORAGE_LD_DIR):$$LD_LIBRARY_PATH \
		CGO_CFLAGS="$(CPPFLAGS)" \
		CGO_LDFLAGS="$(LDFLAGS) -lmilvus-storage" \
		go test -count=1 -timeout 30s ./... -gcflags "all=-N -l" -o gdb/

proto:
	mkdir -p proto/manifest_proto
	mkdir -p proto/schema_proto
	protoc -I="proto" --go_out=paths=source_relative:./proto/manifest_proto proto/manifest.proto

	protoc -I="proto" --go_out=paths=source_relative:./proto/schema_proto proto/storage_schema.proto
