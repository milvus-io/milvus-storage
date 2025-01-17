include ../cpp/build/conanbuildinfo.mak

MILVUS_STORAGE_ROOT = $(abspath $(CURDIR)/..)
ARROW_INCLUDE_DIR = $(abspath $(MILVUS_STORAGE_ROOT)/go/)
MILVUS_STORAGE_LD_DIR = $(abspath $(MILVUS_STORAGE_ROOT)/cpp/build/Release)

INCLUDE_DIRS        = $(CONAN_INCLUDE_DIRS_ARROW) $(ARROW_INCLUDE_DIR)
CGO_CFLAGS          += $(addprefix -I, $(INCLUDE_DIRS))
CGO_LDFLAGS         += $(addprefix -L, $(MILVUS_STORAGE_LD_DIR)) -Wl,-rpath,$(MILVUS_STORAGE_LD_DIR)

.EXPORT_ALL_VARIABLES:
.PHONY: build test proto

build:
	@echo "CGO_CFLAGS: $(CGO_CFLAGS)"
	@echo "CGO_LDFLAGS: $(CGO_LDFLAGS)"
	@. $(PWD)/scripts/setenv.sh && \
	CGO_CFLAGS="$(CGO_CFLAGS)" CGO_LDFLAGS="$(CGO_LDFLAGS) -lmilvus-storage" go build ./...

test:
	@. $(PWD)/scripts/setenv.sh && LD_LIBRARY_PATH=$(MILVUS_STORAGE_LD_DIR):$$LD_LIBRARY_PATH \
		CGO_CFLAGS="$(CGO_CFLAGS)" \
		CGO_LDFLAGS="$(CGO_LDFLAGS) -lmilvus-storage" \
		go test -count=1 -timeout 30s ./... -gcflags "all=-N -l" -o gdb/

proto:
	mkdir -p proto/manifest_proto
	mkdir -p proto/schema_proto
	protoc -I="proto" --go_out=paths=source_relative:./proto/manifest_proto proto/manifest.proto

	protoc -I="proto" --go_out=paths=source_relative:./proto/schema_proto proto/storage_schema.proto
