MILVUS_STORAGE_LD_DIR = $(abspath $(CURDIR)/build/Release)

.EXPORT_ALL_VARIABLES:
.PHONY: build test proto

build:
	@echo "CGO_CFLAGS: $(CGO_CFLAGS)"
	@echo "CGO_LDFLAGS: $(CGO_LDFLAGS)"
	@. $(PWD)/scripts/setenv.sh && \
	CGO_CFLAGS="$(CGO_CFLAGS)" CGO_LDFLAGS="$(CGO_LDFLAGS)" go build ./...

test:
	@. $(PWD)/scripts/setenv.sh && \
		CGO_CFLAGS="$(CGO_CFLAGS)" \
		CGO_LDFLAGS="$(CGO_LDFLAGS)" \
		go test -count=1 -timeout 30s ./... -gcflags "all=-N -l" -o gdb/

proto:
	mkdir -p proto/manifest_proto
	mkdir -p proto/schema_proto
	protoc -I="proto" --go_out=paths=source_relative:./proto/manifest_proto proto/manifest.proto

	protoc -I="proto" --go_out=paths=source_relative:./proto/schema_proto proto/storage_schema.proto
