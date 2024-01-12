.EXPORT_ALL_VARIABLES:
.PHONY: proto 

proto:
	mkdir -p proto/manifest_proto
	mkdir -p proto/schema_proto
	protoc -I="proto" --go_out=paths=source_relative:./proto/manifest_proto proto/manifest.proto
	protoc -I="proto" --go_out=paths=source_relative:./proto/schema_proto proto/storage_schema.proto

