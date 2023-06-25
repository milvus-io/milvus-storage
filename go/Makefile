.EXPORT_ALL_VARIABLES:

define HELP_INFO
Usage:
        make <Target>

Target:
        all    	build all executables (default)
        protos  compile server protobuf files
        prepare	prepare dependencies
        clean   clean artifacts
endef


.PHONY: all
all: prepare\
	 protos \

.PHONY: prepare
prepare:
	go install google.golang.org/protobuf/cmd/protoc-gen-go@latest
	go install google.golang.org/grpc/cmd/protoc-gen-go-grpc@latest
	go install github.com/grpc-ecosystem/grpc-gateway/v2/protoc-gen-grpc-gateway@latest


.PHONY: protos
protos:
	$(MAKE) -C proto/mainfest
	$(MAKE) -C proto/scheme

.PHONY: clean-protos
clean-protos:
	$(MAKE) -C proto/mainfest clean
	$(MAKE) -C proto/scheme clean


.PHONY: clean
clean: clean-protos \

