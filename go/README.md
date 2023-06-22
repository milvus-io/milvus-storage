A lib to store and query scalar and vector data.


## install dependencies

```bash
go install google.golang.org/protobuf/cmd/protoc-gen-go@latest
go install google.golang.org/grpc/cmd/protoc-gen-go-grpc@latest
````
## generate proto

```bash
cd    proto
mkdir manifest_proto
mkdir schema_proto
protoc --go_out=./manifest_proto --go_opt=paths=source_relative manifest_proto 
protoc --go_out=./schema_proto --go_opt=paths=source_relative schema.proto

```
