package blob

import "github.com/milvus-io/milvus-storage/go/proto/manifest_proto"

type Blob struct {
	Name string
	Size int64
	File string
}

func (b Blob) ToProtobuf() *manifest_proto.Blob {
	blob := &manifest_proto.Blob{}
	blob.Name = b.Name
	blob.Size = b.Size
	blob.File = b.File
	return blob
}

func FromProtobuf(blob *manifest_proto.Blob) Blob {
	return Blob{
		Name: blob.Name,
		Size: blob.Size,
		File: blob.File,
	}
}
