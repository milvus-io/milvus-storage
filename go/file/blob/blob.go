// Copyright 2023 Zilliz
// 
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// 
//     http://www.apache.org/licenses/LICENSE-2.0
// 
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

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
