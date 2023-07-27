package fragment

import (
	"github.com/milvus-io/milvus-storage-format/io/fs"
	"github.com/milvus-io/milvus-storage-format/storage/schema"
)

type pkType any
type DeleteFragmentVector []DeleteFragment
type DeleteFragment struct {
	id     int64
	schema *schema.Schema
	fs     fs.Fs
	data   map[pkType][]int64
}

func NewDeleteFragment(id int64, schema *schema.Schema, fs fs.Fs) *DeleteFragment {
	return &DeleteFragment{
		id:     id,
		schema: schema,
		fs:     fs,
		data:   make(map[pkType][]int64),
	}
}

func Make(f fs.Fs, s *schema.Schema, frag Fragment) DeleteFragment {
	// TODO: implement
	panic("implement me")
}
