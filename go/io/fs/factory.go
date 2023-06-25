package fs

import (
	"github.com/milvus-io/milvus-storage-format/storage/options"
)

type Factory struct {
}

func (f *Factory) Create(fsType options.FsType) Fs {
	switch fsType {
	case options.InMemory:
		return NewMemoryFs()
	case options.LocalFS:
		return NewLocalFs()
	default:
		panic("unknown fs type")
	}
}

func NewFsFactory() *Factory {
	return &Factory{}
}
