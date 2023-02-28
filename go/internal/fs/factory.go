package fs

import (
	"github.com/milvus-io/milvus-storage-format/options"
)

type FsFactory struct {
}

func (f *FsFactory) Create(fsType options.FsType) Fs {
	switch fsType {
	case options.InMemory:
		return NewMemoryFs()
	default:
		panic("unknown fs type")
	}
}

func NewFsFactory() *FsFactory {
	return &FsFactory{}
}
