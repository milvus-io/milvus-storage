package fs

import (
	"github.com/milvus-io/milvus-storage-format/storage/options/option"
)

type Factory struct {
}

func (f *Factory) Create(fsType option.FsType) Fs {
	switch fsType {
	case option.InMemory:
		return NewMemoryFs()
	case option.LocalFS:
		return NewLocalFs()
	default:
		panic("unknown fs type")
	}
}

func NewFsFactory() *Factory {
	return &Factory{}
}
