package fs

import (
	"net/url"

	"github.com/milvus-io/milvus-storage/storage/options/option"
)

type Factory struct {
}

func (f *Factory) Create(fsType option.FsType, uri *url.URL) (Fs, error) {
	switch fsType {
	case option.InMemory:
		return NewMemoryFs(), nil
	case option.LocalFS:
		return NewLocalFs(), nil
	case option.S3:
		return NewMinioFs(uri)
	default:
		panic("unknown fs type")
	}
}

func NewFsFactory() *Factory {
	return &Factory{}
}
