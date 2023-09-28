package fs

import (
	"net/url"

	"github.com/milvus-io/milvus-storage/go/storage/options"
)

type Factory struct {
}

func (f *Factory) Create(fsType options.FsType, uri *url.URL) (Fs, error) {
	switch fsType {
	case options.InMemory:
		return NewMemoryFs(), nil
	case options.LocalFS:
		return NewLocalFs(uri), nil
	case options.S3:
		return NewMinioFs(uri)
	default:
		panic("unknown fs type")
	}
}

func NewFsFactory() *Factory {
	return &Factory{}
}
