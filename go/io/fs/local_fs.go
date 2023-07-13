package fs

import (
	"github.com/milvus-io/milvus-storage-format/io/fs/file"
	"os"
)

type LocalFS struct{}

func (l *LocalFS) OpenFile(path string) (file.File, error) {
	open, err := os.OpenFile(path, os.O_RDWR|os.O_CREATE, 0666)
	if err != nil {
		return nil, err
	}
	return file.NewLocalFile(open), nil
}

func (l *LocalFS) Rename(src string, dst string) error {
	return os.Rename(src, dst)
}

func (l *LocalFS) DeleteFile(path string) error {
	return os.Remove(path)
}

func NewLocalFs() *LocalFS {
	return &LocalFS{}
}
