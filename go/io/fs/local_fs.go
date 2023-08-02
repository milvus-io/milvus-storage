package fs

import (
	"github.com/milvus-io/milvus-storage-format/common/log"
	"github.com/milvus-io/milvus-storage-format/io/fs/file"
	"os"
	"path/filepath"
)

type LocalFS struct{}

func (l *LocalFS) OpenFile(path string) (file.File, error) {
	// Extract the directory from the path
	dir := filepath.Dir(path)
	// Create the directory (including all necessary parent directories)
	err := os.MkdirAll(dir, os.ModePerm)
	if err != nil {
		return nil, err
	}
	open, err := os.OpenFile(path, os.O_RDWR|os.O_CREATE, 0666)
	if err != nil {
		return nil, err
	}
	return file.NewLocalFile(open), nil
}

// Rename renames (moves) a file. If newpath already exists and is not a directory, Rename replaces it.
func (l *LocalFS) Rename(src string, dst string) error {
	return os.Rename(src, dst)
}

func (l *LocalFS) DeleteFile(path string) error {
	return os.Remove(path)
}

func (l *LocalFS) CreateDir(path string) error {
	err := os.MkdirAll(path, os.ModePerm)
	if err != nil && !os.IsExist(err) {
		log.Error(err.Error())
	}
	return nil
}

func (l *LocalFS) List(path string) ([]os.DirEntry, error) {
	entry, err := os.ReadDir(path)
	if err != nil {
		log.Error(err.Error())
		return nil, err
	}
	return entry, nil
}

func (l *LocalFS) ReadFile(path string) ([]byte, error) {
	return os.ReadFile(path)
}

func NewLocalFs() *LocalFS {
	return &LocalFS{}
}
