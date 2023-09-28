package fs

import (
	"os"
	"path/filepath"

	"github.com/milvus-io/milvus-storage/go/common/log"
	"github.com/milvus-io/milvus-storage/go/io/fs/file"
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

func (l *LocalFS) List(path string) ([]FileEntry, error) {
	entries, err := os.ReadDir(path)
	if err != nil {
		log.Error(err.Error())
		return nil, err
	}

	ret := make([]FileEntry, 0, len(entries))
	for _, entry := range entries {
		ret = append(ret, FileEntry{Path: filepath.Join(path, entry.Name())})
	}

	return ret, nil
}

func (l *LocalFS) ReadFile(path string) ([]byte, error) {
	return os.ReadFile(path)
}

func (l *LocalFS) Exist(path string) (bool, error) {
	panic("not implemented")
}

func (l *LocalFS) Path() string {
	return ""
}

func NewLocalFs() *LocalFS {
	return &LocalFS{}
}
