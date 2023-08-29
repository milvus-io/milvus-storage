package fs

import (
	"github.com/milvus-io/milvus-storage/go/io/fs/file"
)

type Fs interface {
	OpenFile(path string) (file.File, error)
	Rename(src string, dst string) error
	DeleteFile(path string) error
	CreateDir(path string) error
	List(path string) ([]FileEntry, error)
	ReadFile(path string) ([]byte, error)
	Exist(path string) (bool, error)
}
type FileEntry struct {
	Path string
}
