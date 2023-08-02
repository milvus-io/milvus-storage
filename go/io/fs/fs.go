package fs

import (
	"github.com/milvus-io/milvus-storage-format/io/fs/file"
	"os"
)

type Fs interface {
	OpenFile(path string) (file.File, error)
	Rename(src string, dst string) error
	DeleteFile(path string) error
	CreateDir(path string) error
	List(path string) ([]os.DirEntry, error)
	ReadFile(path string) ([]byte, error)
}
