package fs

import "github.com/milvus-io/milvus-storage-format/io/fs/file"

type Fs interface {
	OpenFile(path string) (file.File, error)
	Rename(src string, dst string) error
	DeleteFile(path string) error
}
