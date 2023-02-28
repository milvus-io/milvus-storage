package fs

import "github.com/milvus-io/milvus-storage-format/internal/fs/file"

type Fs interface {
	OpenFile(path string) (file.File, error)
}
