package fs

import "github.com/milvus-io/milvus-storage-format/internal/fs/file"

type MemoryFs struct {
	files map[string]*file.MemoryFile
}

func (m *MemoryFs) OpenFile(path string) (file.File, error) {
	if f, ok := m.files[path]; ok {
		return file.NewMemoryFile(f.Bytes()), nil
	}
	f := file.NewMemoryFile(nil)
	m.files[path] = f
	return f, nil
}

func NewMemoryFs() *MemoryFs {
	return &MemoryFs{
		files: make(map[string]*file.MemoryFile),
	}
}
