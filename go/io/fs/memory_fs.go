package fs

import (
	"github.com/milvus-io/milvus-storage/io/fs/file"
)

type MemoryFs struct {
	files map[string]*file.MemoryFile
}

func (m *MemoryFs) List(path string) ([]FileEntry, error) {
	//TODO implement me
	panic("implement me")
}

func (m *MemoryFs) OpenFile(path string) (file.File, error) {
	if f, ok := m.files[path]; ok {
		return file.NewMemoryFile(f.Bytes()), nil
	}
	f := file.NewMemoryFile(nil)
	m.files[path] = f
	return f, nil
}

func (m *MemoryFs) Rename(path string, path2 string) error {
	if _, ok := m.files[path]; !ok {
		return nil
	}
	m.files[path2] = m.files[path]
	delete(m.files, path)
	return nil
}

func (m *MemoryFs) DeleteFile(path string) error {
	delete(m.files, path)
	return nil
}

func (m *MemoryFs) CreateDir(path string) error {
	return nil
}

func (m *MemoryFs) ReadFile(path string) ([]byte, error) {
	panic("implement me")
}

func (m *MemoryFs) Exist(path string) (bool, error) {
	panic("not implemented")
}

func NewMemoryFs() *MemoryFs {
	return &MemoryFs{
		files: make(map[string]*file.MemoryFile),
	}
}
