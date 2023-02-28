package manifest

import (
	"github.com/milvus-io/milvus-storage-format/internal/fs"
)

type DataFile struct {
	path string
}

func (d *DataFile) Path() string {
	return d.path
}

func NewDataFile(path string) *DataFile {
	return &DataFile{path: path}
}

type Manifest struct {
	dataFiles []*DataFile
}

func (m *Manifest) AddDataFile(file *DataFile) {
	m.dataFiles = append(m.dataFiles, file)
}

func (m *Manifest) DataFiles() []*DataFile {
	return m.dataFiles
}

func NewManifest() *Manifest {
	return &Manifest{}
}

func WriteManifestFile(fs fs.Fs, manifest *Manifest) error {
	return nil
}
