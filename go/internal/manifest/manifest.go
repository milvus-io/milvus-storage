package manifest

import (
	"github.com/apache/arrow/go/v12/arrow"
	"github.com/milvus-io/milvus-storage-format/internal/fs"
)

type DataFile struct {
	path string
	cols []string
}

func (d *DataFile) Path() string {
	return d.path
}

func NewDataFile(path string) *DataFile {
	return &DataFile{path: path}
}

type ManifestV1 struct {
	dataFiles []*DataFile
}

func (m *ManifestV1) AddDataFiles(files ...*DataFile) {
	m.dataFiles = append(m.dataFiles, files...)
}

func (m *ManifestV1) DataFiles() []*DataFile {
	return m.dataFiles
}

func NewManifest() *ManifestV1 {
	return &ManifestV1{}
}

func WriteManifestFile(fs fs.Fs, manifest *ManifestV1) error {
	// TODO
	return nil
}

type ManifestV2 struct {
	schema  *arrow.Schema
	schemas []*arrow.Schema
	files   []*DataFile
}

func (m *ManifestV2) Schema() *arrow.Schema {
	return m.schema
}
