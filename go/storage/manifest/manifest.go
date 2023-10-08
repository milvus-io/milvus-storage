package manifest

import (
	"fmt"

	"github.com/apache/arrow/go/v12/arrow"
	"github.com/golang/protobuf/proto"
	"github.com/milvus-io/milvus-storage/go/common/log"
	"github.com/milvus-io/milvus-storage/go/file/blob"
	"github.com/milvus-io/milvus-storage/go/file/fragment"
	"github.com/milvus-io/milvus-storage/go/io/fs"
	"github.com/milvus-io/milvus-storage/go/io/fs/file"
	"github.com/milvus-io/milvus-storage/go/proto/manifest_proto"
	"github.com/milvus-io/milvus-storage/go/storage/schema"
)

type Manifest struct {
	schema          *schema.Schema
	ScalarFragments fragment.FragmentVector
	vectorFragments fragment.FragmentVector
	deleteFragments fragment.FragmentVector
	blobs           []blob.Blob
	version         int64
}

func NewManifest(schema *schema.Schema) *Manifest {
	return &Manifest{
		schema: schema,
	}
}

func Init() *Manifest {
	return &Manifest{
		schema: schema.NewSchema(arrow.NewSchema(nil, nil), schema.DefaultSchemaOptions()),
	}
}

func (m *Manifest) Copy() *Manifest {
	copied := *m
	return &copied
}

func (m *Manifest) GetSchema() *schema.Schema {
	return m.schema
}

func (m *Manifest) AddScalarFragment(fragment fragment.Fragment) {
	m.ScalarFragments = append(m.ScalarFragments, fragment)
}

func (m *Manifest) AddVectorFragment(fragment fragment.Fragment) {
	m.vectorFragments = append(m.vectorFragments, fragment)
}

func (m *Manifest) AddDeleteFragment(fragment fragment.Fragment) {
	m.deleteFragments = append(m.deleteFragments, fragment)
}

func (m *Manifest) GetScalarFragments() fragment.FragmentVector {
	return m.ScalarFragments
}

func (m *Manifest) GetVectorFragments() fragment.FragmentVector {
	return m.vectorFragments
}

func (m *Manifest) GetDeleteFragments() fragment.FragmentVector {
	return m.deleteFragments
}

func (m *Manifest) Version() int64 {
	return m.version
}

func (m *Manifest) SetVersion(version int64) {
	m.version = version
}

func (m *Manifest) ToProtobuf() (*manifest_proto.Manifest, error) {
	manifest := &manifest_proto.Manifest{}
	manifest.Version = m.version
	for _, vectorFragment := range m.vectorFragments {
		manifest.VectorFragments = append(manifest.VectorFragments, vectorFragment.ToProtobuf())
	}
	for _, scalarFragment := range m.ScalarFragments {
		manifest.ScalarFragments = append(manifest.ScalarFragments, scalarFragment.ToProtobuf())
	}
	for _, deleteFragment := range m.deleteFragments {
		manifest.DeleteFragments = append(manifest.DeleteFragments, deleteFragment.ToProtobuf())
	}

	for _, blob := range m.blobs {
		manifest.Blobs = append(manifest.Blobs, blob.ToProtobuf())
	}

	schemaProto, err := m.schema.ToProtobuf()
	if err != nil {
		return nil, err
	}
	manifest.Schema = schemaProto

	return manifest, nil
}

func (m *Manifest) FromProtobuf(manifest *manifest_proto.Manifest) error {
	err := m.schema.FromProtobuf(manifest.Schema)
	if err != nil {
		return err
	}

	for _, vectorFragment := range manifest.VectorFragments {
		m.vectorFragments = append(m.vectorFragments, fragment.FromProtobuf(vectorFragment))
	}

	for _, scalarFragment := range manifest.ScalarFragments {
		m.ScalarFragments = append(m.ScalarFragments, fragment.FromProtobuf(scalarFragment))
	}

	for _, deleteFragment := range manifest.DeleteFragments {
		m.deleteFragments = append(m.deleteFragments, fragment.FromProtobuf(deleteFragment))
	}

	for _, b := range manifest.Blobs {
		m.blobs = append(m.blobs, blob.FromProtobuf(b))
	}

	m.version = manifest.Version
	return nil
}

func WriteManifestFile(manifest *Manifest, output file.File) error {
	protoManifest, err := manifest.ToProtobuf()
	if err != nil {
		return err
	}

	bytes, err := proto.Marshal(protoManifest)
	if err != nil {
		return fmt.Errorf("write manifest file: %w", err)
	}
	write, err := output.Write(bytes)
	if err != nil {
		return fmt.Errorf("write manifest file: %w", err)
	}
	if write != len(bytes) {
		return fmt.Errorf("failed to write whole file, expect: %v, actual: %v", len(bytes), write)
	}
	if err = output.Close(); err != nil {
		return err
	}
	return nil
}

func (m *Manifest) HasBlob(name string) bool {
	for _, b := range m.blobs {
		if b.Name == name {
			return true
		}
	}

	return false
}

func (m *Manifest) AddBlob(blob blob.Blob) {
	m.blobs = append(m.blobs, blob)
}

func (m *Manifest) RemoveBlobIfExist(name string) {
	idx := -1
	for i, b := range m.blobs {
		if b.Name == name {
			idx = i
			break
		}
	}

	m.blobs = append(m.blobs[0:idx], m.blobs[idx+1:]...)
}

func (m *Manifest) GetBlob(name string) (blob.Blob, bool) {
	for _, b := range m.blobs {
		if b.Name == name {
			return b, true
		}
	}

	return blob.Blob{}, false
}

func ParseFromFile(f fs.Fs, path string) (*Manifest, error) {
	manifest := Init()
	manifestProto := &manifest_proto.Manifest{}

	buf, err := f.ReadFile(path)
	if err != nil {
		return nil, err
	}
	err = proto.Unmarshal(buf, manifestProto)
	if err != nil {
		log.Error("Failed to unmarshal manifest proto", log.String("err", err.Error()))
		return nil, fmt.Errorf("parse from file: %w", err)
	}
	manifest.FromProtobuf(manifestProto)

	return manifest, nil
}

// TODO REMOVE BELOW CODE

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
