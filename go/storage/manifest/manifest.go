package manifest

import (
	"github.com/milvus-io/milvus-storage-format/common/result"
	"github.com/milvus-io/milvus-storage-format/common/status"
	"github.com/milvus-io/milvus-storage-format/file/fragment"
	"github.com/milvus-io/milvus-storage-format/io/fs"
	"github.com/milvus-io/milvus-storage-format/io/fs/file"
	"github.com/milvus-io/milvus-storage-format/proto/manifest_proto"
	"github.com/milvus-io/milvus-storage-format/storage/options"
	"github.com/milvus-io/milvus-storage-format/storage/schema"
	"google.golang.org/protobuf/proto"
)

type Manifest struct {
	schema          *schema.Schema
	options         *options.Options
	ScalarFragments fragment.FragmentVector
	vectorFragments fragment.FragmentVector
	deleteFragments fragment.FragmentVector

	version int64
}

func NewManifest(schema *schema.Schema, options *options.Options) *Manifest {

	return &Manifest{
		schema:  schema,
		options: options,
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

func (m *Manifest) SpaceOptions() *options.Options {
	return m.options
}

func (m *Manifest) ToProtobuf() *result.Result[*manifest_proto.Manifest] {
	manifest := &manifest_proto.Manifest{}
	manifest.Version = m.version
	manifest.Options = m.options.ToProtobuf()
	for _, vectorFragment := range m.vectorFragments {
		manifest.VectorFragments = append(manifest.VectorFragments, vectorFragment.ToProtobuf())
	}
	for _, scalarFragment := range m.ScalarFragments {
		manifest.ScalarFragments = append(manifest.ScalarFragments, scalarFragment.ToProtobuf())
	}
	for _, deleteFragment := range m.deleteFragments {
		manifest.DeleteFragments = append(manifest.DeleteFragments, deleteFragment.ToProtobuf())
	}

	schemaProto := m.schema.ToProtobuf()
	if !schemaProto.Ok() {
		return result.NewResultFromStatus[*manifest_proto.Manifest](*schemaProto.Status())
	}
	manifest.Schema = schemaProto.Value()
	return result.NewResult[*manifest_proto.Manifest](manifest)
}

func (m *Manifest) FromProtobuf(manifest *manifest_proto.Manifest) {

	m.options.FromProtobuf(manifest.Options)

	m.schema.FromProtobuf(manifest.Schema)

	for _, vectorFragment := range manifest.VectorFragments {
		m.vectorFragments = append(m.vectorFragments, *fragment.FromProtobuf(vectorFragment))
	}

	for _, scalarFragment := range manifest.ScalarFragments {
		m.ScalarFragments = append(m.ScalarFragments, *fragment.FromProtobuf(scalarFragment))
	}

	for _, deleteFragment := range manifest.DeleteFragments {
		m.deleteFragments = append(m.deleteFragments, *fragment.FromProtobuf(deleteFragment))
	}

	m.version = manifest.Version
}

func WriteManifestFile(manifest *Manifest, output file.File) status.Status {
	protoManifestTmp := manifest.ToProtobuf()

	if !protoManifestTmp.Ok() {
		return *protoManifestTmp.Status()
	}
	protoManifest := protoManifestTmp.Value()

	bytes, err := proto.Marshal(protoManifest)
	if err != nil {
		return status.InternalStateError("Failed to marshal manifest proto")
	}
	output.Write(bytes)

	return status.OK()

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

type ManifestV1 struct {
	dataFiles []*DataFile
}

func (m *ManifestV1) AddDataFiles(files ...*DataFile) {
	m.dataFiles = append(m.dataFiles, files...)
}

func (m *ManifestV1) DataFiles() []*DataFile {
	return m.dataFiles
}

func NewManifestV1() *ManifestV1 {
	return &ManifestV1{}
}

func WriteManifestFileV1(fs fs.Fs, manifest *ManifestV1) error {
	// TODO
	return nil
}
