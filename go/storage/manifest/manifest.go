package manifest

import (
	"github.com/apache/arrow/go/v12/arrow"
	"github.com/milvus-io/milvus-storage-format/common/log"
	"github.com/milvus-io/milvus-storage-format/common/result"
	"github.com/milvus-io/milvus-storage-format/common/status"
	"github.com/milvus-io/milvus-storage-format/file/fragment"
	"github.com/milvus-io/milvus-storage-format/io/fs"
	"github.com/milvus-io/milvus-storage-format/io/fs/file"
	"github.com/milvus-io/milvus-storage-format/proto/manifest_proto"
	"github.com/milvus-io/milvus-storage-format/storage/options/schema_option"
	"github.com/milvus-io/milvus-storage-format/storage/schema"
	"google.golang.org/protobuf/proto"
)

type Manifest struct {
	schema          *schema.Schema
	ScalarFragments fragment.FragmentVector
	vectorFragments fragment.FragmentVector
	deleteFragments fragment.FragmentVector
	version         int64
}

func NewManifest(schema *schema.Schema) *Manifest {
	return &Manifest{
		schema: schema,
	}
}

func Init() *Manifest {
	return &Manifest{
		schema: schema.NewSchema(arrow.NewSchema(nil, nil), schema_option.Init()),
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

func (m *Manifest) ToProtobuf() *result.Result[*manifest_proto.Manifest] {
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

	schemaProto := m.schema.ToProtobuf()
	if !schemaProto.Ok() {
		return result.NewResultFromStatus[*manifest_proto.Manifest](*schemaProto.Status())
	}
	manifest.Schema = schemaProto.Value()

	return result.NewResult[*manifest_proto.Manifest](manifest, status.OK())
}

func (m *Manifest) FromProtobuf(manifest *manifest_proto.Manifest) {

	schemaResult := m.schema.FromProtobuf(manifest.Schema)
	if !schemaResult.IsOK() {
		log.Error("Failed to unmarshal schema proto")
		return
	}

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
	write, err := output.Write(bytes)
	if err != nil {
		return status.InternalStateError("Failed to write manifest file")
	}
	if write != len(bytes) {
		return status.InternalStateError("Failed to write manifest file")
	}

	return status.OK()
}

func ParseFromFile(f fs.Fs, path string) *result.Result[*Manifest] {
	manifest := Init()
	manifestProto := &manifest_proto.Manifest{}

	buf, err := f.ReadFile(path)
	err = proto.Unmarshal(buf, manifestProto)
	if err != nil {
		log.Error("Failed to unmarshal manifest proto", log.String("err", err.Error()))
		return result.NewResultFromStatus[*Manifest](status.InternalStateError(err.Error()))
	}
	manifest.FromProtobuf(manifestProto)

	return result.NewResult[*Manifest](manifest, status.OK())
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
