package manifest

import (
	"github.com/apache/arrow/go/v12/arrow"
	"github.com/milvus-io/milvus-storage/go/common/utils"
	"github.com/milvus-io/milvus-storage/go/file/fragment"
	"github.com/milvus-io/milvus-storage/go/io/fs"
	"github.com/milvus-io/milvus-storage/go/storage/lock"
	"github.com/milvus-io/milvus-storage/go/storage/schema"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"sync"
	"testing"
)

// Test Manifest
func TestManifest(t *testing.T) {
	pkField := arrow.Field{
		Name:     "pk_field",
		Type:     arrow.DataType(&arrow.Int64Type{}),
		Nullable: false,
	}
	vsField := arrow.Field{
		Name:     "vs_field",
		Type:     arrow.DataType(&arrow.Int64Type{}),
		Nullable: false,
	}
	vecField := arrow.Field{
		Name:     "vec_field",
		Type:     arrow.DataType(&arrow.FixedSizeBinaryType{ByteWidth: 16}),
		Nullable: false,
	}
	fields := []arrow.Field{pkField, vsField, vecField}

	as := arrow.NewSchema(fields, nil)
	schemaOptions := &schema.SchemaOptions{
		PrimaryColumn: "pk_field",
		VersionColumn: "vs_field",
		VectorColumn:  "vec_field",
	}

	sc := schema.NewSchema(as, schemaOptions)
	err := sc.Validate()
	assert.NoError(t, err)

	maniFest := NewManifest(sc)

	f1 := fragment.NewFragment()
	f1.SetFragmentId(1)
	f1.AddFile("scalar1")
	f1.AddFile("scalar2")
	maniFest.AddScalarFragment(f1)

	f2 := fragment.NewFragment()
	f2.SetFragmentId(2)
	f2.AddFile("vector1")
	f2.AddFile("vector2")
	maniFest.AddVectorFragment(f2)

	f3 := fragment.NewFragment()
	f3.SetFragmentId(3)
	f3.AddFile("delete1")
	maniFest.AddDeleteFragment(f3)

	require.Equal(t, len(maniFest.GetScalarFragments()), 1)
	require.Equal(t, len(maniFest.GetVectorFragments()), 1)
	require.Equal(t, len(maniFest.GetDeleteFragments()), 1)
	require.Equal(t, sc, maniFest.GetSchema())
}

// Test ManifestCommitOp
func TestManifestCommitOp(t *testing.T) {
	tmpDir := t.TempDir()
	f, err := fs.BuildFileSystem("file:///" + tmpDir)

	// create manifest path
	err = f.MkdirAll(utils.GetManifestDir(tmpDir), 0755)
	assert.NoError(t, err)

	// create manifest file
	manifest := NewManifest(schema.NewSchema(arrow.NewSchema(nil, nil), schema.DefaultSchemaOptions()))
	manifest.SetVersion(0)

	mc := ManifestCommit{
		ops:  []ManifestCommitOp{},
		rw:   NewManifestReaderWriter(f, tmpDir),
		lock: lock.NewMemoryLockManager(),
	}

	err = mc.rw.Write(manifest)
	assert.NoError(t, err)

	mc.AddOp(AddScalarFragmentOp{ScalarFragment: fragment.NewFragment()})
	mc.AddOp(AddVectorFragmentOp{VectorFragment: fragment.NewFragment()})
	mc.AddOp(AddDeleteFragmentOp{DeleteFragment: fragment.NewFragment()})
	err = mc.Commit()
	assert.NoError(t, err)
}

// Test ManifestReaderWriter Read
func TestManifestReaderWriter_Read(t *testing.T) {
	tmpDir := t.TempDir()
	f, err := fs.BuildFileSystem("file:///" + tmpDir)

	// create manifest path
	err = f.MkdirAll(utils.GetManifestDir(tmpDir), 0755)
	assert.NoError(t, err)

	// create manifest file
	manifest := NewManifest(schema.NewSchema(arrow.NewSchema(nil, nil), schema.DefaultSchemaOptions()))
	manifest.SetVersion(0)
	err = NewManifestReaderWriter(f, tmpDir).Write(manifest)
	assert.NoError(t, err)

	// read manifest file
	m, err := NewManifestReaderWriter(f, tmpDir).Read(0)
	assert.NoError(t, err)
	assert.Equal(t, manifest.version, m.version)
}

// Test ManifestReaderWriter MaxVersion
func TestManifestReaderWriter_MaxVersion(t *testing.T) {
	tmpDir := t.TempDir()
	f, err := fs.BuildFileSystem("file:///" + tmpDir)

	// create manifest path
	err = f.MkdirAll(utils.GetManifestDir(tmpDir), 0755)
	assert.NoError(t, err)

	// create manifest file
	manifest := NewManifest(schema.NewSchema(arrow.NewSchema(nil, nil), schema.DefaultSchemaOptions()))
	manifest.SetVersion(0)
	err = NewManifestReaderWriter(f, tmpDir).Write(manifest)
	assert.NoError(t, err)

	// read manifest file
	m, err := NewManifestReaderWriter(f, tmpDir).MaxVersion()
	assert.NoError(t, err)
	assert.Equal(t, manifest.version, m)
}

// Test ManifestReaderWriter Write
func TestManifestReaderWriter_Write(t *testing.T) {
	tmpDir := t.TempDir()
	f, err := fs.BuildFileSystem("file:///" + tmpDir)

	// create manifest path
	err = f.MkdirAll(utils.GetManifestDir(tmpDir), 0755)
	assert.NoError(t, err)

	// create manifest file
	manifest := NewManifest(schema.NewSchema(arrow.NewSchema(nil, nil), schema.DefaultSchemaOptions()))
	manifest.SetVersion(0)
	err = NewManifestReaderWriter(f, tmpDir).Write(manifest)
	assert.NoError(t, err)
}

// Test ManifestReaderWriter concurrency write
func TestManifestReaderWriter_concurrency(t *testing.T) {
	tmpDir := t.TempDir()
	f, err := fs.BuildFileSystem("file:///" + tmpDir)

	// create manifest path
	err = f.MkdirAll(utils.GetManifestDir(tmpDir), 0755)
	assert.NoError(t, err)

	// create manifest file
	manifest := NewManifest(schema.NewSchema(arrow.NewSchema(nil, nil), schema.DefaultSchemaOptions()))
	manifest.SetVersion(0)
	err = NewManifestReaderWriter(f, tmpDir).Write(manifest)
	assert.NoError(t, err)

	// read manifest file
	m, err := NewManifestReaderWriter(f, tmpDir).Read(0)
	assert.NoError(t, err)
	assert.Equal(t, manifest.version, m.version)

	// write manifest file
	manifest.SetVersion(1)
	err = NewManifestReaderWriter(f, tmpDir).Write(manifest)
	assert.NoError(t, err)

	// read manifest file
	m, err = NewManifestReaderWriter(f, tmpDir).Read(1)
	assert.NoError(t, err)

	// write manifest file concurrently
	wg := sync.WaitGroup{}

	for i := 0; i < 100; i++ {
		wg.Add(1)
		i := i
		go func() {
			defer wg.Done()
			manifest.SetVersion(int64(i))
			err = NewManifestReaderWriter(f, tmpDir).Write(manifest)
			assert.NoError(t, err)
		}()
	}

	wg.Wait()

	// read manifest file
	m, err = NewManifestReaderWriter(f, tmpDir).Read(99)
	assert.NoError(t, err)
	assert.NotEqual(t, manifest.version, m.version)
}

// Test Manifest commit concurrency
func TestManifestCommit_concurrency(t *testing.T) {

	tmpDir := t.TempDir()
	f, err := fs.BuildFileSystem("file:///" + tmpDir)

	// create manifest path
	err = f.MkdirAll(utils.GetManifestDir(tmpDir), 0755)
	assert.NoError(t, err)

	sc := createNewSchema()
	// create manifest file
	manifest := NewManifest(sc)
	manifest.SetVersion(0)
	mrw := NewManifestReaderWriter(f, tmpDir)
	err = mrw.Write(manifest)
	assert.NoError(t, err)

	l := lock.NewMemoryLockManager()

	// use commit to write manifest file concurrently
	wg := sync.WaitGroup{}
	for i := 0; i < 5; i++ {
		wg.Add(1)
		go func() {
			mc := ManifestCommit{
				ops:  []ManifestCommitOp{},
				rw:   mrw,
				lock: l,
			}
			mc.AddOp(AddScalarFragmentOp{ScalarFragment: fragment.NewFragment()})
			mc.AddOp(AddVectorFragmentOp{VectorFragment: fragment.NewFragment()})
			mc.AddOp(AddDeleteFragmentOp{DeleteFragment: fragment.NewFragment()})
			err = mc.Commit()
			wg.Done()
		}()
	}
	wg.Wait()

}

func createNewSchema() *schema.Schema {
	pkField := arrow.Field{
		Name:     "pk_field",
		Type:     arrow.DataType(&arrow.Int64Type{}),
		Nullable: false,
	}
	vsField := arrow.Field{
		Name:     "vs_field",
		Type:     arrow.DataType(&arrow.Int64Type{}),
		Nullable: false,
	}
	vecField := arrow.Field{
		Name:     "vec_field",
		Type:     arrow.DataType(&arrow.FixedSizeBinaryType{ByteWidth: 10}),
		Nullable: false,
	}
	fields := []arrow.Field{pkField, vsField, vecField}

	as := arrow.NewSchema(fields, nil)
	schemaOptions := &schema.SchemaOptions{
		PrimaryColumn: "pk_field",
		VersionColumn: "vs_field",
		VectorColumn:  "vec_field",
	}

	sc := schema.NewSchema(as, schemaOptions)
	return sc
}
