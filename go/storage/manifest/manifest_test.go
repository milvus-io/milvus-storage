package manifest

import (
	"github.com/apache/arrow/go/v12/arrow"
	"github.com/milvus-io/milvus-storage-format/file/fragment"
	"github.com/milvus-io/milvus-storage-format/storage/options/schema_option"
	"github.com/milvus-io/milvus-storage-format/storage/schema"
	"github.com/stretchr/testify/require"
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
	schemaOptions := &schema_option.SchemaOptions{
		PrimaryColumn: "pk_field",
		VersionColumn: "vs_field",
		VectorColumn:  "vec_field",
	}

	sc := schema.NewSchema(as, schemaOptions)
	validate := sc.Validate()
	require.Equal(t, validate.IsOK(), true)

	maniFest := NewManifest(sc)

	f1 := fragment.NewFragment(1)
	f1.AddFile("scalar1")
	f1.AddFile("scalar2")
	maniFest.AddScalarFragment(*f1)

	f2 := fragment.NewFragment(2)
	f2.AddFile("vector1")
	f2.AddFile("vector2")
	maniFest.AddVectorFragment(*f2)

	f3 := fragment.NewFragment(3)
	f3.AddFile("delete1")
	maniFest.AddDeleteFragment(*f3)

	require.Equal(t, len(maniFest.GetScalarFragments()), 1)
	require.Equal(t, len(maniFest.GetVectorFragments()), 1)
	require.Equal(t, len(maniFest.GetDeleteFragments()), 1)
	require.Equal(t, sc, maniFest.GetSchema())
}
