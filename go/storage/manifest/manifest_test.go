package manifest

import (
	"testing"

	"github.com/apache/arrow/go/v12/arrow"
	"github.com/milvus-io/milvus-storage/go/file/fragment"
	"github.com/milvus-io/milvus-storage/go/storage/options/schema_option"
	"github.com/milvus-io/milvus-storage/go/storage/schema"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
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
	err := sc.Validate()
	assert.NoError(t, err)

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
