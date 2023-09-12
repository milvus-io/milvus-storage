package record_reader

import (
	"github.com/apache/arrow/go/v12/arrow/array"
	"github.com/milvus-io/milvus-storage/go/file/fragment"
	"github.com/milvus-io/milvus-storage/go/filter"
	"github.com/milvus-io/milvus-storage/go/io/fs"
	"github.com/milvus-io/milvus-storage/go/storage/manifest"
	"github.com/milvus-io/milvus-storage/go/storage/options/option"
	"github.com/milvus-io/milvus-storage/go/storage/schema"
)

func MakeRecordReader(
	m *manifest.Manifest,
	s *schema.Schema,
	f fs.Fs,
	deleteFragments fragment.DeleteFragmentVector,
	options *option.ReadOptions,
) array.RecordReader {
	relatedColumns := make([]string, 0)
	for _, column := range options.Columns {
		relatedColumns = append(relatedColumns, column)
	}

	for _, filter := range options.Filters {
		relatedColumns = append(relatedColumns, filter.GetColumnName())
	}

	scalarData := m.GetScalarFragments()
	vectorData := m.GetVectorFragments()

	onlyScalar := onlyContainScalarColumns(s, relatedColumns)
	onlyVector := onlyContainVectorColumns(s, relatedColumns)

	if onlyScalar || onlyVector {
		var dataFragments fragment.FragmentVector
		if onlyScalar {
			dataFragments = scalarData
		} else {
			dataFragments = vectorData
		}
		return NewScanRecordReader(s, options, f, dataFragments, deleteFragments)
	}
	if len(options.Filters) > 0 && filtersOnlyContainPKAndVersion(s, options.FiltersV2) {
		return NewMergeRecordReader(s, options, f, scalarData, vectorData, deleteFragments)
	}
	return NewFilterQueryReader(s, options, f, scalarData, vectorData, deleteFragments)
}

func onlyContainVectorColumns(schema *schema.Schema, relatedColumns []string) bool {
	for _, column := range relatedColumns {
		if schema.Options().VectorColumn != column && schema.Options().PrimaryColumn != column && schema.Options().VersionColumn != column {
			return false
		}
	}
	return true
}

func onlyContainScalarColumns(schema *schema.Schema, relatedColumns []string) bool {
	for _, column := range relatedColumns {
		if schema.Options().VectorColumn == column {
			return false
		}
	}
	return true
}

func filtersOnlyContainPKAndVersion(s *schema.Schema, filters []filter.Filter) bool {
	for _, f := range filters {
		if f.GetColumnName() != s.Options().PrimaryColumn &&
			f.GetColumnName() != s.Options().VersionColumn {
			return false
		}
	}
	return true
}

func MakeScanDeleteReader(manifest *manifest.Manifest, fs fs.Fs) array.RecordReader {
	return NewMultiFilesSequentialReader(fs, manifest.GetDeleteFragments(), manifest.GetSchema().DeleteSchema(), option.NewReadOptions())
}
