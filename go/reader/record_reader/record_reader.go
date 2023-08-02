package record_reader

import (
	"github.com/apache/arrow/go/v12/arrow/array"
	"github.com/milvus-io/milvus-storage-format/file/fragment"
	"github.com/milvus-io/milvus-storage-format/filter"
	"github.com/milvus-io/milvus-storage-format/io/fs"
	"github.com/milvus-io/milvus-storage-format/storage/manifest"
	"github.com/milvus-io/milvus-storage-format/storage/options/option"
	"github.com/milvus-io/milvus-storage-format/storage/schema"
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
