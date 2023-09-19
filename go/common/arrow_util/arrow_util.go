package arrow_util

import (
	"context"

	"github.com/apache/arrow/go/v12/arrow/array"
	"github.com/apache/arrow/go/v12/arrow/memory"
	"github.com/apache/arrow/go/v12/parquet/file"
	"github.com/apache/arrow/go/v12/parquet/pqarrow"
	"github.com/milvus-io/milvus-storage/go/io/fs"
	"github.com/milvus-io/milvus-storage/go/storage/options"
)

func MakeArrowFileReader(fs fs.Fs, filePath string) (*pqarrow.FileReader, error) {
	f, err := fs.OpenFile(filePath)
	if err != nil {
		return nil, err
	}
	parquetReader, err := file.NewParquetReader(f, nil)
	if err != nil {
		return nil, err
	}
	return pqarrow.NewFileReader(parquetReader, pqarrow.ArrowReadProperties{}, memory.DefaultAllocator)
}

func MakeArrowRecordReader(reader *pqarrow.FileReader, opts *options.ReadOptions) (array.RecordReader, error) {
	var rowGroupsIndices []int
	var columnIndices []int
	metadata := reader.ParquetReader().MetaData()
	for _, c := range opts.Columns {
		columnIndices = append(columnIndices, metadata.Schema.ColumnIndexByName(c))
	}
	for _, f := range opts.Filters {
		columnIndices = append(columnIndices, metadata.Schema.ColumnIndexByName(f.GetColumnName()))
	}

	for i := 0; i < int(metadata.NumRows); i++ {
		rg := metadata.RowGroup(i)
		var canIgnored bool
		for _, filter := range opts.Filters {
			columnIndex := rg.Schema.ColumnIndexByName(filter.GetColumnName())
			columnChunk, err := rg.ColumnChunk(columnIndex)
			if err != nil {
				return nil, err
			}
			columnStats, err := columnChunk.Statistics()
			if err != nil {
				return nil, err
			}
			if columnStats == nil || !columnStats.HasMinMax() {
				continue
			}
			if filter.CheckStatistics(columnStats) {
				canIgnored = true
				break
			}
		}
		if !canIgnored {
			rowGroupsIndices = append(rowGroupsIndices, i)
		}
	}

	return reader.GetRecordReader(context.TODO(), columnIndices, rowGroupsIndices)
}
