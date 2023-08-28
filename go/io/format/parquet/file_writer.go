package parquet

import (
	"github.com/apache/arrow/go/v12/arrow"
	"github.com/apache/arrow/go/v12/parquet"
	"github.com/apache/arrow/go/v12/parquet/pqarrow"
	"github.com/milvus-io/milvus-storage/io/format"
	"github.com/milvus-io/milvus-storage/io/fs"
)

var _ format.Writer = (*FileWriter)(nil)

type FileWriter struct {
	writer *pqarrow.FileWriter
	count  int64
}

func (f *FileWriter) Write(record arrow.Record) error {
	if err := f.writer.Write(record); err != nil {
		return err
	}
	f.count += record.NumRows()
	return nil
}

func (f *FileWriter) Count() int64 {
	return f.count
}

func (f *FileWriter) Close() error {
	return f.writer.Close()
}

func NewFileWriter(schema *arrow.Schema, fs fs.Fs, filePath string) (*FileWriter, error) {
	file, err := fs.OpenFile(filePath)
	if err != nil {
		return nil, err
	}

	w, err := pqarrow.NewFileWriter(schema, file, parquet.NewWriterProperties(), pqarrow.DefaultWriterProps())
	if err != nil {
		return nil, err
	}

	return &FileWriter{writer: w}, nil
}
