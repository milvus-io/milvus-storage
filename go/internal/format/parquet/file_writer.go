package parquet

import (
	"github.com/apache/arrow/go/v12/arrow"
	"github.com/apache/arrow/go/v12/parquet"
	"github.com/apache/arrow/go/v12/parquet/pqarrow"
	"github.com/milvus-io/milvus-storage-format/internal/format"
	"github.com/milvus-io/milvus-storage-format/internal/fs"
)

var _ format.Writer = (*FileWriter)(nil)

type FileWriter struct {
	writer *pqarrow.FileWriter
}

func (f *FileWriter) Write(record arrow.Record) error {
	if err := f.writer.Write(record); err != nil {
		return err
	}
	// FIXME: should not close here
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
