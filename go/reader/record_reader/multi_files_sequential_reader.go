package record_reader

import (
	"sync/atomic"

	"github.com/apache/arrow/go/v12/arrow"
	"github.com/milvus-io/milvus-storage/go/file/fragment"
	"github.com/milvus-io/milvus-storage/go/io/format/parquet"
	"github.com/milvus-io/milvus-storage/go/io/fs"
	"github.com/milvus-io/milvus-storage/go/storage/options/option"
)

type MultiFilesSequentialReader struct {
	fs         fs.Fs
	schema     *arrow.Schema
	files      []string
	nextPos    int
	options    *option.ReadOptions
	currReader *parquet.FileReader
	err        error
	ref        int64
}

func (m MultiFilesSequentialReader) Retain() {
	atomic.AddInt64(&m.ref, 1)
}

func (m MultiFilesSequentialReader) Release() {
	if atomic.AddInt64(&m.ref, -1) == 0 {
		if m.currReader != nil {
			m.currReader.Close()
		}
	}
}

func (m MultiFilesSequentialReader) Schema() *arrow.Schema {
	return m.schema
}

func (m MultiFilesSequentialReader) Next() bool {
	//TODO implement me
	panic("implement me")
}

func (m MultiFilesSequentialReader) Record() arrow.Record {
	//TODO implement me
	panic("implement me")
}

func (m MultiFilesSequentialReader) Err() error {
	return m.err
}

func NewMultiFilesSequentialReader(fs fs.Fs, fragments fragment.FragmentVector, schema *arrow.Schema, options *option.ReadOptions) *MultiFilesSequentialReader {
	files := make([]string, 0, len(fragments))
	for _, f := range fragments {
		files = append(files, f.Files()...)
	}

	return &MultiFilesSequentialReader{
		fs:      fs,
		schema:  schema,
		options: options,
		files:   files,
		nextPos: 0,
		ref:     1,
	}
}
