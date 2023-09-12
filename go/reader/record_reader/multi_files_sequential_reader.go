package record_reader

import (
	"sync/atomic"

	"github.com/apache/arrow/go/v12/arrow"
	"github.com/apache/arrow/go/v12/arrow/array"
	"github.com/apache/arrow/go/v12/parquet/pqarrow"
	"github.com/milvus-io/milvus-storage/go/common/arrow_util"
	"github.com/milvus-io/milvus-storage/go/file/fragment"
	"github.com/milvus-io/milvus-storage/go/io/fs"
	"github.com/milvus-io/milvus-storage/go/storage/options/option"
)

type MultiFilesSequentialReader struct {
	fs         fs.Fs
	schema     *arrow.Schema
	files      []string
	nextPos    int
	options    *option.ReadOptions
	currReader array.RecordReader
	err        error
	ref        int64
}

func (m *MultiFilesSequentialReader) Retain() {
	atomic.AddInt64(&m.ref, 1)
}

func (m *MultiFilesSequentialReader) Release() {
	if atomic.AddInt64(&m.ref, -1) == 0 {
		if m.currReader != nil {
			m.currReader.Release()
			m.currReader = nil
		}
	}
}

func (m *MultiFilesSequentialReader) Schema() *arrow.Schema {
	return m.schema
}

func (m *MultiFilesSequentialReader) Next() bool {
	for true {
		if m.currReader == nil {
			if m.nextPos >= len(m.files) {
				return false
			}

			m.nextReader()
			if m.err != nil {
				return false
			}
			m.nextPos++
		}
		if m.currReader.Next() {
			return true
		}
		if m.currReader.Err() != nil {
			m.err = m.currReader.Err()
			return false
		}
		if m.currReader != nil {
			m.currReader.Release()
			m.currReader = nil
		}
	}
	return false
}

func (m *MultiFilesSequentialReader) Record() arrow.Record {
	if m.currReader != nil {
		return m.currReader.Record()
	}
	return nil
}

func (m *MultiFilesSequentialReader) Err() error {
	return m.err
}

func (m *MultiFilesSequentialReader) nextReader() {
	var fileReader *pqarrow.FileReader
	fileReader, m.err = arrow_util.MakeArrowFileReader(m.fs, m.files[m.nextPos])
	if m.err != nil {
		return
	}
	m.currReader, m.err = arrow_util.MakeArrowRecordReader(fileReader, m.options)
	return
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
