package record_reader

import (
	"github.com/apache/arrow/go/v12/arrow"
	"github.com/apache/arrow/go/v12/arrow/array"
	"github.com/milvus-io/milvus-storage/file/fragment"
	"github.com/milvus-io/milvus-storage/io/format"
	"github.com/milvus-io/milvus-storage/io/fs"
	"github.com/milvus-io/milvus-storage/storage/options/option"
)

type MultiFilesSequentialReader struct {
	fs                fs.Fs
	schema            *arrow.Schema
	files             []string
	nextPos           int
	currReader        array.RecordReader
	holdingFileReader format.Reader
	err               error
	options           *option.ReadOptions
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
	}
}
