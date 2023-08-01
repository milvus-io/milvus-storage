package record_reader

import (
	"github.com/apache/arrow/go/v12/arrow"
	"github.com/apache/arrow/go/v12/arrow/array"
	"github.com/milvus-io/milvus-storage-format/file/fragment"
	"github.com/milvus-io/milvus-storage-format/io/fs"
	"github.com/milvus-io/milvus-storage-format/storage/options/option"
	"github.com/milvus-io/milvus-storage-format/storage/schema"
)

type FilterQueryRecordReader struct {
	//TODO implement me
	ref             int64
	schema          *schema.Schema
	options         *option.ReadOptions
	fs              fs.Fs
	scalarFragment  fragment.FragmentVector
	vectorFragment  fragment.FragmentVector
	deleteFragments fragment.DeleteFragmentVector
	record          arrow.Record
}

func NewFilterQueryReader(
	s *schema.Schema,
	options *option.ReadOptions,
	f fs.Fs,
	scalarFragment fragment.FragmentVector,
	vectorFragment fragment.FragmentVector,
	deleteFragments fragment.DeleteFragmentVector) array.RecordReader {
	//TODO implement me
	panic("implement me")
}
