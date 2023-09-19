package record_reader

import (
	"github.com/apache/arrow/go/v12/arrow"
	"github.com/apache/arrow/go/v12/arrow/array"
	"github.com/milvus-io/milvus-storage/go/file/fragment"
	"github.com/milvus-io/milvus-storage/go/io/fs"
	"github.com/milvus-io/milvus-storage/go/storage/options"
	"github.com/milvus-io/milvus-storage/go/storage/schema"
)

type FilterQueryRecordReader struct {
	//TODO implement me
	ref             int64
	schema          *schema.Schema
	options         *options.ReadOptions
	fs              fs.Fs
	scalarFragment  fragment.FragmentVector
	vectorFragment  fragment.FragmentVector
	deleteFragments fragment.DeleteFragmentVector
	record          arrow.Record
}

func NewFilterQueryReader(
	s *schema.Schema,
	options *options.ReadOptions,
	f fs.Fs,
	scalarFragment fragment.FragmentVector,
	vectorFragment fragment.FragmentVector,
	deleteFragments fragment.DeleteFragmentVector) array.RecordReader {
	//TODO implement me
	panic("implement me")
}
