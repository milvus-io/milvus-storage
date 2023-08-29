package common_reader

import (
	"github.com/apache/arrow/go/v12/arrow/array"
	"github.com/milvus-io/milvus-storage/go/storage/options/option"
)

type FilterReader struct {
	recordReader               array.RecordReader
	option                     *option.ReadOptions
	currentFilteredBatchReader array.RecordReader
}

func MakeFilterReader(recordReader array.RecordReader, option *option.ReadOptions) *FilterReader {
	return &FilterReader{
		recordReader: recordReader,
		option:       option,
	}
}

func (r *FilterReader) Next() bool {
	//for {
	//	if r.currentFilteredBatchReader != nil {
	//		filteredBatch := r.currentFilteredBatchReader.Next()
	//		if err != nil {
	//			return false
	//		}
	//		if filteredBatch == nil {
	//			r.currentFilteredBatchReader = nil
	//			continue
	//		}
	//		return filteredBatch, nil
	//	}
	//	err := r.NextFilteredBatchReader()
	//	if err != nil {
	//		return nil
	//	}
	//	if r.currentFilteredBatchReader == nil {
	//		return nil
	//	}
	//}
	return false
}
