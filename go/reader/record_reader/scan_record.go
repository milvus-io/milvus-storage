package record_reader

import (
	"github.com/apache/arrow/go/v12/arrow"
	"github.com/apache/arrow/go/v12/arrow/array"
	"github.com/milvus-io/milvus-storage/common/log"
	"github.com/milvus-io/milvus-storage/common/utils"
	"github.com/milvus-io/milvus-storage/file/fragment"
	"github.com/milvus-io/milvus-storage/io/format"
	"github.com/milvus-io/milvus-storage/io/format/parquet"
	"github.com/milvus-io/milvus-storage/io/fs"
	"github.com/milvus-io/milvus-storage/storage/options/option"
	"github.com/milvus-io/milvus-storage/storage/schema"
	"go.uber.org/zap"
	"io"
	"sync/atomic"
)

type ScanRecordReader struct {
	ref             int64
	schema          *schema.Schema
	options         *option.ReadOptions
	fs              fs.Fs
	dataFragments   fragment.FragmentVector
	deleteFragments fragment.DeleteFragmentVector
	rec             arrow.Record
	curReader       format.Reader
	reader          array.RecordReader
	nextPos         int
	err             error
}

func NewScanRecordReader(
	s *schema.Schema,
	options *option.ReadOptions,
	f fs.Fs,
	dataFragments fragment.FragmentVector,
	deleteFragments fragment.DeleteFragmentVector,
) *ScanRecordReader {
	return &ScanRecordReader{
		ref:             1,
		schema:          s,
		options:         options,
		fs:              f,
		dataFragments:   dataFragments,
		deleteFragments: deleteFragments,
	}
}

func (r *ScanRecordReader) Schema() *arrow.Schema {
	return utils.ProjectSchema(r.schema.Schema(), r.options.OutputColumns())
}

func (r *ScanRecordReader) Retain() {
	atomic.AddInt64(&r.ref, 1)
}

func (r *ScanRecordReader) Release() {
	if atomic.AddInt64(&r.ref, -1) == 0 {
		if r.rec != nil {
			r.rec.Release()
			r.rec = nil
		}
		if r.curReader != nil {
			r.curReader.Close()
			r.curReader = nil
		}
	}
}

func (r *ScanRecordReader) Next() bool {
	datafiles := fragment.ToFilesVector(r.dataFragments)
	log.Debug("ScanRecordReader Next", zap.Any("datafiles", datafiles))
	if r.rec != nil {
		r.rec.Release()
		r.rec = nil
	}
	for {
		if r.curReader == nil {
			if r.nextPos >= len(datafiles) {
				return false
			}
			// FIXME: nil options
			reader, err := parquet.NewFileReader(r.fs, datafiles[r.nextPos], r.options)
			if err != nil {
				r.err = err
				return false
			}
			r.nextPos++
			r.curReader = reader
		}

		rec, err := r.curReader.Read()
		if err != nil {
			if err == io.EOF {
				r.curReader.Close()
				r.curReader = nil
				continue
			}
			// if error occurs in the middle of reading, return false
			r.curReader.Close()
			r.curReader = nil
			r.err = err
			return false
		}
		r.rec = rec
		return true
	}
}

func (r *ScanRecordReader) Record() arrow.Record {
	return r.rec
}

func (r *ScanRecordReader) Err() error {
	//TODO implement me
	panic("implement me")
}

func (r *ScanRecordReader) MakeInnerReader() array.RecordReader {
	//TODO implement me
	//	reader := NewMultiFilesSequentialReader(r.fs, r.dataFragments, r.Schema(), r.options)
	return nil
}
