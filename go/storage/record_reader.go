package storage

import (
	"io"
	"sync/atomic"

	"github.com/apache/arrow/go/v12/arrow"
	"github.com/milvus-io/milvus-storage-format/io/format"
	"github.com/milvus-io/milvus-storage-format/io/format/parquet"
	"github.com/milvus-io/milvus-storage-format/storage/options"
)

type DefaultRecordReader struct {
	ref       int64
	space     *ReferenceSpace
	options   *options.ReadOptions
	curReader format.Reader
	nextPos   int
	rec       arrow.Record
	err       error
}

func NewDefaultRecordReader(space *ReferenceSpace, options *options.ReadOptions) *DefaultRecordReader {
	return &DefaultRecordReader{
		space:   space,
		options: options,
		ref:     1,
	}
}

func (r *DefaultRecordReader) Retain() {
	atomic.AddInt64(&r.ref, 1)
}

func (r *DefaultRecordReader) Release() {
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

func (r *DefaultRecordReader) Schema() *arrow.Schema {
	return r.space.schema
}

func (r *DefaultRecordReader) Next() bool {
	// FIXME: use cloned space
	datafiles := r.space.manifest.DataFiles()
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
			reader, err := parquet.NewFileReader(r.space.fs, datafiles[r.nextPos].Path(), r.options)
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

func (r *DefaultRecordReader) Record() arrow.Record {
	return r.rec
}

func (r *DefaultRecordReader) Err() error {
	return nil
}
