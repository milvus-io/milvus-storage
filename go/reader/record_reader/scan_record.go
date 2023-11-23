// Copyright 2023 Zilliz
// 
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// 
//     http://www.apache.org/licenses/LICENSE-2.0
// 
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package record_reader

import (
	"io"
	"sync/atomic"

	"github.com/apache/arrow/go/v12/arrow"
	"github.com/apache/arrow/go/v12/arrow/array"
	"github.com/milvus-io/milvus-storage/go/common/log"
	"github.com/milvus-io/milvus-storage/go/common/utils"
	"github.com/milvus-io/milvus-storage/go/file/fragment"
	"github.com/milvus-io/milvus-storage/go/io/format"
	"github.com/milvus-io/milvus-storage/go/io/format/parquet"
	"github.com/milvus-io/milvus-storage/go/io/fs"
	"github.com/milvus-io/milvus-storage/go/reader/common_reader"
	"github.com/milvus-io/milvus-storage/go/storage/options"
	"github.com/milvus-io/milvus-storage/go/storage/schema"
	"go.uber.org/zap"
)

type ScanRecordReader struct {
	ref             int64
	schema          *schema.Schema
	options         *options.ReadOptions
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
	options *options.ReadOptions,
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

		if rec.NumRows() == 0 {
			continue
		}

		r.rec = rec
		return true
	}
}

func (r *ScanRecordReader) Record() arrow.Record {
	return r.rec
}

func (r *ScanRecordReader) Err() error {
	return r.err
}

func (r *ScanRecordReader) MakeInnerReader() array.RecordReader {
	//TODO implement me
	reader := NewMultiFilesSequentialReader(r.fs, r.dataFragments, r.Schema(), r.options)

	filterReader := common_reader.MakeFilterReader(reader, r.options)

	deleteReader := common_reader.NewDeleteReader(filterReader, r.schema.Options(), r.deleteFragments, r.options)

	res := common_reader.NewProjectionReader(deleteReader, r.options, r.schema.Schema())
	return res
}
