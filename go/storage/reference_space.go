package storage

import (
	"fmt"
	"github.com/apache/arrow/go/v12/arrow"
	"github.com/apache/arrow/go/v12/arrow/array"
	"github.com/google/uuid"
	"github.com/milvus-io/milvus-storage-format/io/format"
	"github.com/milvus-io/milvus-storage-format/io/format/parquet"
	"github.com/milvus-io/milvus-storage-format/io/fs"
	"github.com/milvus-io/milvus-storage-format/storage/manifest"
	"github.com/milvus-io/milvus-storage-format/storage/options/option"
)

type ReferenceSpace struct {
	schema   *arrow.Schema
	fs       fs.Fs
	options  *option.SpaceOptions
	manifest *manifest.ManifestV1
}

func (s *ReferenceSpace) Write(reader array.RecordReader, options *option.WriteOptions) error {
	// check schema consistency
	if !s.schema.Equal(reader.Schema()) {
		return ErrSchemaNotMatch
	}

	var dataFiles []*manifest.DataFile
	var writer format.Writer
	var err error
	// write data
	for reader.Next() {
		rec := reader.Record()

		if rec.NumRows() == 0 {
			continue
		}

		if writer == nil {
			filePath := uuid.NewString() + ".parquet"
			writer, err = parquet.NewFileWriter(s.schema, s.fs, filePath)
			if err != nil {
				return err
			}
			dataFiles = append(dataFiles, manifest.NewDataFile(filePath))
		}

		if err := writer.Write(rec); err != nil {
			return err
		}

		if writer.Count() >= options.MaxRecordPerFile {
			if err := writer.Close(); err != nil {
				return err
			}
			writer = nil
		}
	}

	if writer != nil {
		if err := writer.Close(); err != nil {
			return err
		}
	}

	// update manifest
	if len(dataFiles) != 0 {
		s.manifest.AddDataFiles(dataFiles...)
		if err := manifest.WriteManifestFileV1(s.fs, s.manifest); err != nil {
			return err
		}
	}

	return nil
}

// Read return a RecordReader. Remember to call Release after using the RecordReader
func (s *ReferenceSpace) Read(options *option.ReadOptions) (array.RecordReader, error) {
	// check read options
	for _, col := range options.Columns {
		if !s.schema.HasField(col) {
			return nil, fmt.Errorf("%w: %s", ErrColumnNotExist, col)
		}
	}

	return NewDefaultRecordReader(s, options), nil
}

func NewReferenceSpace(schema *arrow.Schema, options *option.SpaceOptions) *ReferenceSpace {
	fsFactory := fs.NewFsFactory()
	fs := fsFactory.Create(options.Fs)
	return &ReferenceSpace{
		schema:   schema,
		fs:       fs,
		options:  options,
		manifest: manifest.NewManifestV1(),
	}
}
