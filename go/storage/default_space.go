package storage

import (
	"errors"
	"github.com/apache/arrow/go/v12/arrow"
	"github.com/apache/arrow/go/v12/arrow/array"
	"github.com/apache/arrow/go/v12/arrow/memory"
	"github.com/milvus-io/milvus-storage-format/common/log"
	"github.com/milvus-io/milvus-storage-format/common/status"
	"github.com/milvus-io/milvus-storage-format/common/utils"
	"github.com/milvus-io/milvus-storage-format/file/fragment"
	"github.com/milvus-io/milvus-storage-format/io/format"
	"github.com/milvus-io/milvus-storage-format/io/format/parquet"
	"github.com/milvus-io/milvus-storage-format/io/fs"
	mnf "github.com/milvus-io/milvus-storage-format/storage/manifest"
	"github.com/milvus-io/milvus-storage-format/storage/options"
	"github.com/milvus-io/milvus-storage-format/storage/schema"
	"sync"
)

type DefaultSpace struct {
	basePath        string
	fs              fs.Fs
	schema          *schema.Schema
	deleteFragments fragment.DeleteFragmentVector
	manifest        *mnf.Manifest
	options         *options.Options
	lock            sync.RWMutex
}

func NewDefaultSpace(schema *schema.Schema, op *options.Options) *DefaultSpace {
	fsFactory := fs.NewFsFactory()
	f := fsFactory.Create(options.LocalFS)
	// TODO: implement uri parser
	uri := op.Uri
	maniFest := mnf.NewManifest(schema, op)
	// TODO: implement delete fragment
	deleteFragments := fragment.DeleteFragmentVector{}

	return &DefaultSpace{
		basePath:        uri,
		fs:              f,
		schema:          schema,
		options:         op,
		manifest:        maniFest,
		deleteFragments: deleteFragments,
	}
}

func (s *DefaultSpace) Write(reader array.RecordReader, options *options.WriteOptions) error {
	// check schema consistency
	if !s.schema.Schema().Equal(reader.Schema()) {
		return ErrSchemaNotMatch
	}

	scalarSchema, vectorSchema := s.schema.ScalarSchema(), s.schema.VectorSchema()
	var (
		scalarWriter format.Writer
		vectorWriter format.Writer
	)
	scalarFragment := fragment.NewFragment(s.manifest.Version())
	vectorFragment := fragment.NewFragment(s.manifest.Version())

	for reader.Next() {
		rec := reader.Record()

		if rec.NumRows() == 0 {
			continue
		}

		var (
			err error
		)

		scalarWriter, err = s.write(scalarSchema, rec, scalarWriter, scalarFragment, options, true)
		if err != nil {
			return err
		}

		vectorWriter, err = s.write(vectorSchema, rec, vectorWriter, vectorFragment, options, false)
		if err != nil {
			return err
		}
	}

	if scalarWriter != nil {
		if err := scalarWriter.Close(); err != nil {
			return err
		}
	}

	if vectorWriter != nil {
		if err := vectorWriter.Close(); err != nil {
			return err
		}
	}

	s.lock.Lock()
	defer s.lock.Unlock()
	copiedManifest := s.manifest
	oldVersion := s.manifest.Version()
	scalarFragment.SetFragmentId(oldVersion + 1)
	vectorFragment.SetFragmentId(oldVersion + 1)
	copiedManifest.AddScalarFragment(*scalarFragment)
	copiedManifest.AddVectorFragment(*vectorFragment)
	copiedManifest.SetVersion(oldVersion + 1)
	saveManifest := s.SafeSaveManifest(copiedManifest)
	if !saveManifest.IsOK() {
		return errors.New(saveManifest.Msg())
	}
	s.manifest = mnf.NewManifest(s.schema, s.options)

	return nil
}

func (s *DefaultSpace) SafeSaveManifest(manifest *mnf.Manifest) status.Status {
	tmpManifestFilePath := utils.GetManifestTmpFilePath(manifest.SpaceOptions().Uri, manifest.Version())
	manifestFilePath := utils.GetManifestFilePath(manifest.SpaceOptions().Uri, manifest.Version())
	log.Debug("path", log.String("tmpManifestFilePath", tmpManifestFilePath), log.String("manifestFilePath", manifestFilePath))
	output, err := s.fs.OpenFile(tmpManifestFilePath)
	if err != nil {
		return status.InternalStateError(err.Error())
	}
	writeManifestFileStatus := mnf.WriteManifestFile(manifest, output)
	if !writeManifestFileStatus.IsOK() {
		return writeManifestFileStatus
	}
	err = s.fs.Rename(tmpManifestFilePath, manifestFilePath)
	if err != nil {
		return status.InternalStateError(err.Error())
	}
	log.Debug("save manifest file success", log.String("path", manifestFilePath))
	return status.OK()
}

func (s *DefaultSpace) write(
	schema *arrow.Schema,
	rec arrow.Record,
	writer format.Writer,
	fragment *fragment.Fragment,
	opt *options.WriteOptions,
	isScalar bool,
) (format.Writer, error) {

	var columns []arrow.Array
	cols := rec.Columns()
	for k := range cols {
		_, has := schema.FieldsByName(rec.ColumnName(k))
		if has {
			columns = append(columns, cols[k])
		}
	}

	if isScalar {
		// add offset column for scalar
		offsetValues := make([]int64, rec.NumRows())
		for i := 0; i < int(rec.NumRows()); i++ {
			offsetValues[i] = int64(i)
		}
		builder := array.NewInt64Builder(memory.DefaultAllocator)
		builder.AppendValues(offsetValues, nil)
		offsetColumn := builder.NewArray()
		columns = append(columns, offsetColumn)
	}

	var err error

	record := array.NewRecord(schema, columns, rec.NumRows())

	if writer == nil {
		filePath := utils.GetNewParquetFilePath(s.manifest.SpaceOptions().Uri)
		writer, err = parquet.NewFileWriter(schema, s.fs, filePath)
		if err != nil {
			return nil, err
		}
		fragment.AddFile(filePath)
	}

	err = writer.Write(record)
	if err != nil {
		return nil, err
	}

	if writer.Count() >= opt.MaxRecordPerFile {
		log.Debug("close writer", log.Any("count", writer.Count()))
		err := writer.Close()
		if err != nil {
			return nil, err
		}
		writer = nil
	}

	return writer, nil
}

func (s *DefaultSpace) Read(options *options.ReadOptions) (array.RecordReader, error) {
	panic("not implemented") // TODO: Implement
}
