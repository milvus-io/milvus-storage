package storage

import (
	"github.com/apache/arrow/go/v12/arrow"
	"github.com/apache/arrow/go/v12/arrow/array"
	"github.com/milvus-io/milvus-storage-format/common/status"
	"github.com/milvus-io/milvus-storage-format/common/utils"
	"github.com/milvus-io/milvus-storage-format/file/fragment"
	"github.com/milvus-io/milvus-storage-format/io/format"
	"github.com/milvus-io/milvus-storage-format/io/format/parquet"
	"github.com/milvus-io/milvus-storage-format/io/fs"
	mnf "github.com/milvus-io/milvus-storage-format/storage/manifest"
	"github.com/milvus-io/milvus-storage-format/storage/options"
	"github.com/milvus-io/milvus-storage-format/storage/schema"
)

type DefaultSpace struct {
	basePath string
	fs       fs.Fs
	schema   *schema.Schema

	deleteFragments fragment.DeleteFragmentVector
	manifest        *mnf.Manifest
	options         *options.Options
}

func NewSeparateVectorSpace(schema *schema.Schema, op *options.Options) *DefaultSpace {
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
		scalarWriter   format.Writer
		vectorWriter   format.Writer
		scalarFragment *fragment.Fragment
		vectorFragment *fragment.Fragment
	)

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
		if err := vectorWriter.Close(); err != nil {
			return err
		}
	}

	copiedManifest := s.manifest
	oldVersion := s.manifest.Version()
	scalarFragment.SetFragmentId(oldVersion + 1)
	vectorFragment.SetFragmentId(oldVersion + 1)
	copiedManifest.AddScalarFragment(*scalarFragment)
	copiedManifest.AddVectorFragment(*vectorFragment)
	copiedManifest.SetVersion(oldVersion + 1)

	s.SafeSaveManifest(copiedManifest)
	s.manifest = new(mnf.Manifest)

	return nil
}

func (s *DefaultSpace) SafeSaveManifest(manifest *mnf.Manifest) status.Status {
	tmpManifestFilePath := utils.GetManifestTmpFilePath(manifest.SpaceOptions().Uri)
	manifestFilePath := utils.GetManifestFilePath(manifest.SpaceOptions().Uri)
	output, _ := s.fs.OpenFile(tmpManifestFilePath)
	mnf.WriteManifestFile(manifest, output)
	s.fs.Rename(tmpManifestFilePath, manifestFilePath)
	s.fs.DeleteFile(tmpManifestFilePath)
	return status.OK()
}

func (s *DefaultSpace) write(
	scalarSchema *arrow.Schema,
	rec arrow.Record,
	writer format.Writer,
	scalarFragment *fragment.Fragment,
	opt *options.WriteOptions,
	isScalar bool,
) (format.Writer, error) {

	var scalarCols []arrow.Array
	for i := 0; i < int(rec.NumCols()); i++ {
		if scalarSchema.HasField(rec.ColumnName(i)) {
			scalarCols = append(scalarCols, rec.Column(i))
		}
	}

	if isScalar {
		// add offset column for scalar
		offsetValues := make([]int64, rec.NumRows())
		for i := 0; i < int(rec.NumRows()); i++ {
			offsetValues[i] = int64(i)
		}
		builder := array.Int64Builder{}
		builder.AppendValues(offsetValues, nil)
		scalarCols = append(scalarCols, builder.NewArray())
	}

	var err error

	scalarRecord := array.NewRecord(scalarSchema, scalarCols, rec.NumRows())

	if writer == nil {
		scalarFilePath := utils.GetNewParquetFilePath(s.manifest.SpaceOptions().Uri)
		writer, err = parquet.NewFileWriter(scalarSchema, s.fs, scalarFilePath)
		if err != nil {
			return nil, err
		}
		scalarFragment.AddFile(scalarFilePath)
	}

	writer.Write(scalarRecord)

	if writer.Count() >= opt.MaxRecordPerFile {
		writer.Close()
		writer = nil
	}

	return writer, nil
}

func (s *DefaultSpace) Read(options *options.ReadOptions) (array.RecordReader, error) {
	panic("not implemented") // TODO: Implement
}
