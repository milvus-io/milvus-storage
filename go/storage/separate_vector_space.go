package storage

import (
	"github.com/apache/arrow/go/v12/arrow"
	"github.com/apache/arrow/go/v12/arrow/array"
	"github.com/google/uuid"
	"github.com/milvus-io/milvus-storage-format/internal/format"
	"github.com/milvus-io/milvus-storage-format/internal/format/parquet"
	"github.com/milvus-io/milvus-storage-format/internal/fs"
	"github.com/milvus-io/milvus-storage-format/internal/manifest"
	"github.com/milvus-io/milvus-storage-format/options"
)

type SeparateVectorSpace struct {
	manifest *manifest.ManifestV2
	fs       fs.Fs
	options  *options.SpaceOptions
}

func (s *SeparateVectorSpace) Write(reader array.RecordReader, options *options.WriteOptions) error {
	// check schema consistency
	if !s.manifest.Schema().Equal(reader.Schema()) {
		return ErrSchemaNotMatch
	}

	scalarSchema, vectorSchema := s.manifest.ScalarSchema(), s.manifest.VectorSchema()
	var (
		scalarWriter format.Writer
		vectorWriter format.Writer
		scalarFiles  []*manifest.DataFile
		vectorFiles  []*manifest.DataFile
	)

	for reader.Next() {
		rec := reader.Record()

		if rec.NumRows() == 0 {
			continue
		}

		var (
			err            error
			scalarDataFile *manifest.DataFile
			vectorDataFile *manifest.DataFile
		)

		scalarWriter, scalarDataFile, err = s.write(scalarSchema, rec, scalarWriter, options)
		if err != nil {
			return err
		}
		if scalarDataFile != nil {
			scalarFiles = append(scalarFiles, scalarDataFile)
		}

		vectorWriter, vectorDataFile, err = s.write(vectorSchema, rec, vectorWriter, options)
		if err != nil {
			return err
		}
		if vectorDataFile != nil {
			vectorFiles = append(vectorFiles, vectorDataFile)
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

	if len(scalarFiles) != 0 {
		s.manifest.AddScalarDataFiles(scalarFiles...)
		s.manifest.AddVectorDataFiles(vectorFiles...)
		if err := manifest.WriteManifestV2File(s.fs, s.manifest); err != nil {
			return err
		}
	}

	return nil
}

func (s *SeparateVectorSpace) write(schema *arrow.Schema, rec arrow.Record, writer format.Writer, opt *options.WriteOptions) (format.Writer, *manifest.DataFile, error) {
	var arrs []arrow.Array
	for i := 0; i < int(rec.NumCols()); i++ {
		if schema.HasField(rec.ColumnName(i)) {
			arrs = append(arrs, rec.Column(i))
		}
	}

	var err error
	var dataFile *manifest.DataFile
	if writer == nil {
		filePath := uuid.NewString() + ".parquet"
		writer, err = parquet.NewFileWriter(schema, s.fs, filePath)
		if err != nil {
			return nil, nil, err
		}
		dataFile = manifest.NewDataFile(filePath)
	}

	rec = array.NewRecord(schema, arrs, int64(rec.NumRows()))
	if err := writer.Write(rec); err != nil {
		return nil, nil, err
	}

	if writer.Count() >= opt.MaxRowsPerFile {
		if err := writer.Close(); err != nil {
			return nil, nil, err
		}
		writer = nil
	}

	return writer, dataFile, nil
}

func (s *SeparateVectorSpace) Read(options *options.ReadOptions) (array.RecordReader, error) {
	panic("not implemented") // TODO: Implement
}
