package storage

import (
	"errors"
	"fmt"
	"math"
	"net/url"

	"github.com/apache/arrow/go/v12/arrow"
	"github.com/apache/arrow/go/v12/arrow/array"
	"github.com/apache/arrow/go/v12/arrow/memory"
	"github.com/milvus-io/milvus-storage/go/common/log"
	"github.com/milvus-io/milvus-storage/go/common/utils"
	"github.com/milvus-io/milvus-storage/go/file/blob"
	"github.com/milvus-io/milvus-storage/go/file/fragment"
	"github.com/milvus-io/milvus-storage/go/filter"
	"github.com/milvus-io/milvus-storage/go/io/format"
	"github.com/milvus-io/milvus-storage/go/io/format/parquet"
	"github.com/milvus-io/milvus-storage/go/io/fs"
	"github.com/milvus-io/milvus-storage/go/reader/record_reader"
	"github.com/milvus-io/milvus-storage/go/storage/lock"
	"github.com/milvus-io/milvus-storage/go/storage/manifest"
	"github.com/milvus-io/milvus-storage/go/storage/options"
)

var (
	ErrSchemaIsNil      = errors.New("schema is nil")
	ErrBlobAlreadyExist = errors.New("blob already exist")
	ErrBlobNotExist     = errors.New("blob not exist")
	ErrSchemaNotMatch   = errors.New("schema not match")
	ErrColumnNotExist   = errors.New("column not exist")
)

type Space struct {
	path            string
	fs              fs.Fs
	deleteFragments fragment.DeleteFragmentVector
	manifest        *manifest.Manifest
	lockManager     lock.LockManager
}

func (s *Space) init() error {
	for _, f := range s.manifest.GetDeleteFragments() {
		deleteFragment := fragment.Make(s.fs, s.manifest.GetSchema(), f)
		s.deleteFragments = append(s.deleteFragments, deleteFragment)
	}
	return nil
}

func NewSpace(f fs.Fs, path string, m *manifest.Manifest, lockManager lock.LockManager) *Space {
	deleteFragments := fragment.DeleteFragmentVector{}
	return &Space{
		fs:              f,
		path:            path,
		manifest:        m,
		deleteFragments: deleteFragments,
		lockManager:     lockManager,
	}
}

func (s *Space) Write(reader array.RecordReader, options *options.WriteOptions) error {
	// check schema consistency
	if !s.manifest.GetSchema().Schema().Equal(reader.Schema()) {
		return ErrSchemaNotMatch
	}

	scalarSchema, vectorSchema := s.manifest.GetSchema().ScalarSchema(), s.manifest.GetSchema().VectorSchema()
	var (
		scalarWriter format.Writer
		vectorWriter format.Writer
	)
	scalarFragment := fragment.NewFragment()
	vectorFragment := fragment.NewFragment()

	for reader.Next() {
		rec := reader.Record()

		if rec.NumRows() == 0 {
			continue
		}
		var err error
		scalarWriter, err = s.write(scalarSchema, rec, scalarWriter, &scalarFragment, options, true)
		if err != nil {
			return err
		}
		vectorWriter, err = s.write(vectorSchema, rec, vectorWriter, &vectorFragment, options, false)
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

	if scalarWriter == nil {
		return nil
	}

	op1 := manifest.AddScalarFragmentOp{ScalarFragment: scalarFragment}
	op2 := manifest.AddVectorFragmentOp{VectorFragment: vectorFragment}
	commit := manifest.NewManifestCommit([]manifest.ManifestCommitOp{op1, op2}, s.lockManager, manifest.NewManifestReaderWriter(s.fs, s.path))
	commit.Commit()
	return nil
}

func (s *Space) Delete(reader array.RecordReader) error {
	// TODO: add delete frament
	schema := s.manifest.GetSchema().DeleteSchema()
	fragment := fragment.NewFragment()
	var (
		err        error
		writer     format.Writer
		deleteFile string
	)

	for reader.Next() {
		rec := reader.Record()
		if rec.NumRows() == 0 {
			continue
		}

		if writer == nil {
			deleteFile = utils.GetNewParquetFilePath(utils.GetDeleteDataDir(s.path))
			writer, err = parquet.NewFileWriter(schema, s.fs, deleteFile)
			if err != nil {
				return err
			}
		}

		if err = writer.Write(rec); err != nil {
			return err
		}
	}

	if writer != nil {
		if err = writer.Close(); err != nil {
			return err
		}

		op := manifest.AddDeleteFragmentOp{DeleteFragment: fragment}
		commit := manifest.NewManifestCommit([]manifest.ManifestCommitOp{op}, s.lockManager, manifest.NewManifestReaderWriter(s.fs, s.path))
		commit.Commit()
	}
	return nil
}

func (s *Space) write(
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

	var rootPath string
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
		rootPath = utils.GetScalarDataDir(s.path)
	} else {
		rootPath = utils.GetVectorDataDir(s.path)
	}

	var err error

	record := array.NewRecord(schema, columns, rec.NumRows())

	if writer == nil {
		filePath := utils.GetNewParquetFilePath(rootPath)
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
		err = writer.Close()
		if err != nil {
			return nil, err
		}
		writer = nil
	}

	return writer, nil
}

// Open opened a space or create if the space does not exist.
// If space does not exist. schema should not be nullptr, or an error will be returned.
// If space exists and version is specified, it will restore to the state at this version,
// or it will choose the latest version.
func Open(uri string, opt options.Options) (*Space, error) {
	var f fs.Fs
	var m *manifest.Manifest
	var path string
	f, err := fs.BuildFileSystem(uri)
	if err != nil {
		return nil, err
	}

	parsedUri, err := url.Parse(uri)
	if err != nil {
		return nil, err
	}
	path = parsedUri.Path
	log.Debug("open space", log.String("path", path))

	log.Debug(utils.GetManifestDir(path))
	// create if not exist
	if err = f.CreateDir(utils.GetManifestDir(path)); err != nil {
		return nil, err
	}
	if err = f.CreateDir(utils.GetScalarDataDir(path)); err != nil {
		return nil, err
	}
	if err = f.CreateDir(utils.GetVectorDataDir(path)); err != nil {
		return nil, err
	}
	if err = f.CreateDir(utils.GetBlobDir(path)); err != nil {
		return nil, err
	}
	if err = f.CreateDir(utils.GetDeleteDataDir(path)); err != nil {
		return nil, err
	}

	rw := manifest.NewManifestReaderWriter(f, path)
	m, err = rw.Read(opt.Version)

	if err != nil {
		// create the first manifest file
		if err == manifest.ErrManifestNotFound {
			if opt.Schema == nil {
				log.Error("schema is nil")
				return nil, ErrSchemaIsNil
			}
			m = manifest.NewManifest(opt.Schema)
			m.SetVersion(0) //TODO: check if this is necessary
			if err = rw.Write(m); err != nil {
				return nil, err
			}
		} else {
			return nil, err
		}
	}
	space := NewSpace(f, path, m, opt.LockManager)
	return space, nil
}

func (s *Space) readManifest(version int64) error {
	rw := manifest.NewManifestReaderWriter(s.fs, s.path)
	manifest, err := rw.Read(version)
	if err != nil {
		return err
	}
	s.manifest = manifest
	return nil
}

func (s *Space) Read(readOptions *options.ReadOptions) (array.RecordReader, error) {
	if s.manifest == nil || readOptions.ManifestVersion != s.manifest.Version() {
		if err := s.readManifest(readOptions.ManifestVersion); err != nil {
			return nil, err
		}
	}
	if s.manifest.GetSchema().Options().HasVersionColumn() {
		f := filter.NewConstantFilter(filter.LessThanOrEqual, s.manifest.GetSchema().Options().VersionColumn, int64(math.MaxInt64))
		readOptions.AddFilter(f)
		readOptions.AddColumn(s.manifest.GetSchema().Options().VersionColumn)
	}
	log.Debug("read", log.Any("readOption", readOptions))

	return record_reader.MakeRecordReader(s.manifest, s.manifest.GetSchema(), s.fs, s.deleteFragments, readOptions), nil
}

func (s *Space) WriteBlob(content []byte, name string, replace bool) error {
	if !replace && s.manifest.HasBlob(name) {
		return ErrBlobAlreadyExist
	}

	blobFile := utils.GetBlobFilePath(utils.GetBlobDir(s.path))
	f, err := s.fs.OpenFile(blobFile)
	if err != nil {
		return err
	}

	n, err := f.Write(content)
	if err != nil {
		return err
	}

	if n != len(content) {
		return fmt.Errorf("blob not writen completely, writen %d but expect %d", n, len(content))
	}

	if err = f.Close(); err != nil {
		return err
	}

	op := manifest.AddBlobOp{Blob: blob.Blob{
		Name: name,
		Size: int64(len(content)),
		File: blobFile,
	}}
	commit := manifest.NewManifestCommit([]manifest.ManifestCommitOp{op}, s.lockManager, manifest.NewManifestReaderWriter(s.fs, s.path))
	commit.Commit()
	return nil
}

func (s *Space) ReadBlob(name string, output []byte) (int, error) {
	blob, ok := s.manifest.GetBlob(name)
	if !ok {
		return -1, ErrBlobNotExist
	}

	f, err := s.fs.OpenFile(blob.File)
	if err != nil {
		return -1, err
	}

	return f.Read(output)
}

func (s *Space) GetBlobByteSize(name string) (int64, error) {
	blob, ok := s.manifest.GetBlob(name)
	if !ok {
		return -1, ErrBlobNotExist
	}
	return blob.Size, nil
}

func (s *Space) GetCurrentVersion() int64 {
	return s.manifest.Version()
}

func (s *Space) ScanDelete() (array.RecordReader, error) {
	return record_reader.MakeScanDeleteReader(s.manifest, s.fs), nil
}
