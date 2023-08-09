package storage

import (
	"errors"
	"fmt"
	"math"
	"net/url"
	"os"
	"sort"
	"sync"
	"sync/atomic"

	"github.com/apache/arrow/go/v12/arrow"
	"github.com/apache/arrow/go/v12/arrow/array"
	"github.com/apache/arrow/go/v12/arrow/memory"
	"github.com/milvus-io/milvus-storage-format/common/log"
	"github.com/milvus-io/milvus-storage-format/common/utils"
	"github.com/milvus-io/milvus-storage-format/file/blob"
	"github.com/milvus-io/milvus-storage-format/file/fragment"
	"github.com/milvus-io/milvus-storage-format/filter"
	"github.com/milvus-io/milvus-storage-format/io/format"
	"github.com/milvus-io/milvus-storage-format/io/format/parquet"
	"github.com/milvus-io/milvus-storage-format/io/fs"
	"github.com/milvus-io/milvus-storage-format/reader/record_reader"
	"github.com/milvus-io/milvus-storage-format/storage/manifest"
	"github.com/milvus-io/milvus-storage-format/storage/options/option"
)

var (
	ErrSchemaIsNil      = errors.New("schema is nil")
	ErrManifestNotFound = errors.New("manifest not found")
	ErrBlobAlreadyExist = errors.New("blob already exist")
	ErrBlobNotExist     = errors.New("blob not exist")
)

type DefaultSpace struct {
	path                string
	fs                  fs.Fs
	deleteFragments     fragment.DeleteFragmentVector
	manifest            *manifest.Manifest
	lock                sync.RWMutex
	nextManifestVersion int64
}

func (s *DefaultSpace) init() error {
	for _, f := range s.manifest.GetDeleteFragments() {
		deleteFragment := fragment.Make(s.fs, s.manifest.GetSchema(), f)
		s.deleteFragments = append(s.deleteFragments, deleteFragment)
	}
	return nil
}

func NewDefaultSpace(f fs.Fs, path string, m *manifest.Manifest, nv int64) *DefaultSpace {
	deleteFragments := fragment.DeleteFragmentVector{}
	return &DefaultSpace{
		fs:                  f,
		path:                path,
		manifest:            m,
		nextManifestVersion: nv,
		deleteFragments:     deleteFragments,
	}
}

func (s *DefaultSpace) Write(reader array.RecordReader, options *option.WriteOptions) error {
	// check schema consistency
	if !s.manifest.GetSchema().Schema().Equal(reader.Schema()) {
		return ErrSchemaNotMatch
	}

	scalarSchema, vectorSchema := s.manifest.GetSchema().ScalarSchema(), s.manifest.GetSchema().VectorSchema()
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
		var err error
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
	copied := s.manifest.Copy()

	nextVersion := s.nextManifestVersion
	currentVersion := s.manifest.Version()
	log.Debug("s.manifest.Version()", log.Int64("current version", currentVersion))
	log.Debug("s.nextManifestVersion", log.Int64("next version", nextVersion))

	scalarFragment.SetFragmentId(nextVersion)
	vectorFragment.SetFragmentId(nextVersion)

	copied.SetVersion(nextVersion)
	copied.AddScalarFragment(*scalarFragment)
	copied.AddVectorFragment(*vectorFragment)

	log.Debug("check copied set version", log.Int64("copied version", copied.Version()))
	if err := safeSaveManifest(s.fs, s.path, copied); err != nil {
		return err
	}
	s.manifest = copied
	atomic.AddInt64(&s.nextManifestVersion, 1)

	return nil
}

func safeSaveManifest(fs fs.Fs, path string, m *manifest.Manifest) error {
	tmpManifestFilePath := utils.GetManifestTmpFilePath(path, m.Version())
	manifestFilePath := utils.GetManifestFilePath(path, m.Version())
	log.Debug("path", log.String("tmpManifestFilePath", tmpManifestFilePath), log.String("manifestFilePath", manifestFilePath))
	output, err := fs.OpenFile(tmpManifestFilePath)
	if err != nil {
		return fmt.Errorf("save manfiest: %w", err)
	}
	if err = manifest.WriteManifestFile(m, output); err != nil {
		return err
	}
	err = fs.Rename(tmpManifestFilePath, manifestFilePath)
	if err != nil {
		return fmt.Errorf("save manfiest: %w", err)
	}
	log.Debug("save manifest file success", log.String("path", manifestFilePath))
	return nil
}

func (s *DefaultSpace) write(
	schema *arrow.Schema,
	rec arrow.Record,
	writer format.Writer,
	fragment *fragment.Fragment,
	opt *option.WriteOptions,
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
		filePath := utils.GetNewParquetFilePath(s.path)
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
func Open(uri string, op option.Options) (*DefaultSpace, error) {
	var f fs.Fs
	var m *manifest.Manifest
	var path string
	var nextManifestVersion int64
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
	err = f.CreateDir(utils.GetManifestDir(path))
	if err != nil {
		log.Error("create dir error", log.String("path", utils.GetManifestDir(path)))
		return nil, err
	}

	manifestFileInfoVec, err := findAllManifest(f, utils.GetManifestDir(path))
	if err != nil {
		log.Error("find all manifest file error", log.String("path", utils.GetManifestDir(path)))
		return nil, err
	}

	var filteredInfoVec []os.DirEntry
	for _, info := range manifestFileInfoVec {
		if utils.ParseVersionFromFileName(info.Name()) != -1 {
			filteredInfoVec = append(filteredInfoVec, info)
		}
	}
	sort.Slice(filteredInfoVec, func(i, j int) bool {
		return utils.ParseVersionFromFileName(filteredInfoVec[i].Name()) < utils.ParseVersionFromFileName(filteredInfoVec[j].Name())
	})

	// not exist manifest file, create new manifest file
	if len(filteredInfoVec) == 0 {
		if op.Schema == nil {
			log.Error("schema is nil")
			return nil, ErrSchemaIsNil
		}
		m = manifest.NewManifest(op.Schema)
		m.SetVersion(0) //TODO: check if this is necessary
		if err = safeSaveManifest(f, path, m); err != nil {
			return nil, err
		}
		atomic.AddInt64(&nextManifestVersion, 1)
	} else {
		var fileInfo os.DirEntry
		var version int64
		// not assign version to restore to the latest version manifest
		if op.Version == -1 {
			maxVersion := int64(-1)
			var maxManifest os.DirEntry
			for _, info := range filteredInfoVec {
				version := utils.ParseVersionFromFileName(info.Name())
				if version > maxVersion {
					maxVersion = version
					maxManifest = info
				}
			}
			// the last one
			fileInfo = maxManifest
			version = maxVersion
			atomic.AddInt64(&nextManifestVersion, version+1)

		} else {
			// assign version to restore to the specified version manifest
			for _, info := range filteredInfoVec {
				ver := utils.ParseVersionFromFileName(info.Name())
				if ver == op.Version {
					fileInfo = info
					atomic.AddInt64(&nextManifestVersion, ver+1)
				}
			}
			if fileInfo == nil {
				return nil, fmt.Errorf("open manifest: %w", ErrManifestNotFound)
			}
			version = op.Version
		}
		manifestFilePath := utils.GetManifestFilePath(path, version)

		m, err = manifest.ParseFromFile(f, manifestFilePath)
		if err != nil {
			return nil, err
		}
	}
	space := NewDefaultSpace(f, path, m, nextManifestVersion)
	space.init()
	return space, nil
}

func findAllManifest(fs fs.Fs, path string) ([]os.DirEntry, error) {
	log.Debug("find all manifest", log.String("path", path))
	files, err := fs.List(path)
	for _, file := range files {
		log.Debug("find all manifest", log.String("file", file.Name()))
	}
	if err != nil {
		return nil, err
	}
	return files, nil
}

func (s *DefaultSpace) Read(readOption *option.ReadOptions) (array.RecordReader, error) {

	if s.manifest.GetSchema().Options().HasVersionColumn() {
		f := filter.NewConstantFilter(filter.LessThanOrEqual, s.manifest.GetSchema().Options().VersionColumn, int64(math.MaxInt64))
		readOption.AddFilter(f)
		readOption.AddColumn(s.manifest.GetSchema().Options().VersionColumn)
	}
	log.Debug("read", log.Any("readOption", readOption))

	return record_reader.MakeRecordReader(s.manifest, s.manifest.GetSchema(), s.fs, s.deleteFragments, readOption), nil
}

func (s *DefaultSpace) WriteBlob(content []byte, name string, replace bool) error {
	if !replace && s.manifest.HasBlob(name) {
		return ErrBlobAlreadyExist
	}

	blobFile := utils.GetBlobFilePath(s.path)
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

	s.lock.Lock()
	defer s.lock.Unlock()
	copied := s.manifest.Copy()

	nextVersion := s.nextManifestVersion
	copied.SetVersion(nextVersion)
	copied.AddBlob(blob.Blob{
		Name: name,
		Size: int64(len(content)),
		File: blobFile,
	})

	if err := safeSaveManifest(s.fs, s.path, copied); err != nil {
		return err
	}
	s.manifest = copied
	atomic.AddInt64(&s.nextManifestVersion, 1)
	return nil
}

func (s *DefaultSpace) ReadBlob(name string, output []byte) (int, error) {
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

func (s *DefaultSpace) GetBlobByteSize(name string) (int64, error) {
	blob, ok := s.manifest.GetBlob(name)
	if !ok {
		return -1, ErrBlobNotExist
	}
	return blob.Size, nil
}
