package storage

import (
	"math"
	"net/url"

	"github.com/apache/arrow/go/v12/arrow/array"
	"github.com/milvus-io/milvus-storage/go/common/errors"
	"github.com/milvus-io/milvus-storage/go/common/log"
	"github.com/milvus-io/milvus-storage/go/common/utils"
	"github.com/milvus-io/milvus-storage/go/file/fragment"
	"github.com/milvus-io/milvus-storage/go/filter"
	"github.com/milvus-io/milvus-storage/go/io/fs"
	"github.com/milvus-io/milvus-storage/go/reader/record_reader"
	"github.com/milvus-io/milvus-storage/go/storage/lock"
	"github.com/milvus-io/milvus-storage/go/storage/manifest"
	"github.com/milvus-io/milvus-storage/go/storage/options"
	"github.com/milvus-io/milvus-storage/go/storage/transaction"
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

func (s *Space) NewTransaction() transaction.Transaction {
	return transaction.NewConcurrentWriteTransaction(s)
}

func (s *Space) Write(reader array.RecordReader, options *options.WriteOptions) error {
	return transaction.NewConcurrentWriteTransaction(s).Write(reader, options).Commit()
}

func (s *Space) Delete(reader array.RecordReader) error {
	return transaction.NewConcurrentWriteTransaction(s).Delete(reader).Commit()
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
				return nil, errors.ErrSchemaIsNil
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
	return transaction.NewConcurrentWriteTransaction(s).WriteBlob(content, name, replace).Commit()
}

func (s *Space) ReadBlob(name string, output []byte) (int, error) {
	blob, ok := s.manifest.GetBlob(name)
	if !ok {
		return -1, errors.ErrBlobNotExist
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
		return -1, errors.ErrBlobNotExist
	}
	return blob.Size, nil
}

func (s *Space) GetCurrentVersion() int64 {
	return s.manifest.Version()
}

func (s *Space) ScanDelete() (array.RecordReader, error) {
	return record_reader.MakeScanDeleteReader(s.manifest, s.fs), nil
}

func (s *Space) Path() string {
	return s.path
}

func (s *Space) Fs() fs.Fs {
	return s.fs
}

func (s *Space) Manifest() *manifest.Manifest {
	return s.manifest
}

func (s *Space) LockManager() lock.LockManager {
	return s.lockManager
}
