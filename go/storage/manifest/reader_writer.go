package manifest

import (
	"errors"
	"fmt"
	"path/filepath"

	"github.com/milvus-io/milvus-storage/go/common/constant"
	"github.com/milvus-io/milvus-storage/go/common/log"
	"github.com/milvus-io/milvus-storage/go/common/utils"
	"github.com/milvus-io/milvus-storage/go/io/fs"
)

var ErrManifestNotFound = errors.New("manifest not found")

type ManifestReaderWriter struct {
	fs   fs.Fs
	root string
}

func findAllManifest(fs fs.Fs, path string) ([]fs.FileEntry, error) {
	log.Debug("find all manifest", log.String("path", path))
	files, err := fs.List(path)
	for _, file := range files {
		log.Debug("find all manifest", log.String("file", file.Path))
	}
	if err != nil {
		return nil, err
	}
	return files, nil
}
func (rw ManifestReaderWriter) Read(version int64) (*Manifest, error) {
	manifests, err := findAllManifest(rw.fs, utils.GetManifestDir(rw.root))
	if err != nil {
		return nil, err
	}

	var maxVersionManifest string
	var maxVersion int64 = -1
	for _, m := range manifests {
		ver := utils.ParseVersionFromFileName(filepath.Base(m.Path))
		if ver == -1 {
			continue
		}

		if version != constant.LatestManifestVersion {
			if ver == version {
				return ParseFromFile(rw.fs, m.Path)
			}
		} else if ver > maxVersion {
			maxVersion = ver
			maxVersionManifest = m.Path
		}
	}

	if maxVersion != -1 {
		return ParseFromFile(rw.fs, maxVersionManifest)
	}
	return nil, ErrManifestNotFound
}

func (rw ManifestReaderWriter) MaxVersion() (int64, error) {
	manifests, err := findAllManifest(rw.fs, utils.GetManifestDir(rw.root))
	if err != nil {
		return -1, err
	}
	var max int64 = -1
	for _, m := range manifests {
		ver := utils.ParseVersionFromFileName(filepath.Base(m.Path))
		if ver == -1 {
			continue
		}

		if ver > max {
			max = ver
		}

	}

	if max == -1 {
		return -1, ErrManifestNotFound
	}
	return max, nil
}

func (rw ManifestReaderWriter) Write(m *Manifest) error {
	tmpManifestFilePath := utils.GetManifestTmpFilePath(rw.root, m.Version())
	manifestFilePath := utils.GetManifestFilePath(rw.root, m.Version())
	log.Debug("path", log.String("tmpManifestFilePath", tmpManifestFilePath), log.String("manifestFilePath", manifestFilePath))
	output, err := rw.fs.OpenFile(tmpManifestFilePath)
	if err != nil {
		return fmt.Errorf("save manfiest: %w", err)
	}
	if err = WriteManifestFile(m, output); err != nil {
		return err
	}
	err = rw.fs.Rename(tmpManifestFilePath, manifestFilePath)
	if err != nil {
		return fmt.Errorf("save manfiest: %w", err)
	}
	log.Debug("save manifest file success", log.String("path", manifestFilePath))
	return nil
}

func NewManifestReaderWriter(fs fs.Fs, root string) ManifestReaderWriter {
	return ManifestReaderWriter{fs, root}
}
