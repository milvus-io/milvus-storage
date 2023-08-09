package fs

import (
	"errors"
	"fmt"
	"net/url"

	"github.com/milvus-io/milvus-storage-format/storage/options/option"
)

var (
	ErrInvalidFsType = errors.New("invalid fs type")
)

func BuildFileSystem(uri string) (Fs, error) {
	parsedUri, err := url.Parse(uri)
	if err != nil {
		return nil, fmt.Errorf("build file system with uri %s: %w", uri, err)
	}
	switch parsedUri.Scheme {
	case "file":
		return NewFsFactory().Create(option.LocalFS), nil
	default:
		return nil, fmt.Errorf("build file system with uri %s: %w", uri, ErrInvalidFsType)
	}
}
