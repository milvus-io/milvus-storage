package fs

import (
	"errors"
	"fmt"
	"net/url"

	"github.com/milvus-io/milvus-storage/go/storage/options"
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
		return NewFsFactory().Create(options.LocalFS, parsedUri)
	case "s3":
		return NewFsFactory().Create(options.S3, parsedUri)

	default:
		return nil, fmt.Errorf("build file system with uri %s: %w", uri, ErrInvalidFsType)
	}
}
