package fs

import (
	"github.com/milvus-io/milvus-storage-format/common/result"
	"github.com/milvus-io/milvus-storage-format/common/status"
	"github.com/milvus-io/milvus-storage-format/storage/options"
	"net/url"
)

func BuildFileSystem(uri string) *result.Result[Fs] {
	parsedUri, err := url.Parse(uri)
	if err != nil {
		return nil
	}
	switch parsedUri.Scheme {
	case "file":
		return result.NewResult(NewFsFactory().Create(options.LocalFS), status.OK())
	default:
		return result.NewResultFromStatus[Fs](status.InvalidArgument("unknown fs type"))
	}
}
