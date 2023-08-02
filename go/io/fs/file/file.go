package file

import "io"

type File interface {
	io.Writer
	io.ReaderAt
	io.Seeker
	io.Reader
	io.Closer
}
