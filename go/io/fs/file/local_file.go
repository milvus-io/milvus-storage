package file

import (
	"io"
	"os"
)

var EOF = io.EOF

type LocalFile struct {
	file os.File
}

func (l *LocalFile) Read(p []byte) (n int, err error) {
	return l.file.Read(p)
}

func (l *LocalFile) Write(p []byte) (n int, err error) {
	return l.file.Write(p)
}

func (l *LocalFile) ReadAt(p []byte, off int64) (n int, err error) {
	return l.file.ReadAt(p, off)
}

func (l *LocalFile) Seek(offset int64, whence int) (int64, error) {
	return l.file.Seek(offset, whence)
}

func (l *LocalFile) Close() error {
	return l.file.Close()
}

func NewLocalFile(f *os.File) *LocalFile {
	return &LocalFile{
		file: *f,
	}
}
