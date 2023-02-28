package file

import (
	"errors"
	"io"
)

var errInvalid = errors.New("invalid argument")

type MemoryFile struct {
	b []byte
	i int
}

func (f *MemoryFile) Write(b []byte) (int, error) {
	n, err := f.writeAt(b, int64(f.i))
	f.i += n
	return n, err
}
func (f *MemoryFile) writeAt(b []byte, off int64) (int, error) {
	if off < 0 || int64(int(off)) < off {
		return 0, errInvalid
	}
	if off > int64(len(f.b)) {
		f.truncate(off)
	}
	n := copy(f.b[off:], b)
	f.b = append(f.b, b[n:]...)
	return len(b), nil
}

func (f *MemoryFile) truncate(n int64) error {
	switch {
	case n < 0 || int64(int(n)) < n:
		return errInvalid
	case n <= int64(len(f.b)):
		f.b = f.b[:n]
		return nil
	default:
		f.b = append(f.b, make([]byte, int(n)-len(f.b))...)
		return nil
	}
}

func (f *MemoryFile) ReadAt(b []byte, off int64) (n int, err error) {
	if off < 0 || int64(int(off)) < off {
		return 0, errInvalid
	}
	if off > int64(len(f.b)) {
		return 0, io.EOF
	}
	n = copy(b, f.b[off:])
	f.i += n
	if n < len(b) {
		return n, io.EOF
	}
	return n, nil
}

func (f *MemoryFile) Seek(offset int64, whence int) (int64, error) {
	var abs int64
	switch whence {
	case io.SeekStart:
		abs = offset
	case io.SeekCurrent:
		abs = int64(f.i) + offset
	case io.SeekEnd:
		abs = int64(len(f.b)) + offset
	default:
		return 0, errInvalid
	}
	if abs < 0 {
		return 0, errInvalid
	}
	f.i = int(abs)
	return abs, nil
}

func (f *MemoryFile) Bytes() []byte {
	return f.b
}

func NewMemoryFile(b []byte) *MemoryFile {
	return &MemoryFile{
		b: b,
	}
}
