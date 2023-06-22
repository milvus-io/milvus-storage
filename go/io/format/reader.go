package format

import (
	"github.com/apache/arrow/go/v12/arrow"
)

type Reader interface {
	Read() (arrow.Record, error)
	Close() error
}
