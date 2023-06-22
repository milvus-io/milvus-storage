package format

import "github.com/apache/arrow/go/v12/arrow"

type Writer interface {
	Write(record arrow.Record) error
	Count() int64
	Close() error
}
