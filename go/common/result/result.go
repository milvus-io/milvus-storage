package result

import (
	"github.com/milvus-io/milvus-storage-format/common/status"
)

type Result[T any] struct {
	value  *T
	status *status.Status
}

func NewResult[T any](value T) *Result[T] {
	return &Result[T]{value: &value}
}

func NewResultFromStatus[T any](status status.Status) *Result[T] {
	return &Result[T]{status: &status}
}

func (r *Result[T]) Ok() bool {
	return r.value != nil
}

func (r *Result[T]) HasValue() bool {
	return r.value != nil
}

func (r *Result[T]) Value() T {
	if r.value == nil {
		panic("value is nil")
	}
	return *r.value
}

func (r *Result[T]) Status() *status.Status {
	return r.status
}
