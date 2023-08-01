package filter

import (
	"github.com/apache/arrow/go/v12/arrow"
	"github.com/apache/arrow/go/v12/parquet/metadata"
	"github.com/bits-and-blooms/bitset"
)

type FilterType int8

const (
	And FilterType = iota
	Or
	Constant
	Range
)

type Filter interface {
	CheckStatistics(metadata.TypedStatistics) bool
	Type() FilterType
	Apply(colData arrow.Array, filterBitSet *bitset.BitSet)
	GetColumnName() string
}

type ComparisonType int8

const (
	Equal ComparisonType = iota
	NotEqual
	LessThan
	LessThanOrEqual
	GreaterThan
	GreaterThanOrEqual
)
