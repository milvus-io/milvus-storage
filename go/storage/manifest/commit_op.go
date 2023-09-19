package manifest

import (
	"github.com/milvus-io/milvus-storage/go/file/blob"
	"github.com/milvus-io/milvus-storage/go/file/fragment"
)

type ManifestCommitOp interface {
	commit(manifest *Manifest)
}

type AddScalarFragmentOp struct {
	ScalarFragment fragment.Fragment
}

func (op AddScalarFragmentOp) commit(manifest *Manifest) {
	op.ScalarFragment.SetFragmentId(manifest.Version())
	manifest.AddScalarFragment(op.ScalarFragment)
}

type AddVectorFragmentOp struct {
	VectorFragment fragment.Fragment
}

func (op AddVectorFragmentOp) commit(manifest *Manifest) {
	op.VectorFragment.SetFragmentId(manifest.Version())
	manifest.AddVectorFragment(op.VectorFragment)
}

type AddDeleteFragmentOp struct {
	DeleteFragment fragment.Fragment
}

func (op AddDeleteFragmentOp) commit(manifest *Manifest) {
	op.DeleteFragment.SetFragmentId(manifest.Version())
	manifest.AddDeleteFragment(op.DeleteFragment)
}

type AddBlobOp struct {
	Blob blob.Blob
}

func (op AddBlobOp) commit(manifest *Manifest) {
	manifest.AddBlob(op.Blob)
}
