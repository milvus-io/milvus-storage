package manifest

import (
	"github.com/milvus-io/milvus-storage/go/common/errors"
	"github.com/milvus-io/milvus-storage/go/file/blob"
	"github.com/milvus-io/milvus-storage/go/file/fragment"
)

type ManifestCommitOp interface {
	commit(manifest *Manifest) error
}

type AddScalarFragmentOp struct {
	ScalarFragment fragment.Fragment
}

func (op AddScalarFragmentOp) commit(manifest *Manifest) error {
	op.ScalarFragment.SetFragmentId(manifest.Version())
	manifest.AddScalarFragment(op.ScalarFragment)
	return nil
}

type AddVectorFragmentOp struct {
	VectorFragment fragment.Fragment
}

func (op AddVectorFragmentOp) commit(manifest *Manifest) error {
	op.VectorFragment.SetFragmentId(manifest.Version())
	manifest.AddVectorFragment(op.VectorFragment)
	return nil
}

type AddDeleteFragmentOp struct {
	DeleteFragment fragment.Fragment
}

func (op AddDeleteFragmentOp) commit(manifest *Manifest) error {
	op.DeleteFragment.SetFragmentId(manifest.Version())
	manifest.AddDeleteFragment(op.DeleteFragment)
	return nil
}

type AddBlobOp struct {
	Replace bool
	Blob    blob.Blob
}

func (op AddBlobOp) commit(manifest *Manifest) error {
	if !op.Replace && manifest.HasBlob(op.Blob.Name) {
		return errors.ErrBlobAlreadyExist
	}
	manifest.AddBlob(op.Blob)
	return nil
}
