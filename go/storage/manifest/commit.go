package manifest

import (
	"github.com/milvus-io/milvus-storage/go/common/constant"
	"github.com/milvus-io/milvus-storage/go/storage/lock"
)

type ManifestCommit struct {
	ops  []ManifestCommitOp
	lock lock.LockManager
	rw   ManifestReaderWriter
}

func (m ManifestCommit) Commit() error {
	ver, latest := m.lock.Acquire()
	var err error
	var version int64
	defer func() {
		if err != nil {
			m.lock.Release(-1, false)
		} else {
			m.lock.Release(version, true)
		}
	}()
	var base *Manifest
	if latest {
		base, err = m.rw.Read(constant.LatestManifestVersion)
		if err != nil {
			return err
		}
		base.version++
	} else {
		base, err = m.rw.Read(ver)
		if err != nil {
			return err
		}
		maxVersion, err := m.rw.MaxVersion()
		if err != nil {
			return err
		}
		base.version = maxVersion + 1
	}

	for _, op := range m.ops {
		op.commit(base)
	}
	version = base.version

	return m.rw.Write(base)
}

func NewManifestCommit(ops []ManifestCommitOp, lock lock.LockManager, rw ManifestReaderWriter) ManifestCommit {
	return ManifestCommit{ops, lock, rw}
}
