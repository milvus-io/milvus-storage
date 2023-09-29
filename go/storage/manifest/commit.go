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

func (m *ManifestCommit) AddOp(op ...ManifestCommitOp) {
	m.ops = append(m.ops, op...)
}

func (m ManifestCommit) Commit() (err error) {
	ver, latest, err := m.lock.Acquire()
	if err != nil {
		return err
	}
	var version int64
	defer func() {
		if err != nil {
			if err2 := m.lock.Release(-1, false); err2 != nil {
				err = err2
			}
		} else {
			err = m.lock.Release(version, true)
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

	err = m.rw.Write(base)
	if err != nil {
		return err
	}
	return nil
}

func NewManifestCommit(lock lock.LockManager, rw ManifestReaderWriter) ManifestCommit {
	return ManifestCommit{nil, lock, rw}
}
