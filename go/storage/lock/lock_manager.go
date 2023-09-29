package lock

import (
	"github.com/milvus-io/milvus-storage/go/common/constant"
	"github.com/milvus-io/milvus-storage/go/common/log"
	"sync"
)

type LockManager interface {
	// Acquire the lock, wait until the lock is available, return the version to be modified or use the newest version
	Acquire() (version int64, useLatestVersion bool, err error)
	// Release the lock, accepts the new allocated manifest version and success state of operations between Acquire and Release as parameters
	Release(version int64, success bool) error
}

type EmptyLockManager struct {
	lock sync.Mutex
}

func (h *EmptyLockManager) Acquire() (version int64, useLatestVersion bool, err error) {
	log.Debug("acquire lock")
	h.lock.Lock()
	return constant.LatestManifestVersion, true, nil
}

func (h *EmptyLockManager) Release(_ int64, _ bool) error {
	log.Debug("release lock")
	h.lock.Unlock()
	return nil
}
