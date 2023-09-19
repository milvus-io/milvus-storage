package lock

import "github.com/milvus-io/milvus-storage/go/common/constant"

type LockManager interface {
	// Acquire the lock, wait until the lock is available, return the version to be modified or use the newest version
	Acquire() (version int64, useLatestVersion bool)
	// Release the lock, accepts the new allocated manifest version and success state of operations between Acquire and Release as parameters
	Release(version int64, success bool)
}

type EmptyLockManager struct{}

func (h *EmptyLockManager) Acquire() (version int64, useLatestVersion bool) {
	return constant.LatestManifestVersion, true
}

func (h *EmptyLockManager) Release(_ int64, _ bool) {}
