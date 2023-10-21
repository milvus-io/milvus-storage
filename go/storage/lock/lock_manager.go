package lock

import (
	"github.com/milvus-io/milvus-storage/go/common/constant"
	"github.com/milvus-io/milvus-storage/go/common/log"
	"github.com/pkg/errors"
	"go.uber.org/zap"
	"sync"
)

type LockManager interface {
	// Acquire the lock, wait until the lock is available, return the version to be modified or use the newest version
	Acquire() (version int64, useLatestVersion bool, err error)
	// Release the lock, accepts the new allocated manifest version and success state of operations between Acquire and Release as parameters
	Release(version int64, success bool) error
}

type EmptyLockManager struct{}

func (h *EmptyLockManager) Acquire() (version int64, useLatestVersion bool, err error) {
	return constant.LatestManifestVersion, true, nil
}

func (h *EmptyLockManager) Release(_ int64, _ bool) error {
	return nil
}

type MemoryLockManager struct {
	mu          sync.Mutex
	locks       map[int64]bool
	nextVersion int64
}

func NewMemoryLockManager() *MemoryLockManager {
	return &MemoryLockManager{
		mu:          sync.Mutex{},
		locks:       make(map[int64]bool),
		nextVersion: 0,
	}
}

func (m *MemoryLockManager) Acquire() (version int64, useLatestVersion bool, err error) {
	m.mu.Lock()
	defer m.mu.Unlock()

	version = m.nextVersion

	if m.locks[version] {
		log.Warn("lock is already acquired", zap.Int64("version", version))
		return version, false, errors.New("lock is already acquired")
	}

	if version == constant.LatestManifestVersion {
		useLatestVersion = true
	} else {
		useLatestVersion = false
	}
	m.locks[version] = true
	log.Info("acquire lock", zap.Int64("version", version), zap.Bool("useLatestVersion", useLatestVersion))

	return version, useLatestVersion, nil
}

func (m *MemoryLockManager) Release(version int64, success bool) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	realVersion := int64(0)
	realVersion = version - 1
	if !m.locks[realVersion] {
		return errors.New("lock is already released or does not exist")
	}
	m.locks[realVersion] = false
	log.Info("release lock", zap.Int64("version", realVersion), zap.Bool("success", success))
	if success {
		m.nextVersion = version
	} else {
		m.nextVersion = constant.LatestManifestVersion
	}

	return nil
}
