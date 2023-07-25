package log

import (
	"testing"
)

func TestLogger(t *testing.T) {
	defer Sync()
	Info("Testing")
	Debug("Testing")
	Warn("Testing")
	Error("Testing")
	defer func() {
		if err := recover(); err != nil {
			Debug("logPanic recover")
		}
	}()
	Panic("Testing")
}
