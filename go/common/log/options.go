package log

import "go.uber.org/zap"

type Option = zap.Option

var (
	WrapCore      = zap.WrapCore
	Hooks         = zap.Hooks
	Fields        = zap.Fields
	ErrorOutput   = zap.ErrorOutput
	Development   = zap.Development
	AddCaller     = zap.AddCaller
	WithCaller    = zap.WithCaller
	AddCallerSkip = zap.AddCallerSkip
	AddStacktrace = zap.AddStacktrace
	IncreaseLevel = zap.IncreaseLevel
	WithFatalHook = zap.WithFatalHook
	WithClock     = zap.WithClock
)
