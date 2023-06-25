package status

type Code int32

const (
	KOk                 Code = 0
	kArrowError         Code = 1
	kInvalidArgument    Code = 2
	kInternalStateError Code = 3
)

type Status struct {
	code Code
	msg  string
}

func NewStatus(code Code, msg string) *Status {
	return &Status{
		code: code,
		msg:  msg,
	}
}

func (s *Status) Code() Code {
	return s.code
}

func (s *Status) Msg() string {
	return s.msg
}

func OK() Status {
	return Status{
		code: KOk,
	}
}

func ArrowError(msg string) Status {
	return Status{
		code: kArrowError,
		msg:  msg,
	}
}

func InvalidArgument(msg string) Status {
	return Status{
		code: kInvalidArgument,
		msg:  msg,
	}
}

func InternalStateError(msg string) Status {
	return Status{
		code: kInternalStateError,
		msg:  msg,
	}
}

func (s *Status) IsOK() bool {
	return s.code == KOk
}

func (s *Status) IsArrowError() bool {
	return s.code == kArrowError
}

func (s *Status) IsInvalidArgument() bool {
	return s.code == kInvalidArgument
}

func (s *Status) IsInternalStateError() bool {
	return s.code == kInternalStateError
}
