package errors

import "errors"

var (
	ErrSchemaIsNil      = errors.New("schema is nil")
	ErrBlobAlreadyExist = errors.New("blob already exist")
	ErrBlobNotExist     = errors.New("blob not exist")
	ErrSchemaNotMatch   = errors.New("schema not match")
	ErrColumnNotExist   = errors.New("column not exist")
)
