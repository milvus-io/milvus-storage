// Copyright 2026 Zilliz. Licensed under the Apache License, Version 2.0.
package main

/*
#cgo pkg-config: libstorage
#include <stdlib.h>
#include "milvus-storage/ffi_c.h"
extern int32_t manifestSubmit(void*, LoonAsyncTask, void*);
static inline void runManifestTask(LoonAsyncTask task, void* data) { task(data); }
extern void manifestBeginComplete(uintptr_t, LoonFFIResult, LoonTransactionHandle);
extern void manifestCommitComplete(uintptr_t, LoonFFIResult, int32_t, int64_t);
*/
import "C"

import (
	"context"
	"fmt"
	"os"
	"runtime"
	"runtime/cgo"
	"strconv"
	"sync"
	"time"
	"unsafe"
)

// The application owns these bounded worker queues. Storage only schedules work.
type executorTask struct {
	run  C.LoonAsyncTask
	data unsafe.Pointer
}
type executor struct {
	tasks   chan executorTask
	workers sync.WaitGroup
	token   cgo.Handle
	context unsafe.Pointer
}

func newExecutor(workers int) *executor {
	pool := &executor{tasks: make(chan executorTask, 2048)}
	pool.token = cgo.NewHandle(pool)
	pool.context = C.malloc(C.size_t(unsafe.Sizeof(C.uintptr_t(0))))
	*(*C.uintptr_t)(pool.context) = C.uintptr_t(pool.token)
	for i := 0; i < workers; i++ {
		pool.workers.Add(1)
		go func() {
			defer pool.workers.Done()
			for task := range pool.tasks {
				C.runManifestTask(task.run, task.data)
			}
		}()
	}
	return pool
}
func (pool *executor) descriptor() C.LoonAsyncExecutor {
	return C.LoonAsyncExecutor{struct_size: C.uint32_t(C.sizeof_LoonAsyncExecutor),
		context: pool.context, submit: (C.LoonAsyncSubmit)(C.manifestSubmit)}
}
func (pool *executor) close() {
	close(pool.tasks)
	pool.workers.Wait()
	pool.token.Delete()
	C.free(pool.context)
}

//export manifestSubmit
func manifestSubmit(context unsafe.Pointer, task C.LoonAsyncTask, data unsafe.Pointer) C.int32_t {
	pool := cgo.Handle(*(*C.uintptr_t)(context)).Value().(*executor)
	select {
	case pool.tasks <- executorTask{task, data}:
		return 0
	default:
		return 1
	}
}

type completion struct {
	transaction C.LoonTransactionHandle
	err         error
}

type storageError struct {
	Code    int
	Message string
}

func (e storageError) Error() string { return fmt.Sprintf("storage code %d: %s", e.Code, e.Message) }

func consume(result C.LoonFFIResult) error {
	defer C.loon_ffi_free_result(&result)
	if result.err_code == 0 {
		return nil
	}
	return storageError{int(result.err_code), C.GoString(result.message)}
}

//export manifestBeginComplete
func manifestBeginComplete(token C.uintptr_t, result C.LoonFFIResult, transaction C.LoonTransactionHandle) {
	handle := cgo.Handle(token)
	done := handle.Value().(chan completion)
	value := completion{transaction, consume(result)}
	handle.Delete()
	done <- value // Buffered before submission; early completion cannot lose notification.
}

func begin(ctx context.Context, ioContext C.LoonIOContextHandle, path string, properties *C.LoonProperties) (C.LoonTransactionHandle, error) {
	return beginVersion(ctx, ioContext, path, properties, -1)
}

func beginVersion(ctx context.Context, ioContext C.LoonIOContextHandle, path string, properties *C.LoonProperties, version int64) (C.LoonTransactionHandle, error) {
	done := make(chan completion, 1)
	token := cgo.NewHandle(done)
	nativePath := C.CString(path)
	defer C.free(unsafe.Pointer(nativePath))
	var operation C.LoonAsyncHandle
	result := C.loon_transaction_begin_async(ioContext, nativePath, properties, C.int64_t(version), 0, 1, nil,
		(C.LoonTransactionBeginCallback)(C.manifestBeginComplete), C.uintptr_t(token), &operation)
	if err := consume(result); err != nil {
		token.Delete()
		return 0, err
	}
	defer C.loon_async_release(operation)
	var value completion
	select {
	case value = <-done:
	case <-ctx.Done():
		C.loon_async_cancel(operation)
		value = <-done // Keep transaction/token ownership until the synchronous worker operation completes.
	}
	return value.transaction, value.err
}

type commitCompletion struct {
	Outcome int32
	Version int64
	Err     error
}

//export manifestCommitComplete
func manifestCommitComplete(token C.uintptr_t, result C.LoonFFIResult, outcome C.int32_t, version C.int64_t) {
	handle := cgo.Handle(token)
	done := handle.Value().(chan commitCompletion)
	value := commitCompletion{int32(outcome), int64(version), consume(result)}
	handle.Delete()
	done <- value
}

func commit(ctx context.Context, ioContext C.LoonIOContextHandle, transaction C.LoonTransactionHandle) commitCompletion {
	done := make(chan commitCompletion, 1)
	token := cgo.NewHandle(done)
	var operation C.LoonAsyncHandle
	result := C.loon_transaction_commit_async(ioContext, transaction, nil,
		(C.LoonTransactionCommitCallback)(C.manifestCommitComplete), C.uintptr_t(token), &operation)
	if err := consume(result); err != nil {
		token.Delete()
		return commitCompletion{0, -1, err}
	}
	defer C.loon_async_release(operation)
	select {
	case value := <-done:
		return value
	case <-ctx.Done():
		C.loon_async_cancel(operation)
		return <-done // UNKNOWN cannot be converted into context cancellation alone.
	}
}

func main() {
	runtime.GOMAXPROCS(1)
	pool := newExecutor(1)
	defer pool.close()
	descriptor := pool.descriptor()
	var ioContext C.LoonIOContextHandle
	if err := consume(C.loon_io_context_create(&descriptor, &ioContext)); err != nil {
		panic(err)
	}
	defer C.loon_io_context_destroy(ioContext)
	pairs := [][2]string{
		{"fs.storage_type", "remote"}, {"fs.address", os.Getenv("S3_ENDPOINT")},
		{"fs.bucket_name", os.Getenv("S3_BUCKET")}, {"fs.access_key_id", os.Getenv("S3_ACCESS_KEY")},
		{"fs.access_key_value", os.Getenv("S3_SECRET_KEY")}, {"fs.region", "us-east-1"},
	}
	keys := make([]*C.char, len(pairs))
	values := make([]*C.char, len(pairs))
	for i, pair := range pairs {
		keys[i] = C.CString(pair[0])
		values[i] = C.CString(pair[1])
		defer C.free(unsafe.Pointer(keys[i]))
		defer C.free(unsafe.Pointer(values[i]))
	}
	var properties C.LoonProperties
	if err := consume(C.loon_properties_create(&keys[0], &values[0], C.size_t(len(pairs)), &properties)); err != nil {
		panic(err)
	}
	defer C.loon_properties_free(&properties)
	if value := os.Getenv("MANIFEST_CANCEL_ROUNDS"); value != "" {
		rounds, err := strconv.Atoi(value)
		if err != nil || rounds < 1 {
			panic("invalid MANIFEST_CANCEL_ROUNDS")
		}
		successes, cancelled := 0, 0
		for i := 0; i < rounds; i++ {
			ctx, cancel := context.WithCancel(context.Background())
			go func() { runtime.Gosched(); cancel() }()
			transaction, err := beginVersion(ctx, ioContext, "cancel-example", &properties, 0)
			cancel()
			if transaction != 0 {
				C.loon_transaction_destroy(transaction)
			}
			if err == nil {
				successes++
			} else if native, ok := err.(storageError); ok && native.Code == 113 {
				cancelled++
			} else {
				panic(err)
			}
		}
		fmt.Printf("callback/context races: success=%d cancelled=%d total=%d\n", successes, cancelled, rounds)
		return
	}

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	// The harness delays the backend. Capture the actual Go waiting stack while a
	// second goroutine runs; synchronous cgo would show a different waiting frame.
	observed := make(chan struct{})
	go func() {
		time.Sleep(100 * time.Millisecond)
		stack := make([]byte, 64<<10)
		n := runtime.Stack(stack, true)
		fmt.Printf("concurrent goroutine ran; waiting stacks:\n%s\n", stack[:n])
		close(observed)
	}()
	transaction, err := begin(ctx, ioContext, os.Getenv("MANIFEST_PATH"), &properties)
	if transaction != 0 {
		defer C.loon_transaction_destroy(transaction)
		if os.Getenv("MANIFEST_COMMIT") == "1" {
			path := C.CString("_delta/example")
			mutationErr := consume(C.loon_transaction_add_delta_log(transaction, path, 1))
			C.free(unsafe.Pointer(path))
			if mutationErr != nil {
				panic(mutationErr)
			}
			value := commit(ctx, ioContext, transaction)
			fmt.Printf("commit outcome=%d version=%d error=%v\n", value.Outcome, value.Version, value.Err)
			// UNKNOWN (2) is preserved even for a transient error. Do not blindly retry.
			if value.Err != nil {
				panic(value.Err)
			}
		}
	}
	<-observed
	if err != nil {
		panic(err)
	}
	fmt.Println("async begin completed")
}
