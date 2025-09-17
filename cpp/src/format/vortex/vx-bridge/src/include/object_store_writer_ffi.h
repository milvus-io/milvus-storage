#ifndef OBJECT_STORE_WRITER_FFI_H
#define OBJECT_STORE_WRITER_FFI_H

#include <stdint.h>
#include "error_ffi.h"
#include "object_store_ffi.h"

#ifdef __cplusplus
extern "C" {
#endif

// ptr in rust
typedef struct ObjectStoreWrapper ObjectStoreWrapper;
typedef struct ObjectStoreWriterWrapper ObjectStoreWriterWrapper;

C_BRIDGE_STATUS create_object_store_writer(const ObjectStoreWrapper* object_store,
                                           const char* location,
                                           ObjectStoreWriterWrapper** out_writer);

void free_object_store_writer(ObjectStoreWriterWrapper* writer);

C_BRIDGE_STATUS write_array_stream(const ObjectStoreWrapper* object_store, uint8_t* input_stream, const char* location);

#ifdef __cplusplus
}
#endif

#endif  // OBJECT_STORE_WRITER_FFI_H