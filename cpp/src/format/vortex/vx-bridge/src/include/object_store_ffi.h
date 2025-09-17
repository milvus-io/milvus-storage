#ifndef OBJECT_STORE_FFI_H
#define OBJECT_STORE_FFI_H

#include <stdint.h>
#include "error_ffi.h"

#ifdef __cplusplus
extern "C" {
#endif

// ptr in rust
typedef struct ObjectStoreWrapper ObjectStoreWrapper;

C_BRIDGE_STATUS create_object_store(const char* ostype,
                                    const char* endpoint,
                                    const char* access_key_id,
                                    const char* secret_access_key,
                                    const char* region,
                                    const char* bucket_name,
                                    ObjectStoreWrapper** out_store);

void free_object_store_wrapper(ObjectStoreWrapper* wrapper);

#ifdef __cplusplus
}
#endif

#endif  // OBJECT_STORE_FFI_H