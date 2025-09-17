#ifndef TEST_H
#define TEST_H

#include <stdint.h>
#include "error_ffi.h"
#include "object_store_ffi.h"

#ifdef __cplusplus
extern "C" {
#endif

// ptr in rust
typedef struct ObjectStoreWrapper ObjectStoreWrapper;

C_BRIDGE_STATUS test_bridge_object_store_async_to_sync(ObjectStoreWrapper* wrapper, const char* path_raw);

#ifdef __cplusplus
}
#endif

#endif  // TEST_H