#ifndef ERROR_FFI_H
#define ERROR_FFI_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef int32_t C_BRIDGE_STATUS;

// success
const C_BRIDGE_STATUS C_SUCCESS = 0;
const C_BRIDGE_STATUS C_ERR_INVALID_ARGS = 1;
const C_BRIDGE_STATUS C_ERR_UTF8_CONVERSION = 2;
const C_BRIDGE_STATUS C_ERR_OBJ_STORE_BUILD_FAILED = 3;

#ifdef __cplusplus
}
#endif

#endif  // ERROR_FFI_H