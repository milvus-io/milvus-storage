use libc::{c_int};

pub type ErrCode = c_int;
pub const SUCCESS: ErrCode = 0;
pub const ERR_INVALID_ARGS: ErrCode = 1;
pub const ERR_UTF8_CONVERSION: ErrCode = 2;
pub const ERR_BUILD_FAILED: ErrCode = 3;
