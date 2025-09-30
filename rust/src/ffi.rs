use std::ffi::{CStr, CString};
use std::os::raw::c_char;
use std::ptr;
use crate::error::{MilvusError, Result};

// Include the generated bindings
include!(concat!(env!("OUT_DIR"), "/bindings.rs"));

// Helper implementations for the generated types
impl ArrowSchema {
    pub fn empty() -> Self {
        // Return a default/zero-initialized struct
        unsafe { std::mem::zeroed() }
    }
}

impl ArrowArray {
    pub fn empty() -> Self {
        // Return a default/zero-initialized struct
        unsafe { std::mem::zeroed() }
    }
}

impl ArrowArrayStream {
    pub fn empty() -> Self {
        // Return a default/zero-initialized struct
        unsafe { std::mem::zeroed() }
    }
}

// Helper function to check FFIResult and convert to Rust Result
fn check_ffi_result(result: FFIResult) -> Result<()> {
    unsafe {
        let mut result = result;
        if IsSuccess(&mut result) != 0 {
            Ok(())
        } else {
            let error_msg = GetErrorMessage(&mut result);
            let error_str = if error_msg.is_null() {
                "Unknown FFI error".to_string()
            } else {
                CStr::from_ptr(error_msg).to_string_lossy().to_string()
            };
            FreeFFIResult(&mut result);
            Err(MilvusError::Ffi(error_str))
        }
    }
}

/// Safe wrapper around ReaderHandle
pub struct Reader {
    handle: ReaderHandle,
}

impl Reader {
    /// Create a new reader
    pub fn new(
        manifest: &str,
        schema: &ArrowSchema,
        columns: Option<&[&str]>,
        properties: Option<&Properties>,
    ) -> Result<Self> {
        let manifest_cstr = CString::new(manifest)
            .map_err(|e| MilvusError::Ffi(format!("Invalid manifest string: {}", e)))?;

        // Convert column names to C strings if provided
        let (column_ptrs, _column_cstrs): (Vec<*const std::os::raw::c_char>, Vec<CString>) = if let Some(cols) = columns {
            let cstrs: Result<Vec<CString>> = cols
                .iter()
                .map(|s| CString::new(*s).map_err(|e| MilvusError::Ffi(format!("Invalid column name: {}", e))))
                .collect();
            let cstrs = cstrs?;
            let ptrs: Vec<*const std::os::raw::c_char> = cstrs.iter().map(|s| s.as_ptr()).collect();
            (ptrs, cstrs)
        } else {
            (Vec::new(), Vec::new())
        };

        let column_ptr = if column_ptrs.is_empty() {
            ptr::null()
        } else {
            column_ptrs.as_ptr()
        };

        let properties_ptr = properties.map_or(ptr::null(), |p| p as *const Properties);

        let mut handle: ReaderHandle = 0;
        let result = unsafe {
            reader_new(
                manifest_cstr.as_ptr() as *mut c_char,
                schema as *const ArrowSchema as *mut ArrowSchema,
                column_ptr,
                column_ptrs.len(),
                properties_ptr,
                &mut handle,
            )
        };

        check_ffi_result(result)?;
        Ok(Reader { handle })
    }

    /// Get a record batch reader for streaming data
    pub fn get_record_batch_reader(
        &self,
        predicate: Option<&str>,
        batch_size: i64,
        buffer_size: i64,
    ) -> Result<*mut ArrowArrayStream> {
        let predicate_ptr = if let Some(pred) = predicate {
            let predicate_cstr = CString::new(pred)
                .map_err(|e| MilvusError::Ffi(format!("Invalid predicate string: {}", e)))?;
            predicate_cstr.as_ptr()
        } else {
            ptr::null()
        };

        let mut stream = ArrowArrayStream::empty();
        let result = unsafe {
            get_record_batch_reader(self.handle, predicate_ptr, batch_size, buffer_size, &mut stream)
        };

        check_ffi_result(result)?;

        Ok(&mut stream as *mut ArrowArrayStream)
    }

    /// Take specific rows by indices
    pub fn take(&self, row_indices: &[i64], parallelism: i64) -> Result<*mut ArrowArray> {
        let mut array = ArrowArray::empty();
        let result = unsafe {
            take(
                self.handle,
                row_indices.as_ptr(),
                row_indices.len(),
                parallelism,
                &mut array,
            )
        };

        check_ffi_result(result)?;

        Ok(&mut array as *mut ArrowArray)
    }

    /// Get a chunk reader for a specific column group
    pub fn get_chunk_reader(&self, column_group_id: i64) -> Result<ChunkReader> {
        let mut handle: ChunkReaderHandle = 0;
        let result = unsafe { get_chunk_reader(self.handle, column_group_id, &mut handle) };

        check_ffi_result(result)?;

        Ok(ChunkReader { handle })
    }
}

impl Drop for Reader {
    fn drop(&mut self) {
        unsafe {
            reader_destroy(self.handle);
        }
    }
}

// Thread safety markers
unsafe impl Send for Reader {}
unsafe impl Sync for Reader {}

/// Safe wrapper around ChunkReaderHandle
pub struct ChunkReader {
    handle: ChunkReaderHandle,
}

impl ChunkReader {
    /// Get chunk indices for given row indices
    pub fn get_chunk_indices(&self, row_indices: &[i64]) -> Result<Vec<i64>> {
        let mut chunk_indices_ptr: *mut i64 = ptr::null_mut();
        let mut num_chunk_indices: usize = 0;

        let result = unsafe {
            get_chunk_indices(
                self.handle,
                row_indices.as_ptr(),
                row_indices.len(),
                &mut chunk_indices_ptr,
                &mut num_chunk_indices,
            )
        };

        check_ffi_result(result)?;

        if chunk_indices_ptr.is_null() {
            return Ok(Vec::new());
        }

        let chunk_indices = unsafe {
            std::slice::from_raw_parts(chunk_indices_ptr, num_chunk_indices).to_vec()
        };

        unsafe {
            free_chunk_indices(chunk_indices_ptr);
        }

        Ok(chunk_indices)
    }

    /// Get a single chunk by index
    pub fn get_chunk(&self, chunk_index: i64) -> Result<*mut ArrowArray> {
        let mut array = ArrowArray::empty();
        let result = unsafe { get_chunk(self.handle, chunk_index, &mut array) };

        check_ffi_result(result)?;

        Ok(&mut array as *mut ArrowArray)
    }

    /// Get multiple chunks by indices
    pub fn get_chunks(&self, chunk_indices: &[i64], parallelism: i64) -> Result<Vec<*mut ArrowArray>> {
        let mut arrays_ptr: *mut ArrowArray = ptr::null_mut();
        let mut num_arrays: usize = 0;

        let result = unsafe {
            get_chunks(
                self.handle,
                chunk_indices.as_ptr(),
                chunk_indices.len(),
                parallelism,
                &mut arrays_ptr,
                &mut num_arrays,
            )
        };

        check_ffi_result(result)?;

        if arrays_ptr.is_null() {
            return Ok(Vec::new());
        }

        let arrays: Vec<*mut ArrowArray> = unsafe {
            std::slice::from_raw_parts_mut(arrays_ptr, num_arrays)
                .iter_mut()
                .map(|arr| arr as *mut ArrowArray)
                .collect()
        };

        // Note: The caller is responsible for freeing the individual arrays
        // We only free the container array here
        unsafe {
            free_chunk_arrays(arrays_ptr, num_arrays);
        }

        Ok(arrays)
    }
}

impl Drop for ChunkReader {
    fn drop(&mut self) {
        unsafe {
            chunk_reader_destroy(self.handle);
        }
    }
}

// Thread safety markers
unsafe impl Send for ChunkReader {}
unsafe impl Sync for ChunkReader {}

/// Builder for Properties
pub struct PropertiesBuilder {
    keys: Vec<CString>,
    values: Vec<CString>,
}

impl PropertiesBuilder {
    pub fn new() -> Self {
        PropertiesBuilder {
            keys: Vec::new(),
            values: Vec::new(),
        }
    }

    pub fn add_property(mut self, key: &str, value: &str) -> Result<Self> {
        let key_cstr = CString::new(key)
            .map_err(|e| MilvusError::Ffi(format!("Invalid property key: {}", e)))?;
        let value_cstr = CString::new(value)
            .map_err(|e| MilvusError::Ffi(format!("Invalid property value: {}", e)))?;
        
        self.keys.push(key_cstr);
        self.values.push(value_cstr);
        Ok(self)
    }

    pub fn build(self) -> Result<Properties> {
        if self.keys.is_empty() {
            return Ok(Properties {
                properties: ptr::null_mut(),
                count: 0,
            });
        }

        let key_ptrs: Vec<*const c_char> = self.keys.iter().map(|s| s.as_ptr()).collect();
        let value_ptrs: Vec<*const c_char> = self.values.iter().map(|s| s.as_ptr()).collect();

        let mut properties = Properties {
            properties: ptr::null_mut(),
            count: 0,
        };

        let result = unsafe {
            properties_create(
                key_ptrs.as_ptr(),
                value_ptrs.as_ptr(),
                self.keys.len(),
                &mut properties,
            )
        };

        check_ffi_result(result)?;

        Ok(properties)
    }
}

impl Properties {
    /// Get a property value by key
    pub fn get(&self, key: &str) -> Option<String> {
        let key_cstr = CString::new(key).ok()?;
        let value_ptr = unsafe { properties_get(self, key_cstr.as_ptr()) };
        
        if value_ptr.is_null() {
            None
        } else {
            unsafe {
                Some(CStr::from_ptr(value_ptr).to_string_lossy().to_string())
            }
        }
    }
}

impl Drop for Properties {
    fn drop(&mut self) {
        if !self.properties.is_null() {
            unsafe {
                properties_free(self);
            }
        }
    }
}

// Thread safety markers
unsafe impl Send for Properties {}
unsafe impl Sync for Properties {}