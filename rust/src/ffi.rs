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
            Err(MilvusError::FFI(error_str))
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
            .map_err(|e| MilvusError::InvalidManifest(e.to_string()))?;

        // Convert column names to C strings if provided
        let (column_ptrs, _column_cstrs): (Vec<*const std::os::raw::c_char>, Vec<CString>) = if let Some(cols) = columns {
            let cstrs: Result<Vec<CString>> = cols
                .iter()
                .map(|s| CString::new(*s).map_err(|e| MilvusError::FFI(format!("Invalid column name: {}", e))))
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
                .map_err(|e| MilvusError::FFI(format!("Invalid predicate string: {}", e)))?;
            predicate_cstr.as_ptr()
        } else {
            ptr::null()
        };

        let mut raw_stream = ArrowArrayStream::empty();
        let result = unsafe {
            get_record_batch_reader(
                self.handle, 
                predicate_ptr,
                &mut raw_stream as *mut ArrowArrayStream
            )
        };

        check_ffi_result(result)?;

        let boxed_stream = Box::new(raw_stream);
        Ok(Box::into_raw(boxed_stream) as *mut ArrowArrayStream)
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
            .map_err(|e| MilvusError::FFI(format!("Invalid property key: {}", e)))?;
        let value_cstr = CString::new(value)
            .map_err(|e| MilvusError::FFI(format!("Invalid property value: {}", e)))?;
        
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use tokio::task;

    #[tokio::test(flavor = "multi_thread")]
    async fn test_properties_builder_empty() {
        let builder = PropertiesBuilder::new();
        let properties = builder.build().expect("Should build empty properties successfully");
        
        // Empty properties should have null pointer and zero count
        assert!(properties.properties.is_null());
        assert_eq!(properties.count, 0);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_properties_builder_single_property() {
        let builder = PropertiesBuilder::new();
        let properties = builder
            .add_property("test_key", "test_value")
            .expect("Should add property successfully")
            .build()
            .expect("Should build properties successfully");
        
        // Should have non-null pointer and count of 1
        assert!(!properties.properties.is_null());
        assert_eq!(properties.count, 1);
        
        // Test getting the property value
        let value = properties.get("test_key");
        assert_eq!(value, Some("test_value".to_string()));
        
        // Test getting non-existent property
        let missing = properties.get("missing_key");
        assert_eq!(missing, None);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_properties_builder_multiple_properties() {
        let builder = PropertiesBuilder::new();
        let properties = builder
            .add_property("fs.storage_type", "local")
            .expect("Should add storage_type property")
            .add_property("fs.root_path", "/tmp/test")
            .expect("Should add root_path property")
            .add_property("fs.use_ssl", "false")
            .expect("Should add use_ssl property")
            .build()
            .expect("Should build properties successfully");
        
        // Should have non-null pointer and count of 3
        assert!(!properties.properties.is_null());
        assert_eq!(properties.count, 3);
        
        // Test all property values
        assert_eq!(properties.get("fs.storage_type"), Some("local".to_string()));
        assert_eq!(properties.get("fs.root_path"), Some("/tmp/test".to_string()));
        assert_eq!(properties.get("fs.use_ssl"), Some("false".to_string()));
        
        // Test non-existent property
        assert_eq!(properties.get("non_existent"), None);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_properties_builder_special_characters() {
        let builder = PropertiesBuilder::new();
        let properties = builder
            .add_property("key_with_spaces", "value with spaces")
            .expect("Should handle spaces")
            .add_property("key.with.dots", "value.with.dots")
            .expect("Should handle dots")
            .add_property("key-with-dashes", "value-with-dashes")
            .expect("Should handle dashes")
            .add_property("key_with_numbers123", "value_with_numbers456")
            .expect("Should handle numbers")
            .build()
            .expect("Should build properties successfully");
        
        assert_eq!(properties.count, 4);
        assert_eq!(properties.get("key_with_spaces"), Some("value with spaces".to_string()));
        assert_eq!(properties.get("key.with.dots"), Some("value.with.dots".to_string()));
        assert_eq!(properties.get("key-with-dashes"), Some("value-with-dashes".to_string()));
        assert_eq!(properties.get("key_with_numbers123"), Some("value_with_numbers456".to_string()));
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_properties_builder_empty_values() {
        let builder = PropertiesBuilder::new();
        let properties = builder
            .add_property("empty_key", "")
            .expect("Should handle empty value")
            .add_property("normal_key", "normal_value")
            .expect("Should handle normal value")
            .build()
            .expect("Should build properties successfully");
        
        assert_eq!(properties.count, 2);
        assert_eq!(properties.get("empty_key"), Some("".to_string()));
        assert_eq!(properties.get("normal_key"), Some("normal_value".to_string()));
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_properties_builder_invalid_strings() {
        let builder = PropertiesBuilder::new();
        
        // Test key with null byte (should fail)
        let result = builder.add_property("key\0with_null", "value");
        assert!(result.is_err());
        
        let builder = PropertiesBuilder::new();
        // Test value with null byte (should fail)
        let result = builder.add_property("key", "value\0with_null");
        assert!(result.is_err());
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_properties_builder_chaining() {
        // Test that the builder pattern works correctly with method chaining
        let properties = PropertiesBuilder::new()
            .add_property("key1", "value1")
            .expect("Should add key1")
            .add_property("key2", "value2")
            .expect("Should add key2")
            .add_property("key3", "value3")
            .expect("Should add key3")
            .build()
            .expect("Should build properties");
        
        assert_eq!(properties.count, 3);
        assert_eq!(properties.get("key1"), Some("value1".to_string()));
        assert_eq!(properties.get("key2"), Some("value2".to_string()));
        assert_eq!(properties.get("key3"), Some("value3".to_string()));
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_properties_drop_cleanup() {
        for i in 0..10 {
            let properties = PropertiesBuilder::new()
                .add_property(&format!("key_{}", i), &format!("value_{}", i))
                .expect("Should add property")
                .build()
                .expect("Should build properties");
            
            assert_eq!(properties.count, 1);
            assert_eq!(properties.get(&format!("key_{}", i)), Some(format!("value_{}", i)));
        }
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_properties_filesystem_config() {
        // Test the specific properties that were causing crashes
        let properties = PropertiesBuilder::new()
            .add_property("fs.storage_type", "local")
            .expect("Should add storage_type")
            .add_property("fs.root_path", "/tmp/")
            .expect("Should add root_path")
            .build()
            .expect("Should build filesystem properties");
        
        assert_eq!(properties.count, 2);
        assert_eq!(properties.get("fs.storage_type"), Some("local".to_string()));
        assert_eq!(properties.get("fs.root_path"), Some("/tmp/".to_string()));
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_properties_concurrent_access() {
        // Test that properties can be safely accessed from multiple threads
        let properties = Arc::new(
            PropertiesBuilder::new()
                .add_property("shared_key", "shared_value")
                .expect("Should add property")
                .build()
                .expect("Should build properties")
        );
        
        let mut handles = vec![];
        
        // Spawn multiple tasks that access the properties concurrently
        for i in 0..5 {
            let props = properties.clone();
            let handle = task::spawn(async move {
                let value = props.get("shared_key");
                assert_eq!(value, Some("shared_value".to_string()));
                i // Return task id for verification
            });
            handles.push(handle);
        }
        
        // Wait for all tasks to complete
        for (i, handle) in handles.into_iter().enumerate() {
            let result = handle.await.expect("Task should complete successfully");
            assert_eq!(result, i);
        }
    }
}