mod ffi;
mod table_provider;
mod execution_plan;
mod record_batch_stream;
mod error;

pub use table_provider::MilvusTableProvider;
pub use error::{MilvusError, Result};

// Re-export FFI types for testing
pub use ffi::{PropertiesBuilder, Properties};

// Re-export commonly used types
pub use datafusion::prelude::{DataFrame, SessionContext, col, lit};
pub use arrow::array::{Array, RecordBatch, StringArray, Int32Array, Float64Array};
pub use arrow::datatypes::*;
