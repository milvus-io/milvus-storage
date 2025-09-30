use thiserror::Error;

pub type Result<T> = std::result::Result<T, MilvusError>;

#[derive(Error, Debug)]
pub enum MilvusError {
    #[error("FFI error: {0}")]
    Ffi(String),
    
    #[error("Arrow error: {0}")]
    Arrow(#[from] arrow::error::ArrowError),
    
    #[error("DataFusion error: {0}")]
    DataFusion(#[from] datafusion::error::DataFusionError),
    
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    
    #[error("Invalid schema: {0}")]
    InvalidSchema(String),
    
    #[error("Reader creation failed: {0}")]
    ReaderCreation(String),
    
    #[error("Invalid manifest: {0}")]
    InvalidManifest(String),
    
    #[error("Null pointer error: {0}")]
    NullPointer(String),
}

impl From<MilvusError> for datafusion::error::DataFusionError {
    fn from(err: MilvusError) -> Self {
        datafusion::error::DataFusionError::External(Box::new(err))
    }
}
