use thiserror::Error;

pub type Result<T> = std::result::Result<T, MilvusError>;

#[derive(Error, Debug)]
pub enum MilvusError {
    #[error("FFI error: {0}")]
    FFI(String),
    
    #[error("Invalid manifest: {0}")]
    InvalidManifest(String),
}

impl From<MilvusError> for datafusion::error::DataFusionError {
    fn from(err: MilvusError) -> Self {
        datafusion::error::DataFusionError::External(Box::new(err))
    }
}
