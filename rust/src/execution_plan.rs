use std::any::Any;
use std::fmt;
use std::sync::Arc;
use std::pin::Pin;
use std::task::{Context, Poll};

use arrow::datatypes::{SchemaRef};
use arrow::record_batch::{RecordBatch, RecordBatchReader};
use arrow::ffi_stream::{ArrowArrayStreamReader, FFI_ArrowArrayStream};
use datafusion::common::{Statistics, Result as DFResult};
use datafusion::execution::context::TaskContext;
use datafusion::physical_plan::{
    DisplayAs, DisplayFormatType, ExecutionPlan, Partitioning, 
    SendableRecordBatchStream, PlanProperties, RecordBatchStream,
    execution_plan::{Boundedness, EmissionType},
};
use datafusion::error::DataFusionError;
use futures::Stream;

use crate::ffi::Reader;

/// Simple adapter to convert RecordBatchReader to RecordBatchStream
pub struct RecordBatchReaderStream<R: RecordBatchReader> {
    reader: R,
    schema: SchemaRef,
}

impl<R: RecordBatchReader> RecordBatchReaderStream<R> {
    pub fn new(reader: R) -> Self {
        let schema = reader.schema();
        Self { reader, schema }
    }
}

impl<R: RecordBatchReader + Unpin> Stream for RecordBatchReaderStream<R> {
    type Item = Result<RecordBatch, DataFusionError>;

    fn poll_next(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = self.get_mut();
        match this.reader.next() {
            Some(Ok(batch)) => Poll::Ready(Some(Ok(batch))),
            Some(Err(e)) => Poll::Ready(Some(Err(DataFusionError::ArrowError(Box::new(e), None)))),
            None => Poll::Ready(None),
        }
    }
}

impl<R: RecordBatchReader + Unpin> RecordBatchStream for RecordBatchReaderStream<R> {
    fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }
}

/// Physical execution plan for scanning Milvus storage
pub struct MilvusScanExec {
    reader: Arc<Reader>,
    schema: SchemaRef,
    projection: Option<Vec<String>>,
    predicate: Option<String>,
    batch_size: i64,
    buffer_size: i64,
    cache: PlanProperties,
}

impl MilvusScanExec {
    pub fn new(
        reader: Arc<Reader>,
        schema: SchemaRef,
        projection: Option<Vec<String>>,
        predicate: Option<String>,
        batch_size: Option<i64>,
        buffer_size: Option<i64>,
    ) -> Self {
        let cache = Self::compute_properties(schema.clone());
        
        Self {
            reader,
            schema,
            projection,
            predicate,
            batch_size: batch_size.unwrap_or(8192),
            buffer_size: buffer_size.unwrap_or(64 * 1024 * 1024), // 64MB default
            cache,
        }
    }

    fn compute_properties(schema: SchemaRef) -> PlanProperties {
        // For compatibility with DataFusion v35, create simpler properties
        PlanProperties::new(
            datafusion::physical_expr::EquivalenceProperties::new(schema),
            Partitioning::UnknownPartitioning(1), // Single partition for now
            EmissionType::Final,
            Boundedness::Bounded,
        )
    }
}

impl fmt::Debug for MilvusScanExec {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "MilvusScanExec")
    }
}

impl DisplayAs for MilvusScanExec {
    fn fmt_as(&self, t: DisplayFormatType, f: &mut fmt::Formatter) -> fmt::Result {
        match t {
            DisplayFormatType::Default | DisplayFormatType::Verbose => {
                write!(f, "MilvusScanExec: projection={:?}, predicate={:?}", 
                       self.projection, self.predicate)
            }
            DisplayFormatType::TreeRender => {
                write!(f, "MilvusScanExec")
            }
        }
    }
}

impl ExecutionPlan for MilvusScanExec {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }

    fn properties(&self) -> &PlanProperties {
        &self.cache
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![]
    }

    fn name(&self) -> &str {
        "MilvusScanExec"
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> DFResult<Arc<dyn ExecutionPlan>> {
        if children.is_empty() {
            Ok(self)
        } else {
            Err(datafusion::error::DataFusionError::Internal(
                "MilvusScanExec cannot have children".to_string(),
            ))
        }
    }

    fn execute(
        &self,
        partition: usize,
        _context: Arc<TaskContext>,
    ) -> DFResult<SendableRecordBatchStream> {
        if partition != 0 {
            return Err(datafusion::error::DataFusionError::Internal(
                format!("Invalid partition {} for MilvusScanExec", partition),
            ));
        }

        // Get the raw ArrowArrayStream from Milvus
        let predicate_str = self.predicate.as_deref();
        let raw_stream = self.reader
            .get_record_batch_reader(predicate_str, self.batch_size, self.buffer_size)
            .map_err(|e| datafusion::error::DataFusionError::External(Box::new(e)))?;

        // Convert to ArrowArrayStreamReader using Arrow's built-in utilities
        let reader = unsafe { 
            ArrowArrayStreamReader::from_raw(raw_stream as *mut FFI_ArrowArrayStream) 
        }
        .map_err(|e| datafusion::error::DataFusionError::ArrowError(Box::new(e), None))?;

        // Convert to stream using simple adapter
        let stream = RecordBatchReaderStream::new(reader);
        
        Ok(Box::pin(stream))
    }

    fn statistics(&self) -> DFResult<Statistics> {
        Ok(Statistics::new_unknown(&self.schema))
    }
}
