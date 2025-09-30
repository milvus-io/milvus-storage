use std::any::Any;
use std::fmt;
use std::sync::Arc;

use arrow::datatypes::{SchemaRef};
use datafusion::common::{Statistics, Result as DFResult};
use datafusion::execution::context::TaskContext;
use datafusion::physical_plan::{
    DisplayAs, DisplayFormatType, ExecutionPlan, Partitioning, 
    SendableRecordBatchStream, PlanProperties,
    execution_plan::{Boundedness, EmissionType},
};

use crate::ffi::Reader;
use crate::record_batch_stream::MilvusRecordBatchStream;

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

        // Get the record batch reader from Milvus
        let predicate_str = self.predicate.as_deref();
        let stream = self.reader
            .get_record_batch_reader(predicate_str, self.batch_size, self.buffer_size)
            .map_err(|e| datafusion::error::DataFusionError::External(Box::new(e)))?;

        let milvus_stream = MilvusRecordBatchStream::new(self.schema.clone(), stream);
        
        Ok(Box::pin(milvus_stream))
    }

    fn statistics(&self) -> DFResult<Statistics> {
        Ok(Statistics::new_unknown(&self.schema))
    }
}
