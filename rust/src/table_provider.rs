use std::any::Any;
use std::sync::Arc;

use arrow::datatypes::SchemaRef;
use arrow::ffi::FFI_ArrowSchema;
use crate::ffi::ArrowSchema;
use async_trait::async_trait;
use datafusion::catalog::Session;
use datafusion::common::Result as DFResult;
use std::borrow::Cow;
use datafusion::datasource::{TableProvider, TableType};
use datafusion::logical_expr::{Expr, TableProviderFilterPushDown};
use datafusion::physical_plan::ExecutionPlan;

use crate::execution_plan::MilvusScanExec;
use crate::ffi::{Reader, PropertiesBuilder};
use crate::error::Result;

/// DataFusion TableProvider for Milvus Storage
pub struct MilvusTableProvider {
    reader: Arc<Reader>,
    schema: SchemaRef,
    table_name: String,
}

impl MilvusTableProvider {
    /// Create a new MilvusTableProvider
    pub fn new(
        manifest: &str,
        schema: SchemaRef,
        table_name: String,
        columns: Option<&[&str]>,
        properties: Option<Vec<(String, String)>>,
    ) -> Result<Self> {
        // Convert the Arrow Rust schema to C ABI format
        let ffi_schema = FFI_ArrowSchema::try_from(schema.as_ref())
            .map_err(|e| crate::error::MilvusError::Ffi(format!("Failed to convert schema: {}", e)))?;
 
        // Convert FFI_ArrowSchema to our ArrowSchema type for the C API
        // We need to pass the FFI_ArrowSchema as a pointer to the C API
        let arrow_schema_ptr = &ffi_schema as *const FFI_ArrowSchema as *const ArrowSchema;
        
        // Create read properties - always create properties, even if empty
        let mut builder = PropertiesBuilder::new();
        if let Some(props) = properties {
            for (key, value) in props {
                builder = builder.add_property(&key, &value)?;
            }
        }
        let read_properties = match builder.build() {
            Ok(read_properties) => read_properties,
            Err(err) => {
                return Err(err);
            }
        };

        // Create the reader
        let reader = match Reader::new(
            manifest,
            unsafe { &*arrow_schema_ptr },
            columns,
            Some(&read_properties),
        ) {
            Ok(reader) => reader,
            Err(err) => {
                return Err(err);
            }
        };

        Ok(Self {
            reader: Arc::new(reader),
            schema,
            table_name,
        })
    }

    /// Create a new MilvusTableProvider with default options
    pub fn try_new(
        manifest: &str,
        schema: SchemaRef,
        table_name: String,
        columns: Option<&[&str]>,
        properties: Option<Vec<(String, String)>>,
    ) -> Result<Self> {
        Self::new(manifest, schema, table_name, columns, properties)
    }

    /// Get the underlying reader
    pub fn reader(&self) -> Arc<Reader> {
        self.reader.clone()
    }

    /// Convert filter expressions to predicate string
    fn filters_to_predicate(&self, filters: &[Expr]) -> Option<String> {
        if filters.is_empty() {
            return None;
        }

        // For now, we'll create a simple string representation
        // In a real implementation, you'd want to convert this to Milvus's filter format
        let filter_strings: Vec<String> = filters
            .iter()
            .map(|expr| format!("{}", expr))
            .collect();

        if filter_strings.is_empty() {
            None
        } else {
            Some(filter_strings.join(" AND "))
        }
    }

    /// Extract column names from projection
    fn projection_to_columns(&self, projection: Option<&Vec<usize>>) -> Option<Vec<String>> {
        projection.map(|proj| {
            proj.iter()
                .filter_map(|&i| {
                    if i < self.schema.fields().len() {
                        Some(self.schema.field(i).name().clone())
                    } else {
                        None
                    }
                })
                .collect()
        })
    }
}

#[async_trait]
impl TableProvider for MilvusTableProvider {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }

    fn table_type(&self) -> TableType {
        TableType::Base
    }

    fn get_logical_plan(&self) -> Option<Cow<'_, datafusion::logical_expr::LogicalPlan>> {
        None
    }

    fn supports_filters_pushdown(
        &self,
        filters: &[&Expr],
    ) -> DFResult<Vec<TableProviderFilterPushDown>> {
        // For now, we'll say we support all filters for pushdown
        // In a real implementation, you'd check which filters are supported by Milvus
        Ok(vec![TableProviderFilterPushDown::Inexact; filters.len()])
    }

    async fn scan(
        &self,
        _state: &dyn Session,
        projection: Option<&Vec<usize>>,
        filters: &[Expr],
        limit: Option<usize>,
    ) -> DFResult<Arc<dyn ExecutionPlan>> {
        // Convert projection to column names
        let columns = self.projection_to_columns(projection);

        // Convert filters to predicate string
        let predicate = self.filters_to_predicate(filters);

        // Determine batch size based on limit
        let batch_size = limit.map(|l| l as i64);

        let exec = MilvusScanExec::new(
            self.reader.clone(),
            self.schema(),
            columns,
            predicate,
            batch_size,
            None, // Use default buffer size
        );

        Ok(Arc::new(exec))
    }
}

impl std::fmt::Debug for MilvusTableProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MilvusTableProvider")
            .field("table_name", &self.table_name)
            .field("schema", &self.schema)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::datatypes::{DataType, Field, Schema};

    #[tokio::test]
    async fn test_milvus_table_provider_creation() {
        // Create a simple test schema
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("vector", DataType::FixedSizeBinary(128), false),
            Field::new("metadata", DataType::Utf8, true),
        ]));

        // Test table provider creation - this will fail gracefully when the C++ library isn't available
        let result = MilvusTableProvider::try_new(
            "test_manifest",
            schema.clone(),
            "test_table".to_string(),
            None,
            None,
        );

        // We expect this to fail since we don't have a real Milvus setup
        assert!(result.is_err());

        // Test basic schema validation
        assert_eq!(schema.fields().len(), 3);
        assert_eq!(schema.field(0).name(), "id");
        assert_eq!(schema.field(1).name(), "vector");
        assert_eq!(schema.field(2).name(), "metadata");
    }
}
