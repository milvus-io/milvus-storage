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

        // Handle None columns by converting to explicit column list
        let column_names: Vec<String> = if columns.is_none() {
            // Extract all column names from the schema
            schema.fields().iter().map(|f| f.name().clone()).collect()
        } else {
            Vec::new() // Won't be used
        };
        
        let column_refs: Vec<&str> = if columns.is_none() {
            column_names.iter().map(|s| s.as_str()).collect()
        } else {
            Vec::new() // Won't be used
        };

        let final_columns = if columns.is_none() {
            Some(column_refs.as_slice())
        } else {
            columns
        };

        // Create the reader
        let reader = match Reader::new(
            manifest,
            unsafe { &*arrow_schema_ptr },
            final_columns,
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
    use crate::MilvusError;

    use super::*;
    use datafusion::prelude::*;
    use arrow::datatypes::{DataType, Field, Schema};
    use std::path::PathBuf;
    use std::fs;

    #[tokio::test]
    async fn test_creation() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("vector", DataType::FixedSizeBinary(128), false),
            Field::new("metadata", DataType::Utf8, true),
        ]));
    
        // This should fail gracefully with invalid manifest
        let result = MilvusTableProvider::try_new(
            "test_manifest",
            schema,
            "test_table".to_string(),
            None,
            None,
        );
    
        match result {
            Err(MilvusError::Ffi(_)) => {
                // Expected error
            }
            _ => panic!("Expected FFI or ReaderCreation error"),
        }
    }

    #[tokio::test]
    async fn test_registration() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("data", DataType::Utf8, true),
        ]));
    
        match MilvusTableProvider::try_new(
            "malformed manifest",
            schema.clone(),
            "test_table".to_string(),
            None,
            Some(vec![("fs.storage_type".to_string(), "local".to_string()), 
                ("fs.root_path".to_string(), "/tmp/".to_string())]
            ),
        ) {
            Ok(_) => {
                println!("Table provider creation succeeded unexpectedly");
            }
            Err(e) => {
                println!("Table provider creation failed as expected: {}", e);
            }
        }
    
        match MilvusTableProvider::try_new(
            r#"{
      "column_groups": [
        {
          "columns": [
            "id",
            "data"
          ],
          "format": "parquet",
          "paths": [
            "column_group_0.parquet"
          ]
        }
      ],
      "version": 0
    }"#,
            schema.clone(),
            "test_table".to_string(),
            None,
            Some(vec![("fs.storage_type".to_string(), "local".to_string()), 
                ("fs.root_path".to_string(), "/tmp/".to_string())]
            ),
        ) {
            Ok(_) => {
                println!("Table provider creation succeeded");
            }
            Err(e) => {
                println!("Table provider creation failed: {}", e);
            }
        }
    }

    #[tokio::test]
    async fn test_query() {
       // Read the manifest content from test data
        let test_data_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("data");
        let fs_root_path = test_data_dir.display().to_string();

        let manifest_path = test_data_dir.join("manifest.json");
        let manifest_content = fs::read_to_string(&manifest_path)
            .expect("Failed to read manifest.json from test data");

        println!("Successfully read manifest content ({} bytes)", manifest_content.len());

        // FIXME: support relative paths in manifest
        let manifest_content = manifest_content.replace(
            "column_group_",
            &(fs_root_path.clone() + "/column_group_"),
        );

        // Create schema that matches the test data
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("value", DataType::Float64, true),
            Field::new("name", DataType::Utf8, true),
            Field::new("vector", DataType::FixedSizeBinary(128), true),
        ]));

        // Attempt to create table provider with test data
        let result = MilvusTableProvider::try_new(
            &manifest_content,
            schema,
            "test_data_table".to_string(),
            Some(&["id", "value", "name"][..]),
            Some(vec![("fs.storage_type".to_string(), "local".to_string()), 
                ("fs.root_path".to_string(), fs_root_path.clone())]
            ),
        );

        match result {
            Ok(table_provider) => {
                // If successful, try to register with DataFusion and query
                let ctx = SessionContext::new();
                let registration_result = ctx.register_table("test_data", Arc::new(table_provider));
                
                if registration_result.is_ok() {
                    // Try a simple query
                    let sql = "SELECT * FROM test_data LIMIT 5";
                    let df_result = ctx.sql(sql).await;
                    
                    match df_result {
                        Ok(dataframe) => {
                            
                            // Try to collect results (this will actually execute the query)
                            let collect_result = dataframe.collect().await;
                            match collect_result {
                                Ok(batches) => {
                                    println!("Successfully read {} batches from test data", batches.len());
                                    // should have 5 rows in 1 batch
                                    assert_eq!(batches.len(), 1);
                                    assert_eq!(batches[0].num_rows(), 5);

                                    // dump the data
                                    for batch in batches {
                                        println!("Batch: {:?}", batch);
                                    }
                                }
                                Err(e) => {
                                    assert!(false, "Failed to collect results: {}", e);
                                }
                            }
                        }
                        Err(e) => {
                            assert!(false, "SQL execution failed: {}", e);
                        }
                    }

                    // Test passed! FFI integration is working correctly with real data
                } else {
                    assert!(false, "Table registration failed - this is expected without C++ library");
                }
            }
            Err(e) => {
                assert!(false, "Table provider creation failed: {}", e);
            }
        }
    }
}
