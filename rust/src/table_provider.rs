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
    manifest: String,
    schema: SchemaRef,
    table_name: String,
    properties: Option<Vec<(String, String)>>,
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
        // Note: columns parameter is ignored since we derive columns from scan projection
        let _ = columns; // Acknowledge the parameter but don't use it

        Ok(Self {
            manifest: manifest.to_string(),
            schema,
            table_name,
            properties,
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

    /// Create a Reader with the stored arguments and projection
    fn create_reader(&self, projection: Option<&Vec<usize>>) -> Result<Reader> {
        // Convert the Arrow Rust schema to C ABI format
        let ffi_schema = FFI_ArrowSchema::try_from(self.schema.as_ref())
            .map_err(|e| crate::error::MilvusError::Ffi(format!("Failed to convert schema: {}", e)))?;
 
        // Convert FFI_ArrowSchema to our ArrowSchema type for the C API
        // We need to pass the FFI_ArrowSchema as a pointer to the C API
        let arrow_schema_ptr = &ffi_schema as *const FFI_ArrowSchema as *const ArrowSchema;
        
        // Create read properties - always create properties, even if empty
        let mut builder = PropertiesBuilder::new();
        if let Some(ref props) = self.properties {
            for (key, value) in props {
                builder = builder.add_property(key, value)?;
            }
        }
        let read_properties = builder.build()?;

        // Derive columns from projection - handle lifetime properly
        // Only include columns that exist in the base schema (filter out computed columns)
        let projected_columns: Vec<String>;
        let column_refs: Vec<&str>;
        let final_columns = if let Some(proj) = projection {
            // Use projection to determine which columns to read, but only base table columns
            projected_columns = proj.iter()
                .filter_map(|&i| {
                    if i < self.schema.fields().len() {
                        Some(self.schema.field(i).name().clone())
                    } else {
                        // Skip computed columns that don't exist in base schema
                        None
                    }
                })
                .collect();
            column_refs = projected_columns.iter().map(|s| s.as_str()).collect();
            Some(column_refs.as_slice())
        } else {
            // No projection means read all columns - let Reader handle this with None
            None
        };

        // Create the reader
        Reader::new(
            &self.manifest,
            unsafe { &*arrow_schema_ptr },
            final_columns,
            Some(&read_properties),
        )
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
        // Create the Reader here with stored arguments and projection
        let reader = self.create_reader(projection)
            .map_err(|e| datafusion::error::DataFusionError::External(Box::new(e)))?;

        // Convert projection to column names
        let columns = self.projection_to_columns(projection);

        // Create projected schema that matches what the Reader will produce
        let projected_schema = if let Some(proj) = projection {
            let projected_fields: Vec<_> = proj.iter()
                .filter_map(|&i| {
                    if i < self.schema.fields().len() {
                        Some(self.schema.field(i).clone())
                    } else {
                        None
                    }
                })
                .collect();
            Arc::new(arrow::datatypes::Schema::new(projected_fields))
        } else {
            // No projection means all columns
            self.schema()
        };

        // Convert filters to predicate string
        let predicate = self.filters_to_predicate(filters);

        // Determine batch size based on limit
        let batch_size = limit.map(|l| l as i64);

        let exec = MilvusScanExec::new(
            Arc::new(reader),
            projected_schema, // Use projected schema instead of full table schema
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
    
        // Table provider creation should now succeed since Reader creation is delayed
        let result = MilvusTableProvider::try_new(
            "test_manifest",
            schema,
            "test_table".to_string(),
            None,
            None,
        );
    
        match result {
            Ok(table_provider) => {
                // Creation should succeed, but scanning with invalid manifest should fail
                let ctx = SessionContext::new();
                if let Ok(_) = ctx.register_table("test", Arc::new(table_provider)) {
                    // Try to scan which should fail due to invalid manifest
                    let scan_result = ctx.sql("SELECT * FROM test LIMIT 1").await;
                    match scan_result {
                        Ok(dataframe) => {
                            // Try to collect which should trigger the error
                            let collect_result = dataframe.collect().await;
                            match collect_result {
                                Err(_) => {
                                    // Expected error during scanning/collection
                                }
                                Ok(_) => panic!("Expected error during scanning with invalid manifest"),
                            }
                        }
                        Err(_) => {
                            // Also acceptable - error during SQL parsing/planning
                        }
                    }
                }
            }
            Err(_) => panic!("Table provider creation should succeed with delayed Reader creation"),
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
            Some(&["id", "value", "name", "vector"][..]),
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
                    let sql = r#"SELECT id + 1000, value, digest(name, 'md5') FROM test_data WHERE id > 10 LIMIT 5"#;
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
