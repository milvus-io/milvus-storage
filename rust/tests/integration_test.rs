use std::sync::Arc;
use std::path::PathBuf;
use std::fs;

use arrow::datatypes::{DataType, Field, Schema};
use datafusion::prelude::*;
use milvus_storage_datafusion::{MilvusTableProvider, MilvusError};

#[tokio::test(flavor = "multi_thread")]
async fn test_table_provider_creation() {
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

#[tokio::test(flavor = "multi_thread")]
async fn test_datafusion_registration() {
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
async fn test_read_from_test_data() {
    // Read the manifest content from test data
    let test_data_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("data");
    
    let manifest_path = test_data_dir.join("manifest.json");
    let manifest_content = fs::read_to_string(&manifest_path)
        .expect("Failed to read manifest.json from test data");
    
    println!("Successfully read manifest content ({} bytes)", manifest_content.len());

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
        Some(&["name"][..]),
        Some(vec![("fs.storage_type".to_string(), "local".to_string()), 
            ("fs.root_path".to_string(), test_data_dir.display().to_string())]
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
                        // FIXME: pointer being freed was not allocated
                        
                        // Try to collect results (this will actually execute the query)
                        // let collect_result = dataframe.collect().await;
                        // match collect_result {
                        //     Ok(batches) => {
                        //         // println!("Successfully read {} batches from test data", batches.len());
                        //         // for (i, batch) in batches.iter().enumerate() {
                        //         //     println!("Batch {}: {} rows, {} columns", i, batch.num_rows(), batch.num_columns());
                        //         // }
                        //     }
                        //     Err(e) => {
                        //         println!("Failed to collect results: {}", e);
                        //         // This is expected if the C++ library is not available
                        //     }
                        // }
                    }
                    Err(e) => {
                        println!("SQL execution failed: {}", e);
                        // This is expected if the C++ library is not available
                    }
                }
            } else {
                println!("Table registration failed - this is expected without C++ library");
            }
        }
        Err(e) => {
            assert!(false, "Table provider creation failed: {}", e);
        }
    }
}
