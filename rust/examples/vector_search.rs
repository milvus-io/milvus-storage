use std::sync::Arc;

use arrow::array::{Float32Array, Int64Array, StringArray, FixedSizeBinaryArray};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use datafusion::prelude::*;
use datafusion::logical_expr::{create_udf, Volatility, ColumnarValue};
use datafusion::arrow::array::Array;
use datafusion::error::DataFusionError;
use milvus_storage_datafusion::Result;

fn base64_encode(bytes: &[u8]) -> String {
    // Simple base64-like encoding for demo purposes
    // In practice, you'd use a proper base64 library
    format!("vector_{}", bytes.len())
}

/// Example demonstrating vector similarity search with DataFusion and Milvus Storage
#[tokio::main]
async fn main() -> Result<()> {
    env_logger::init();
    
    let ctx = SessionContext::new();

    // Register custom vector similarity functions
    register_vector_functions(&ctx)?;

    // Define schema for vector data
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("embedding", DataType::FixedSizeBinary(512), false), // 128 * 4 bytes (float32)
        Field::new("category", DataType::Utf8, true),
        Field::new("score", DataType::Float32, true),
        Field::new("metadata", DataType::Utf8, true),
    ]));

    // Create table provider (in real usage, you'd have proper filesystem setup)
    let _manifest = create_sample_manifest();

    // For demonstration, we'll create a mock table
    // In practice, you'd use:
    // let table_provider = MilvusTableProvider::try_new(&manifest, schema, "vectors".to_string(), None, None)?;
    
    println!("Creating sample vector table...");
    let table_provider = create_sample_vector_table(schema.clone());
    ctx.register_table("vectors", Arc::new(table_provider))?;

    // Example 1: Basic vector similarity search
    println!("\n=== Example 1: Vector Similarity Search ===");
    let query_vector = vec![0.1f32; 128]; // 128-dimensional query vector
    let query_bytes = vector_to_bytes(&query_vector);
    
    let df = ctx.sql(&format!(r#"
        SELECT id, category, score,
               cosine_similarity(embedding, CAST('{}' AS BINARY)) as similarity
        FROM vectors 
        WHERE cosine_similarity(embedding, CAST('{}' AS BINARY)) > 0.8
        ORDER BY similarity DESC
        LIMIT 10
    "#, base64_encode(&query_bytes), base64_encode(&query_bytes))).await?;
    
    df.show().await?;

    // Example 2: Vector search with category filtering
    println!("\n=== Example 2: Filtered Vector Search ===");
    let df = ctx.sql(&format!(r#"
        SELECT id, category, metadata,
               euclidean_distance(embedding, CAST('{}' AS BINARY)) as distance
        FROM vectors 
        WHERE category = 'image' 
          AND euclidean_distance(embedding, CAST('{}' AS BINARY)) < 10.0
        ORDER BY distance ASC
        LIMIT 5
    "#, base64_encode(&query_bytes), base64_encode(&query_bytes))).await?;
    
    df.show().await?;

    // Example 3: Aggregate vector analysis
    println!("\n=== Example 3: Vector Analytics ===");
    let df = ctx.sql(&format!(r#"
        SELECT category,
               COUNT(*) as count,
               AVG(cosine_similarity(embedding, CAST('{}' AS BINARY))) as avg_similarity,
               MIN(euclidean_distance(embedding, CAST('{}' AS BINARY))) as min_distance,
               MAX(euclidean_distance(embedding, CAST('{}' AS BINARY))) as max_distance
        FROM vectors 
        GROUP BY category
        ORDER BY avg_similarity DESC
    "#, base64_encode(&query_bytes), base64_encode(&query_bytes), base64_encode(&query_bytes))).await?;
    
    df.show().await?;

    // Example 4: Vector clustering analysis
    println!("\n=== Example 4: Vector Clustering ===");
    let df = ctx.sql(r#"
        WITH similarities AS (
            SELECT id, category,
                   cosine_similarity(embedding, (SELECT embedding FROM vectors WHERE id = 1)) as sim_to_1,
                   cosine_similarity(embedding, (SELECT embedding FROM vectors WHERE id = 2)) as sim_to_2
            FROM vectors
        )
        SELECT category,
               CASE 
                   WHEN sim_to_1 > sim_to_2 THEN 'cluster_1'
                   ELSE 'cluster_2'
               END as cluster,
               COUNT(*) as count,
               AVG(sim_to_1) as avg_sim_1,
               AVG(sim_to_2) as avg_sim_2
        FROM similarities
        GROUP BY category, cluster
        ORDER BY category, cluster
    "#).await?;
    
    df.show().await?;

    Ok(())
}

/// Register custom vector similarity functions with DataFusion
fn register_vector_functions(ctx: &SessionContext) -> Result<()> {
    // Cosine similarity function
    let cosine_similarity = create_udf(
        "cosine_similarity",
        vec![DataType::Binary, DataType::Binary],
        DataType::Float32,
        Volatility::Immutable,
        Arc::new(|args| {
            // Extract arrays from ColumnarValue
            let array1 = match &args[0] {
                ColumnarValue::Array(arr) => arr.clone(),
                ColumnarValue::Scalar(_) => return Err(DataFusionError::Execution("Expected array argument".to_string())),
            };
            let array2 = match &args[1] {
                ColumnarValue::Array(arr) => arr.clone(),
                ColumnarValue::Scalar(_) => return Err(DataFusionError::Execution("Expected array argument".to_string())),
            };

            let vec1 = array1.as_any().downcast_ref::<FixedSizeBinaryArray>()
                .or_else(|| array1.as_any().downcast_ref::<arrow::array::BinaryArray>().map(|_| panic!("Need fixed size binary")))
                .unwrap();
            let vec2 = array2.as_any().downcast_ref::<FixedSizeBinaryArray>()
                .or_else(|| array2.as_any().downcast_ref::<arrow::array::BinaryArray>().map(|_| panic!("Need fixed size binary")))
                .unwrap();

            let mut results = Vec::with_capacity(vec1.len());
            
            for i in 0..vec1.len() {
                if vec1.is_null(i) || vec2.is_null(i) {
                    results.push(None);
                } else {
                    let bytes1 = vec1.value(i);
                    let bytes2 = vec2.value(i);
                    
                    let floats1 = bytes_to_f32_vec(bytes1);
                    let floats2 = bytes_to_f32_vec(bytes2);
                    
                    let similarity = cosine_similarity_calc(&floats1, &floats2);
                    results.push(Some(similarity));
                }
            }

            Ok(ColumnarValue::Array(Arc::new(Float32Array::from(results))))
        }),
    );

    // Euclidean distance function
    let euclidean_distance = create_udf(
        "euclidean_distance",
        vec![DataType::Binary, DataType::Binary],
        DataType::Float32,
        Volatility::Immutable,
        Arc::new(|args| {
            // Extract arrays from ColumnarValue
            let array1 = match &args[0] {
                ColumnarValue::Array(arr) => arr.clone(),
                ColumnarValue::Scalar(_) => return Err(DataFusionError::Execution("Expected array argument".to_string())),
            };
            let array2 = match &args[1] {
                ColumnarValue::Array(arr) => arr.clone(),
                ColumnarValue::Scalar(_) => return Err(DataFusionError::Execution("Expected array argument".to_string())),
            };

            let vec1 = array1.as_any().downcast_ref::<FixedSizeBinaryArray>()
                .or_else(|| array1.as_any().downcast_ref::<arrow::array::BinaryArray>().map(|_| panic!("Need fixed size binary")))
                .unwrap();
            let vec2 = array2.as_any().downcast_ref::<FixedSizeBinaryArray>()
                .or_else(|| array2.as_any().downcast_ref::<arrow::array::BinaryArray>().map(|_| panic!("Need fixed size binary")))
                .unwrap();

            let mut results = Vec::with_capacity(vec1.len());
            
            for i in 0..vec1.len() {
                if vec1.is_null(i) || vec2.is_null(i) {
                    results.push(None);
                } else {
                    let bytes1 = vec1.value(i);
                    let bytes2 = vec2.value(i);
                    
                    let floats1 = bytes_to_f32_vec(bytes1);
                    let floats2 = bytes_to_f32_vec(bytes2);
                    
                    let distance = euclidean_distance_calc(&floats1, &floats2);
                    results.push(Some(distance));
                }
            }

            Ok(ColumnarValue::Array(Arc::new(Float32Array::from(results))))
        }),
    );

    ctx.register_udf(cosine_similarity);
    ctx.register_udf(euclidean_distance);

    Ok(())
}

/// Convert f32 vector to bytes
fn vector_to_bytes(vector: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(vector.len() * 4);
    for &f in vector {
        bytes.extend_from_slice(&f.to_le_bytes());
    }
    bytes
}

/// Convert bytes to f32 vector
fn bytes_to_f32_vec(bytes: &[u8]) -> Vec<f32> {
    bytes.chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect()
}

/// Calculate cosine similarity between two vectors
fn cosine_similarity_calc(vec1: &[f32], vec2: &[f32]) -> f32 {
    if vec1.len() != vec2.len() {
        return 0.0;
    }

    let dot_product: f32 = vec1.iter().zip(vec2.iter()).map(|(a, b)| a * b).sum();
    let norm1: f32 = vec1.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm2: f32 = vec2.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm1 == 0.0 || norm2 == 0.0 {
        0.0
    } else {
        dot_product / (norm1 * norm2)
    }
}

/// Calculate Euclidean distance between two vectors
fn euclidean_distance_calc(vec1: &[f32], vec2: &[f32]) -> f32 {
    if vec1.len() != vec2.len() {
        return f32::INFINITY;
    }

    vec1.iter()
        .zip(vec2.iter())
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f32>()
        .sqrt()
}

/// Create a sample manifest for demonstration
fn create_sample_manifest() -> String {
    serde_json::json!({
        "version": 1,
        "column_groups": [
            {
                "columns": ["id", "category", "score", "metadata"],
                "paths": ["data/scalars.parquet"],
                "format": "parquet"
            },
            {
                "columns": ["embedding"],
                "paths": ["data/vectors.bin"],
                "format": "binary"
            }
        ]
    }).to_string()
}

/// Create a sample vector table for demonstration
/// In practice, you would use MilvusTableProvider
fn create_sample_vector_table(schema: Arc<Schema>) -> MockVectorTable {
    MockVectorTable { schema }
}

// Mock table for demonstration purposes
#[derive(Debug)]
struct MockVectorTable {
    schema: Arc<Schema>,
}

#[async_trait::async_trait]
impl datafusion::datasource::TableProvider for MockVectorTable {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn schema(&self) -> arrow::datatypes::SchemaRef {
        self.schema.clone()
    }

    fn table_type(&self) -> datafusion::datasource::TableType {
        datafusion::datasource::TableType::Base
    }

    fn get_logical_plan(&self) -> Option<std::borrow::Cow<'_, datafusion::logical_expr::LogicalPlan>> {
        None
    }

    async fn scan(
        &self,
        _state: &dyn datafusion::catalog::Session,
        _projection: Option<&Vec<usize>>,
        _filters: &[datafusion::logical_expr::Expr],
        _limit: Option<usize>,
    ) -> datafusion::common::Result<Arc<dyn datafusion::physical_plan::ExecutionPlan>> {
        // Create sample data
        let ids = Int64Array::from(vec![1, 2, 3, 4, 5]);
        let categories = StringArray::from(vec!["image", "text", "image", "audio", "text"]);
        let scores = Float32Array::from(vec![0.95, 0.87, 0.92, 0.78, 0.89]);
        let metadata = StringArray::from(vec!["cat.jpg", "doc1.txt", "dog.jpg", "song.mp3", "doc2.txt"]);
        
        // Create sample embeddings (128-dimensional vectors)
        let mut embedding_data = Vec::new();
        for i in 0..5 {
            let vector: Vec<f32> = (0..128).map(|j| (i as f32 + j as f32) * 0.01).collect();
            embedding_data.extend(vector_to_bytes(&vector));
        }
        let embeddings = FixedSizeBinaryArray::try_new(512, embedding_data.into(), None)?;

        let batch = RecordBatch::try_new(
            self.schema.clone(),
            vec![
                Arc::new(ids),
                Arc::new(embeddings),
                Arc::new(categories),
                Arc::new(scores),
                Arc::new(metadata),
            ],
        )?;

        Ok(Arc::new(MockVectorExecutionPlan {
            schema: self.schema.clone(),
            batch,
        }))
    }
}

// Mock execution plan that returns sample data
struct MockVectorExecutionPlan {
    schema: Arc<Schema>,
    batch: RecordBatch,
}

impl std::fmt::Debug for MockVectorExecutionPlan {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "MockVectorExecutionPlan")
    }
}

impl datafusion::physical_plan::DisplayAs for MockVectorExecutionPlan {
    fn fmt_as(
        &self,
        _t: datafusion::physical_plan::DisplayFormatType,
        f: &mut std::fmt::Formatter,
    ) -> std::fmt::Result {
        write!(f, "MockVectorExecutionPlan")
    }
}

impl datafusion::physical_plan::ExecutionPlan for MockVectorExecutionPlan {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn schema(&self) -> arrow::datatypes::SchemaRef {
        self.schema.clone()
    }

    fn properties(&self) -> &datafusion::physical_plan::PlanProperties {
        unimplemented!("Mock implementation")
    }

    fn children(&self) -> Vec<&Arc<dyn datafusion::physical_plan::ExecutionPlan>> {
        vec![]
    }

    fn name(&self) -> &str {
        "MockVectorExecutionPlan"
    }

    fn with_new_children(
        self: Arc<Self>,
        _children: Vec<Arc<dyn datafusion::physical_plan::ExecutionPlan>>,
    ) -> datafusion::common::Result<Arc<dyn datafusion::physical_plan::ExecutionPlan>> {
        Ok(self)
    }

    fn execute(
        &self,
        _partition: usize,
        _context: Arc<datafusion::execution::context::TaskContext>,
    ) -> datafusion::common::Result<datafusion::physical_plan::SendableRecordBatchStream> {
        use datafusion::physical_plan::memory::MemoryStream;
        
        let batch = self.batch.clone();
        let batches = vec![batch];
        let record_batch_stream = Box::pin(MemoryStream::try_new(batches, self.schema.clone(), None)?);
        
        Ok(record_batch_stream)
    }

    fn statistics(&self) -> datafusion::common::Result<datafusion::common::Statistics> {
        Ok(datafusion::common::Statistics::new_unknown(&self.schema))
    }
}
