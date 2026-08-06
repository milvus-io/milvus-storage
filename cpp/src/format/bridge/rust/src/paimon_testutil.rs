// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! Repository-owned Paimon fixtures used by C++ integration tests and loon.
//!
//! The fixture is produced through paimon-rust's public writer and commit APIs;
//! it deliberately does not depend on a machine-local Java table generator.

use std::sync::Arc;

use anyhow::{Result, bail, ensure};
use arrow_array58::builder::{Float32Builder, ListBuilder};
use arrow_array58::{ArrayRef, Float64Array, Int32Array, Int64Array, RecordBatch, StringArray};
use arrow_schema58::{DataType as ArrowDataType, Field as ArrowField, Schema as ArrowSchema};
use bytes::Bytes;
use paimon::Table;
use paimon::catalog::Identifier;
use paimon::io::FileIO;
use paimon::spec::{
    ArrayType, BigIntType, DataType, DoubleType, FloatType, IntType, Schema, TableSchema,
    VarCharType,
};

use crate::TOKIO_RT;
fn fixture_schema(mode: &str, file_format: &str, dimension: u32) -> Result<TableSchema> {
    ensure!(
        matches!(file_format, "parquet" | "vortex"),
        "unsupported Paimon fixture file format '{file_format}'; expected parquet or vortex"
    );
    let vector_fixture = dimension > 0;
    let mut builder = if vector_fixture {
        Schema::builder()
            .column("pk", DataType::BigInt(BigIntType::with_nullable(false)))
            .column(
                "label",
                DataType::VarChar(VarCharType::with_nullable(false, VarCharType::MAX_LENGTH)?),
            )
            .column(
                "vector",
                DataType::Array(ArrayType::with_nullable(
                    false,
                    DataType::Float(FloatType::with_nullable(false)),
                )),
            )
    } else {
        Schema::builder()
            .column("id", DataType::Int(IntType::new()))
            .column("name", DataType::VarChar(VarCharType::string_type()))
            .column("value", DataType::Double(DoubleType::new()))
    };
    builder = builder
        // Keep test splits intentionally small and deterministic.
        .option("source.split.target-size", "1b")
        .option("source.split.open-file-cost", "1b");
    if file_format != "parquet" {
        builder = builder.option("file.format", file_format);
    }
    match mode {
        "append" => {}
        "mor" => {
            builder = builder
                .primary_key([if vector_fixture { "pk" } else { "id" }])
                .option("bucket", "1");
        }
        "deletion-vector" => {
            builder = builder
                .option("row-tracking.enabled", "true")
                .option("data-evolution.enabled", "true")
                .option("deletion-vectors.enabled", "true");
        }
        other => bail!(
            "unsupported Paimon fixture mode '{other}'; expected append, mor, or deletion-vector"
        ),
    }
    Ok(TableSchema::new(0, &builder.build()?))
}

fn scalar_fixture_batch(start: u64, rows: u64, value_multiplier: i32) -> Result<RecordBatch> {
    let end = start
        .checked_add(rows)
        .ok_or_else(|| anyhow::anyhow!("Paimon fixture row range overflow"))?;
    ensure!(
        end <= i32::MAX as u64,
        "Paimon fixture supports at most {} rows",
        i32::MAX
    );
    let ids: Vec<i32> = (start..end).map(|value| value as i32).collect();
    let names: Vec<String> = ids
        .iter()
        .map(|id| format!("row_{}", id.saturating_mul(value_multiplier)))
        .collect();
    let values: Vec<f64> = ids
        .iter()
        .map(|id| f64::from(*id) * 1.5 * f64::from(value_multiplier))
        .collect();
    let schema = Arc::new(ArrowSchema::new(vec![
        ArrowField::new("id", ArrowDataType::Int32, false),
        ArrowField::new("name", ArrowDataType::Utf8, false),
        ArrowField::new("value", ArrowDataType::Float64, false),
    ]));
    Ok(RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int32Array::from(ids)),
            Arc::new(StringArray::from(names)),
            Arc::new(Float64Array::from(values)),
        ],
    )?)
}

fn vector_fixture_batch(
    start: u64,
    rows: u64,
    value_multiplier: i32,
    dimension: u32,
) -> Result<RecordBatch> {
    ensure!(
        dimension > 0,
        "Paimon vector fixture dimension must be positive"
    );
    let end = start
        .checked_add(rows)
        .ok_or_else(|| anyhow::anyhow!("Paimon fixture row range overflow"))?;
    ensure!(
        end <= i64::MAX as u64,
        "Paimon fixture supports at most {} rows",
        i64::MAX
    );
    let ids: Vec<i64> = (start..end).map(|value| value as i64).collect();
    let labels: Vec<String> = ids
        .iter()
        .map(|id| format!("label_{}", id.saturating_mul(i64::from(value_multiplier))))
        .collect();
    let element_field = Arc::new(ArrowField::new("element", ArrowDataType::Float32, false));
    let mut vectors = ListBuilder::new(Float32Builder::new()).with_field(element_field.clone());
    for id in &ids {
        let base = id.saturating_mul(i64::from(value_multiplier)) as f32;
        for component in 0..dimension {
            vectors
                .values()
                .append_value(base + component as f32 / dimension as f32);
        }
        vectors.append(true);
    }
    let schema = Arc::new(ArrowSchema::new(vec![
        ArrowField::new("pk", ArrowDataType::Int64, false),
        ArrowField::new("label", ArrowDataType::Utf8, false),
        ArrowField::new("vector", ArrowDataType::List(element_field), false),
    ]));
    Ok(RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int64Array::from(ids)) as ArrayRef,
            Arc::new(StringArray::from(labels)) as ArrayRef,
            Arc::new(vectors.finish()) as ArrayRef,
        ],
    )?)
}

fn fixture_batch(
    start: u64,
    rows: u64,
    value_multiplier: i32,
    dimension: u32,
) -> Result<RecordBatch> {
    if dimension == 0 {
        scalar_fixture_batch(start, rows, value_multiplier)
    } else {
        vector_fixture_batch(start, rows, value_multiplier, dimension)
    }
}

async fn commit_batch(table: &Table, batch: &RecordBatch) -> Result<i64> {
    let write_builder = table.new_write_builder();
    let mut writer = write_builder.new_write()?;
    writer.write_arrow_batch(batch).await?;
    let messages = writer.prepare_commit().await?;
    write_builder.new_commit().commit(messages).await?;
    latest_snapshot_id(table).await
}

async fn latest_snapshot_id(table: &Table) -> Result<i64> {
    table
        .snapshot_manager()
        .get_latest_snapshot()
        .await?
        .map(|snapshot| snapshot.id())
        .ok_or_else(|| anyhow::anyhow!("Paimon fixture has no committed snapshot"))
}

async fn write_schema(file_io: &FileIO, table_location: &str, schema: &TableSchema) -> Result<()> {
    file_io.mkdirs(&format!("{table_location}/schema")).await?;
    file_io
        .new_output(&format!("{table_location}/schema/schema-0"))?
        .write(Bytes::from(serde_json::to_vec(schema)?))
        .await?;
    Ok(())
}

pub fn paimon_create_test_table(
    table_location: &str,
    num_rows: u64,
    mode: &str,
    deleted_positions: Vec<i64>,
    file_format: &str,
    dimension: u32,
) -> Result<i64> {
    let mode = mode.to_string();
    let file_format = file_format.to_string();
    TOKIO_RT.block_on(async move {
        let file_io = FileIO::from_path(table_location)?.build()?;
        file_io
            .mkdirs(&format!("{table_location}/snapshot"))
            .await?;
        file_io
            .mkdirs(&format!("{table_location}/manifest"))
            .await?;
        let schema = fixture_schema(&mode, &file_format, dimension)?;
        write_schema(&file_io, table_location, &schema).await?;
        let table = Table::new(
            file_io,
            Identifier::new("default", "milvus_storage_fixture"),
            table_location.to_string(),
            schema,
            None,
        );

        let first_batch = fixture_batch(0, num_rows, 1, dimension)?;
        commit_batch(&table, &first_batch).await?;
        if mode == "mor" && num_rows > 1 {
            // Overwrite the upper half of the key range in a second commit so
            // the resulting split requires Paimon's merge reader.
            let start = num_rows / 2;
            commit_batch(
                &table,
                &fixture_batch(start, num_rows - start, 10, dimension)?,
            )
            .await?;
        } else if mode == "deletion-vector" && !deleted_positions.is_empty() {
            let write_builder = table.new_write_builder();
            let mut writer = write_builder.new_delete()?;
            writer.add_row_ids(deleted_positions)?;
            let messages = writer.prepare_commit().await?;
            write_builder.new_commit().commit(messages).await?;
        }

        latest_snapshot_id(&table).await
    })
}
