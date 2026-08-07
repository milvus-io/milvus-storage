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

use std::collections::HashMap;
use std::sync::Arc;

use anyhow::{Result, bail, ensure};
use arrow_array58::builder::{Float32Builder, ListBuilder};
use arrow_array58::{ArrayRef, Float64Array, Int64Array, RecordBatch, StringArray};
use arrow_schema58::{DataType as ArrowDataType, Field as ArrowField, Schema as ArrowSchema};
use bytes::Bytes;
use paimon::Table;
use paimon::catalog::Identifier;
use paimon::io::FileIO;
use paimon::spec::{
    ArrayType, BigIntType, DataType, DoubleType, FloatType, Schema, TableSchema, VarCharType,
};

use crate::TOKIO_RT;
use crate::paimon_test_ffi::PaimonTestTableInfo;

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
            .column("id", DataType::BigInt(BigIntType::new()))
            .column("name", DataType::VarChar(VarCharType::string_type()))
            .column("value", DataType::Double(DoubleType::new()))
    };
    if vector_fixture && mode == "mor" {
        builder = builder.column(
            "op",
            DataType::VarChar(VarCharType::with_nullable(false, VarCharType::MAX_LENGTH)?),
        );
    }
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
            if vector_fixture {
                builder = builder.option("rowkind.field", "op");
            }
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
        end <= i64::MAX as u64,
        "Paimon fixture supports at most {} rows",
        i64::MAX
    );
    let ids: Vec<i64> = (start..end).map(|value| value as i64).collect();
    let names: Vec<String> = ids
        .iter()
        .map(|id| format!("row_{}", id.saturating_mul(i64::from(value_multiplier))))
        .collect();
    let values: Vec<f64> = ids
        .iter()
        .map(|id| *id as f64 * 1.5 * f64::from(value_multiplier))
        .collect();
    let schema = Arc::new(ArrowSchema::new(vec![
        ArrowField::new("id", ArrowDataType::Int64, false),
        ArrowField::new("name", ArrowDataType::Utf8, false),
        ArrowField::new("value", ArrowDataType::Float64, false),
    ]));
    Ok(RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int64Array::from(ids)),
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

fn vector_mor_batch(rows: &[(i64, &str, &str)], dimension: u32) -> Result<RecordBatch> {
    let ids = rows.iter().map(|(pk, _, _)| *pk).collect::<Vec<_>>();
    let labels = rows.iter().map(|(_, label, _)| *label).collect::<Vec<_>>();
    let operations = rows.iter().map(|(_, _, op)| *op).collect::<Vec<_>>();
    let element_field = Arc::new(ArrowField::new("element", ArrowDataType::Float32, false));
    let mut vectors = ListBuilder::new(Float32Builder::new()).with_field(element_field.clone());
    for id in &ids {
        for component in 0..dimension {
            vectors
                .values()
                .append_value(*id as f32 + component as f32 / dimension as f32);
        }
        vectors.append(true);
    }
    let schema = Arc::new(ArrowSchema::new(vec![
        ArrowField::new("pk", ArrowDataType::Int64, false),
        ArrowField::new("label", ArrowDataType::Utf8, false),
        ArrowField::new("vector", ArrowDataType::List(element_field), false),
        ArrowField::new("op", ArrowDataType::Utf8, false),
    ]));
    Ok(RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int64Array::from(ids)) as ArrayRef,
            Arc::new(StringArray::from(labels)) as ArrayRef,
            Arc::new(vectors.finish()) as ArrayRef,
            Arc::new(StringArray::from(operations)) as ArrayRef,
        ],
    )?)
}

fn vector_mor_range_batch(
    start: i64,
    rows: u64,
    dimension: u32,
    operation: &str,
) -> Result<RecordBatch> {
    let rows_i64 = i64::try_from(rows).map_err(|_| anyhow::anyhow!("row count exceeds i64"))?;
    let end = start
        .checked_add(rows_i64)
        .ok_or_else(|| anyhow::anyhow!("fixture row range exceeds i64"))?;
    let values = (start..end)
        .map(|pk| (pk, format!("label_{pk}"), operation))
        .collect::<Vec<_>>();
    let borrowed = values
        .iter()
        .map(|(pk, label, op)| (*pk, label.as_str(), *op))
        .collect::<Vec<_>>();
    vector_mor_batch(&borrowed, dimension)
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
    storage_options_keys: Vec<String>,
    storage_options_values: Vec<String>,
    file_format: &str,
    dimension: u32,
) -> Result<PaimonTestTableInfo> {
    let mode = mode.to_string();
    let file_format = file_format.to_string();
    TOKIO_RT.block_on(async move {
        ensure!(
            storage_options_keys.len() == storage_options_values.len(),
            "Paimon fixture storage option key/value count mismatch"
        );
        let storage_options: HashMap<String, String> = storage_options_keys
            .into_iter()
            .zip(storage_options_values)
            .collect();
        let file_io = FileIO::from_path(table_location)?
            .with_props(storage_options)
            .build()?;
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

        let first_batch = if mode == "mor" && dimension > 0 {
            ensure!(
                num_rows >= 3,
                "Paimon vector merge-on-read fixture requires at least three rows"
            );
            vector_mor_range_batch(0, num_rows, dimension, "+I")?
        } else {
            fixture_batch(0, num_rows, 1, dimension)?
        };
        let mut snapshot_ids = vec![commit_batch(&table, &first_batch).await?];
        if mode == "mor" && dimension > 0 {
            let inserted_pk =
                i64::try_from(num_rows).map_err(|_| anyhow::anyhow!("row count exceeds i64"))?;
            let inserted_label = format!("label_{inserted_pk}");
            let changes = [
                (1, "label_1_updated", "+U"),
                (2, "label_2_deleted", "-D"),
                (inserted_pk, inserted_label.as_str(), "+I"),
            ];
            snapshot_ids.push(commit_batch(&table, &vector_mor_batch(&changes, dimension)?).await?);
        } else if mode == "mor" && num_rows > 1 {
            // Overwrite the upper half of the key range in a second commit so
            // the resulting split requires Paimon's merge reader.
            let start = num_rows / 2;
            snapshot_ids.push(
                commit_batch(
                    &table,
                    &fixture_batch(start, num_rows - start, 10, dimension)?,
                )
                .await?,
            );
        } else if mode == "deletion-vector" && !deleted_positions.is_empty() {
            let write_builder = table.new_write_builder();
            let mut writer = write_builder.new_delete()?;
            writer.add_row_ids(deleted_positions)?;
            let messages = writer.prepare_commit().await?;
            write_builder.new_commit().commit(messages).await?;
            snapshot_ids.push(latest_snapshot_id(&table).await?);
        }

        Ok(PaimonTestTableInfo { snapshot_ids })
    })
}
