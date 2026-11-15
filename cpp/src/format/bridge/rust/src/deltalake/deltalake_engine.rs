// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright Zilliz

//! Keep Kernel's snapshot/replay and default I/O, while making checkpoint row
//! counts available to its unfiltered JSON-only metadata scan.

use std::sync::Arc;

use delta_kernel::arrow::array::{
    Array, ArrayRef, Int64Array, StringArray, StringBuilder, StructArray,
};
use delta_kernel::arrow::datatypes::{DataType as ArrowType, Schema as ArrowSchema};
use delta_kernel::arrow::record_batch::RecordBatch;
use delta_kernel::engine::arrow_data::{ArrowEngineData, EngineDataArrowExt};
use delta_kernel::expressions::PredicateRef;
use delta_kernel::schema::{DataType, SchemaRef, StructField, StructType};
use delta_kernel::{
    Engine, EngineData, EvaluationHandler, FileDataReadResultIterator, FileMeta, FileSize,
    JsonHandler, KernelError, KernelResult, KernelResultIterator, ParquetFooter, ParquetHandler,
    StorageHandler,
};
use delta_kernel_default_engine::DefaultEngine;
use delta_kernel_default_engine::executor::tokio::TokioMultiThreadExecutor;

use crate::deltalake_object_store::DeltaPathBinding;

pub(crate) struct DeltaEngine {
    inner: DefaultEngine<TokioMultiThreadExecutor>,
    parquet: Arc<CheckpointParquetHandler>,
}

impl DeltaEngine {
    pub(crate) fn new(
        inner: DefaultEngine<TokioMultiThreadExecutor>,
        binding: DeltaPathBinding,
    ) -> Self {
        let parquet = Arc::new(CheckpointParquetHandler {
            inner: inner.parquet_handler(),
            binding,
        });
        Self { inner, parquet }
    }
}

impl Engine for DeltaEngine {
    fn evaluation_handler(&self) -> Arc<dyn EvaluationHandler> {
        self.inner.evaluation_handler()
    }

    fn storage_handler(&self) -> Arc<dyn StorageHandler> {
        self.inner.storage_handler()
    }

    fn json_handler(&self) -> Arc<dyn JsonHandler> {
        self.inner.json_handler()
    }

    fn parquet_handler(&self) -> Arc<dyn ParquetHandler> {
        self.parquet.clone()
    }
}

struct CheckpointParquetHandler {
    inner: Arc<dyn ParquetHandler>,
    binding: DeltaPathBinding,
}

impl CheckpointParquetHandler {
    fn validate_file(&self, file: &FileMeta) -> KernelResult<()> {
        // The default handler turns URLs into keys and drops their authority.
        // Validate checkpoint/sidecar identities before that can rebase a URL.
        self.binding
            .validate_kernel_url(&file.location)
            .map_err(|error| KernelError::unsupported(format!("[delta:error=unsupported] {error}")))
    }
}

impl ParquetHandler for CheckpointParquetHandler {
    fn read_parquet_files(
        &self,
        files: &[FileMeta],
        physical_schema: SchemaRef,
        predicate: Option<PredicateRef>,
    ) -> KernelResult<FileDataReadResultIterator> {
        for file in files {
            self.validate_file(file)?;
        }
        // This engine reads log/checkpoint actions; payloads use the C++ reader.
        // Match the requested action schema, not filenames: a sidecar can use
        // an absolute path outside _delta_log within the bound filesystem.
        let projection = CheckpointCountProjection::new(&physical_schema)?;
        let Some(projection) = projection else {
            return self
                .inner
                .read_parquet_files(files, physical_schema, predicate);
        };
        let batches =
            self.inner
                .read_parquet_files(files, projection.read_schema.clone(), predicate)?;
        Ok(Box::new(batches.map(move |batch| {
            Ok(Box::new(ArrowEngineData::new(
                projection.restore(batch?.try_into_record_batch()?)?,
            )) as Box<dyn EngineData>)
        })))
    }

    fn read_parquet_footer(&self, file: &FileMeta) -> KernelResult<ParquetFooter> {
        self.validate_file(file)?;
        self.inner.read_parquet_footer(file)
    }

    fn write_parquet_file(
        &self,
        _: url::Url,
        _: Box<dyn KernelResultIterator<Box<dyn EngineData>>>,
    ) -> KernelResult<FileSize> {
        Err(KernelError::unsupported("Delta engine is read-only"))
    }
}

struct CheckpointCountProjection {
    read_schema: SchemaRef,
    add_index: usize,
    stats_index: usize,
    parsed_index: usize,
}

impl CheckpointCountProjection {
    fn new(schema: &SchemaRef) -> KernelResult<Option<Self>> {
        let Some((add_index, add_field)) = schema.field_with_index("add") else {
            return Ok(None);
        };
        let DataType::Struct(add) = add_field.data_type() else {
            return Ok(None);
        };
        let Some((stats_index, stats)) = add.field_with_index("stats") else {
            return Ok(None);
        };
        if *stats.data_type() != DataType::STRING || add.field("stats_parsed").is_some() {
            return Ok(None);
        }
        // json_only + no predicate does not request a typed stats schema in
        // Kernel, so its checkpoint discovery cannot synthesize this count.
        // Request just numRecords, independent of data-column statistics and
        // schema evolution. Preserve Kernel's ordinary scan-row/cache shape.
        let parsed = StructField::nullable(
            "stats_parsed",
            StructType::try_new([StructField::nullable("numRecords", DataType::LONG)])?,
        );
        let fields = add.fields().cloned().chain(std::iter::once(parsed));
        let augmented = StructField::new(
            add_field.name(),
            StructType::try_new(fields)?,
            add_field.is_nullable(),
        )
        .with_metadata(add_field.metadata().clone());
        let read_schema = Arc::new(StructType::try_new(schema.fields().map(|field| {
            if field.name() == "add" {
                augmented.clone()
            } else {
                field.clone()
            }
        }))?);
        Ok(Some(Self {
            read_schema,
            add_index,
            stats_index,
            parsed_index: add.num_fields(),
        }))
    }

    fn restore(&self, batch: RecordBatch) -> KernelResult<RecordBatch> {
        let add = batch
            .column(self.add_index)
            .as_any()
            .downcast_ref::<StructArray>()
            .ok_or_else(|| KernelError::generic("checkpoint add is not a struct"))?;
        let stats = add.column(self.stats_index);
        let parsed = add
            .column(self.parsed_index)
            .as_any()
            .downcast_ref::<StructArray>()
            .ok_or_else(|| KernelError::generic("checkpoint stats_parsed is not a struct"))?;
        let counts = parsed
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .ok_or_else(|| KernelError::generic("checkpoint numRecords is not an integer"))?;
        let needs_fill = (0..add.len()).any(|row| {
            add.is_valid(row) && stats.is_null(row) && parsed.is_valid(row) && counts.is_valid(row)
        });
        let mut columns = add.columns()[..self.parsed_index].to_vec();
        let mut fields = add.fields()[..self.parsed_index].to_vec();
        if needs_fill {
            let normalized = delta_kernel::arrow::compute::cast(stats, &ArrowType::Utf8)?;
            let stats = normalized
                .as_any()
                .downcast_ref::<StringArray>()
                .ok_or_else(|| KernelError::generic("checkpoint stats is not a string"))?;
            let mut values = StringBuilder::new();
            for row in 0..add.len() {
                if add.is_null(row) {
                    values.append_null();
                } else if stats.is_valid(row) {
                    values.append_value(stats.value(row));
                } else if parsed.is_valid(row) && counts.is_valid(row) {
                    let count = counts.value(row);
                    if count < 0 {
                        return Err(KernelError::generic(
                            "[delta:error=invalid] negative checkpoint numRecords",
                        ));
                    }
                    values.append_value(format!("{{\"numRecords\":{count}}}"));
                } else {
                    values.append_null();
                }
            }
            columns[self.stats_index] = Arc::new(values.finish());
            fields[self.stats_index] = Arc::new(
                fields[self.stats_index]
                    .as_ref()
                    .clone()
                    .with_data_type(ArrowType::Utf8),
            );
        }
        let restored: ArrayRef = Arc::new(StructArray::try_new(
            fields.into(),
            columns,
            add.nulls().cloned(),
        )?);
        let mut columns = batch.columns().to_vec();
        let schema = batch.schema();
        let mut fields = schema.fields().to_vec();
        fields[self.add_index] = Arc::new(
            fields[self.add_index]
                .as_ref()
                .clone()
                .with_data_type(restored.data_type().clone()),
        );
        columns[self.add_index] = restored;
        let schema = Arc::new(ArrowSchema::new_with_metadata(
            fields,
            schema.metadata().clone(),
        ));
        Ok(RecordBatch::try_new(schema, columns)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use delta_kernel::arrow::buffer::NullBuffer;
    use delta_kernel::engine::arrow_conversion::TryIntoArrow;

    fn batch(
        stats: Vec<Option<&str>>,
        counts: Vec<Option<i64>>,
        nulls: Option<NullBuffer>,
    ) -> (CheckpointCountProjection, RecordBatch) {
        let schema = Arc::new(
            StructType::try_new([StructField::nullable(
                "add",
                StructType::try_new([StructField::nullable("stats", DataType::STRING)]).unwrap(),
            )])
            .unwrap(),
        );
        let projection = CheckpointCountProjection::new(&schema).unwrap().unwrap();
        let read_schema: ArrowSchema = projection.read_schema.as_ref().try_into_arrow().unwrap();
        let ArrowType::Struct(add_fields) = read_schema.field(0).data_type() else {
            unreachable!()
        };
        let ArrowType::Struct(count_fields) = add_fields[1].data_type() else {
            unreachable!()
        };
        let parsed = StructArray::new(
            count_fields.clone(),
            vec![Arc::new(Int64Array::from(counts))],
            None,
        );
        let add = StructArray::new(
            add_fields.clone(),
            vec![Arc::new(StringArray::from(stats)), Arc::new(parsed)],
            nulls,
        );
        let batch = RecordBatch::try_new(Arc::new(read_schema), vec![Arc::new(add)]).unwrap();
        (projection, batch)
    }

    #[test]
    fn checkpoint_counts_preserve_json_precedence_and_parent_nulls() {
        let original = "{\"numRecords\":2,\"minValues\":{\"id\":1}}";
        let (projection, input) = batch(
            vec![Some(original), None, None],
            vec![Some(99), Some(4), Some(6)],
            Some(NullBuffer::from(vec![true, true, false])),
        );
        let output = projection.restore(input).unwrap();
        let add = output
            .column(0)
            .as_any()
            .downcast_ref::<StructArray>()
            .unwrap();
        assert_eq!(add.num_columns(), 1); // Restore exactly the original action schema.
        let stats = add
            .column(0)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        assert_eq!(stats.value(0), original);
        assert_eq!(stats.value(1), "{\"numRecords\":4}");
        assert!(add.is_null(2));
        assert!(stats.is_null(2));
    }

    #[test]
    fn negative_checkpoint_counts_are_rejected() {
        let (projection, input) = batch(vec![None], vec![Some(-1)], None);
        assert!(projection.restore(input).is_err());
    }
}
