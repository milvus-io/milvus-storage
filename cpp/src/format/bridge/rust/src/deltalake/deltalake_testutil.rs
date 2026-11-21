// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright Zilliz

//! Generate table and deletion-vector fixtures using Kernel's writer.

use std::collections::HashMap;
use std::ops::Range;
use std::sync::Arc;

use crate::deltalake_ffi::{
    DeltaDvPositions, DeltaTestTableKind, GeneratedDeltaDv, GeneratedDeltaDvContainer,
    GeneratedDeltaDvDescriptor,
};
use anyhow::{Result, anyhow, bail, ensure};
use delta_kernel::actions::deletion_vector_writer::{
    KernelDeletionVector, StreamingDeletionVectorWriter,
};
use delta_kernel::arrow::array::{ArrayRef, Float64Array, Int64Array, StringArray};
use delta_kernel::arrow::record_batch::RecordBatch;
use delta_kernel::committer::FileSystemCommitter;
use delta_kernel::engine::arrow_conversion::TryFromArrow;
use delta_kernel::engine::arrow_data::ArrowEngineData;
use delta_kernel::expressions::Scalar;
use delta_kernel::schema::StructType;
use delta_kernel::transaction::create_table::create_table;
use delta_kernel::transaction::data_layout::DataLayout;
use delta_kernel::transaction::{CommitResult, UpdateTableOperation};
use delta_kernel_default_engine::DefaultEngineBuilder;
use delta_kernel_default_engine::executor::tokio::TokioMultiThreadExecutor;
use delta_kernel_default_engine::storage::store_from_url;
use url::Url;

pub fn delta_test_write_local_table(
    path: &str,
    num_rows: u64,
    kind: DeltaTestTableKind,
) -> Result<()> {
    let table_url = Url::from_directory_path(path)
        .map_err(|_| anyhow!("Invalid local Delta fixture directory: {path}"))?;
    let executor = Arc::new(TokioMultiThreadExecutor::new(
        crate::TOKIO_RT.handle().clone(),
    ));
    let engine = DefaultEngineBuilder::new(store_from_url(&table_url)?)
        .with_task_executor(executor)
        .build();

    let num_rows = i64::try_from(num_rows)?;
    let batch = fixture_batch(0..num_rows)?;
    let schema = Arc::new(StructType::try_from_arrow(batch.schema().as_ref())?);
    let mut builder = create_table(table_url.as_str(), schema, "milvus-storage/test", true);
    if kind == DeltaTestTableKind::Partitioned {
        builder = builder.with_data_layout(DataLayout::partitioned(["name"]));
    }
    if kind == DeltaTestTableKind::Checkpointed {
        builder = builder.with_table_properties([
            ("delta.checkpoint.writeStatsAsJson", "false"),
            ("delta.checkpoint.writeStatsAsStruct", "true"),
        ]);
    }

    let mut txn = builder.build(&engine, Box::new(FileSystemCommitter::new()))?;
    let mut context = txn.write_state()?.write_context_builder();
    let batch = if kind == DeltaTestTableKind::Partitioned {
        context = context.with_partition_values(HashMap::from([(
            "name".to_string(),
            Scalar::String("east/west".to_string()),
        )]));
        batch.project(&[0, 2])?
    } else {
        batch
    };
    let context = context.build()?;
    let file =
        crate::TOKIO_RT.block_on(engine.write_parquet(&ArrowEngineData::new(batch), &context))?;
    txn.add_files(file);
    let result = txn.commit(&engine)?;
    let CommitResult::Committed(committed) = result else {
        bail!("Could not commit Delta fixture: {result:?}");
    };
    let snapshot = committed
        .post_commit_snapshot()
        .ok_or_else(|| anyhow!("Delta fixture commit did not return a snapshot"))?;

    match kind {
        DeltaTestTableKind::Basic | DeltaTestTableKind::Partitioned => {}
        DeltaTestTableKind::Appended => {
            let mut append = snapshot
                .clone()
                .update_table(UpdateTableOperation::Write, "milvus-storage/test", true)
                .with_blind_append()
                .build(&engine, Box::new(FileSystemCommitter::new()))?;
            let context = append.write_state()?.write_context_builder().build()?;
            let end = num_rows
                .checked_mul(2)
                .ok_or_else(|| anyhow!("Too many fixture rows"))?;
            let batch = ArrowEngineData::new(fixture_batch(num_rows..end)?);
            let file = crate::TOKIO_RT.block_on(engine.write_parquet(&batch, &context))?;
            append.add_files(file);
            let result = append.commit(&engine)?;
            ensure!(
                result.is_committed(),
                "Could not append to Delta fixture: {result:?}"
            );
        }
        DeltaTestTableKind::Checkpointed => {
            snapshot.checkpoint(&engine, None)?;
        }
        _ => bail!("Unknown Delta test table kind"),
    }
    Ok(())
}

fn fixture_batch(rows: Range<i64>) -> Result<RecordBatch> {
    let ids = Int64Array::from_iter_values(rows.clone());
    let names = StringArray::from_iter_values(rows.clone().map(|id| format!("name_{id}")));
    let values = Float64Array::from_iter_values(rows.map(|id| id as f64 * 1.5));
    Ok(RecordBatch::try_from_iter([
        ("id", Arc::new(ids) as ArrayRef),
        ("name", Arc::new(names)),
        ("value", Arc::new(values)),
    ])?)
}

pub fn delta_test_deletion_vector(positions: &[u64]) -> Result<GeneratedDeltaDv> {
    let generated = delta_test_deletion_vectors(vec![DeltaDvPositions {
        rows: positions.to_vec(),
    }])?;
    let descriptor = &generated.descriptors[0];
    Ok(GeneratedDeltaDv {
        bytes: generated.bytes,
        offset: descriptor.offset,
        size: descriptor.size,
        cardinality: descriptor.cardinality,
    })
}

pub fn delta_test_deletion_vectors(
    vectors: Vec<DeltaDvPositions>,
) -> Result<GeneratedDeltaDvContainer> {
    let mut bytes = Vec::new();
    let mut descriptors = Vec::with_capacity(vectors.len());
    {
        let mut writer = StreamingDeletionVectorWriter::new(&mut bytes);
        for positions in vectors {
            let mut vector = KernelDeletionVector::new();
            vector.add_deleted_row_indexes(positions.rows);
            let descriptor = writer.write_deletion_vector(vector)?;
            descriptors.push(GeneratedDeltaDvDescriptor {
                offset: descriptor.offset,
                size: descriptor.size_in_bytes,
                cardinality: descriptor.cardinality,
                inline_data: String::new(),
            });
        }
        writer.finalize()?;
    }
    for descriptor in &mut descriptors {
        // The offset points at the on-disk size prefix. Inline DVs contain only
        // the magic + bitmap, excluding that prefix, the file header and CRC.
        let start = usize::try_from(descriptor.offset)? + size_of::<u32>();
        let end = start + usize::try_from(descriptor.size)?;
        descriptor.inline_data = z85::encode(&bytes[start..end]);
    }
    Ok(GeneratedDeltaDvContainer { bytes, descriptors })
}

#[cfg(test)]
mod tests {
    use super::*;
    use roaring::RoaringTreemap;

    #[test]
    fn shared_container_offsets_and_inline_payloads() -> Result<()> {
        let rows = [vec![], vec![1, 2, 3], vec![4, (1u64 << 32) + 5]];
        let generated = delta_test_deletion_vectors(
            rows.iter()
                .map(|rows| DeltaDvPositions { rows: rows.clone() })
                .collect(),
        )?;
        assert_eq!(generated.bytes[0], 1);
        assert_eq!(generated.descriptors.len(), rows.len());
        let mut offset = 1;
        for (descriptor, expected) in generated.descriptors.iter().zip(rows) {
            assert_eq!(descriptor.offset as usize, offset);
            assert_eq!(descriptor.cardinality as usize, expected.len());
            let size = descriptor.size as usize;
            let payload = z85::decode(&descriptor.inline_data)?;
            assert_eq!(payload.len(), size);
            assert_eq!(payload, generated.bytes[offset + 4..offset + 4 + size]);
            assert_eq!(&payload[..4], &1681511377u32.to_le_bytes());
            let bitmap = RoaringTreemap::deserialize_from(&payload[4..])?;
            assert_eq!(bitmap.iter().collect::<Vec<_>>(), expected);
            let crc = u32::from_be_bytes(
                generated.bytes[offset + 4 + size..offset + 8 + size].try_into()?,
            );
            assert_eq!(crc, crc32fast::hash(&payload));
            offset += size + 8;
        }
        assert_eq!(generated.bytes.len(), offset);
        Ok(())
    }

    #[test]
    fn single_vector_helper_preserves_container_format() -> Result<()> {
        let single = delta_test_deletion_vector(&[5, 1, 5])?;
        let generated = delta_test_deletion_vectors(vec![DeltaDvPositions { rows: vec![1, 5] }])?;
        let descriptor = &generated.descriptors[0];
        assert_eq!(single.bytes, generated.bytes);
        assert_eq!(single.offset, descriptor.offset);
        assert_eq!(single.size, descriptor.size);
        assert_eq!(single.cardinality, 2);
        Ok(())
    }
}
