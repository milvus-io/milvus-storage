// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright Zilliz

//! Adapt ordinary C++ Parquet schemas/batches to Kernel's selected-leaf input.

use std::sync::Arc;

use anyhow::{Result, ensure};
use delta_kernel::arrow::array::{Array, ArrayData, make_array};
use delta_kernel::arrow::datatypes::{DataType, FieldRef, Schema};
use delta_kernel::arrow::record_batch::{RecordBatch, RecordBatchOptions};
use delta_kernel::parquet::arrow::arrow_reader::{ArrowReaderMetadata, ArrowReaderOptions};
use delta_kernel::parquet::arrow::{ArrowSchemaConverter, ProjectionMask};
use delta_kernel::parquet::file::metadata::{FileMetaData, ParquetMetaData};

const INVALID: &str = "[delta:error=invalid]";

pub(crate) fn planning_metadata(schema: Arc<Schema>, rows: u64) -> Result<ArrowReaderMetadata> {
    // Kernel's planner needs Arrow fields and physical leaf ordinals, not file
    // offsets. Build a descriptor from the already-decoded C++ schema. Supplying
    // that same schema preserves Arrow 17 names/layouts (notably map wrappers).
    // This metadata has no row groups and must never be used for payload I/O.
    let descriptor = Arc::new(ArrowSchemaConverter::new().convert(&schema)?);
    let file = FileMetaData::new(1, rows.try_into()?, None, None, descriptor, None);
    Ok(ArrowReaderMetadata::try_new(
        Arc::new(ParquetMetaData::new(file, vec![])),
        ArrowReaderOptions::new().with_schema(schema),
    )?)
}

pub(crate) struct FileProjection {
    input_schema: Arc<Schema>,
    schema: Arc<Schema>,
    fields: Vec<ArrayProjection>,
    columns: Vec<String>,
}

enum ArrayProjection {
    Identity,
    Nested {
        data_type: DataType,
        children: Vec<(usize, ArrayProjection)>,
    },
}

impl FileProjection {
    pub(crate) fn new(schema: &Arc<Schema>, mask: Option<&ProjectionMask>) -> Self {
        if mask.is_none() {
            return Self {
                input_schema: schema.clone(),
                schema: schema.clone(),
                fields: schema
                    .fields()
                    .iter()
                    .map(|_| ArrayProjection::Identity)
                    .collect(),
                columns: schema
                    .fields()
                    .iter()
                    .map(|field| field.name().clone())
                    .collect(),
            };
        }
        let mut leaf = 0;
        let mut inputs = Vec::new();
        let mut outputs = Vec::new();
        let mut fields = Vec::new();
        for field in schema.fields() {
            if let Some((output, projection)) = project_field(field, mask, &mut leaf) {
                inputs.push(field.clone());
                outputs.push(output);
                fields.push(projection);
            }
        }
        Self {
            columns: inputs.iter().map(|field| field.name().clone()).collect(),
            // Kernel fixup replaces root metadata. Share field metadata without
            // copying potentially large file-level key/value maps per plan.
            input_schema: Arc::new(Schema::new(inputs)),
            schema: Arc::new(Schema::new(outputs)),
            fields,
        }
    }

    pub(crate) fn columns(&self) -> &[String] {
        &self.columns
    }

    pub(crate) fn empty_batch(&self) -> Result<RecordBatch> {
        self.apply(RecordBatch::new_empty(self.input_schema.clone()))
    }

    pub(crate) fn apply(&self, batch: RecordBatch) -> Result<RecordBatch> {
        ensure!(
            batch.schema().fields() == self.input_schema.fields(),
            "{INVALID} Parquet batch schema differs from planned physical columns"
        );
        if self
            .fields
            .iter()
            .all(|field| matches!(field, ArrayProjection::Identity))
        {
            return Ok(batch);
        }
        let arrays = self
            .fields
            .iter()
            .zip(batch.columns())
            .map(|(plan, array)| match plan {
                ArrayProjection::Identity => Ok(array.clone()),
                _ => Ok(make_array(plan.apply(&array.to_data())?)),
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(RecordBatch::try_new_with_options(
            self.schema.clone(),
            arrays,
            &RecordBatchOptions::new().with_row_count(Some(batch.num_rows())),
        )?)
    }
}

impl ArrayProjection {
    fn apply(&self, data: &ArrayData) -> Result<ArrayData> {
        match self {
            Self::Identity => Ok(data.clone()),
            Self::Nested {
                data_type,
                children,
            } => {
                let children = children
                    .iter()
                    .map(|(index, child)| child.apply(&data.child_data()[*index]))
                    .collect::<Result<Vec<_>>>()?;
                // Rebuild only structural metadata. Value buffers, offsets,
                // validity and slice offsets remain owned by the input data.
                Ok(data
                    .clone()
                    .into_builder()
                    .data_type(data_type.clone())
                    .child_data(children)
                    .build()?)
            }
        }
    }
}

fn project_field(
    field: &FieldRef,
    mask: Option<&ProjectionMask>,
    leaf: &mut usize,
) -> Option<(FieldRef, ArrayProjection)> {
    let children: Vec<FieldRef> = match field.data_type() {
        DataType::Struct(fields) => fields.to_vec(),
        DataType::List(element)
        | DataType::LargeList(element)
        | DataType::FixedSizeList(element, _)
        | DataType::Map(element, _) => vec![element.clone()],
        _ => {
            let included = mask.is_none_or(|mask| mask.leaf_included(*leaf));
            *leaf += 1;
            return included.then(|| (field.clone(), ArrayProjection::Identity));
        }
    };
    let mut selected = Vec::new();
    let mut projections = Vec::new();
    for (index, child) in children.iter().enumerate() {
        if let Some((field, projection)) = project_field(child, mask, leaf) {
            selected.push(field);
            projections.push((index, projection));
        }
    }
    if selected.is_empty() {
        return None;
    }
    let data_type = match field.data_type() {
        DataType::Struct(_) => DataType::Struct(selected.into()),
        DataType::List(_) => DataType::List(selected.remove(0)),
        DataType::LargeList(_) => DataType::LargeList(selected.remove(0)),
        DataType::FixedSizeList(_, size) => DataType::FixedSizeList(selected.remove(0), *size),
        DataType::Map(entries, sorted) => {
            let projected = selected.remove(0);
            // Match Arrow C++'s projected-map representation. Missing entries
            // or partial struct keys become list<struct<...>>, preserving the
            // same input shape as the former physical-leaf reader adapter.
            match (entries.data_type(), projected.data_type()) {
                (DataType::Struct(original), DataType::Struct(fields))
                    if fields.len() == 2 && fields[0].data_type() == original[0].data_type() =>
                {
                    DataType::Map(projected, *sorted)
                }
                _ => DataType::List(projected),
            }
        }
        _ => unreachable!(),
    };
    if &data_type == field.data_type() {
        Some((field.clone(), ArrayProjection::Identity))
    } else {
        Some((
            Arc::new(field.as_ref().clone().with_data_type(data_type.clone())),
            ArrayProjection::Nested {
                data_type,
                children: projections,
            },
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use delta_kernel::arrow::array::cast::AsArray;
    use delta_kernel::arrow::array::{
        ArrayRef, Int64Array, ListArray, MapArray, StringArray, StructArray,
    };
    use delta_kernel::arrow::buffer::{NullBuffer, OffsetBuffer};
    use delta_kernel::arrow::datatypes::{Field, TimeUnit};
    use delta_kernel::engine::arrow_conversion::TryIntoKernel;
    use delta_kernel::engine::arrow_utils::{fixup_parquet_read, parquet_read_plan};
    use delta_kernel::schema::StructType;
    use std::collections::HashMap;

    fn values() -> StructArray {
        StructArray::new(
            vec![
                Arc::new(Field::new("keep", DataType::Int64, true)),
                Arc::new(Field::new("dropped", DataType::Utf8, true)),
            ]
            .into(),
            vec![
                Arc::new(Int64Array::from(vec![7, 8, 9])),
                Arc::new(StringArray::from(vec!["a", "b", "c"])),
            ],
            Some(NullBuffer::from(vec![true, false, true])),
        )
    }

    fn projection(schema: Arc<Schema>, leaves: &[usize]) -> Result<FileProjection> {
        let metadata = planning_metadata(schema.clone(), 3)?;
        let mask = ProjectionMask::leaves(metadata.parquet_schema(), leaves.iter().copied());
        Ok(FileProjection::new(&schema, Some(&mask)))
    }

    fn batch(field: FieldRef, array: ArrayRef) -> Result<RecordBatch> {
        Ok(RecordBatch::try_new(
            Arc::new(Schema::new(vec![field])),
            vec![array],
        )?)
    }

    #[test]
    fn struct_projection_preserves_nulls_slices_and_value_buffers() -> Result<()> {
        let values = values().slice(1, 2);
        let field = Arc::new(
            Field::new("profile", values.data_type().clone(), true)
                .with_metadata(HashMap::from([("PARQUET:field_id".into(), "4".into())])),
        );
        let schema = Arc::new(Schema::new(vec![
            Field::new("unread", DataType::Int64, true),
            field.as_ref().clone(),
        ]));
        let plan = projection(schema, &[1])?;
        assert_eq!(plan.columns(), &["profile"]);
        let input = batch(field, Arc::new(values))?;
        let output = plan.apply(input.clone())?;
        let projected = output.column(0).as_struct();
        assert_eq!(projected.num_columns(), 1);
        assert_eq!(projected.nulls(), input.column(0).nulls());
        assert_eq!(
            projected.column(0).to_data().buffers()[0].as_ptr(),
            input.column(0).as_struct().column(0).to_data().buffers()[0].as_ptr()
        );
        assert_eq!(output.schema().field(0).metadata()["PARQUET:field_id"], "4");
        assert_eq!(output.num_rows(), 2);
        Ok(())
    }

    #[test]
    fn list_projection_preserves_offsets_validity_and_child_buffers() -> Result<()> {
        let values = Arc::new(values());
        let element = Arc::new(Field::new(
            "producer_element",
            values.data_type().clone(),
            true,
        ));
        let list = ListArray::try_new(
            element,
            OffsetBuffer::new(vec![0, 1, 1, 3].into()),
            values,
            Some(NullBuffer::from(vec![true, false, true])),
        )?
        .slice(1, 2);
        let field = Arc::new(Field::new("items", list.data_type().clone(), true));
        let plan = projection(Arc::new(Schema::new(vec![field.clone()])), &[0])?;
        let input = batch(field, Arc::new(list))?;
        let output = plan.apply(input.clone())?;
        let before = input.column(0).as_list::<i32>();
        let after = output.column(0).as_list::<i32>();
        assert_eq!(after.value_offsets(), before.value_offsets());
        assert_eq!(
            after.value_offsets().as_ptr(),
            before.value_offsets().as_ptr()
        );
        assert_eq!(after.nulls(), before.nulls());
        assert_eq!(
            after.values().as_struct().column(0).to_data().buffers()[0].as_ptr(),
            before.values().as_struct().column(0).to_data().buffers()[0].as_ptr()
        );
        assert_eq!(after.len(), 2);
        Ok(())
    }

    #[test]
    fn map_projection_preserves_shape_and_matches_partial_map_representation() -> Result<()> {
        // Map values are nullable structs; map entries themselves are required.
        let values = Arc::new(values());
        let entries = StructArray::new(
            vec![
                Arc::new(Field::new("key", DataType::Utf8, false)),
                Arc::new(Field::new("value", values.data_type().clone(), true)),
            ]
            .into(),
            vec![Arc::new(StringArray::from(vec!["a", "b", "c"])), values],
            None,
        );
        let entry_field = Arc::new(Field::new("key_value", entries.data_type().clone(), false));
        let map = MapArray::try_new(
            entry_field,
            OffsetBuffer::new(vec![0, 1, 1, 3].into()),
            entries,
            Some(NullBuffer::from(vec![true, false, true])),
            false,
        )?
        .slice(1, 2);
        let field = Arc::new(Field::new("lookup", map.data_type().clone(), true));
        let schema = Arc::new(Schema::new(vec![field.clone()]));
        let input = batch(field, Arc::new(map))?;
        for leaves in [&[0, 1][..], &[0][..], &[1][..]] {
            let plan = projection(schema.clone(), leaves)?;
            let output = plan.apply(input.clone())?;
            let before = input.column(0).as_map();
            assert_eq!(output.column(0).nulls(), before.nulls());
            let (offsets, entries) = if leaves.len() == 2 {
                let map = output.column(0).as_map();
                assert_eq!(map.entries().column(1).as_struct().num_columns(), 1);
                (map.value_offsets(), map.entries())
            } else {
                let list = output.column(0).as_list::<i32>();
                assert_eq!(list.values().as_struct().num_columns(), 1);
                (list.value_offsets(), list.values().as_struct())
            };
            assert_eq!(offsets, before.value_offsets());
            assert_eq!(offsets.as_ptr(), before.value_offsets().as_ptr());
            if leaves.contains(&0) {
                assert_eq!(
                    entries.column(0).to_data().buffers()[1].as_ptr(),
                    before.entries().column(0).to_data().buffers()[1].as_ptr()
                );
            }
        }
        let requested = Schema::new(vec![Field::new(
            "lookup",
            DataType::Map(
                Arc::new(Field::new(
                    "entries",
                    DataType::Struct(
                        vec![
                            Arc::new(Field::new("key", DataType::Utf8, false)),
                            Arc::new(Field::new(
                                "value",
                                DataType::Struct(
                                    vec![Arc::new(Field::new("added", DataType::Utf8, true))]
                                        .into(),
                                ),
                                true,
                            )),
                        ]
                        .into(),
                    ),
                    false,
                )),
                false,
            ),
            true,
        )]);
        let kernel = Arc::new((&requested).try_into_kernel()?);
        let metadata = planning_metadata(schema.clone(), 3)?;
        let (ordering, mask) = parquet_read_plan(&kernel, &metadata)?;
        let plan = FileProjection::new(&schema, mask.as_ref());
        // With no selected value leaves, Arrow C++'s projected map is a list.
        // Keep the pinned Kernel's rejection of that representation, rather
        // than retaining an unselected value child as a container workaround.
        let result = fixup_parquet_read(plan.apply(input)?, &ordering, None, None, Some(&kernel));
        assert!(matches!(result, Err(error) if error.to_string().contains("isn't a MapArray")));
        Ok(())
    }

    #[test]
    fn empty_projection_preserves_row_count_and_rejects_unplanned_columns() -> Result<()> {
        let field = Arc::new(Field::new("id", DataType::Int64, false));
        let plan = projection(Arc::new(Schema::new(vec![field.clone()])), &[])?;
        assert!(plan.columns().is_empty());
        let empty = RecordBatch::try_new_with_options(
            Arc::new(Schema::empty()),
            vec![],
            &RecordBatchOptions::new().with_row_count(Some(3)),
        )?;
        assert_eq!(plan.apply(empty)?.num_rows(), 3);
        assert!(
            plan.apply(batch(field, Arc::new(Int64Array::from(vec![1, 2, 3])))?)
                .is_err()
        );
        Ok(())
    }

    #[test]
    fn zero_leaf_struct_retains_upstream_missing_container_behavior() -> Result<()> {
        let original = values();
        assert_eq!(original.null_count(), 1);
        let schema = Arc::new(Schema::new(vec![Field::new(
            "profile",
            original.data_type().clone(),
            true,
        )]));
        let metadata = planning_metadata(schema, 3)?;
        let requested = Schema::new(vec![Field::new(
            "profile",
            DataType::Struct(vec![Arc::new(Field::new("added", DataType::Utf8, true))].into()),
            true,
        )]);
        let kernel = Arc::new((&requested).try_into_kernel()?);
        let (ordering, mask) = parquet_read_plan(&kernel, &metadata)?;
        let plan = FileProjection::new(metadata.schema(), mask.as_ref());
        assert!(plan.columns().is_empty());
        let input = RecordBatch::try_new_with_options(
            Arc::new(Schema::empty()),
            vec![],
            &RecordBatchOptions::new().with_row_count(Some(3)),
        )?;
        let output: RecordBatch =
            fixup_parquet_read(plan.apply(input)?, &ordering, None, None, Some(&kernel))?.into();
        // Keep Kernel's all-null parent rather than reading an otherwise
        // unselected physical child to recover the original parent validity.
        assert_eq!(output.column(0).null_count(), 3);
        Ok(())
    }

    #[test]
    fn planning_adapter_preserves_field_ids_map_wrappers_and_kernel_matching() -> Result<()> {
        let values = values();
        let map = DataType::Map(
            Arc::new(Field::new(
                "key_value",
                DataType::Struct(
                    vec![
                        Arc::new(Field::new("key", DataType::Utf8, false)),
                        Arc::new(Field::new("value", values.data_type().clone(), true)),
                    ]
                    .into(),
                ),
                false,
            )),
            false,
        );
        let id = HashMap::from([("PARQUET:field_id".into(), "7".into())]);
        let schema = Arc::new(Schema::new(vec![
            Field::new("lookup", map, true).with_metadata(id.clone()),
        ]));
        let metadata = planning_metadata(schema.clone(), 3)?;
        assert_eq!(metadata.schema(), &schema);
        let requested = Schema::new(vec![
            Field::new(
                "mapped_lookup",
                DataType::Map(
                    Arc::new(Field::new(
                        "entries",
                        DataType::Struct(
                            vec![
                                Arc::new(Field::new("key", DataType::Utf8, false)),
                                Arc::new(Field::new(
                                    "value",
                                    DataType::Struct(
                                        vec![Arc::new(Field::new("keep", DataType::Int64, true))]
                                            .into(),
                                    ),
                                    true,
                                )),
                            ]
                            .into(),
                        ),
                        false,
                    )),
                    false,
                ),
                true,
            )
            .with_metadata(id),
        ]);
        let kernel: StructType = (&requested).try_into_kernel()?;
        let (_, mask) = parquet_read_plan(&Arc::new(kernel), &metadata)?;
        let plan = FileProjection::new(metadata.schema(), mask.as_ref());
        assert_eq!(plan.columns(), &["lookup"]);
        assert_eq!(plan.empty_batch()?.num_columns(), 1);
        Ok(())
    }

    #[test]
    fn planning_adapter_accepts_exposed_arrow_layouts() -> Result<()> {
        let child = Arc::new(Field::new("item", DataType::Int32, true));
        let layouts = [
            DataType::Null,
            DataType::Boolean,
            DataType::Int8,
            DataType::Int16,
            DataType::Int32,
            DataType::Int64,
            DataType::Float32,
            DataType::Float64,
            DataType::Utf8,
            DataType::LargeUtf8,
            DataType::Binary,
            DataType::LargeBinary,
            DataType::Date32,
            DataType::Decimal128(12, 2),
            DataType::Timestamp(TimeUnit::Second, None),
            DataType::Timestamp(TimeUnit::Millisecond, Some("UTC".into())),
            DataType::Timestamp(TimeUnit::Microsecond, None),
            DataType::Timestamp(TimeUnit::Nanosecond, Some("UTC".into())),
            DataType::List(child.clone()),
            DataType::LargeList(child.clone()),
            DataType::FixedSizeList(child, 2),
        ];
        for layout in layouts {
            let schema = Arc::new(Schema::new(vec![Field::new("value", layout, true)]));
            assert_eq!(planning_metadata(schema.clone(), 3)?.schema(), &schema);
        }
        Ok(())
    }
}
