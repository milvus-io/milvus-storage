// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright Zilliz

//! Align full Parquet columns to Kernel's physical schema before fixup.

use std::collections::HashMap;
use std::sync::Arc;

use anyhow::{Result, ensure};
use delta_kernel::arrow::array::{Array, ArrayData, make_array, new_null_array};
use delta_kernel::arrow::datatypes::{DataType, Field, FieldRef, Fields, Schema};
use delta_kernel::arrow::record_batch::{RecordBatch, RecordBatchOptions};
use delta_kernel::parquet::arrow::PARQUET_FIELD_ID_META_KEY;

const INVALID: &str = "[delta:error=invalid]";

pub(crate) struct FileProjection {
    input_schema: Arc<Schema>,
    schema: Arc<Schema>,
    fields: Vec<FieldProjection>,
    columns: Vec<String>,
    identity: bool,
}

enum FieldProjection {
    Source(usize, ArrayProjection),
    Missing(DataType),
}

enum ArrayProjection {
    Identity,
    Nested(DataType, Vec<FieldProjection>),
}

impl FileProjection {
    pub(crate) fn new(file: &Schema, requested: &Schema) -> Result<Self> {
        let (outputs, mut fields) = align_struct(file.fields(), requested.fields())?;
        let mut read_positions = vec![None; file.fields().len()];
        for field in &fields {
            if let FieldProjection::Source(index, _) = field {
                read_positions[*index] = Some(0);
            }
        }
        // The Parquet reader consumes physical top-level names. Keep file order
        // for I/O, then align the decoded columns to Kernel's physical order.
        let mut inputs = Vec::new();
        for (field, position) in file.fields().iter().zip(&mut read_positions) {
            if position.is_some() {
                *position = Some(inputs.len());
                inputs.push(field.clone());
            }
        }
        for field in &mut fields {
            if let FieldProjection::Source(index, _) = field {
                *index = read_positions[*index].expect("selected physical column");
            }
        }
        let identity = fields
            .iter()
            .enumerate()
            .all(|(i, field)| field.is_identity(i));
        Ok(Self {
            columns: inputs.iter().map(|field| field.name().clone()).collect(),
            input_schema: Arc::new(Schema::new(inputs)),
            schema: Arc::new(Schema::new(outputs)),
            fields,
            identity,
        })
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
        if self.identity {
            return Ok(batch);
        }
        let arrays = self
            .fields
            .iter()
            .map(|field| match field {
                FieldProjection::Source(index, ArrayProjection::Identity) => {
                    Ok(batch.column(*index).clone())
                }
                FieldProjection::Source(index, plan) => {
                    Ok(make_array(plan.apply(&batch.column(*index).to_data())?))
                }
                FieldProjection::Missing(data_type) => {
                    Ok(new_null_array(data_type, batch.num_rows()))
                }
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(RecordBatch::try_new_with_options(
            self.schema.clone(),
            arrays,
            &RecordBatchOptions::new().with_row_count(Some(batch.num_rows())),
        )?)
    }
}

impl FieldProjection {
    fn is_identity(&self, index: usize) -> bool {
        matches!(self, Self::Source(source, ArrayProjection::Identity) if *source == index)
    }
}

impl ArrayProjection {
    fn apply(&self, data: &ArrayData) -> Result<ArrayData> {
        match self {
            Self::Identity => Ok(data.clone()),
            Self::Nested(data_type, children) => {
                let children = children
                    .iter()
                    .map(|child| match child {
                        FieldProjection::Source(index, plan) => {
                            plan.apply(&data.child_data()[*index])
                        }
                        FieldProjection::Missing(data_type) => {
                            Ok(new_null_array(data_type, data.offset() + data.len()).to_data())
                        }
                    })
                    .collect::<Result<Vec<_>>>()?;
                // Keep the source null buffer and offsets. In particular, an
                // existing container may have no physical children in the result.
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

fn field_id(field: &Field) -> Option<i64> {
    field
        .metadata()
        .get(PARQUET_FIELD_ID_META_KEY)?
        .parse()
        .ok()
}

fn align_struct(
    file: &Fields,
    requested: &Fields,
) -> Result<(Vec<FieldRef>, Vec<FieldProjection>)> {
    let by_name: HashMap<_, _> = requested
        .iter()
        .enumerate()
        .map(|(i, f)| (f.name(), i))
        .collect();
    let by_id: HashMap<_, _> = requested
        .iter()
        .enumerate()
        .filter_map(|(i, f)| Some((field_id(f)?, i)))
        .collect();
    let mut matches = vec![None; requested.len()];
    for (index, field) in file.iter().enumerate() {
        // Keep the pinned planner's ID-first matching and name fallback.
        let target = field_id(field)
            .and_then(|id| by_id.get(&id))
            .or_else(|| by_name.get(field.name()));
        if let Some(&target) = target {
            ensure!(
                matches[target].replace(index).is_none(),
                "{INVALID} multiple Parquet fields match Delta field {}",
                requested[target].name()
            );
        }
    }
    requested
        .iter()
        .zip(matches)
        .map(|(target, index)| match index {
            Some(index) => {
                let (field, array) = align_field(&file[index], target)?;
                Ok((field, FieldProjection::Source(index, array)))
            }
            None => {
                ensure!(
                    target.is_nullable(),
                    "{INVALID} required Delta field missing from Parquet schema: {}",
                    target.name()
                );
                Ok((
                    target.clone(),
                    FieldProjection::Missing(target.data_type().clone()),
                ))
            }
        })
        .collect::<Result<Vec<_>>>()
        .map(|fields| fields.into_iter().unzip())
}

fn align_field(file: &FieldRef, target: &FieldRef) -> Result<(FieldRef, ArrayProjection)> {
    // Keep physical leaf types, names and field IDs on every matched field.
    // Kernel fixup validates these against the target and performs casts.
    let (data_type, projections) = match (file.data_type(), target.data_type()) {
        (DataType::Struct(file), DataType::Struct(target)) => {
            let (fields, projections) = align_struct(file, target)?;
            (DataType::Struct(fields.into()), projections)
        }
        (DataType::List(element) | DataType::LargeList(element), DataType::List(target)) => {
            let (field, plan) = align_field(element, target)?;
            let data_type = match file.data_type() {
                DataType::LargeList(_) => DataType::LargeList(field),
                _ => DataType::List(field),
            };
            (data_type, vec![FieldProjection::Source(0, plan)])
        }
        (DataType::Map(entries, sorted), DataType::Map(target, _)) => {
            // The entries wrapper always has a key and a value. Match these
            // positions structurally even when producer wrapper names differ.
            let (DataType::Struct(file_fields), DataType::Struct(target_fields)) =
                (entries.data_type(), target.data_type())
            else {
                anyhow::bail!("{INVALID} invalid Delta map entries");
            };
            ensure!(
                file_fields.len() == 2 && target_fields.len() == 2,
                "{INVALID} a Delta map must contain a key and a value"
            );
            let (key, key_plan) = align_field(&file_fields[0], &target_fields[0])?;
            let (value, value_plan) = align_field(&file_fields[1], &target_fields[1])?;
            let (entries, plan) = rebuild(
                entries,
                DataType::Struct(vec![key, value].into()),
                vec![
                    FieldProjection::Source(0, key_plan),
                    FieldProjection::Source(1, value_plan),
                ],
            );
            (
                DataType::Map(entries, *sorted),
                vec![FieldProjection::Source(0, plan)],
            )
        }
        _ => return Ok((file.clone(), ArrayProjection::Identity)),
    };
    Ok(rebuild(file, data_type, projections))
}

fn rebuild(
    file: &FieldRef,
    data_type: DataType,
    children: Vec<FieldProjection>,
) -> (FieldRef, ArrayProjection) {
    if &data_type == file.data_type()
        && children
            .iter()
            .enumerate()
            .all(|(i, field)| field.is_identity(i))
    {
        (file.clone(), ArrayProjection::Identity)
    } else {
        let field = Arc::new(file.as_ref().clone().with_data_type(data_type.clone()));
        (field, ArrayProjection::Nested(data_type, children))
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
    use delta_kernel::engine::arrow_conversion::TryIntoKernel;
    use delta_kernel::engine::arrow_utils::fixup_parquet_read;
    use delta_kernel::schema::StructType;

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

    fn struct_type(name: &str, data_type: DataType) -> DataType {
        DataType::Struct(vec![Arc::new(Field::new(name, data_type, true))].into())
    }

    fn fixup(plan: &FileProjection, input: RecordBatch, schema: &Schema) -> Result<RecordBatch> {
        let target: StructType = schema.try_into_kernel()?;
        Ok(
            fixup_parquet_read(plan.apply(input)?, &[], None, None, Some(&Arc::new(target)))?
                .into(),
        )
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
        let requested = Schema::new(vec![Field::new(
            "profile",
            struct_type("keep", DataType::Int64),
            true,
        )]);
        let plan = FileProjection::new(&schema, &requested)?;
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
        let requested = Schema::new(vec![Field::new(
            "items",
            DataType::new_list(struct_type("keep", DataType::Int64), true),
            true,
        )]);
        let plan = FileProjection::new(&Schema::new(vec![field.clone()]), &requested)?;
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
    fn map_projection_preserves_keys_and_nullable_values_with_no_matching_leaves() -> Result<()> {
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
            OffsetBuffer::new(vec![0, 1, 1, 1, 3].into()),
            entries,
            Some(NullBuffer::from(vec![true, false, true, true])),
            false,
        )?
        .slice(1, 3);
        let field = Arc::new(Field::new("lookup", map.data_type().clone(), true));
        let input = batch(field, Arc::new(map))?;
        let requested = Schema::new(vec![Field::new_map(
            "lookup",
            "entries",
            Field::new("key", DataType::Utf8, false),
            Field::new("value", struct_type("added", DataType::Utf8), true),
            false,
            true,
        )]);
        let plan = FileProjection::new(&input.schema(), &requested)?;
        assert_eq!(plan.columns(), &["lookup"]);
        let output = fixup(&plan, input.clone(), &requested)?;
        let before = input.column(0).as_map();
        let after = output.column(0).as_map();
        assert_eq!(after.nulls(), before.nulls());
        assert_eq!(after.value_offsets(), &[1, 1, 1, 3]);
        assert_eq!(
            after.value_offsets().as_ptr(),
            before.value_offsets().as_ptr()
        );
        assert_eq!(after.keys(), before.keys());
        assert_eq!(
            after.keys().to_data().buffers()[1].as_ptr(),
            before.keys().to_data().buffers()[1].as_ptr()
        );
        let values = after.values().as_struct();
        assert_eq!(values.nulls(), before.values().nulls());
        assert_eq!(values.column(0).null_count(), 3);
        Ok(())
    }

    #[test]
    fn empty_projection_preserves_row_count_and_rejects_unplanned_columns() -> Result<()> {
        let field = Arc::new(Field::new("id", DataType::Int64, false));
        let plan = FileProjection::new(&Schema::new(vec![field.clone()]), &Schema::empty())?;
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
    fn zero_leaf_struct_keeps_the_parent_and_missing_top_level_fields_stay_null() -> Result<()> {
        let original = values().slice(1, 2);
        let input = batch(
            Arc::new(Field::new("profile", original.data_type().clone(), true)),
            Arc::new(original),
        )?;
        let requested = Schema::new(vec![
            Field::new("absent", struct_type("added", DataType::Utf8), true),
            Field::new("profile", struct_type("added", DataType::Utf8), true),
        ]);
        let plan = FileProjection::new(&input.schema(), &requested)?;
        assert_eq!(plan.columns(), &["profile"]);
        let output = fixup(&plan, input, &requested)?;
        assert_eq!(output.column(0).null_count(), 2);
        let profile = output.column(1).as_struct();
        assert!(profile.is_null(0));
        assert!(profile.is_valid(1));
        assert_eq!(profile.column(0).null_count(), 2);
        Ok(())
    }

    #[test]
    fn nested_lists_keep_offsets_and_validity_even_when_all_values_are_new_fields() -> Result<()> {
        let values = Arc::new(values());
        let inner = ListArray::try_new(
            Arc::new(Field::new("item", values.data_type().clone(), true)),
            OffsetBuffer::new(vec![0, 1, 1, 1, 3].into()),
            values,
            Some(NullBuffer::from(vec![true, false, true, true])),
        )?;
        let list = ListArray::try_new(
            Arc::new(Field::new("item", inner.data_type().clone(), true)),
            OffsetBuffer::new(vec![0, 3, 3, 3, 4].into()),
            Arc::new(inner),
            Some(NullBuffer::from(vec![true, false, true, true])),
        )?
        .slice(1, 3);
        let input = batch(
            Arc::new(Field::new("items", list.data_type().clone(), true)),
            Arc::new(list),
        )?;
        let expected = DataType::new_list(struct_type("added", DataType::Utf8), true);
        let requested = Schema::new(vec![Field::new(
            "items",
            DataType::new_list(expected, true),
            true,
        )]);
        let plan = FileProjection::new(&input.schema(), &requested)?;
        assert_eq!(plan.columns(), &["items"]);
        let output = fixup(&plan, input.clone(), &requested)?;
        let before = input.column(0).as_list::<i32>();
        let after = output.column(0).as_list::<i32>();
        assert_eq!(after.value_offsets(), &[3, 3, 3, 4]);
        assert_eq!(after.nulls(), before.nulls());
        let before = before.values().as_list::<i32>();
        let after = after.values().as_list::<i32>();
        assert_eq!(after.value_offsets(), before.value_offsets());
        assert_eq!(after.nulls(), before.nulls());
        assert_eq!(after.values().nulls(), before.values().nulls());
        assert_eq!(after.values().as_struct().column(0).null_count(), 3);
        Ok(())
    }

    #[test]
    fn alignment_retains_source_ids_and_validates_required_children() -> Result<()> {
        let id = |n: u64| HashMap::from([(PARQUET_FIELD_ID_META_KEY.into(), n.to_string())]);
        let profile = Arc::new(values());
        let input = batch(
            Arc::new(Field::new("actual", profile.data_type().clone(), true).with_metadata(id(7))),
            profile,
        )?;
        let requested = |name: &str, field_id: u64, nullable: bool| {
            Schema::new(vec![
                Field::new_struct(
                    name,
                    vec![Arc::new(Field::new("added", DataType::Utf8, nullable))],
                    true,
                )
                .with_metadata(id(field_id)),
            ])
        };

        let schema = requested("mapped", 7, true);
        let plan = FileProjection::new(&input.schema(), &schema)?;
        assert_eq!(plan.columns(), &["actual"]);
        let output = fixup(&plan, input.clone(), &schema)?;
        assert_eq!(output.schema().field(0).name(), "mapped");
        assert_eq!(output.column(0).nulls(), input.column(0).nulls());

        let schema = requested("actual", 8, true);
        let plan = FileProjection::new(&input.schema(), &schema)?;
        let target = Arc::new((&schema).try_into_kernel()?);
        assert!(fixup_parquet_read(plan.empty_batch()?, &[], None, None, Some(&target)).is_err());

        assert!(FileProjection::new(&input.schema(), &requested("mapped", 7, false)).is_err());
        Ok(())
    }

    #[test]
    fn reads_roots_in_file_order_and_leaves_nested_casts_to_kernel() -> Result<()> {
        let original = Arc::new(values());
        let file = Schema::new(vec![
            Field::new("profile", original.data_type().clone(), true),
            Field::new("ignored", DataType::Utf8, true),
            Field::new("id", DataType::Int64, false),
        ]);
        let requested = Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new(
                "profile",
                struct_type("keep", DataType::Decimal128(20, 0)),
                true,
            ),
        ]);
        let plan = FileProjection::new(&file, &requested)?;
        assert_eq!(plan.columns(), &["profile", "id"]);
        let input = RecordBatch::try_new(
            Arc::new(Schema::new(vec![
                file.field(0).clone(),
                file.field(2).clone(),
            ])),
            vec![original, Arc::new(Int64Array::from(vec![0, 1, 2]))],
        )?;
        let output = fixup(&plan, input, &requested)?;
        assert_eq!(output.schema().as_ref(), &requested);
        let profile = output.column(1).as_struct();
        assert!(profile.is_null(1));
        let kept = profile
            .column(0)
            .as_primitive::<delta_kernel::arrow::datatypes::Decimal128Type>();
        assert_eq!(kept.value(0), 7);
        assert_eq!(kept.value(2), 9);
        Ok(())
    }
}
