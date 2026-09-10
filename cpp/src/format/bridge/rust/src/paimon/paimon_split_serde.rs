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

//! Persistence boundary for Paimon `DataSplit`.
//!
//! The payload uses paimon-rust's Java-compatible `SplitSerializer` frame.
//! The outer Milvus metadata selects the codec; the payload owns its wire
//! version, magic, and split type.

use paimon::DataSplit;
use paimon::spec::POSTPONE_BUCKET;

pub(crate) const NATIVE_CODEC: &str = "paimon-split";
pub(crate) const CURRENT_CODEC: &str = NATIVE_CODEC;

pub(crate) const MAX_DESCRIPTOR_BYTES: usize = 8 * 1024 * 1024;
pub(crate) const MAX_DESCRIPTOR_DATA_FILES: usize = 4096;
pub(crate) const MAX_DESCRIPTOR_ROW_RANGES: usize = 65536;

#[derive(Debug)]
pub(crate) struct SplitSerdeError(String);

impl SplitSerdeError {
    fn new(message: impl Into<String>) -> Self {
        Self(message.into())
    }
}

impl std::fmt::Display for SplitSerdeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for SplitSerdeError {}

pub(crate) struct EncodedDataSplit {
    pub(crate) codec: &'static str,
    pub(crate) payload: Vec<u8>,
}

/// Validate fields that Milvus relies on before the split reaches storage I/O.
pub(crate) fn validate_split(split: &DataSplit) -> Result<(), SplitSerdeError> {
    if split.snapshot_id() <= 0 {
        return Err(SplitSerdeError::new(format!(
            "Paimon split descriptor pins non-positive snapshot id {}",
            split.snapshot_id()
        )));
    }
    if split.bucket() < POSTPONE_BUCKET {
        return Err(SplitSerdeError::new(format!(
            "Paimon split descriptor has invalid bucket {}",
            split.bucket()
        )));
    }
    if split.bucket_path().is_empty() {
        return Err(SplitSerdeError::new(
            "Paimon split descriptor has an empty bucket path",
        ));
    }
    if split.total_buckets() < POSTPONE_BUCKET {
        return Err(SplitSerdeError::new(format!(
            "Paimon split descriptor has invalid total bucket count {}",
            split.total_buckets()
        )));
    }
    if split.total_buckets() > 0 && (split.bucket() < 0 || split.bucket() >= split.total_buckets())
    {
        return Err(SplitSerdeError::new(format!(
            "Paimon split descriptor bucket {} is outside [0, {})",
            split.bucket(),
            split.total_buckets()
        )));
    }

    let files = split.data_files();
    if files.is_empty() {
        return Err(SplitSerdeError::new(
            "Paimon split descriptor contains no data files",
        ));
    }
    if files.len() > MAX_DESCRIPTOR_DATA_FILES {
        return Err(SplitSerdeError::new(format!(
            "Paimon split descriptor references {} data files, above the {} limit",
            files.len(),
            MAX_DESCRIPTOR_DATA_FILES
        )));
    }

    let mut total_rows = 0i64;
    for file in files {
        if file.file_name.is_empty() {
            return Err(SplitSerdeError::new(
                "Paimon split descriptor contains a data file without a name",
            ));
        }
        if file.row_count < 0 {
            return Err(SplitSerdeError::new(format!(
                "Paimon data file '{}' has a negative row count: {}",
                file.file_name, file.row_count
            )));
        }
        if file.file_size < 0 {
            return Err(SplitSerdeError::new(format!(
                "Paimon data file '{}' has a negative file size: {}",
                file.file_name, file.file_size
            )));
        }
        total_rows = total_rows.checked_add(file.row_count).ok_or_else(|| {
            SplitSerdeError::new("Paimon split descriptor total row count overflows int64")
        })?;
        if let Some(first_row_id) = file.first_row_id {
            if first_row_id < 0 || file.row_count == 0 {
                return Err(SplitSerdeError::new(format!(
                    "Paimon data file '{}' has invalid row-id range start {} for {} rows",
                    file.file_name, first_row_id, file.row_count
                )));
            }
            if first_row_id.checked_add(file.row_count - 1).is_none() {
                return Err(SplitSerdeError::new(format!(
                    "Paimon data file '{}' row-id range overflows int64",
                    file.file_name
                )));
            }
        }
        if let Some(count) = file.delete_row_count {
            if count < 0 || count > file.row_count {
                return Err(SplitSerdeError::new(format!(
                    "Paimon data file '{}' has delete row count {} outside [0, {}]",
                    file.file_name, count, file.row_count
                )));
            }
        }
    }

    if let Some(deletion_files) = split.data_deletion_files() {
        if deletion_files.len() != files.len() {
            return Err(SplitSerdeError::new(format!(
                "Paimon split descriptor has {} deletion-file slots for {} data files",
                deletion_files.len(),
                files.len()
            )));
        }
        for (index, slot) in deletion_files.iter().enumerate() {
            if let Some(deletion_file) = slot {
                if deletion_file.path().is_empty()
                    || deletion_file.offset() < 0
                    || deletion_file.length() <= 0
                {
                    return Err(SplitSerdeError::new(format!(
                        "Paimon split descriptor deletion slot {index} has an invalid region"
                    )));
                }
                if let Some(cardinality) = deletion_file.cardinality() {
                    if cardinality < 0 || cardinality > files[index].row_count {
                        return Err(SplitSerdeError::new(format!(
                            "Paimon split descriptor deletion slot {index} has cardinality \
                             {cardinality} outside [0, {}]",
                            files[index].row_count
                        )));
                    }
                }
            }
        }
    }

    if let Some(ranges) = split.row_ranges() {
        if ranges.len() > MAX_DESCRIPTOR_ROW_RANGES {
            return Err(SplitSerdeError::new(format!(
                "Paimon split descriptor carries {} row ranges, above the {} limit",
                ranges.len(),
                MAX_DESCRIPTOR_ROW_RANGES
            )));
        }
        for range in ranges {
            if range.from() < 0
                || range.to() < range.from()
                || range
                    .to()
                    .checked_sub(range.from())
                    .and_then(|span| span.checked_add(1))
                    .is_none()
            {
                return Err(SplitSerdeError::new(format!(
                    "Paimon split descriptor has an invalid row range [{}, {}]",
                    range.from(),
                    range.to()
                )));
            }
        }
    }
    Ok(())
}

/// Drop planner-only file statistics before persisting the split.
// TODO: Replace this serde round-trip when paimon-rust exposes a stats API.
fn split_without_stats(split: &DataSplit) -> Result<DataSplit, SplitSerdeError> {
    let mut value = serde_json::to_value(split)
        .map_err(|error| SplitSerdeError::new(format!("failed to copy Paimon split: {error}")))?;
    let empty_stats = serde_json::json!({
        "_MIN_VALUES": [],
        "_MAX_VALUES": [],
        "_NULL_COUNTS": []
    });
    if let Some(files) = value
        .get_mut("data_files")
        .and_then(serde_json::Value::as_array_mut)
    {
        for file in files {
            let Some(fields) = file.as_object_mut() else {
                continue;
            };
            for key in ["_KEY_STATS", "_VALUE_STATS"] {
                if fields.contains_key(key) {
                    fields.insert(key.to_string(), empty_stats.clone());
                }
            }
            if fields.contains_key("_VALUE_STATS_COLS") {
                fields.insert("_VALUE_STATS_COLS".to_string(), serde_json::Value::Null);
            }
        }
    }
    serde_json::from_value(value).map_err(|error| {
        SplitSerdeError::new(format!("failed to strip Paimon split stats: {error}"))
    })
}

fn serialize_native(split: &DataSplit) -> Result<Vec<u8>, SplitSerdeError> {
    let payload = split.serialize_split_v1().map_err(|error| {
        SplitSerdeError::new(format!("failed to serialize Paimon split: {error}"))
    })?;
    if payload.len() > MAX_DESCRIPTOR_BYTES {
        return Err(SplitSerdeError::new(format!(
            "serialized Paimon split descriptor is {} bytes, above the {} limit",
            payload.len(),
            MAX_DESCRIPTOR_BYTES
        )));
    }
    Ok(payload)
}

fn deserialize_native(payload: &[u8]) -> Result<DataSplit, SplitSerdeError> {
    if payload.is_empty() {
        return Err(SplitSerdeError::new(
            "Paimon split descriptor must not be empty",
        ));
    }
    if payload.len() > MAX_DESCRIPTOR_BYTES {
        return Err(SplitSerdeError::new(format!(
            "Paimon split descriptor is {} bytes, above the {} limit",
            payload.len(),
            MAX_DESCRIPTOR_BYTES
        )));
    }
    let split = DataSplit::deserialize_split_v1(payload).map_err(|error| {
        SplitSerdeError::new(format!(
            "invalid Paimon split descriptor ({error}); refresh the external table"
        ))
    })?;
    validate_split(&split)?;
    Ok(split)
}

pub(crate) fn serialize(split: &DataSplit) -> Result<EncodedDataSplit, SplitSerdeError> {
    validate_split(split)?;
    Ok(EncodedDataSplit {
        codec: CURRENT_CODEC,
        payload: serialize_native(&split_without_stats(split)?)?,
    })
}

pub(crate) fn deserialize(codec: &str, payload: &[u8]) -> Result<DataSplit, SplitSerdeError> {
    match codec {
        NATIVE_CODEC => deserialize_native(payload),
        codec => Err(SplitSerdeError::new(format!(
            "unsupported Paimon split codec '{codec}'"
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use paimon::spec::{BinaryRow, DataFileMeta};
    use paimon::{DataSplitBuilder, DeletionFile, RowRange};

    fn data_file(name: &str, rows: i64) -> DataFileMeta {
        serde_json::from_value(serde_json::json!({
            "_FILE_NAME": name,
            "_FILE_SIZE": 128,
            "_ROW_COUNT": rows,
            "_MIN_KEY": [],
            "_MAX_KEY": [],
            "_KEY_STATS": {"_MIN_VALUES": [], "_MAX_VALUES": [], "_NULL_COUNTS": []},
            "_VALUE_STATS": {"_MIN_VALUES": [], "_MAX_VALUES": [], "_NULL_COUNTS": []},
            "_MIN_SEQUENCE_NUMBER": 0,
            "_MAX_SEQUENCE_NUMBER": 0,
            "_SCHEMA_ID": 0,
            "_LEVEL": 1,
            "_EXTRA_FILES": [],
            "_CREATION_TIME": null,
            "_DELETE_ROW_COUNT": null,
            "_EMBEDDED_FILE_INDEX": null
        }))
        .unwrap()
    }

    fn test_split(partition: BinaryRow) -> DataSplit {
        DataSplitBuilder::new()
            .with_snapshot(5)
            .with_partition(partition)
            .with_bucket(0)
            .with_bucket_path("file:/warehouse/db.db/tbl/bucket-0".to_string())
            .with_total_buckets(1)
            .with_data_files(vec![data_file("data-native.parquet", 20)])
            .with_data_deletion_files(vec![Some(DeletionFile::new(
                "file:/warehouse/db.db/tbl/index/index-dv-native".to_string(),
                4,
                24,
                Some(2),
            ))])
            .with_row_ranges(vec![RowRange::new(2, 7)])
            .with_raw_convertible(false)
            .build()
            .unwrap()
    }

    fn native_test_split() -> DataSplit {
        test_split(BinaryRow::from_bytes(
            0,
            vec![0; BinaryRow::cal_fix_part_size_in_bytes(0) as usize],
        ))
    }

    #[test]
    fn native_round_trip_preserves_split_fields() {
        let encoded = serialize(&native_test_split()).unwrap();
        assert_eq!(encoded.codec, NATIVE_CODEC);
        let decoded = deserialize(encoded.codec, &encoded.payload).unwrap();
        assert_eq!(decoded.snapshot_id(), 5);
        assert_eq!(decoded.row_ranges().unwrap(), &[RowRange::new(2, 7)]);
        let deletion = decoded.deletion_file_for_data_file_index(0).unwrap();
        assert_eq!((deletion.offset(), deletion.length()), (4, 24));
    }

    #[test]
    fn native_round_trip_supports_empty_partition() {
        let encoded = serialize(&test_split(BinaryRow::new(0))).unwrap();
        let decoded = deserialize(encoded.codec, &encoded.payload).unwrap();
        assert_eq!(decoded.partition().arity(), 0);
        assert_eq!(
            decoded.partition().to_serialized_bytes().len(),
            std::mem::size_of::<i32>() + BinaryRow::cal_fix_part_size_in_bytes(0) as usize
        );
    }

    #[test]
    fn native_decoder_rejects_truncation_and_trailing_bytes() {
        let encoded = serialize(&native_test_split()).unwrap();
        let truncated = &encoded.payload[..encoded.payload.len() - 1];
        assert!(deserialize(encoded.codec, truncated).is_err());

        let mut trailing = encoded.payload;
        trailing.push(0);
        assert!(deserialize(NATIVE_CODEC, &trailing).is_err());
    }

    #[test]
    fn stats_are_not_persisted() {
        let mut value = serde_json::to_value(native_test_split()).unwrap();
        let filler: Vec<u8> = (0..4096).map(|byte| (byte % 251) as u8).collect();
        let stats = serde_json::json!({
            "_MIN_VALUES": filler,
            "_MAX_VALUES": filler,
            "_NULL_COUNTS": [0, 1, 0],
        });
        value["data_files"][0]["_KEY_STATS"] = stats.clone();
        value["data_files"][0]["_VALUE_STATS"] = stats;
        value["data_files"][0]["_VALUE_STATS_COLS"] = serde_json::json!(["id", "name", "value"]);
        let split: DataSplit = serde_json::from_value(value).unwrap();

        let with_stats = serialize_native(&split).unwrap();
        let encoded = serialize(&split).unwrap();
        assert!(encoded.payload.len() * 4 < with_stats.len());

        let decoded = deserialize(encoded.codec, &encoded.payload).unwrap();
        let decoded = serde_json::to_value(decoded).unwrap();
        assert_eq!(
            decoded["data_files"][0]["_KEY_STATS"]["_MIN_VALUES"],
            serde_json::json!([])
        );
        assert_eq!(
            decoded["data_files"][0]["_VALUE_STATS_COLS"],
            serde_json::Value::Null
        );
    }

    #[test]
    fn codec_and_size_limits_are_enforced() {
        let payload = serialize(&native_test_split()).unwrap().payload;
        assert!(
            deserialize("unknown", &payload)
                .unwrap_err()
                .to_string()
                .contains("unsupported")
        );

        let oversized = vec![0; MAX_DESCRIPTOR_BYTES + 1];
        assert!(
            deserialize(NATIVE_CODEC, &oversized)
                .unwrap_err()
                .to_string()
                .contains("limit")
        );
    }

    #[test]
    fn structural_validation_rejects_invalid_file_and_deletion_metadata() {
        let mut value = serde_json::to_value(native_test_split()).unwrap();
        value["data_files"][0]["_ROW_COUNT"] = serde_json::json!(-1);
        let split: DataSplit = serde_json::from_value(value).unwrap();
        assert!(
            validate_split(&split)
                .unwrap_err()
                .to_string()
                .contains("negative row count")
        );

        let error = DataSplitBuilder::new()
            .with_snapshot(5)
            .with_partition(BinaryRow::new(0))
            .with_bucket(0)
            .with_bucket_path("file:/warehouse/db.db/tbl/bucket-0".to_string())
            .with_total_buckets(1)
            .with_data_files(vec![data_file("data.parquet", 10)])
            .with_data_deletion_files(vec![None, None])
            .build()
            .unwrap_err();
        assert!(error.to_string().contains("must match data_files length"));
    }

    #[test]
    fn structural_validation_accepts_known_bucket_sentinels() {
        let split = DataSplitBuilder::new()
            .with_snapshot(5)
            .with_partition(BinaryRow::new(0))
            .with_bucket(POSTPONE_BUCKET)
            .with_bucket_path("file:/warehouse/db.db/tbl".to_string())
            .with_total_buckets(POSTPONE_BUCKET)
            .with_data_files(vec![data_file("data.parquet", 10)])
            .build()
            .unwrap();
        validate_split(&split).unwrap();
    }
}
