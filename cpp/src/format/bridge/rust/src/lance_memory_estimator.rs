// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Best-effort decoded Arrow buffer size estimation from Lance file metadata.
//!
//! Loading this metadata reads file footers and column/page descriptions only.
//! No data page is read or decompressed.
//!
//! Estimation rules:
//! - A fixed-width leaf, including fixed-size lists, uses its exact logical
//!   value-buffer width.
//! - Every other top-level column is estimated from its physical pages. Each
//!   completed page contributes 8 MiB and the final page is prorated by rows.
//! - Encoded page bytes are only a lower bound, not decoded memory size. A
//!   single-page column uses this lower bound because it has no completed page
//!   from which to infer the decoded page size.
//! - Deletions are applied once using `logical_rows / physical_rows`.
//!
//! Lance normalizes 2.1, 2.2, and 2.3 footers into `ColumnInfo` / `PageInfo`,
//! so this estimator does not contain format-version or encoding-specific logic.

use std::collections::HashMap;
use std::sync::Arc;

use arrow_schema58::DataType;
use lance::dataset::Dataset;
use lance::{Error, Result};
use lance_core::datatypes::{Field, Schema};
use lance_encoding::decoder::{ColumnInfo, PageInfo};
use lance_file::reader::{CachedFileMetadata, FileReader};
use lance_io::scheduler::{ScanScheduler, SchedulerConfig};
use lance_table::format::{DataFile, Fragment};
use object_store::path::Path;

use crate::lance_ffi::LanceColumnMemoryEstimate;

const DATA_DIR: &str = "data";
// Lance's standard writer normally flushes a page after accumulating about
// 8 MiB of Arrow data. This is a heuristic, not a value read from the footer.
const DEFAULT_PAGE_BYTES: u64 = 8 * 1024 * 1024;

fn invalid_input(message: impl Into<String>) -> Error {
    Error::InvalidInput {
        source: message.into().into(),
        location: snafu::location!(),
    }
}

/// Compute `value * numerator / denominator`, rounded to the nearest integer,
/// without overflowing intermediate `u64` multiplication.
fn mul_div_round(value: u64, numerator: u64, denominator: u64) -> u64 {
    if value == 0 || numerator == 0 || denominator == 0 {
        return 0;
    }
    let value = value as u128;
    let numerator = numerator as u128;
    let denominator = denominator as u128;
    let rounded = (value * numerator + denominator / 2) / denominator;
    rounded.min(u64::MAX as u128) as u64
}

/// Number of bytes used by Arrow's bit-packed Boolean value buffer.
fn bitmap_bytes(values: u64) -> u64 {
    values.saturating_add(7) / 8
}

/// Return the value-buffer estimate for a fixed-width Arrow data type.
fn fixed_data_type_memory(data_type: &DataType, rows: u64) -> Option<u64> {
    match data_type {
        DataType::Null => Some(0),
        DataType::Boolean => Some(bitmap_bytes(rows)),
        DataType::FixedSizeBinary(width) => Some((*width as u64).saturating_mul(rows)),
        DataType::FixedSizeList(item, list_size) => fixed_data_type_memory(
            item.data_type(),
            rows.saturating_mul(*list_size as u64),
        ),
        DataType::Dictionary(_, _) => None,
        data_type => data_type
            .primitive_width()
            .map(|width| (width as u64).saturating_mul(rows)),
    }
}

/// Return the value-buffer estimate for a directly fixed-width leaf.
///
/// `None` means the field must use the page heuristic. Structurally nested,
/// variable-width, and dictionary fields take that path. Nullable validity
/// buffers are deliberately omitted because this estimator only targets a
/// coarse value-memory estimate.
fn fixed_column_memory(field: &Field, rows: u64) -> Option<u64> {
    if !field.is_leaf() {
        return None;
    }

    fixed_data_type_memory(&field.data_type(), rows)
}

/// Resolve a fragment data file, including files stored under an external base
/// path rather than the dataset's own `data/` directory.
fn data_file_path(dataset: &Dataset, data_file: &DataFile) -> Result<Path> {
    let data_dir = match data_file.base_id {
        Some(base_id) => {
            let base_path = dataset
                .manifest()
                .base_paths
                .get(&base_id)
                .ok_or_else(|| invalid_input(format!("base path {} not found", base_id)))?;
            let path = base_path.extract_path(dataset.session().store_registry())?;
            if base_path.is_dataset_root {
                path.join(DATA_DIR)
            } else {
                path
            }
        }
        None => dataset.data_dir(),
    };
    Ok(data_dir.join(data_file.path.as_str()))
}

/// Read the Lance footer and its column/page metadata.
///
/// `FileReader::read_all_metadata` reads file-tail metadata only. It does not
/// schedule any data-page reads or decompression.
async fn load_file_metadata(
    dataset: &Dataset,
    data_file: &DataFile,
    default_scheduler: &Arc<ScanScheduler>,
) -> Result<CachedFileMetadata> {
    let path = data_file_path(dataset, data_file)?;
    let scheduler = if data_file.base_id.is_none() {
        default_scheduler.clone()
    } else {
        let object_store = dataset.object_store(data_file.base_id).await?;
        ScanScheduler::new(
            object_store.clone(),
            SchedulerConfig::max_bandwidth(&object_store),
        )
    };
    let file_scheduler = scheduler
        .open_file_with_priority(&path, 0, &data_file.file_size_bytes)
        .await?;

    // FIXME: `FileFragment::get_file_metadata` is not currently a public API.
    // `FileReader::read_all_metadata` is public, but calling it directly cannot
    // reuse Lance's metadata cache and may cause duplicate footer I/O. After
    // https://github.com/lance-format/lance/pull/7820 is available in the Lance
    // version used here, use `get_file_metadata` to reuse cached metadata and
    // remove this extra I/O.
    FileReader::read_all_metadata(&file_scheduler).await
}

/// Sum the encoded buffers referenced by a page.
///
/// This is never interpreted as decoded Arrow size. It is only a conservative
/// floor and the fallback for a column with no completed page.
fn page_disk_bytes(page: &PageInfo) -> u64 {
    page.buffer_offsets_and_sizes
        .iter()
        .map(|(_, size)| *size)
        .sum()
}

/// Estimate one physical non-fixed-width column from its page metadata.
///
/// All pages except the last are treated as completed 8 MiB pages. The last
/// page is prorated using its row count and the average rows per completed page.
/// Encoded bytes remain a floor so the estimate cannot be smaller than the data
/// physically represented by the page buffers.
fn estimate_page_column(column: &ColumnInfo) -> u64 {
    let pages = &column.page_infos;
    let disk_fallback = pages.iter().map(page_disk_bytes).sum::<u64>();
    if pages.len() <= 1 {
        return disk_fallback;
    }

    let full_pages = &pages[..pages.len() - 1];
    let full_rows = full_pages.iter().map(|page| page.num_rows).sum::<u64>();
    if full_rows == 0 {
        return disk_fallback;
    }

    let full_bytes = DEFAULT_PAGE_BYTES.saturating_mul(full_pages.len() as u64);
    let tail_bytes = mul_div_round(full_bytes, pages.last().unwrap().num_rows, full_rows);
    full_bytes.saturating_add(tail_bytes).max(disk_fallback)
}

/// Return the footer column index associated with each schema field id.
///
/// Current Lance manifests persist this mapping in `column_indices`. Older
/// files derive the same mapping from the file schema's pre-order field list.
/// A negative persisted index means the field has no physical file column.
fn file_field_columns(data_file: &DataFile, metadata: &CachedFileMetadata) -> Vec<(i32, u32)> {
    if !data_file.column_indices.is_empty() {
        return data_file
            .fields
            .iter()
            .copied()
            .zip(data_file.column_indices.iter().copied())
            .filter_map(|(field_id, column)| (column >= 0).then_some((field_id, column as u32)))
            .collect();
    }

    // Older files without column_indices use the file schema's pre-order fields.
    metadata
        .file_schema
        .fields_pre_order()
        .enumerate()
        .map(|(column, field)| (field.id, column as u32))
        .collect()
}

/// Record which top-level dataset column owns `field` and all its descendants.
///
/// Lance footer columns for nested Arrow types are identified by physical leaf
/// field ids, while the public estimator returns one result per top-level field.
fn map_field_owner(field: &Field, top_level_index: usize, owners: &mut HashMap<i32, usize>) {
    owners.insert(field.id, top_level_index);
    for child in &field.children {
        map_field_owner(child, top_level_index, owners);
    }
}

/// Build `schema field id -> top-level field index` for grouping physical
/// footer columns into the public per-column result.
fn field_owners(schema: &Schema) -> HashMap<i32, usize> {
    let mut owners = HashMap::new();
    for (index, field) in schema.fields.iter().enumerate() {
        map_field_owner(field, index, &mut owners);
    }
    owners
}

/// Estimate each top-level dataset column in schema order.
///
/// A result is returned for every top-level schema field. A field not present in
/// this fragment, for example after schema evolution, receives a zero estimate.
pub(crate) async fn estimate_fragment_column_memory(
    dataset: &Dataset,
    fragment: &Fragment,
    default_scheduler: Arc<ScanScheduler>,
) -> Result<Vec<LanceColumnMemoryEstimate>> {
    let physical_rows = fragment.physical_rows.ok_or_else(|| {
        invalid_input(format!(
            "fragment {} has no physical row count",
            fragment.id
        ))
    })? as u64;
    let logical_rows = fragment.num_rows().ok_or_else(|| {
        invalid_input(format!("fragment {} has no logical row count", fragment.id))
    })? as u64;
    let schema = dataset.schema();

    // Footer entries for nested columns use descendant field ids. Group those
    // physical columns back into the top-level fields exposed by the API.
    let top_level_by_field_id = field_owners(schema);

    // Fixed-width estimates depend only on the logical row count. `None` marks
    // columns that must be estimated from footer page metadata.
    let fixed_sizes = schema
        .fields
        .iter()
        .map(|field| fixed_column_memory(field, logical_rows))
        .collect::<Vec<_>>();

    // `None` also preserves whether a schema field is absent from this fragment.
    let mut estimates = vec![None; schema.fields.len()];

    for data_file in &fragment.files {
        let metadata = load_file_metadata(dataset, data_file, &default_scheduler).await?;
        for (field_id, column_index) in file_field_columns(data_file, &metadata) {
            let Some(top_level_index) = top_level_by_field_id.get(&field_id).copied() else {
                continue;
            };
            if let Some(bytes) = fixed_sizes[top_level_index] {
                estimates[top_level_index] = Some(bytes);
                continue;
            }

            let Some(column) = metadata.column_infos.get(column_index as usize) else {
                continue;
            };

            // Page metadata describes physical rows. Apply the fragment's live
            // row ratio once to obtain the estimate after deletions.
            let bytes = mul_div_round(
                estimate_page_column(column),
                logical_rows,
                physical_rows,
            );
            let estimate = estimates[top_level_index].get_or_insert(0_u64);
            *estimate = estimate.saturating_add(bytes);
        }
    }

    Ok(schema
        .fields
        .iter()
        .zip(estimates)
        .map(|(field, memory_size)| LanceColumnMemoryEstimate {
            field_id: field.id,
            memory_size: memory_size.unwrap_or(0),
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use lance_encoding::decoder::PageEncoding;
    use lance_encoding::format::pb21::PageLayout;

    fn test_page(rows: u64, disk_bytes: u64) -> PageInfo {
        PageInfo {
            num_rows: rows,
            priority: 0,
            encoding: PageEncoding::Structural(PageLayout::default()),
            buffer_offsets_and_sizes: Arc::from([(0, disk_bytes)]),
        }
    }

    fn test_column(pages: Vec<PageInfo>) -> ColumnInfo {
        ColumnInfo::new(0, pages.into(), vec![], Default::default())
    }

    #[test]
    fn cumulative_scaling_is_stable() {
        assert_eq!(mul_div_round(100, 3, 4), 75);
        assert_eq!(mul_div_round(7, 1, 2), 4);
        assert_eq!(mul_div_round(0, 1, 2), 0);
    }

    #[test]
    fn completed_pages_drive_variable_estimate() {
        let column = test_column(vec![
            test_page(100, 10),
            test_page(100, 10),
            test_page(50, 5),
        ]);

        assert_eq!(estimate_page_column(&column), 20 * 1024 * 1024);
    }

    #[test]
    fn single_page_uses_metadata_floor() {
        let column = test_column(vec![test_page(10, 100)]);

        assert_eq!(estimate_page_column(&column), 100);
    }
}
