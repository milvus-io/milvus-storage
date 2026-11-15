// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright Zilliz

//! Generate valid DV fixtures using Kernel's writer; no checked-in binary data.

use crate::deltalake_ffi::GeneratedDeltaDv;
use anyhow::Result;
use delta_kernel::actions::deletion_vector_writer::{
    KernelDeletionVector, StreamingDeletionVectorWriter,
};

pub fn delta_test_deletion_vector(positions: &[u64]) -> Result<GeneratedDeltaDv> {
    let mut bytes = Vec::new();
    let mut vector = KernelDeletionVector::new();
    vector.add_deleted_row_indexes(positions.iter().copied());
    let descriptor = {
        let mut writer = StreamingDeletionVectorWriter::new(&mut bytes);
        let descriptor = writer.write_deletion_vector(vector)?;
        writer.finalize()?;
        descriptor
    };
    Ok(GeneratedDeltaDv {
        bytes,
        offset: descriptor.offset,
        size: descriptor.size_in_bytes,
        cardinality: descriptor.cardinality,
    })
}
