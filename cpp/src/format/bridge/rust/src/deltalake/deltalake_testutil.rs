// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright Zilliz

//! Generate valid DV fixtures using Kernel's writer; no checked-in binary data.

use crate::deltalake_ffi::{
    DeltaDvPositions, GeneratedDeltaDv, GeneratedDeltaDvContainer, GeneratedDeltaDvDescriptor,
};
use anyhow::Result;
use delta_kernel::actions::deletion_vector_writer::{
    KernelDeletionVector, StreamingDeletionVectorWriter,
};

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
