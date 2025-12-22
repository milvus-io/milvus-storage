// Copyright 2023 Zilliz
//
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

use std::sync::LazyLock;

pub static RT: LazyLock<tokio::runtime::Runtime> = LazyLock::new(|| {
    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .expect("Failed to create tokio runtime")
});

pub mod bridgeimpl;
pub use bridgeimpl::*;

#[cxx::bridge(namespace = "milvus_storage::lance::ffi")]
mod ffi {
    extern "Rust" {
        type BlockingDataset;
        pub fn open_dataset(uri: &str) -> Result<Box<BlockingDataset>>;
        pub unsafe fn write_dataset(uri: &str, stream_ptr: *mut u8)
        -> Result<Box<BlockingDataset>>;

        pub unsafe fn write_stream(self: &mut BlockingDataset, stream_ptr: *mut u8) -> Result<()>;
        pub fn get_all_fragment_ids(self: &BlockingDataset) -> Vec<u64>;

        type BlockingFragmentReader;
        pub unsafe fn open_fragment_reader(
            dataset: &BlockingDataset,
            fragment_id: u64,
            schema_rawptr: *mut u8,
        ) -> Result<Box<BlockingFragmentReader>>;

        // BlockingFragmentReader functions
        pub fn number_of_rows(self: &BlockingFragmentReader) -> Result<u64>;
        pub unsafe fn take_as_single_batch(
            self: &BlockingFragmentReader,
            indices: &[u32],
            out_array: *mut u8,
        ) -> Result<()>;

        pub unsafe fn take_as_stream(
            self: &BlockingFragmentReader,
            indices: &[u32],
            batch_size: u32,
            out_stream: *mut u8,
        ) -> Result<()>;

        pub unsafe fn read_all_as_stream(
            self: &BlockingFragmentReader,
            batch_size: u32,
            out_stream: *mut u8,
        ) -> Result<()>;

        pub unsafe fn read_ranges_as_stream(
            self: &BlockingFragmentReader,
            row_range_start: u32,
            row_range_end: u32,
            batch_size: u32,
            out_stream: *mut u8,
        ) -> Result<()>;
    }
}
