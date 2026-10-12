"""
Data type tests.

Verify write/read roundtrip for various Arrow data types.
"""

import math
import random

import pyarrow as pa
import pytest
from milvus_storage import Reader, Writer


class TestDataTypes:
    """Test data type roundtrips."""

    def _roundtrip(self, path, schema, batch, props):
        """Write a batch and read it back."""
        writer = Writer(path, schema, props)
        writer.write(batch)
        column_groups = writer.close()

        reader = Reader(column_groups, schema, properties=props)
        result = list(reader.scan())
        total_rows = sum(b.num_rows for b in result)
        return result, total_rows

    @pytest.mark.e2e_smoke
    def test_scalar_boundaries_preserve_each_field(self, temp_case_path, default_properties):
        schema = pa.schema(
            [
                pa.field("signed", pa.int32()),
                pa.field("unsigned", pa.uint64()),
                pa.field("float32", pa.float32()),
                pa.field("float64", pa.float64()),
                pa.field("flag", pa.bool_()),
                pa.field("text", pa.string()),
                pa.field("binary", pa.binary()),
            ]
        )
        values = {
            "signed": [-(2**31), -1, 0, 2**31 - 1],
            "unsigned": [0, 2**32, 2**63, 2**64 - 1],
            "float32": [-0.0, float("inf"), float("-inf"), float("nan")],
            "float64": [1.5, -0.0, float("inf"), float("nan")],
            "flag": [True, False, None, True],
            "text": ["中文", "", None, "🎉"],
            "binary": [b"\x00\xff", b"", None, b"abc"],
        }
        batch = pa.RecordBatch.from_pydict(values, schema=schema)
        with Writer(temp_case_path, schema, default_properties) as writer:
            writer.write(batch)
            groups = writer.close()
        with groups, Reader(groups, schema, properties=default_properties) as reader:
            actual = pa.Table.from_batches(list(reader.scan())).to_pydict()

        for name in ("signed", "unsigned", "flag", "text", "binary"):
            assert actual[name] == values[name]
        for name in ("float32", "float64"):
            assert actual[name][:3] == values[name][:3]
            assert math.isnan(actual[name][3])
        assert math.copysign(1.0, actual["float32"][0]) == -1.0
        assert math.copysign(1.0, actual["float64"][1]) == -1.0

    def test_integer_types(
        self,
        temp_case_path: str,
        default_properties,
    ):
        """All integer types (signed and unsigned)."""
        schema = pa.schema(
            [
                pa.field("i8", pa.int8()),
                pa.field("i16", pa.int16()),
                pa.field("i32", pa.int32()),
                pa.field("i64", pa.int64()),
                pa.field("u8", pa.uint8()),
                pa.field("u16", pa.uint16()),
                pa.field("u32", pa.uint32()),
                pa.field("u64", pa.uint64()),
            ]
        )
        batch = pa.RecordBatch.from_pydict(
            {
                "i8": pa.array([-128, 0, 127], type=pa.int8()),
                "i16": pa.array([-32768, 0, 32767], type=pa.int16()),
                "i32": pa.array([-2147483648, 0, 2147483647], type=pa.int32()),
                "i64": pa.array([-(2**63), 0, 2**63 - 1], type=pa.int64()),
                "u8": pa.array([0, 128, 255], type=pa.uint8()),
                "u16": pa.array([0, 32768, 65535], type=pa.uint16()),
                "u32": pa.array([0, 2147483648, 4294967295], type=pa.uint32()),
                "u64": pa.array([0, 2**32, 2**64 - 1], type=pa.uint64()),
            },
            schema=schema,
        )

        result, total = self._roundtrip(
            temp_case_path, schema, batch, default_properties
        )
        assert total == 3

    def test_float_types(
        self,
        temp_case_path: str,
        default_properties,
    ):
        """Float32 and Float64 types."""
        schema = pa.schema(
            [
                pa.field("f32", pa.float32()),
                pa.field("f64", pa.float64()),
            ]
        )
        batch = pa.RecordBatch.from_pydict(
            {
                "f32": pa.array(
                    [0.0, 1.5, -1.5, float("inf"), float("-inf")], type=pa.float32()
                ),
                "f64": pa.array(
                    [0.0, 1.5, -1.5, float("inf"), float("-inf")], type=pa.float64()
                ),
            },
            schema=schema,
        )

        result, total = self._roundtrip(
            temp_case_path, schema, batch, default_properties
        )
        assert total == 5

    def test_boolean_type(
        self,
        temp_case_path: str,
        default_properties,
    ):
        """Boolean type."""
        schema = pa.schema([pa.field("flag", pa.bool_())])
        batch = pa.RecordBatch.from_pydict(
            {
                "flag": [True, False, True, True, False],
            },
            schema=schema,
        )

        result, total = self._roundtrip(
            temp_case_path, schema, batch, default_properties
        )
        assert total == 5

        flags = []
        for b in result:
            flags.extend(b.column("flag").to_pylist())
        assert flags == [True, False, True, True, False]

    def test_string_and_binary_types(
        self,
        temp_case_path: str,
        default_properties,
    ):
        """String and binary types."""
        schema = pa.schema(
            [
                pa.field("text", pa.string()),
                pa.field("data", pa.binary()),
            ]
        )
        batch = pa.RecordBatch.from_pydict(
            {
                "text": ["hello", "", "world", "中文", "🎉"],
                "data": [b"\x00\x01\x02", b"", b"\xff", b"abc", b"\x00"],
            },
            schema=schema,
        )

        result, total = self._roundtrip(
            temp_case_path, schema, batch, default_properties
        )
        assert total == 5

        texts = []
        for b in result:
            texts.extend(b.column("text").to_pylist())
        assert texts == ["hello", "", "world", "中文", "🎉"]

    def test_list_type(
        self,
        temp_case_path: str,
        default_properties,
    ):
        """Variable-length list type."""
        schema = pa.schema(
            [
                pa.field("ids", pa.list_(pa.int32())),
            ]
        )
        batch = pa.RecordBatch.from_pydict(
            {
                "ids": [[1, 2, 3], [], [4], [5, 6]],
            },
            schema=schema,
        )

        result, total = self._roundtrip(
            temp_case_path, schema, batch, default_properties
        )
        assert total == 4

        all_ids = []
        for b in result:
            all_ids.extend(b.column("ids").to_pylist())
        assert all_ids == [[1, 2, 3], [], [4], [5, 6]]

    @pytest.mark.xfail(
        strict=True,
        reason="D14: Parquet fixed_size_list scan exports an invalid Arrow buffer layout",
        raises=pa.ArrowInvalid,
    )
    def test_fixed_size_list_type(
        self,
        temp_case_path: str,
        default_properties,
    ):
        """Fixed-size list type (common for vectors)."""
        dim = 4
        schema = pa.schema(
            [
                pa.field("vector", pa.list_(pa.float32(), dim)),
            ]
        )

        random.seed(123)
        vectors = [[random.random() for _ in range(dim)] for _ in range(10)]
        batch = pa.RecordBatch.from_pydict(
            {
                "vector": vectors,
            },
            schema=schema,
        )

        properties = {**default_properties, "writer.format": "parquet"}
        result, total = self._roundtrip(temp_case_path, schema, batch, properties)
        assert total == 10

    def test_struct_type(
        self,
        temp_case_path: str,
        default_properties,
    ):
        """Struct type."""
        schema = pa.schema(
            [
                pa.field(
                    "meta",
                    pa.struct(
                        [
                            pa.field("key", pa.string()),
                            pa.field("value", pa.int32()),
                        ]
                    ),
                ),
            ]
        )
        batch = pa.RecordBatch.from_pydict(
            {
                "meta": [
                    {"key": "a", "value": 1},
                    {"key": "b", "value": 2},
                    {"key": "c", "value": 3},
                ],
            },
            schema=schema,
        )

        result, total = self._roundtrip(
            temp_case_path, schema, batch, default_properties
        )
        assert total == 3
        assert [value for item in result for value in item.column("meta").to_pylist()] == [
            {"key": "a", "value": 1},
            {"key": "b", "value": 2},
            {"key": "c", "value": 3},
        ]

    def test_wide_table(
        self,
        temp_case_path: str,
        wide_schema: pa.Schema,
        default_properties,
    ):
        """Wide table with 100 float64 columns."""
        num_rows = 200
        data = {
            f"col_{i}": [float(i * num_rows + j) for j in range(num_rows)]
            for i in range(100)
        }
        batch = pa.RecordBatch.from_pydict(data, schema=wide_schema)

        result, total = self._roundtrip(
            temp_case_path, wide_schema, batch, default_properties
        )
        assert total == num_rows

    @pytest.mark.skip(
        reason="Struct + list combo causes segfault in Parquet reader - skipped to prevent crash"
    )
    def test_complex_nested_types(
        self,
        temp_case_path: str,
        complex_schema: pa.Schema,
        default_properties,
    ):
        """Complex nested types (list, struct)."""
        batch = pa.RecordBatch.from_pydict(
            {
                "id": [1, 2, 3],
                "tags": [["a", "b"], ["c"], []],
                "attributes": [
                    {"key": "k1", "value": "v1"},
                    {"key": "k2", "value": "v2"},
                    {"key": "k3", "value": "v3"},
                ],
                "scores": [[0.1, 0.2], [0.3], [0.4, 0.5, 0.6]],
            },
            schema=complex_schema,
        )

        result, total = self._roundtrip(
            temp_case_path, complex_schema, batch, default_properties
        )
        assert total == 3
