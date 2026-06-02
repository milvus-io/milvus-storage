# Writer Policy Local Formats Design

## Context

The writer policy already has a global `writer.format` property. It is read by
`ColumnGroupPolicy::create_column_group_policy()` and passed into concrete
column group policies as the default format.

That global format is sufficient for `single` and `size_based`, where every
generated column group should share one format. It is not sufficient for
schema-based layouts that intentionally split columns into named pattern groups
and need different formats per matched group.

This design adds policy-local format support only for `schema_based`.
`single` continues to use `writer.format` directly. There is no
`writer.split.single.format` public property, alias, or deprecation path.

## Scope

Add policy-local format support for:

- `schema_based`

Do not add policy-local format support for:

- `single`
- `size_based`

Update external configuration surfaces by adding the schema-based format property
key definition to:

- C/FFI exports
- Python `PropertyKeys`
- Java property constants or key definitions, if present

Java and Python do not need new unit tests for this change. They only need the
new schema-based property key exposed.

## Properties

Add one new property key:

```cpp
#define PROPERTY_WRITER_SCHEMA_BASE_FORMATS "writer.split.schema_based.formats"
```

`writer.split.schema_based.formats`

- Type: `VECTOR_STR`
- Default: empty vector
- Encoding through FFI/Java/Python properties: comma-separated string, matching
  the existing `VECTOR_STR` conversion behavior.
- Valid values: existing writer format constants:
  - `LOON_FORMAT_PARQUET`
  - `LOON_FORMAT_VORTEX`
  - `LOON_FORMAT_LANCE_TABLE`
  - `LOON_FORMAT_ICEBERG_TABLE`
- Behavior:
  - Empty vector: every schema-based pattern group uses `writer.format`.
  - Non-empty vector: length must exactly equal
    `writer.split.schema_based.patterns`.
  - Each pattern group uses the format at the same index.
  - The unmatched/default group is not represented in this vector and always
    uses `writer.format`.

## Priority

For `schema_based` pattern groups, format priority is:

1. `writer.split.schema_based.formats` entry at the matching pattern index
2. Global `writer.format`
3. Existing default value for `writer.format` (`parquet`)

The unmatched/default schema-based group uses the global fallback because it has
no pattern-local entry.

For `single`, format priority is:

1. Global `writer.format`
2. Existing default value for `writer.format` (`parquet`)

## C++ Design

Register `PROPERTY_WRITER_SCHEMA_BASE_FORMATS` in `cpp/src/properties.cpp`.

`writer.split.schema_based.formats` should use `ValidatePropertyType()` plus a
vector enum validator that checks every vector element against supported writer
formats. The empty vector is valid.

Update `ColumnGroupPolicy::create_column_group_policy()`:

- Read `writer.format` as the global fallback.
- For `single`, construct `SingleColumnGroupPolicy` with the global fallback.
- For `schema_based`, read `writer.split.schema_based.patterns` and
  `writer.split.schema_based.formats`.
- If schema-based formats is non-empty and its length differs from patterns,
  return `arrow::Status::Invalid`.
- Construct `SchemaBasedColumnGroupPolicy` with patterns, formats, and the
  global fallback.

Update `SchemaBasedColumnGroupPolicy`:

- Store `column_name_formats_` alongside `column_name_patterns_`.
- When a field matches pattern index `j`, set the column group's format to
  `column_name_formats_[j]` when formats are provided, otherwise
  `default_format_`.
- Unmatched fields continue to use `default_format_`.

## FFI, Python, And Java Surface

FFI:

- Add exported constant `loon_properties_writer_schema_base_formats`.
- Add the symbol to Linux and macOS export maps.
- Do not export `loon_properties_writer_single_format`.

Python:

- Add `PropertyKeys.WRITER_SCHEMA_BASE_FORMATS`.
- Do not add `PropertyKeys.WRITER_SINGLE_FORMAT`.
- No Python unit tests are required.

Java:

- Add matching schema-based property key definitions if the Java side has a
  constants layer.
- Do not add `WRITER_SINGLE_FORMAT`.
- If Java only passes arbitrary string maps today, no JNI API change is needed.
- No Java unit tests are required.

## Error Handling

Invalid schema-based format values fail at property validation.

Schema-based formats length mismatch fails in policy creation with an invalid
status. This failure should include enough context to identify the patterns size
and formats size.

Append commit format incompatibility continues to be enforced by existing
transaction validation, which rejects appended column groups whose format differs
from the corresponding manifest column group.

## Tests

### FFI Tests

Add property key export coverage:

- `loon_properties_writer_schema_base_formats` equals
  `writer.split.schema_based.formats`.

Do not add FFI coverage for `writer.split.single.format`, because that property
does not exist.

### Policy And Properties GTests

Add policy-level tests that do not need to write files:

- `SinglePolicyFallsBackToWriterFormat`
  - Set only `writer.format=vortex`.
  - Assert the generated single column group's format is `vortex`.

- `SchemaBasedFormatsOverrideWriterFormatAndUnmatchedFallsBack`
  - Use a schema with fields such as `id`, `name`, `value`, `vector`.
  - Set `writer.format=parquet`.
  - Set patterns to `id|value,vector`.
  - Set formats to `vortex,parquet`.
  - Assert matched group 0 uses `vortex`.
  - Assert matched group 1 uses `parquet`.
  - Assert unmatched/default group containing `name` uses global `parquet`.

- `SchemaBasedFallsBackToWriterFormatWhenFormatsEmpty`
  - Set `writer.format=vortex`.
  - Set schema-based patterns and no schema-based formats.
  - Assert every generated column group uses `vortex`.

- `SchemaBasedFormatsLengthMismatchFails`
  - Set two patterns and one format.
  - Assert `create_column_group_policy()` returns Invalid.

- `SchemaBasedInvalidFormatFailsValidation`
  - Set formats to include an unsupported value.
  - Assert property validation fails before policy creation, or policy creation
    fails if the existing test helper builds properties directly.

### Transaction GTests

Add real write and manifest commit coverage.

Schema-based mixed-format transaction path:

1. Create a schema-based writer with multiple column groups using different
   formats, write one batch, and commit as manifest version 1.
2. Try to append another batch using formats that differ from the manifest.
   Commit must fail because transaction append validation sees a format mismatch.
3. Append using the same formats as version 1. Commit succeeds and writes
   version 2.
4. Add a new column group with `AddColumnGroup`. Commit succeeds and writes
   version 3.
5. Append again using formats that match all version 3 column groups. Commit
   succeeds and writes version 4.
6. Read manifest version 4 and verify:
   - version is 4
   - column group count is correct
   - columns are assigned to the expected groups
   - each column group's format matches the expected format
   - appended files were added to the expected groups

## Non-Goals

- No new per-column-group generic property model.
- No change to `single` policy format configuration.
- No change to `size_based` policy.
- No Java or Python unit tests beyond exposing the schema-based property key.
- No deprecation path for `writer.split.single.format`.
- No change to transaction format validation semantics.
