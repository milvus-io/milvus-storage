# Writer Policy Local Formats Design

## Context

The writer policy already has a global `writer.format` property that is read by
`ColumnGroupPolicy::create_column_group_policy()` and passed into the concrete
column group policies as the default format. That is not enough for mixed-format
schema-based layouts, and it does not let callers configure a single-policy
format independently from the global fallback.

This change adds policy-local format properties. If a policy-local format is
provided, it wins over `writer.format`. If it is not provided, the existing
global `writer.format` behavior remains unchanged.

## Scope

Add policy-local format support for:

- `single`
- `schema_based`

Do not change `size_based` in this design.

Update external configuration surfaces by adding property key definitions to:

- C/FFI exports
- Python `PropertyKeys`
- Java property constants or key definitions, if present

Java and Python do not need new unit tests for this change. They only need the
new property keys exposed.

## Properties

Add two new property keys:

```cpp
#define PROPERTY_WRITER_SINGLE_FORMAT "writer.split.single.format"
#define PROPERTY_WRITER_SCHEMA_BASE_FORMATS "writer.split.schema_based.formats"
```

`writer.split.single.format`

- Type: `STRING`
- Default: empty string
- Valid non-empty values: existing writer format constants:
  - `LOON_FORMAT_PARQUET`
  - `LOON_FORMAT_VORTEX`
  - `LOON_FORMAT_LANCE_TABLE`
  - `LOON_FORMAT_ICEBERG_TABLE`
- Behavior: if non-empty, this is the format for the single column group.
  Otherwise, use `writer.format`.

`writer.split.schema_based.formats`

- Type: `VECTOR_STR`
- Default: empty vector
- Encoding through FFI/Java/Python properties: comma-separated string, matching
  the existing `VECTOR_STR` conversion behavior.
- Valid values: same existing writer format constants listed above.
- Behavior:
  - Empty vector: every schema-based pattern group uses `writer.format`.
  - Non-empty vector: length must exactly equal
    `writer.split.schema_based.patterns`.
  - Each pattern group uses the format at the same index.
  - The unmatched/default group is not represented in this vector and always
    uses `writer.format`.

## Priority

Format priority is:

1. Policy-local format property
2. Global `writer.format`
3. Existing default `parquet`

For `schema_based`, this priority applies per pattern group. The unmatched group
uses the global fallback because it has no pattern-local entry.

## C++ Design

Register both properties in `cpp/src/properties.cpp`.

`writer.split.single.format` should use a validator that accepts an empty string
or one supported writer format.

`writer.split.schema_based.formats` should use `ValidatePropertyType()` plus a
new vector enum validator that checks every vector element against supported
writer formats. The empty vector is valid.

Update `ColumnGroupPolicy::create_column_group_policy()`:

- Read `writer.format` as the global fallback.
- For `single`, read `writer.split.single.format`; pass the policy-local value
  if non-empty, otherwise pass the global fallback.
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

- Add exported constants:
  - `loon_properties_writer_single_format`
  - `loon_properties_writer_schema_base_formats`
- Add the symbols to export maps.

Python:

- Add `PropertyKeys.WRITER_SINGLE_FORMAT`.
- Add `PropertyKeys.WRITER_SCHEMA_BASE_FORMATS`.
- No Python unit tests are required.

Java:

- Add matching property key definitions if the Java side has a constants layer.
- If Java only passes arbitrary string maps today, no JNI API change is needed.
- No Java unit tests are required.

## Error Handling

Invalid format values fail at property validation.

Schema-based formats length mismatch fails in policy creation with an invalid
status. This failure should include enough context to identify the patterns size
and formats size.

Append commit format incompatibility continues to be enforced by existing
transaction validation, which rejects appended column groups whose format differs
from the corresponding manifest column group.

## Tests

### FFI Tests

Add property key export coverage:

- `loon_properties_writer_single_format` equals
  `writer.split.single.format`.
- `loon_properties_writer_schema_base_formats` equals
  `writer.split.schema_based.formats`.

### Policy And Properties GTests

Add policy-level tests that do not need to write files:

- `SinglePolicyFormatOverridesWriterFormat`
  - Set `writer.format=parquet`.
  - Set `writer.split.single.format=vortex`.
  - Assert the generated single column group's format is `vortex`.

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

Single-policy transaction path:

1. Create a single-policy writer with a policy-local format and commit version 1.
2. Append with the same policy-local format. Commit succeeds.
3. Append with a different policy-local format. Commit fails due to format
   mismatch.

## Non-Goals

- No new per-column-group generic property model.
- No change to `size_based` policy.
- No Java or Python unit tests beyond exposing the new property keys.
- No change to transaction format validation semantics.

