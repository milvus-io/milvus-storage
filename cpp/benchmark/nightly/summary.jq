def target_parts:
  ((.run_name // .name // "") | tostring | split("/")) as $parts
  | if ($parts | length) < 5 or $parts[0] != "NIGHTLY_CI_TARGET"
    then {valid: false, raw_name: (.run_name // .name // "")}
    else {
      valid: true,
      mode: $parts[1],
      operation: $parts[2],
      format: $parts[3],
      dataset: $parts[4]
    }
    end;

def to_ms:
  if (.real_time | type) != "number" then null
  elif .time_unit == "ns" then .real_time / 1000000
  elif .time_unit == "us" then .real_time / 1000
  elif .time_unit == "ms" then .real_time
  elif .time_unit == "s" then .real_time * 1000
  else null
  end;

def markdown_escape:
  tostring | gsub("\\r?\\n"; "<br>") | gsub("\\|"; "\\|");

def fixed($digits):
  . as $number
  | pow(10; $digits) as $scale
  | (($number * $scale) | round) as $scaled
  | (if $scaled < 0 then "-" else "" end) as $sign
  | ($scaled | fabs) as $absolute
  | (($absolute / $scale) | floor | tostring) as $whole
  | (($absolute % $scale) | floor | tostring) as $fraction
  | $sign + $whole + "." + (("000000000000" + $fraction) | .[(-$digits):]);

def benchmark_entries:
  if (.benchmarks? | type) == "array" then .benchmarks else [] end;

def median_cv_shaped_entries:
  [benchmark_entries[] | select(.aggregate_name? == "median" or .aggregate_name? == "cv")];

def aggregate_entries:
  [median_cv_shaped_entries[] | select(.run_type? == "aggregate")];

def target_key:
  "\(.mode)/\(.operation)/\(.format)/\(.dataset)";

def analyzed_records:
  [aggregate_entries[]
   | . as $record
   | ($record | target_parts) as $target
   | ($record | to_ms) as $normalized_ms
   | $record + {
       target: $target,
       normalized_ms: $normalized_ms,
       raw_name: (($record.run_name // $record.name // "") | tostring),
       semantic_key: (if $target.valid then ($target | target_key) else null end)
     }];

def raw_cases:
  analyzed_records
  | sort_by(.raw_name)
  | group_by(.raw_name)
  | map(
      . as $records
      | [$records[] | select(.aggregate_name == "median")] as $medians
      | [$records[] | select(.aggregate_name == "cv")] as $cvs
      | {
          raw_name: $records[0].raw_name,
          target: $records[0].target,
          semantic_key: $records[0].semantic_key,
          medians: $medians,
          cvs: $cvs,
          median: ($medians[0] // null),
          cv: ($cvs[0] // null)
        }
    );

def cases:
  [raw_cases[] | select(.target.valid)]
  | sort_by(.semantic_key)
  | group_by(.semantic_key)
  | map(. as $raw_runs | $raw_runs[0] + {raw_cases: $raw_runs});

def approved_target:
  . as $target
  | $target.valid
    and (
      if $target.mode == "Sync"
      then (["RecordBatchRead", "ChunkRead", "Take"] | index($target.operation) != null)
           and (["Parquet", "Vortex", "Lance"] | index($target.format) != null)
      elif $target.mode == "Async"
      then (["ChunkRead", "Take"] | index($target.operation) != null)
           and (["Parquet", "Vortex"] | index($target.format) != null)
      else false
      end
    )
    and ([
      "SyntheticSmall", "SyntheticMedium", "SyntheticLarge", "ScalarMedium",
      "RandomVector64MiB", "LowEntropyVector256MiB", "RandomVector2GiB"
    ] | index($target.dataset) != null);

def benchmark_errors:
  [benchmark_entries[]
   | select(.error_occurred? == true)
   | {
       name: (.run_name // .name // "<unknown>"),
       message: (.error_message // "benchmark reported an error")
     }];

def completed_cases:
  cases as $cases
  | benchmark_errors as $benchmark_errors
  | [$cases[]
     | . as $case
     | select(
         ($case.target | approved_target)
         and ($case.raw_cases | length) == 1
         and ($case.medians | length) == 1
         and ($case.cvs | length) == 1
         and $case.median.normalized_ms != null
         and $case.cv.normalized_ms != null
         and ([$benchmark_errors[]
               | ({run_name: .name} | target_parts) as $error_target
               | select($error_target.valid
                        and ($error_target | target_key) == $case.semantic_key)]
              | length) == 0
       )];

def expected_count_value:
  if ($expected_count | test("^[0-9]+$"))
  then ($expected_count | tonumber)
  else null
  end;

def validation_issues:
  analyzed_records as $records
  | median_cv_shaped_entries as $median_cv_records
  | raw_cases as $raw_cases
  | cases as $cases
  | benchmark_errors as $benchmark_errors
  | expected_count_value as $expected
  | [
      if (.benchmarks? | type) != "array" then "missing benchmarks array" else empty end,
      if $expected == null then "expected_count is not a number" else empty end,
      ($median_cv_records[]
       | select(.run_type? != "aggregate")
       | "median/CV-shaped record must use run_type aggregate for \((.run_name // .name // "<unknown>") | markdown_escape), found \((.run_type // "<missing>") | tostring | markdown_escape)"),
      ($records[]
       | select(.target.valid | not)
       | "malformed target name: \((.target.raw_name // "<unknown>") | markdown_escape)"),
      ($raw_cases[]
       | select(.target.valid == true and (.target | approved_target | not))
       | "unapproved semantic target: \(.semantic_key | markdown_escape)"),
      ($records[]
       | select(.normalized_ms == null)
       | "unsupported or missing time unit for \((.run_name // .name // "<unknown>") | markdown_escape)"),
      ($raw_cases[]
       | select((.medians | length) != 1)
       | "expected exactly one median for raw run \(.raw_name | markdown_escape), found \(.medians | length)"),
      ($raw_cases[]
       | select((.cvs | length) != 1)
       | "expected exactly one real-time CV for raw run \(.raw_name | markdown_escape), found \(.cvs | length)"),
      ($cases[]
       | select((.raw_cases | length) != 1)
       | "multiple raw runs map to semantic target \(.semantic_key | markdown_escape): \(.raw_cases | map(.raw_name | markdown_escape) | join(", "))"),
      ($benchmark_errors[]
       | "benchmark error for \(.name | markdown_escape): \(.message | markdown_escape)"),
      if $expected != null and ($cases | length) != $expected
      then "expected \($expected) unique semantic targets, found \($cases | length)"
      else empty
      end
    ];

def median_ms:
  if .median == null then null else .median.normalized_ms end;

def cv_percent:
  if .cv == null or (.cv.real_time | type) != "number" then null else .cv.real_time * 100 end;

def throughput_text:
  if .median == null then "—"
  else [
    .median
    | to_entries[]
    | select(.key == "bytes_per_second" or .key == "rows/s" or (.key | startswith("throughput(")))
    | "\(.key): \(.value)"
  ] | if length == 0 then "—" else join("<br>") end
  end;

def parquet_median($all_cases):
  . as $case
  | [$all_cases[]
     | select(
         .target.mode == $case.target.mode
         and .target.operation == $case.target.operation
         and .target.dataset == $case.target.dataset
         and .target.format == "Parquet"
       )
     | median_ms]
  | .[0] // null;

def speedup_text($all_cases):
  median_ms as $median
  | parquet_median($all_cases) as $parquet
  | if $median == null or $parquet == null or $median == 0
    then "—"
    else "\(($parquet / $median) | fixed(3))x"
    end;

def time_text:
  median_ms as $value
  | if $value == null then "—" else "\($value | fixed(3)) ms" end;

def cv_text:
  cv_percent as $value
  | if $value == null then "—"
    elif $value > 10 then "WARN \($value | fixed(2))%"
    else "\($value | fixed(2))%"
    end;

def format_cell($format; $all_cases):
  ([.[] | select(.target.format == $format)] | .[0] // null) as $case
  | if $case == null then "—"
    else "\($case | time_text) (\($case | speedup_text($all_cases)))"
    end;

def comparison_table($mode; $all_cases):
  [$all_cases[] | select(.target.mode == $mode)]
  | sort_by([.target.dataset, .target.operation])
  | group_by([.target.dataset, .target.operation])
  | if $mode == "Async"
    then ["| Dataset | Operation | Parquet | Vortex |",
          "| --- | --- | ---: | ---: |"]
         + map(
             . as $group
             | "| \($group[0].target.dataset | markdown_escape) | \($group[0].target.operation | markdown_escape) | \($group | format_cell("Parquet"; $all_cases) | markdown_escape) | \($group | format_cell("Vortex"; $all_cases) | markdown_escape) |"
           )
    else ["| Dataset | Operation | Parquet | Vortex | Lance |",
          "| --- | --- | ---: | ---: | ---: |"]
         + map(
             . as $group
             | "| \($group[0].target.dataset | markdown_escape) | \($group[0].target.operation | markdown_escape) | \($group | format_cell("Parquet"; $all_cases) | markdown_escape) | \($group | format_cell("Vortex"; $all_cases) | markdown_escape) | \($group | format_cell("Lance"; $all_cases) | markdown_escape) |"
           )
    end
  | join("\n");

def render_markdown:
  cases as $cases
  | completed_cases as $completed_cases
  | validation_issues as $issues
  | expected_count_value as $expected
  | [ $cases[] | cv_percent | select(. != null) ] as $cvs
  | [
      "## C++ Nightly Reader Benchmark",
      "",
      "Commit: `\($commit | markdown_escape)`  ",
      "Trigger: `\($trigger | markdown_escape)`  ",
      "MinIO release: `\($minio_release | markdown_escape)`",
      "",
      "Cases: \($completed_cases | length) completed / \(if $expected == null then "invalid" else $expected end) expected  ",
      "Errors: \($issues | length)  ",
      "Max real-time CV: \(if ($cvs | length) == 0 then "—" else (($cvs | max) as $max | (if $max > 10 then "WARN " else "" end) + ($max | fixed(2)) + "%") end)  ",
      "MinIO CPUs: `\($minio_cpu_list | markdown_escape)`  ",
      "Benchmark CPUs: `\($benchmark_cpu_list | markdown_escape)`  ",
      "Executor threads: `\($executor_threads | markdown_escape)`",
      "",
      "### Sync comparison",
      "",
      comparison_table("Sync"; $cases),
      "",
      "### Async comparison",
      "",
      comparison_table("Async"; $cases),
      "",
      "### Case details",
      "",
      "| Mode | Operation | Format | Dataset | Median real time | Real-time CV | Throughput | Parquet speedup |",
      "| --- | --- | --- | --- | ---: | ---: | --- | ---: |"
    ]
    + ($cases | sort_by([.target.mode, .target.dataset, .target.operation, .target.format]) | map(
        "| \(.target.mode | markdown_escape) | \(.target.operation | markdown_escape) | \(.target.format | markdown_escape) | \(.target.dataset | markdown_escape) | \(time_text | markdown_escape) | \(cv_text | markdown_escape) | \(throughput_text | markdown_escape) | \(speedup_text($cases) | markdown_escape) |"
      ))
    + (if ($issues | length) == 0 then [] else [
        "",
        "### Benchmark errors",
        "",
        ($issues | map("- " + .) | join("\n"))
      ] end)
  | join("\n");

def validate_results:
  validation_issues as $issues
  | if ($issues | length) == 0 then "nightly summary validation passed"
    else error("nightly summary validation failed: \($issues | join("; "))")
    end;

if $command == "render" then render_markdown
elif $command == "validate" then validate_results
else error("unknown nightly summary command")
end
