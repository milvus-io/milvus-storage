#!/usr/bin/env python3
# Copyright 2026 Zilliz
# SPDX-License-Identifier: Apache-2.0
"""Compare already-built baseline/current Release binaries on one fixed dataset.

Run inside the development container after sourcing conanrun.sh. Raw JSON,
process resource use, binary hashes and paired confidence intervals are retained.
The optional sensitivity is a measurement margin, not a claim of zero overhead.
"""
import argparse
import hashlib
import json
import math
import os
import re
from pathlib import Path
import statistics
import subprocess


def digest(path):
    result = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def summarize(raw, sensitivity):
    # Student t 97.5th percentiles; cap df at 30 conservatively.
    critical = [0, 12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306,
                2.262, 2.228, 2.201, 2.179, 2.160, 2.145, 2.131, 2.120,
                2.110, 2.101, 2.093, 2.086, 2.080, 2.074, 2.069, 2.064,
                2.060, 2.056, 2.052, 2.048, 2.045, 2.042]
    output = {}
    for name in raw[0]["baseline"]:
        output[name] = {}
        for metric in ("real_time", "cpu_time"):
            baseline = [pair["baseline"][name][metric] for pair in raw]
            current = [pair["current"][name][metric] for pair in raw]
            ratios = [math.log(b / a) for a, b in zip(baseline, current)]
            mean = statistics.mean(ratios)
            df = len(ratios) - 1
            margin = critical[min(df, 30)] * statistics.stdev(ratios) / math.sqrt(len(ratios))
            low, high = math.exp(mean - margin), math.exp(mean + margin)
            if low > 1:
                verdict = "detected_regression"
            elif high <= 1 + sensitivity:
                verdict = "within_sensitivity_margin"
            else:
                verdict = "inconclusive"
            output[name][metric] = {
                "baseline_median_ns": statistics.median(baseline),
                "current_median_ns": statistics.median(current),
                "paired_geomean_change_pct": 100 * (math.exp(mean) - 1),
                "paired_95pct_ci_change_pct": [100 * (low - 1), 100 * (high - 1)],
                "verdict": verdict,
            }
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--current", required=True, type=Path)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--pairs", type=int, default=9)
    parser.add_argument("--seconds", type=float, default=0.25)
    parser.add_argument("--cpus", default="8,9")
    parser.add_argument("--warmup", type=float, default=0.25)
    parser.add_argument("--casewise", action="store_true")
    parser.add_argument("--mode", type=int, choices=range(4), default=0)
    parser.add_argument("--filter", default=".")
    parser.add_argument("--sensitivity", type=float, default=0.02)
    parser.add_argument("--require-no-regression", action="store_true",
                        help="fail on a detected regression or an inconclusive wall/CPU result")
    args = parser.parse_args()
    if args.pairs < 2:
        parser.error("at least two pairs are required")
    args.output.mkdir(parents=True, exist_ok=False)
    binaries = {"baseline": args.baseline.resolve(), "current": args.current.resolve()}
    libraries = {key: binary.parent.parent / "libmilvus-storage.so" for key, binary in binaries.items()}
    if not all(path.is_file() for path in libraries.values()):
        parser.error("expected libmilvus-storage.so in each binary's build directory")
    metadata = {"libraries": {k: {"path": str(v), "sha256": digest(v)} for k, v in libraries.items()},
                "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                "binaries": {k: {"path": str(v), "sha256": digest(v)} for k, v in binaries.items()},
                "dataset": {str(p.relative_to(args.data)): digest(p)
                            for p in sorted(args.data.rglob("*")) if p.is_file()}}
    (args.output / "inputs.json").write_text(json.dumps(metadata, indent=2))
    raw = [{"baseline": {}, "current": {}} for _ in range(args.pairs)]
    filters = [args.filter]
    if args.casewise:
        names = subprocess.check_output(
            [str(binaries["baseline"]), "--benchmark_list_tests=true", "--benchmark_filter=" + args.filter],
            text=True).splitlines()
        filters = ["^" + re.escape(name) + "$" for name in names if name]
        if not filters:
            raise RuntimeError("no benchmarks selected")
    for case_index, case_filter in enumerate(filters):
        expected_names = None
        for index in range(args.pairs):
            pair = {}
            # Alternating AB/BA makes elapsed-time drift less likely to favor a version.
            order = ("baseline", "current") if index % 2 == 0 else ("current", "baseline")
            for version in order:
                prefix = args.output / f"{case_index:02d}-{index:02d}-{version}"
                env = dict(os.environ, STORAGE_BENCH_DATA=str(args.data.resolve()),
                           STORAGE_TELEMETRY_MODE=str(args.mode if version == "current" else 0))
                env.pop("STORAGE_BENCH_PREPARE", None)
                command = ["taskset", "-c", args.cpus, str(binaries[version]),
                           "--benchmark_min_time=" + str(args.seconds) + "s",
                           "--benchmark_min_warmup_time=" + str(args.warmup), "--benchmark_filter=" + case_filter,
                           "--benchmark_out=" + str(prefix) + ".json", "--benchmark_out_format=json"]
                print(f"case {case_index + 1}/{len(filters)} pair {index + 1}/{args.pairs}: {version}, mode={env['STORAGE_TELEMETRY_MODE']}", flush=True)
                with open(str(prefix) + ".log", "w") as log:
                    with subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT) as process:
                        _, status, usage = os.wait4(process.pid, 0)
                        process.returncode = os.waitstatus_to_exitcode(status)
                        if process.returncode:
                            raise subprocess.CalledProcessError(process.returncode, command)
                    Path(str(prefix) + ".resources.json").write_text(json.dumps({
                        "max_rss_kb": usage.ru_maxrss, "user_seconds": usage.ru_utime,
                        "system_seconds": usage.ru_stime}, indent=2))
                results = json.loads(Path(str(prefix) + ".json").read_text())["benchmarks"]
                if not results or any(item.get("error_occurred") for item in results):
                    raise RuntimeError(f"failed/empty benchmarks: {prefix}")
                values = {item["name"]: item for item in results if item.get("run_type") == "iteration"}
                if any(item["time_unit"] != "ns" for item in values.values()):
                    raise RuntimeError("expected nanosecond results")
                if expected_names is None:
                    expected_names = set(values)
                if not values or set(values) != expected_names:
                    raise RuntimeError("benchmark sets differ")
                pair[version] = values
            for name in expected_names:
                if pair["baseline"][name].get("rows_per_operation") != pair["current"][name].get("rows_per_operation"):
                    raise RuntimeError("baseline/current returned different row counts")
            for version in pair:
                raw[index][version].update(pair[version])
            (args.output / "pairs.json").write_text(json.dumps(raw, indent=2))
    summary = summarize(raw, args.sensitivity)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
    for name, metrics in summary.items():
        result = metrics["real_time"]
        print(f"{name}: {result['paired_geomean_change_pct']:+.2f}% "
              f"CI={result['paired_95pct_ci_change_pct']} {result['verdict']}")
    if args.require_no_regression and any(
            metric["verdict"] != "within_sensitivity_margin"
            for metrics in summary.values() for metric in metrics.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
