# Predicate Pushdown Benchmark Results

## Setup

- **Format**: Vortex 0.56.0
- **Data**: 40,960 rows, schema `{id: int64, name: utf8, value: float64, vector: list<float32>[128]}`
- **Column group**: Single CG (predicate pushdown requires single CG)
- **Threads**: 1
- **Data generation**: Both sorted and unsorted use the **same values**
  (`id = 0..40959`, `name = "name_00000".."name_40959"`, zero-padded).
  Unsorted data is shuffled with a fixed seed. This ensures identical
  predicate selectivity — the only variable is row ordering.
- **Machine**: Apple M-series, 8 cores, local SSD

## Int64 Predicate (`id > threshold`)

| Data | Predicate | Time (ms) | CPU (ms) | I/O (MB) | Selectivity | Rows/s |
|------|-----------|-----------|----------|----------|-------------|--------|
| Unsorted | None | 21.1 | 19.4 | 17.7 | 1.00 | 1.94M |
| Unsorted | 10% | 26.4 | 20.9 | 19.4 | 0.10 | 156K |
| Unsorted | 50% | 46.5 | 36.8 | 19.4 | 0.50 | 451K |
| Unsorted | 90% | 51.4 | 46.0 | 19.4 | 0.90 | 718K |
| Sorted | None | 17.6 | 15.9 | 16.2 | 1.00 | 2.33M |
| Sorted | 10% | **10.7** | **9.1** | **4.8** | 0.10 | 384K |
| Sorted | 50% | 18.8 | 16.1 | 10.5 | 0.50 | 1.09M |
| Sorted | 90% | 25.9 | 21.2 | 17.8 | 0.90 | 1.42M |

## String Predicate (`name > 'name_XXXXX'`)

| Data | Predicate | Time (ms) | CPU (ms) | I/O (MB) | Selectivity | Rows/s |
|------|-----------|-----------|----------|----------|-------------|--------|
| Unsorted | 10% | 27.2 | 22.1 | 19.4 | 0.10 | 150K |
| Unsorted | 50% | 45.1 | 38.2 | 19.4 | 0.50 | 456K |
| Unsorted | 90% | 57.2 | 49.0 | 19.4 | 0.90 | 648K |
| Sorted | 10% | **11.7** | **10.1** | **4.8** | 0.10 | 350K |
| Sorted | 50% | 20.4 | 17.4 | 10.5 | 0.50 | 1.00M |
| Sorted | 90% | 29.4 | 23.5 | 17.8 | 0.90 | 1.26M |

## Analysis

### 1. Zone-map pruning requires sorted data — same values, different order

Since sorted and unsorted use the same values (just reordered), we can
directly measure the impact of data ordering on zone-map pruning:

| Selectivity | Sorted I/O | Unsorted I/O | I/O saved | Sorted time | Unsorted time | Speedup |
|-------------|-----------|-------------|-----------|-------------|---------------|---------|
| 10% (int64) | **4.8 MB** | 19.4 MB | **75%** | **10.7 ms** | 26.4 ms | **2.5x** |
| 50% (int64) | 10.5 MB | 19.4 MB | 46% | 18.8 ms | 46.5 ms | 2.5x |
| 90% (int64) | 17.8 MB | 19.4 MB | 8% | 25.9 ms | 51.4 ms | 2.0x |

For sorted data at 10% selectivity, zone-map pruning eliminates 75% of
I/O. For unsorted data, I/O is constant at 19.4 MB regardless of
selectivity — all zones must be read because every zone spans the full
value range.

### 2. Unsorted data: predicate overhead scales with selectivity

On unsorted data, I/O is constant (19.4 MB, +10% over the 17.7 MB
baseline due to zone-map statistics reads), but wall time increases
with selectivity:

| Selectivity | Time (ms) | Overhead vs no-predicate |
|-------------|-----------|------------------------|
| No predicate | 21.1 | — |
| 10% | 26.4 | +25% |
| 50% | 46.5 | +120% |
| 90% | 51.4 | +144% |

The overhead comes from per-row predicate evaluation. More matching rows
means more output materialization work. Even at 10% selectivity (most rows
filtered), the predicate adds 25% overhead because all rows must be
evaluated even though only 10% are returned.

### 3. Sorted data: predicate speeds up reads at low selectivity

| Selectivity | Time (ms) | vs no-predicate |
|-------------|-----------|-----------------|
| No predicate | 17.6 | — |
| 10% | **10.7** | **39% faster** |
| 50% | 18.8 | 7% slower |
| 90% | 25.9 | 47% slower |

At 10% selectivity, zone-map pruning skips enough zones that the total
work (I/O + evaluation) is less than a full scan. At 50%, the break-even
point: pruning saves some I/O but evaluation overhead offsets it. At 90%,
nearly all zones are read plus evaluation overhead makes it slower.

### 4. String vs int64: comparable I/O, slightly higher CPU

With zero-padded names, zone-map pruning is equally effective for both
types — identical I/O at every selectivity level:

| Selectivity | int64 I/O | string I/O | int64 time | string time | string overhead |
|-------------|----------|-----------|------------|-------------|-----------------|
| Sorted 10% | 4.8 MB | 4.8 MB | 10.7 ms | 11.7 ms | +9% |
| Sorted 50% | 10.5 MB | 10.5 MB | 18.8 ms | 20.4 ms | +9% |
| Sorted 90% | 17.8 MB | 17.8 MB | 25.9 ms | 29.4 ms | +14% |
| Unsorted 10% | 19.4 MB | 19.4 MB | 26.4 ms | 27.2 ms | +3% |
| Unsorted 50% | 19.4 MB | 19.4 MB | 46.5 ms | 45.1 ms | -3% |
| Unsorted 90% | 19.4 MB | 19.4 MB | 51.4 ms | 57.2 ms | +11% |

String predicates are **9-14% slower** than int64 on sorted data due to
string comparison cost. On unsorted data, the difference is smaller (±3-11%)
because I/O overhead dominates.

### 5. CPU utilization

| Scenario | Wall (ms) | CPU (ms) | CPU/Wall | Note |
|----------|-----------|----------|----------|------|
| Sorted, no pred | 17.6 | 15.9 | 0.90 | Baseline |
| Sorted, int64 10% | 10.7 | 9.1 | 0.85 | Less total work |
| Sorted, int64 90% | 25.9 | 21.2 | 0.82 | CPU-bound evaluation |
| Unsorted, no pred | 21.1 | 19.4 | 0.92 | Baseline |
| Unsorted, int64 10% | 26.4 | 20.9 | 0.79 | I/O dominates |
| Unsorted, int64 90% | 51.4 | 46.0 | 0.90 | CPU-heavy evaluation |
| Unsorted, string 90% | 57.2 | 49.0 | 0.86 | String eval costlier |

CPU/Wall ratio is 0.79-0.92 across all scenarios. The workload is
CPU-bound on local SSD. With remote storage, the ratio would drop
significantly as I/O latency dominates.

### 6. I/O reduction vs selectivity (sorted data)

| Selectivity | I/O (MB) | I/O ratio vs full scan | Expected ratio |
|-------------|----------|----------------------|----------------|
| 100% (none) | 16.2 | 1.00x | 1.00x |
| 90% | 17.8 | 1.10x | ~0.90x |
| 50% | 10.5 | 0.65x | ~0.50x |
| 10% | 4.8 | 0.30x | ~0.10x |

I/O reduction is sublinear — at 10% selectivity, actual I/O is 30% of
full scan (not 10%). Zone-map pruning operates at zone granularity:
boundary zones containing a mix of matching/non-matching rows are read
fully. The 90% case reads *more* than baseline because zone-map
statistics overhead (+1.6 MB) exceeds the small I/O savings from
pruning the bottom 10%.

## Recommendations

1. **Sort data by the primary filter column** before writing to Vortex.
   Same data, sorted vs unsorted: 2.5x faster at 10% selectivity,
   75% less I/O. This is the single most impactful optimization.

2. **Predicate pushdown breaks even at ~50% selectivity** on sorted data.
   Below 50%, it's faster than a full scan. Above 50%, the evaluation
   overhead exceeds the I/O savings. Consider skipping pushdown for
   broad predicates.

3. **On unsorted data, pushdown adds 25-144% overhead** with zero I/O
   benefit. Consider a heuristic to detect poor zone-map pruning potential
   (e.g., check if min/max spans the full value range) and skip pushdown.

4. **String predicates perform within 9-14% of int64** when data is sorted
   and strings have consistent lexicographic ordering (zero-padded).
   Zone-map pruning is equally effective for both types.

5. **For remote storage**, the I/O reduction from zone-map pruning will
   have a much larger impact. The 75% I/O reduction at 10% selectivity
   translates to proportional latency and cost reduction.
