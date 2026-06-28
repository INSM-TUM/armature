# compare_log_dependencies.py

Compare activity dependency distributions across multiple XES event logs.

Runs `armature discover` on each log, then builds a table:
- **Columns** — one per log (named by file stem)
- **Rows** — one per dependency type, split into two sections:
  - `temporal` — ordering relationships (direct, eventual, no_ordering, …)
  - `existential` — co-occurrence relationships (implication, equivalence, …)
- **Cells** — fraction of all directed (A, B) pairs in that log assigned this type

### Normalization

For a log with *n* activities, there are *n×(n−1)* directed pairs. Every pair
gets exactly one temporal type and one existential type — pairs absent from the
sparse matrix default to `no_ordering` / `independence`. Each section therefore
sums to 1.0 per log.

### Usage

```bash
python scripts/compare_log_dependencies.py <log1.xes> [log2.xes ...] [OPTIONS]
```

| Option | Default | Description |
|---|---|---|
| `--format` | `table` | Output format: `table`, `csv`, `latex`, `markdown` |
| `--decimals N` | `4` | Decimal places |
| `--no-zeros` | off | Drop rows where all logs show 0.0 |

### Examples

```bash
# Side-by-side table for two logs
python scripts/compare_log_dependencies.py path/to/a.xes path/to/b.xes

# Drop zero rows and print markdown (e.g. for pasting into docs)
python scripts/compare_log_dependencies.py logs/*.xes --no-zeros --format markdown

# Export to CSV for spreadsheet analysis
python scripts/compare_log_dependencies.py logs/*.xes --format csv > dependencies.csv

# Tighter rounding
python scripts/compare_log_dependencies.py logs/*.xes --decimals 2 --no-zeros
```

### Generating sample logs

`generate_sample_logs.py` writes a set of synthetic XES files to disk:

```bash
python scripts/generate_sample_logs.py --output-dir /tmp/sample_logs
python scripts/compare_log_dependencies.py /tmp/sample_logs/*.xes --no-zeros --format markdown
```

Logs generated (subset of the synthetic suite):

| Name | Structure |
|---|---|
| `S1_strict_sequence` | a→b→c→d→e, all traces identical |
| `S2_parallel_block` | a→(b‖c)→d→e, AND-parallel middle |
| `S3_xor_two_paths` | a→b→(c XOR d)→e, mutually exclusive |
| `S5_sese_loop` | a→[b→(c→back\|exit)]→d, single-entry loop |
| `U1_unstructured_3act` | all 6 permutations of 3 activities |
| `U2_unstructured_5act` | all 120 permutations of 5 activities |
