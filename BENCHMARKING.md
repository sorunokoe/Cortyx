# Benchmarking Cortyx

This file documents the benchmark methodology so any retrieval system can be
evaluated on the same fixtures and compared against Cortyx's results.

## Fixtures

Two external benchmark fixtures are used:

| Fixture | Source | Committed path |
|---|---|---|
| LongMemEval-500 | [arXiv:2410.10813](https://arxiv.org/abs/2410.10813), [HuggingFace](https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned) | `tests/fixtures/longmemeval_500.json` |
| LoCoMo sample (200 entries) | [arXiv:2402.17753](https://arxiv.org/abs/2402.17753), [GitHub](https://github.com/snap-research/LoCoMo) | `tests/fixtures/locomo_sample.json` |

The committed fixtures are frozen regression surfaces. To regenerate from the
upstream oracles (required for the headline 96.8% claim, which uses the
cleaned-oracle regeneration):

```bash
python3 scripts/gen_lme500.py   # requires network; stdlib only
python3 scripts/gen_locomo.py   # requires network; stdlib only
```

## Running the Eval Harnesses

Both `eval_lme.py` and `eval_locomo.py` drive any retrieval binary via the
`CORTYX_BIN` environment variable. The binary must implement the same CLI
interface as Cortyx's `cortyx` command.

### With Cortyx

```bash
cargo build --release
CORTYX_BIN=target/release/cortyx python3 scripts/eval_lme.py --fresh-corpus
CORTYX_BIN=target/release/cortyx python3 scripts/eval_locomo.py --fresh-corpus
```

### With a Different System

Set `CORTYX_BIN` to a wrapper script that accepts the same invocations:

```
cortyx compile <dir>                  → index the files in <dir>
cortyx get-contexts <query> [flags]   → retrieve context for a query
```

The eval harnesses invoke these commands for each fixture entry and score
the retrieved text against gold answers using token-level F1, EM, and R@5.

### Scoring Methodology

**R@5 (Recall@5):** Does the expected evidence session appear in the top-5
retrieved results? Measured by keyword presence in the concatenated retrieved
context. This is the primary headline metric for context delivery engines.

**F1 / EM:** Token-level overlap between the retrieved context and the gold
answer string. Useful as a diagnostic but not directly comparable to
end-to-end synthesis F1 from systems that embed an LLM.

**Abstention accuracy:** For `absent` questions (where the answer is not in
history), does the system correctly return no relevant context?

### Proof-Grade Runs

For claim-grade reproducibility, always run with:

```bash
CORTYX_BIN=target/release/cortyx python3 scripts/eval_lme.py --fresh-corpus
```

`--fresh-corpus` bypasses the local index cache so every run starts from the
raw fixture — this ensures retrieval changes are always reflected in results.

## CI Benchmarks

The `Benchmarks` GitHub Actions workflow runs on manual trigger (`workflow_dispatch`) and executes:
1. `cargo test --test bench bench_retrieval_accuracy_500q bench_locomo -- --ignored --nocapture`
2. `eval_lme.py --fresh-corpus` and `eval_locomo.py --fresh-corpus`
3. `benchmark_registry.py scorecard` and `validate`

Results are uploaded as artifacts (90-day retention) and written to the
workflow step summary.

## Submitting External Results

To add your system's results to the comparison scaffold in
`benchmarks/registry.json`:

1. Run the eval harnesses with your system's binary via `CORTYX_BIN`.
2. Record your R@5, F1, and EM results.
3. Open a PR that adds an entry under `comparison_scaffold.competitors` in
   `benchmarks/registry.json` with:
   - `dimension_evidence`: `"same-surface-baseline"` for dimensions where you
     ran the same fixture
   - An entry in the `comparison_scaffold.per_dimension_outcomes` ledger
4. Attach the raw results JSON as a PR artifact or link to a public run.

Only `same-surface-baseline` evidence (same fixture, same scoring harness)
counts toward the scorecard gates. The comparison rules are documented in the
`counting_rules` and `comparison_rules` sections of `benchmarks/registry.json`.

## Proof Matrix and Scorecard

The registry tracks all claim surfaces:

```bash
python3 scripts/benchmark_registry.py matrix      # full proof matrix
python3 scripts/benchmark_registry.py scorecard   # weighted claim gate
python3 scripts/benchmark_registry.py validate    # integrity check
python3 scripts/benchmark_registry.py list --proof-status proven
python3 scripts/benchmark_registry.py show retrieval
```

The `benchmarks/registry.json` file is the source of truth. Every claim in
the README links to a specific `id` in this registry.
