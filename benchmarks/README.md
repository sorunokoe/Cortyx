# Cortyx Benchmark Fixtures

This directory contains fixtures for the 50-question benchmark.

## Structure

```
benchmarks/
├── fixtures/          ← 50-question Q&A pairs for accuracy measurement
├── bench_activation/  ← hyperfine scripts for latency benchmarking
└── results/           ← Benchmark result tables (auto-generated)
```

## Running Benchmarks

### Activation latency (requires hyperfine)
```bash
cargo build --release
hyperfine 'target/release/cortyx status' --warmup 5
```

### Accuracy benchmark
```bash
cargo test --test bench -- --nocapture
```

### Token savings vs raw RAG
```bash
cargo test --test bench token_savings -- --nocapture
```
