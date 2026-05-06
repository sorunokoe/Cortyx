use clap::Parser;
use cortyx::error::Result;
use cortyx::index::NeuronIndex;
use cortyx::mcp::{CortyxServer, GetContextsInput};
use cortyx::miner;
use cortyx::neuron::estimate_tokens;
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Parser, Debug)]
#[command(
    name = "token_bench",
    about = "Benchmark Cortyx token usage across full, retrieval, capsule, delta, and answer modes"
)]
struct Args {
    /// Fixture to sample from.
    #[arg(long, default_value = "tests/fixtures/longmemeval_500.json")]
    fixture: PathBuf,
    /// Number of entries to sample deterministically from the fixture.
    #[arg(long, default_value_t = 20)]
    sample_size: usize,
    /// Maximum retrieval budget passed to Cortyx.
    #[arg(long, default_value_t = 4000)]
    max_tokens: usize,
    /// Minimum retrieval-context savings vs full-history injection.
    #[arg(long)]
    min_retrieval_savings_pct: Option<f64>,
    /// Maximum average retrieval-context tokens.
    #[arg(long)]
    max_retrieval_avg_tokens: Option<usize>,
    /// Minimum capsule+delta repeat savings vs full-history injection.
    #[arg(long)]
    min_delta_repeat_savings_pct: Option<f64>,
    /// Maximum average capsule+delta repeat tokens.
    #[arg(long)]
    max_delta_repeat_avg_tokens: Option<usize>,
    /// Emit structured JSON instead of the markdown table.
    #[arg(long)]
    json: bool,
}

#[derive(Debug, Deserialize, Clone)]
struct FixtureEntry {
    question: String,
    neuron_source_content: String,
    neuron_filename: String,
    expected_keywords: Vec<String>,
}

#[derive(Default)]
struct ModeTotals {
    full_history: usize,
    retrieval: usize,
    capsule: usize,
    capsule_delta_repeat: usize,
    answer_only: usize,
    count: usize,
}

#[derive(Debug, Serialize)]
struct ModeMetrics {
    avg_tokens: usize,
    savings_pct: f64,
}

#[derive(Debug, Serialize)]
struct BenchResults {
    sample_size: usize,
    full_history: ModeMetrics,
    retrieval: ModeMetrics,
    capsule: ModeMetrics,
    delta_repeat: ModeMetrics,
    answer_only: ModeMetrics,
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();
    let entries = load_entries(&args.fixture, args.sample_size)?;
    let workspace = make_temp_dir("cortyx-token-bench-project")?;
    let staging = make_temp_dir("cortyx-token-bench-staging")?;

    let result = run_bench(&workspace, &staging, &entries, args.max_tokens).await;

    let _ = fs::remove_dir_all(&staging);
    let _ = fs::remove_dir_all(&workspace);

    let results = result?;
    if args.json {
        println!("{}", serde_json::to_string_pretty(&results)?);
    } else {
        print_results(&results);
    }
    print_guardrail_status(&args, &results)?;
    Ok(())
}

async fn run_bench(
    workspace: &Path,
    staging: &Path,
    entries: &[FixtureEntry],
    max_tokens: usize,
) -> Result<BenchResults> {
    for entry in entries {
        fs::write(
            staging.join(&entry.neuron_filename),
            &entry.neuron_source_content,
        )?;
    }

    let mut idx = NeuronIndex::load_or_create(workspace)?;
    miner::mine_path(staging, workspace, &mut idx, None)?;
    let server = CortyxServer::for_benchmark(workspace.to_path_buf(), idx);

    let mut totals = ModeTotals::default();
    for entry in entries {
        let retrieval = server
            .benchmark_get_contexts(base_input(entry, max_tokens))
            .await;
        let capsule = server
            .benchmark_get_contexts(GetContextsInput {
                capsule_mode: Some(true),
                ..base_input(entry, max_tokens)
            })
            .await;
        let delta_seed = server
            .benchmark_get_contexts(GetContextsInput {
                capsule_mode: Some(true),
                delta_mode: Some(true),
                ..base_input(entry, max_tokens)
            })
            .await;
        let delta_handle = extract_context_handle(&delta_seed);
        let capsule_delta_repeat = server
            .benchmark_get_contexts(GetContextsInput {
                capsule_mode: Some(true),
                delta_mode: Some(true),
                context_handle: delta_handle,
                ..base_input(entry, max_tokens)
            })
            .await;
        let answer_only = server
            .benchmark_get_contexts(GetContextsInput {
                answer_mode: Some(true),
                ..base_input(entry, max_tokens)
            })
            .await;

        totals.full_history += estimate_tokens(&entry.neuron_source_content).get();
        totals.retrieval += estimate_tokens(&retrieval).get();
        totals.capsule += estimate_tokens(&capsule).get();
        totals.capsule_delta_repeat += estimate_tokens(&capsule_delta_repeat).get();
        totals.answer_only += estimate_tokens(&answer_only).get();
        totals.count += 1;
    }

    Ok(build_results(&totals))
}

fn load_entries(path: &Path, sample_size: usize) -> Result<Vec<FixtureEntry>> {
    let raw = fs::read_to_string(path)?;
    let mut entries: Vec<FixtureEntry> = serde_json::from_str(&raw)?;
    entries.retain(|entry| !entry.expected_keywords.is_empty());
    if sample_size > 0 && entries.len() > sample_size {
        entries.truncate(sample_size);
    }
    Ok(entries)
}

fn make_temp_dir(prefix: &str) -> Result<PathBuf> {
    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    let dir = std::env::temp_dir().join(format!("{prefix}-{stamp}"));
    fs::create_dir_all(&dir)?;
    Ok(dir)
}

fn base_input(entry: &FixtureEntry, max_tokens: usize) -> GetContextsInput {
    GetContextsInput {
        task: entry.question.clone(),
        max_tokens: Some(max_tokens),
        module: None,
        person: None,
        kind: Some("conversation".to_string()),
        min_confidence: None,
        multi_hop: None,
        previous_response: None,
        open_files: None,
        error_context: None,
        delta_mode: None,
        context_handle: None,
        capsule_mode: None,
        answer_mode: None,
        min_answer_confidence: None,
        provenance_mode: None,
    }
}

fn extract_context_handle(output: &str) -> Option<String> {
    let handle = Regex::new(r"<!-- Context handle: ([^ ]+) -->").ok()?;
    handle
        .captures(output)
        .and_then(|caps| caps.get(1))
        .map(|m| m.as_str().to_string())
}

fn build_results(totals: &ModeTotals) -> BenchResults {
    let full_total = totals.full_history;
    BenchResults {
        sample_size: totals.count,
        full_history: ModeMetrics {
            avg_tokens: average(totals.full_history, totals.count),
            savings_pct: 0.0,
        },
        retrieval: ModeMetrics {
            avg_tokens: average(totals.retrieval, totals.count),
            savings_pct: savings_pct(totals.retrieval, full_total),
        },
        capsule: ModeMetrics {
            avg_tokens: average(totals.capsule, totals.count),
            savings_pct: savings_pct(totals.capsule, full_total),
        },
        delta_repeat: ModeMetrics {
            avg_tokens: average(totals.capsule_delta_repeat, totals.count),
            savings_pct: savings_pct(totals.capsule_delta_repeat, full_total),
        },
        answer_only: ModeMetrics {
            avg_tokens: average(totals.answer_only, totals.count),
            savings_pct: savings_pct(totals.answer_only, full_total),
        },
    }
}

fn print_results(results: &BenchResults) {
    println!(
        "| Mode | Avg tokens | Savings vs full |\n|---|---:|---:|\n| Full history | {} | — |\n| Retrieval context | {} | {} |\n| Capsule mode | {} | {} |\n| Capsule + delta repeat | {} | {} |\n| Answer only | {} | {} |",
        results.full_history.avg_tokens,
        results.retrieval.avg_tokens,
        format_pct(results.retrieval.savings_pct),
        results.capsule.avg_tokens,
        format_pct(results.capsule.savings_pct),
        results.delta_repeat.avg_tokens,
        format_pct(results.delta_repeat.savings_pct),
        results.answer_only.avg_tokens,
        format_pct(results.answer_only.savings_pct),
    );
}

fn print_guardrail_status(args: &Args, results: &BenchResults) -> Result<()> {
    let mut guarded = false;
    if let Some(min_pct) = args.min_retrieval_savings_pct {
        guarded = true;
        cortyx::cortyx_ensure!(
            results.retrieval.savings_pct >= min_pct,
            "Retrieval savings must be ≥{min_pct:.1}%; got {:.1}%",
            results.retrieval.savings_pct
        );
        if !args.json {
            println!(
                "[guardrail] retrieval savings {:.1}% ≥ {:.1}%",
                results.retrieval.savings_pct, min_pct
            );
        }
    }
    if let Some(max_tokens) = args.max_retrieval_avg_tokens {
        guarded = true;
        cortyx::cortyx_ensure!(
            results.retrieval.avg_tokens <= max_tokens,
            "Retrieval average tokens must be ≤{max_tokens}; got {}",
            results.retrieval.avg_tokens
        );
        if !args.json {
            println!(
                "[guardrail] retrieval avg tokens {} ≤ {}",
                results.retrieval.avg_tokens, max_tokens
            );
        }
    }
    if let Some(min_pct) = args.min_delta_repeat_savings_pct {
        guarded = true;
        cortyx::cortyx_ensure!(
            results.delta_repeat.savings_pct >= min_pct,
            "Capsule+delta repeat savings must be ≥{min_pct:.1}%; got {:.1}%",
            results.delta_repeat.savings_pct
        );
        if !args.json {
            println!(
                "[guardrail] capsule+delta repeat savings {:.1}% ≥ {:.1}%",
                results.delta_repeat.savings_pct, min_pct
            );
        }
    }
    if let Some(max_tokens) = args.max_delta_repeat_avg_tokens {
        guarded = true;
        cortyx::cortyx_ensure!(
            results.delta_repeat.avg_tokens <= max_tokens,
            "Capsule+delta repeat average tokens must be ≤{max_tokens}; got {}",
            results.delta_repeat.avg_tokens
        );
        if !args.json {
            println!(
                "[guardrail] capsule+delta repeat avg tokens {} ≤ {}",
                results.delta_repeat.avg_tokens, max_tokens
            );
        }
    }
    if guarded && !args.json {
        println!("[guardrail] token budgets green");
    }
    Ok(())
}

fn average(total: usize, count: usize) -> usize {
    if count == 0 {
        0
    } else {
        total / count
    }
}

fn savings_pct(mode_total: usize, full_total: usize) -> f64 {
    if full_total == 0 {
        return 0.0;
    }
    (1.0 - mode_total as f64 / full_total as f64) * 100.0
}

fn format_pct(value: f64) -> String {
    format!("{value:.1}%")
}
