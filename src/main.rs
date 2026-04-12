mod ast_extractor;
mod cli;
mod embedder;
mod export;
mod import_parser;
mod index;
mod mcp;
mod miner;
mod neuron;
mod watcher;

use anyhow::Result;
use clap::Parser;
use std::path::{Path, PathBuf};
use tracing_subscriber::{EnvFilter, fmt};

use cli::{Cli, Commands};

/// Resolve an optional path argument to a canonical project root.
///
/// Falls back to `.` when no path is given; silently uses the non-canonical
/// path if `canonicalize` fails (e.g. directory does not yet exist).
fn project_root(path: Option<PathBuf>) -> PathBuf {
    let p = path.unwrap_or_else(|| PathBuf::from("."));
    p.canonicalize().unwrap_or(p)
}

#[tokio::main]
async fn main() -> Result<()> {
    fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .with_target(false)
        .init();

    let cli = Cli::parse();

    match cli.command {
        Commands::Serve { project } => {
            mcp::serve(project).await?;
        }
        Commands::Compile { path } => {
            let root = project_root(path);
            let mut idx = index::NeuronIndex::load_or_create(&root)?;
            let count = idx.compile()?;
            println!("✓ Compiled {count} neurons in {}", root.display());
            println!("  Next: call cortyx_evolve_context to fill stubs, or `cortyx serve` to start the MCP server.");
        }
        Commands::Status { path } => {
            let idx = index::NeuronIndex::load_or_create(&project_root(path))?;
            idx.print_status();
        }
        Commands::Invalidate { file } => {
            let root = project_root(None);
            let mut idx = index::NeuronIndex::load_or_create(&root)?;
            idx.invalidate(&file)?;
            println!("✓ Marked {} as stale", file.display());
        }
        Commands::Export { provider, output, path } => {
            let root = project_root(path);
            let idx = index::NeuronIndex::load_or_create(&root)?;
            let json = export::build_prompt_json(&root, &idx, provider)?;
            match output {
                Some(out) => {
                    std::fs::write(&out, &json)?;
                    println!("✓ Prompt JSON written to {}", out.display());
                }
                None => println!("{json}"),
            }
        }
        Commands::Mine { path, module } => {
            let root = project_root(None);
            let mut idx = index::NeuronIndex::load_or_create(&root)?;
            let count = miner::mine_path(&path, &root, &mut idx, module.as_deref())?;
            println!("✓ Mined {count} Verbatim neurons from {}", path.display());
        }
        Commands::Watch { path } => {
            let root = project_root(path);
            let idx = index::NeuronIndex::load_or_create(&root)?;
            let index = std::sync::Arc::new(tokio::sync::RwLock::new(idx));
            let _w = watcher::start_watcher(root.clone(), std::sync::Arc::clone(&index))?;
            println!("✓ Watching {} for changes. Press Ctrl+C to stop.", root.display());
            tokio::signal::ctrl_c().await?;
        }
        Commands::Doctor { path } => {
            let root = project_root(path);
            let code = run_doctor(&root);
            std::process::exit(code);
        }
    }

    Ok(())
}

/// Print a health report for the Cortyx installation at `root`.
///
/// Returns 0 if healthy, 1 if any errors found.
fn run_doctor(root: &Path) -> i32 {
    let sep = "─".repeat(60);
    println!("Cortyx Doctor — {}", root.display());
    println!("{sep}");

    let mut errors = 0u32;
    let mut warnings = 0u32;

    // 1. .cortyx/ directory
    let cortyx_dir = root.join(".cortyx");
    if cortyx_dir.exists() {
        println!("[✅] .cortyx/ directory exists");
    } else {
        println!("[❌] .cortyx/ directory not found — run `cortyx compile` to initialize");
        errors += 1;
    }

    // 2. index.json
    let index_path = cortyx_dir.join("index.json");
    if index_path.exists() {
        match index::NeuronIndex::load_or_create(root) {
            Ok(idx) => {
                let total = idx.neuron_count();
                let synapses = idx.synapse_count();
                println!("[✅] index.json valid ({total} neurons, {synapses} synapses)");

                // 3. Neuron status breakdown — use loaded index (no 2nd disk scan)
                let (fresh, stale, stub) = idx.status_counts();
                if stale > 0 {
                    println!("[⚠️] Neurons: {fresh} fresh, {stale} stale, {stub} stubs");
                    warnings += 1;
                } else {
                    println!("[✅] Neurons: {fresh} fresh, {stale} stale, {stub} stubs");
                }
            }
            Err(e) => {
                println!("[❌] index.json parse error: {e}");
                errors += 1;
            }
        }
    } else {
        println!("[⚠️] index.json not found — run `cortyx compile` to create it");
        warnings += 1;
    }

    // 4. Git availability
    let git_ok = std::process::Command::new("git")
        .args(["rev-parse", "--short", "HEAD"])
        .current_dir(root)
        .output()
        .ok()
        .filter(|o| o.status.success());
    if let Some(out) = git_ok {
        let sha = String::from_utf8_lossy(&out.stdout).trim().to_string();
        println!("[✅] Git available (HEAD: {sha})");
    } else {
        println!("[⚠️] Git not available or not a git repository (confidence scores will be 1.0)");
        warnings += 1;
    }

    // 5. Binary size
    if let Ok(exe) = std::env::current_exe() {
        if let Ok(meta) = std::fs::metadata(&exe) {
            let mb = meta.len() as f64 / 1_048_576.0;
            if mb <= 8.0 {
                println!("[✅] Binary size: {mb:.1} MB (limit: 8 MB)");
            } else {
                println!("[⚠️] Binary size: {mb:.1} MB (exceeds 8 MB limit)");
                warnings += 1;
            }
        }
    }

    println!("{sep}");
    if errors == 0 && warnings == 0 {
        println!("Summary: all checks passed ✓");
        0
    } else {
        println!("Summary: {errors} error(s), {warnings} warning(s)");
        if errors > 0 { 1 } else { 0 }
    }
}
