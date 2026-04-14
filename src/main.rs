mod alias_gen;
mod ast_extractor;
mod cli;
mod embedder;
mod export;
mod git_extractor;
mod global_index;
mod import_parser;
mod index;
mod installer;
mod kg;
mod mcp;
mod miner;
mod neuron;
mod reranker;
mod watcher;

use anyhow::Result;
use clap::Parser;
use std::path::{Path, PathBuf};
use tracing_subscriber::{EnvFilter, fmt};

use cli::{Cli, Commands, ConceptsCommand};

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
        Commands::Compile { path, incremental } => {
            let root = project_root(path);
            let mut idx = index::NeuronIndex::load_or_create(&root)?;
            let count = if incremental {
                let n = idx.compile_dirty()?;
                println!("✓ Incremental compile: {n} neurons updated in {}", root.display());
                n
            } else {
                let n = idx.compile()?;
                println!("✓ Compiled {n} neurons in {}", root.display());
                n
            };
            let _ = count;
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
        Commands::Doctor { path, json } => {
            let root = project_root(path);
            let code = run_doctor(&root, json);
            std::process::exit(code);
        }
        Commands::Prune { path, min_use, older_than, dry_run } => {
            let root = project_root(path);
            let removed = run_prune(&root, min_use, older_than, dry_run)?;
            if dry_run {
                println!("Dry run — {} neuron(s) would be removed (re-run without --dry-run to delete)", removed);
            } else {
                println!("✓ Pruned {removed} neuron(s) from {}", root.display());
            }
        }
        Commands::GetContexts { task, max_tokens, module, kind, min_confidence, multi_hop, path } => {
            let root = project_root(path);
            let idx = index::NeuronIndex::load_or_create(&root)?;
            let min_conf = min_confidence.map(|v| v as f32);
            let (included, _overflow) = idx.get_contexts_with_overflow(
                &task,
                max_tokens,
                module.as_deref(),
                kind.as_deref(),
                min_conf,
                multi_hop,
            );
            for neuron_path in &included {
                if let Ok(content) = std::fs::read_to_string(neuron_path) {
                    println!("=== {} ===", neuron_path.display());
                    println!("{content}");
                }
            }
            if included.is_empty() {
                if min_confidence.is_some() {
                    println!("(no neurons matched — confidence below threshold)");
                } else {
                    println!("(no neurons matched)");
                }
            }
        }
        Commands::Rollback { neuron } => {
            // E1: Git-based neuron versioning — restore previous commit
            let output = std::process::Command::new("git")
                .args(["checkout", "HEAD~1", "--", &neuron.to_string_lossy()])
                .output()?;
            if output.status.success() {
                println!("✓ Rolled back {} to HEAD~1", neuron.display());
            } else {
                let stderr = String::from_utf8_lossy(&output.stderr);
                anyhow::bail!("git rollback failed: {stderr}");
            }
        }
        Commands::RollbackSection { neuron, section } => {
            // E2: Section shadow copy — restore from sidecar shadow_sections
            run_rollback_section(&neuron, &section)?;
        }
        Commands::PublishConcept { neuron } => {
            // D1: Global concept layer — publish neuron to ~/.cortyx/global/
            let root = project_root(None);
            let mut idx = global_index::GlobalIndex::load();
            match idx.publish(&neuron, &root) {
                Ok(dest) => println!("✓ Published concept to {}", dest.display()),
                Err(e) => anyhow::bail!("publish-concept failed: {e}"),
            }
        }
        Commands::ListConcepts => {
            // D1: List all global concepts
            let concepts = global_index::list_global_concepts();
            if concepts.is_empty() {
                println!("No global concepts published yet. Use `cortyx publish-concept <neuron>` to add one.");
            } else {
                println!("Global concepts ({} total):", concepts.len());
                for (path, project) in &concepts {
                    println!("  {} [from {}]", path.display(), project);
                }
            }
        }
        Commands::Install { global } => {
            // S1+S3: Auto-configure LLM clients + write hook scripts.
            match installer::run_install(global) {
                Ok(summary) => println!("{summary}"),
                Err(e) => {
                    eprintln!("cortyx install failed: {e}");
                    std::process::exit(1);
                }
            }
        }
        Commands::CloseTaskHook { project } => {
            let root = project_root(project);
            match index::NeuronIndex::load_or_create(&root) {
                Ok(idx) => println!("close-task-hook: index OK ({} neurons).", idx.neuron_count()),
                Err(e) => eprintln!("close-task-hook: could not load index: {e}"),
            }
        }
        Commands::Concepts(sub) => {
            run_concepts(sub)?;
        }
    }

    Ok(())
}

/// Remove unused or outdated neurons from the index.
///
/// Criteria (OR-combined): use_count < min_use, or neuron file older than `older_than` days.
/// `min_use = 0` means "never activated" — newly-compiled stubs with zero activations.
///
/// Returns the count of neurons removed (or that would be removed in dry-run mode).
fn run_prune(root: &Path, min_use: u32, older_than: Option<u64>, dry_run: bool) -> Result<usize> {
    use std::time::{Duration, SystemTime};

    let mut idx = index::NeuronIndex::load_or_create(root)?;
    let now = SystemTime::now();
    let age_cutoff: Option<SystemTime> = older_than.map(|days| {
        now.checked_sub(Duration::from_secs(days * 86_400))
            .unwrap_or(SystemTime::UNIX_EPOCH)
    });

    let candidates: Vec<std::path::PathBuf> = idx
        .neuron_paths_and_use_counts()
        .into_iter()
        .filter(|(path, use_count)| {
            let too_cold = *use_count <= min_use;
            let too_old = age_cutoff.map_or(false, |cutoff| {
                std::fs::metadata(path)
                    .and_then(|m| m.modified())
                    .map(|mtime| mtime < cutoff)
                    .unwrap_or(false)
            });
            too_cold || too_old
        })
        .map(|(path, _)| path)
        .collect();

    let count = candidates.len();

    if dry_run {
        for p in &candidates {
            println!("  would remove: {}", p.display());
        }
        return Ok(count);
    }

    for path in &candidates {
        idx.evict_entry(path);
        // Remove the .context.md file and its sidecar .json
        let _ = std::fs::remove_file(path);
        let sidecar = crate::neuron::meta_path(path);
        let _ = std::fs::remove_file(sidecar);
    }

    if count > 0 {
        // One single-pass rebuild after all evictions — O(n) not O(n²)
        idx.rebuild_derived_pub();
        idx.save()?;
    }

    Ok(count)
}


/// E2 (TRIZ R14): Restore a single neuron section from its shadow copy in the sidecar JSON.
///
/// Before each evolve_context or evolve_section call, Cortyx saves the previous content
/// to `meta.shadow_sections[key]`. This function reads that shadow and writes it back.
fn run_rollback_section(neuron: &Path, section: &str) -> Result<()> {
    use crate::neuron::{meta_path, NeuronMeta, atomic_write, replace_section};

    let meta_p = meta_path(neuron);
    let data = std::fs::read_to_string(&meta_p)
        .map_err(|e| anyhow::anyhow!("Cannot read sidecar for {}: {e}", neuron.display()))?;
    let meta: NeuronMeta = serde_json::from_str(&data)
        .map_err(|e| anyhow::anyhow!("Cannot parse sidecar: {e}"))?;

    let shadow = meta.shadow_sections.get(section).ok_or_else(|| {
        anyhow::anyhow!(
            "No shadow for section '{}' in {}. Shadows are saved before each evolve call.",
            section,
            neuron.display()
        )
    })?;

    if section == "_full" {
        atomic_write(neuron, shadow.as_bytes())?;
        println!("✓ Restored full neuron {} from shadow.", neuron.display());
    } else {
        let current = std::fs::read_to_string(neuron)
            .map_err(|e| anyhow::anyhow!("Cannot read neuron file: {e}"))?;
        let restored = replace_section(&current, section, shadow);
        atomic_write(neuron, restored.as_bytes())?;
        println!("✓ Restored section '{}' in {} from shadow.", section, neuron.display());
    }
    Ok(())
}

/// Returns 0 if healthy, 1 if any errors found.
fn run_doctor(root: &Path, json_output: bool) -> i32 {
    let sep = "─".repeat(60);

    let mut errors = 0u32;
    let mut warnings = 0u32;
    let mut total_neurons = 0usize;
    let mut stale_count = 0usize;
    let mut synapse_count = 0usize;
    let mut low_quality_count = 0usize;
    let mut index_valid = false;

    // 1. .cortyx/ directory
    let cortyx_dir = root.join(".cortyx");
    let cortyx_exists = cortyx_dir.exists();
    if !json_output {
        if cortyx_exists {
            println!("[✅] .cortyx/ directory exists");
        } else {
            println!("Cortyx Doctor — {}", root.display());
            println!("{sep}");
            println!("[❌] .cortyx/ directory not found — run `cortyx compile` to initialize");
            errors += 1;
        }
    } else if !cortyx_exists {
        errors += 1;
    }

    if !json_output {
        println!("Cortyx Doctor — {}", root.display());
        println!("{sep}");
    }

    // 2. index.json
    let index_path = cortyx_dir.join("index.json");
    if index_path.exists() {
        match index::NeuronIndex::load_or_create(root) {
            Ok(idx) => {
                total_neurons = idx.neuron_count();
                synapse_count = idx.synapse_count();
                low_quality_count = idx.low_quality_count();
                index_valid = true;
                let (fresh, stale, stub) = idx.status_counts();
                stale_count = stale;
                if !json_output {
                    if stale > 0 {
                        println!("[⚠️] Neurons: {fresh} fresh, {stale} stale, {stub} stubs");
                        warnings += 1;
                    } else {
                        println!("[✅] Neurons: {fresh} fresh, {stale} stale, {stub} stubs");
                    }
                    println!("[✅] index.json valid ({total_neurons} neurons, {synapse_count} synapses)");
                    if low_quality_count > 0 {
                        println!("[⚠️] Quality: {low_quality_count} neurons below 40% quality (run cortyx evolve)");
                        warnings += 1;
                    }
                } else if stale > 0 {
                    warnings += 1;
                }
            }
            Err(e) => {
                if !json_output {
                    println!("[❌] index.json parse error: {e}");
                }
                errors += 1;
            }
        }
    } else if !json_output {
        println!("[⚠️] index.json not found — run `cortyx compile` to create it");
        warnings += 1;
    } else {
        warnings += 1;
    }

    // 3. Git availability
    let git_ok = std::process::Command::new("git")
        .args(["rev-parse", "--short", "HEAD"])
        .current_dir(root)
        .output()
        .ok()
        .filter(|o| o.status.success());
    if !json_output {
        if let Some(out) = &git_ok {
            let sha = String::from_utf8_lossy(&out.stdout).trim().to_string();
            println!("[✅] Git available (HEAD: {sha})");
        } else {
            println!("[⚠️] Git not available or not a git repository (confidence scores will be 1.0)");
            warnings += 1;
        }
    } else if git_ok.is_none() {
        warnings += 1;
    }

    // 4. Binary size
    if !json_output {
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
    }

    if json_output {
        // S-IX (R16): Machine-readable output for CI pipelines
        println!("{{");
        println!("  \"total_neurons\": {total_neurons},");
        println!("  \"synapse_count\": {synapse_count},");
        println!("  \"stale_count\": {stale_count},");
        println!("  \"low_quality_count\": {low_quality_count},");
        println!("  \"index_valid\": {index_valid},");
        println!("  \"errors\": {errors},");
        println!("  \"warnings\": {warnings}");
        println!("}}");
    } else {
        println!("{sep}");
        if errors == 0 && warnings == 0 {
            println!("Summary: all checks passed ✓");
        } else {
            println!("Summary: {errors} error(s), {warnings} warning(s)");
        }
    }

    if errors > 0 { 1 } else { 0 }
}

/// S-IV (R16): Git-federated concept library management.
///
/// All git operations run in `~/.cortyx/global/` — the shared concept store.
/// `init` creates the directory and git-initializes it.
/// `pull`/`push` require a configured remote (set via `init --remote` or manually).
fn run_concepts(sub: ConceptsCommand) -> Result<()> {
    let global_dir = global_index::global_dir();

    match sub {
        ConceptsCommand::Init { remote } => {
            std::fs::create_dir_all(&global_dir)?;
            let is_already_git = global_dir.join(".git").exists();

            if !is_already_git {
                let out = std::process::Command::new("git")
                    .args(["init", "-b", "main"])
                    .current_dir(&global_dir)
                    .output()?;
                if !out.status.success() {
                    // Older git versions don't support -b; retry without it
                    std::process::Command::new("git")
                        .arg("init")
                        .current_dir(&global_dir)
                        .status()?;
                }
                println!("Initialized git repo at {}", global_dir.display());
            } else {
                println!("Already a git repo: {}", global_dir.display());
            }

            if let Some(url) = remote {
                let add = std::process::Command::new("git")
                    .args(["remote", "add", "origin", &url])
                    .current_dir(&global_dir)
                    .status()?;
                if add.success() {
                    println!("Remote 'origin' set to {url}");
                } else {
                    // Remote may already exist; update it
                    std::process::Command::new("git")
                        .args(["remote", "set-url", "origin", &url])
                        .current_dir(&global_dir)
                        .status()?;
                    println!("Remote 'origin' updated to {url}");
                }
            }
        }

        ConceptsCommand::Pull => {
            println!("Fetching concepts from remote…");
            let fetch = std::process::Command::new("git")
                .args(["fetch", "origin"])
                .current_dir(&global_dir)
                .status()?;
            if !fetch.success() {
                anyhow::bail!("git fetch failed — check your remote and network");
            }
            let merge = std::process::Command::new("git")
                .args(["merge", "--ff-only", "origin/main"])
                .current_dir(&global_dir)
                .status()
                .or_else(|_| {
                    std::process::Command::new("git")
                        .args(["merge", "--ff-only", "origin/master"])
                        .current_dir(&global_dir)
                        .status()
                })?;
            if merge.success() {
                println!("Concepts updated.");
            } else {
                anyhow::bail!("git merge --ff-only failed — diverged history; manual rebase needed");
            }
        }

        ConceptsCommand::Push => {
            println!("Pushing concepts to remote…");
            let push = std::process::Command::new("git")
                .args(["push", "origin", "main"])
                .current_dir(&global_dir)
                .status()
                .or_else(|_| {
                    std::process::Command::new("git")
                        .args(["push", "origin", "master"])
                        .current_dir(&global_dir)
                        .status()
                })?;
            if push.success() {
                println!("Concepts pushed.");
            } else {
                anyhow::bail!("git push failed — check remote permissions");
            }
        }

        ConceptsCommand::Status => {
            // Count neurons
            let neurons_dir = global_dir.join("neurons");
            let neuron_count = if neurons_dir.exists() {
                std::fs::read_dir(&neurons_dir)
                    .map(|rd| rd.filter_map(|e| e.ok()).count())
                    .unwrap_or(0)
            } else {
                0
            };

            // Get remote URL
            let remote_out = std::process::Command::new("git")
                .args(["remote", "get-url", "origin"])
                .current_dir(&global_dir)
                .output()
                .ok();
            let remote = remote_out
                .filter(|o| o.status.success())
                .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
                .unwrap_or_else(|| "(no remote)".to_string());

            // Get last commit
            let log_out = std::process::Command::new("git")
                .args(["log", "--oneline", "-1"])
                .current_dir(&global_dir)
                .output()
                .ok();
            let last_commit = log_out
                .filter(|o| o.status.success())
                .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
                .filter(|s| !s.is_empty())
                .unwrap_or_else(|| "(no commits yet)".to_string());

            println!("Global concept library: {}", global_dir.display());
            println!("  Remote:      {remote}");
            println!("  Last commit: {last_commit}");
            println!("  Neurons:     {neuron_count}");
        }
    }

    Ok(())
}
