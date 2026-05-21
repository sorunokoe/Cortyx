//! Concepts command - manage global concept registry.

use crate::cli::ConceptsCommand;
use crate::error::Result;
use crate::{global_index, index};
use std::path::Path;

/// Helper to auto-commit changes to global concepts directory
///
/// # Errors
///
/// Returns an error if the underlying operation fails.
pub fn auto_commit_global_concepts(global_dir: &Path, message: &str) -> Result<bool> {
    if !global_dir.join(".git").exists() {
        return Ok(false);
    }

    let status = std::process::Command::new(crate::git_util::git_binary())
        .args(["status", "--porcelain"])
        .current_dir(global_dir)
        .output()?;
    if !status.status.success() {
        let stderr = String::from_utf8_lossy(&status.stderr);
        crate::cortyx_bail!("git status failed in {}: {stderr}", global_dir.display());
    }
    if String::from_utf8_lossy(&status.stdout).trim().is_empty() {
        return Ok(false);
    }

    let add = std::process::Command::new(crate::git_util::git_binary())
        .args(["add", "-A"])
        .current_dir(global_dir)
        .output()?;
    if !add.status.success() {
        let stderr = String::from_utf8_lossy(&add.stderr);
        crate::cortyx_bail!("git add failed in {}: {stderr}", global_dir.display());
    }

    let commit = std::process::Command::new(crate::git_util::git_binary())
        .args(["commit", "-m", message])
        .current_dir(global_dir)
        .output()?;
    if !commit.status.success() {
        let stderr = String::from_utf8_lossy(&commit.stderr);
        crate::cortyx_bail!(
            "git commit failed in {}: {}",
            global_dir.display(),
            stderr.trim()
        );
    }

    Ok(true)
}

/// Helper to collect publish-ready concepts from local index
fn collect_ready_concepts(
    root: &Path,
    global_idx: &global_index::GlobalIndex,
    limit: usize,
    min_use: u32,
    min_hit_rate: f32,
    min_quality: f32,
) -> Result<Vec<index::PublishReadySummary>> {
    let idx = index::NeuronIndex::load_or_create(root)?;
    let mut ready = Vec::new();
    for candidate in idx.publish_ready_candidates(min_use, min_hit_rate, min_quality, 0) {
        if global_idx.contains_neuron(&candidate.path)? {
            continue;
        }
        ready.push(candidate);
        if limit > 0 && ready.len() >= limit {
            break;
        }
    }
    Ok(ready)
}

/// # Errors
///
/// Returns an error if the underlying operation fails.
pub fn run(sub: ConceptsCommand) -> Result<()> {
    let global_dir = global_index::global_dir();
    let global_idx = global_index::GlobalIndex::load();

    match sub {
        ConceptsCommand::Init { remote } => {
            std::fs::create_dir_all(&global_dir)?;
            let is_already_git = global_dir.join(".git").exists();

            if !is_already_git {
                let out = std::process::Command::new(crate::git_util::git_binary())
                    .args(["init", "-b", "main"])
                    .current_dir(&global_dir)
                    .output()?;
                if !out.status.success() {
                    // Older git versions don't support -b; retry without it
                    std::process::Command::new(crate::git_util::git_binary())
                        .arg("init")
                        .current_dir(&global_dir)
                        .status()?;
                }
                println!("Initialized git repo at {}", global_dir.display());
            } else {
                println!("Already a git repo: {}", global_dir.display());
            }

            if let Some(ref url) = remote {
                let add = std::process::Command::new(crate::git_util::git_binary())
                    .args(["remote", "add", "origin", url])
                    .current_dir(&global_dir)
                    .status()?;
                if add.success() {
                    println!("Remote 'origin' set to {url}");
                } else {
                    // Remote may already exist; update it
                    std::process::Command::new(crate::git_util::git_binary())
                        .args(["remote", "set-url", "origin", url])
                        .current_dir(&global_dir)
                        .status()?;
                    println!("Remote 'origin' updated to {url}");
                }
            }
        },

        ConceptsCommand::Pull => {
            println!("Fetching concepts from remote…");
            let fetch = std::process::Command::new(crate::git_util::git_binary())
                .args(["fetch", "origin"])
                .current_dir(&global_dir)
                .status()?;
            if !fetch.success() {
                crate::cortyx_bail!("git fetch failed — check your remote and network");
            }
            let merge = std::process::Command::new(crate::git_util::git_binary())
                .args(["merge", "--ff-only", "origin/main"])
                .current_dir(&global_dir)
                .status()
                .or_else(|_| {
                    std::process::Command::new(crate::git_util::git_binary())
                        .args(["merge", "--ff-only", "origin/master"])
                        .current_dir(&global_dir)
                        .status()
                })?;
            if merge.success() {
                println!("Concepts updated.");
            } else {
                crate::cortyx_bail!(
                    "git merge --ff-only failed — diverged history; manual rebase needed"
                );
            }
        },

        ConceptsCommand::Push => {
            println!("Pushing concepts to remote…");
            let push = std::process::Command::new(crate::git_util::git_binary())
                .args(["push", "origin", "main"])
                .current_dir(&global_dir)
                .status()
                .or_else(|_| {
                    std::process::Command::new(crate::git_util::git_binary())
                        .args(["push", "origin", "master"])
                        .current_dir(&global_dir)
                        .status()
                })?;
            if push.success() {
                println!("Concepts pushed.");
            } else {
                crate::cortyx_bail!("git push failed — check remote or create empty repo first");
            }
        },

        ConceptsCommand::Ready {
            project,
            limit,
            min_use,
            min_hit_rate,
            min_quality,
        } => {
            let project_root = project.as_deref().unwrap_or_else(|| Path::new("."));
            let ready = collect_ready_concepts(
                project_root,
                &global_idx,
                limit,
                min_use,
                min_hit_rate,
                min_quality,
            )?;
            if ready.is_empty() {
                println!("No publish-ready concepts found.");
            } else {
                println!("Found {} publish-ready concept(s):", ready.len());
                for summary in ready {
                    println!(
                        "  {} (use: {}, hit_rate: {:.2}, quality: {:.2})",
                        summary.path.display(),
                        summary.use_count,
                        summary.hit_rate,
                        summary.quality_score
                    );
                }
            }
        },

        ConceptsCommand::PublishReady {
            project,
            limit,
            min_use,
            min_hit_rate,
            min_quality,
        } => {
            let project_root = project.as_deref().unwrap_or_else(|| Path::new("."));
            let ready = collect_ready_concepts(
                project_root,
                &global_idx,
                limit,
                min_use,
                min_hit_rate,
                min_quality,
            )?;
            if ready.is_empty() {
                println!("No publish-ready concepts found.");
                return Ok(());
            }

            println!("Publishing {} concept(s)...", ready.len());
            let neurons_dir = global_dir.join("neurons");
            std::fs::create_dir_all(&neurons_dir)?;

            let mut published = 0usize;
            for summary in &ready {
                let src = project_root.join(&summary.path);
                let Some(file_name) = summary.path.file_name() else {
                    continue;
                };
                let dest = neurons_dir.join(file_name);
                std::fs::copy(&src, &dest)?;
                println!("  ✓ {}", summary.path.display());
                published += 1;
            }

            let skipped = ready.len() - published;
            if skipped > 0 {
                println!("{skipped} neuron(s) skipped by ECS quality gate.");
            }

            if published == 0 {
                println!("No neurons passed ECS quality gate — nothing committed.");
                return Ok(());
            }

            if auto_commit_global_concepts(&global_dir, "publish ready concepts")? {
                println!("Changes committed to global concepts.");
            } else {
                println!("Note: No git repository in global_dir — changes not committed.");
            }
        },

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

            println!("Global concepts: {} neurons", neuron_count);
        },
    }

    Ok(())
}
