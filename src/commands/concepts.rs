//! Concepts command - manage global concept registry.

use crate::cli::ConceptsCommand;
use crate::global_index;
use anyhow::Result;
use std::path::PathBuf;

pub fn run(sub: ConceptsCommand) -> Result<()> {
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

            if let Some(ref url) = remote {
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
        },

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
                anyhow::bail!(
                    "git merge --ff-only failed — diverged history; manual rebase needed"
                );
            }
        },

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
                anyhow::bail!("git push failed — check remote or create empty repo first");
            }
        },

        ConceptsCommand::Ready {
            project: _,
            limit: _,
            min_use: _,
            min_hit_rate: _,
            min_quality: _,
        } => {
            // TODO: Extract collect_ready_concepts helper from main.rs
            anyhow::bail!(
                "Ready command not yet implemented - extract collect_ready_concepts from main.rs"
            );
        },

        ConceptsCommand::PublishReady {
            project: _,
            limit: _,
            min_use: _,
            min_hit_rate: _,
            min_quality: _,
        } => {
            // TODO: Extract collect_ready_concepts and auto_commit_global_concepts helpers from main.rs
            anyhow::bail!(
                "PublishReady command not yet implemented - extract helpers from main.rs"
            );
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
