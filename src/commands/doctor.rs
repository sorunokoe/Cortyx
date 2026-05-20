//! Doctor command - diagnose index health and configuration issues.

use crate::index;
use std::path::Path;

#[must_use]
pub fn run(root: &Path, json_output: bool) -> i32 {
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
                    println!(
                        "[✅] index.json valid ({total_neurons} neurons, {synapse_count} synapses)"
                    );
                    if low_quality_count > 0 {
                        println!("[⚠️] Quality: {low_quality_count} neurons below 40% quality (run cortyx evolve)");
                        warnings += 1;
                    }
                } else if stale > 0 {
                    warnings += 1;
                }
            },
            Err(e) => {
                if !json_output {
                    println!("[❌] index.json parse error: {e}");
                }
                errors += 1;
            },
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
        .map(|out| out.status.success())
        .unwrap_or(false);

    if !json_output {
        if git_ok {
            println!("[✅] Git repository detected");
        } else {
            println!("[⚠️] Not a git repository — commit message extraction will be unavailable");
            warnings += 1;
        }
    } else if !git_ok {
        warnings += 1;
    }

    // Summary
    if json_output {
        println!(
            r#"{{"errors":{errors},"warnings":{warnings},"index_valid":{index_valid},"neurons":{total_neurons},"stale":{stale_count},"synapses":{synapse_count},"low_quality":{low_quality_count}}}"#
        );
    } else {
        println!("{sep}");
        if errors == 0 && warnings == 0 {
            println!("✅ All checks passed");
        } else {
            println!("Errors: {errors}, Warnings: {warnings}");
        }
    }

    if errors > 0 {
        1
    } else {
        0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    #[test]
    fn doctor_detects_missing_cortyx_dir() {
        let temp = TempDir::new().unwrap();
        let exit_code = run(temp.path(), true);
        assert_eq!(exit_code, 1, "Should return error when .cortyx/ missing");
    }

    #[test]
    fn doctor_succeeds_with_valid_index() {
        let temp = TempDir::new().unwrap();
        let cortyx_dir = temp.path().join(".cortyx");
        fs::create_dir_all(&cortyx_dir).unwrap();

        // Create minimal valid index.json
        let index_json = cortyx_dir.join("index.json");
        fs::write(
            &index_json,
            r#"{"version":1,"neurons":{},"inverted_index":{}}"#,
        )
        .unwrap();

        let exit_code = run(temp.path(), true);
        assert_eq!(exit_code, 0, "Should succeed with valid index");
    }
}
