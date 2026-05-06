//! Prune command - remove low-quality or stale neurons.

use crate::error::Result;
use crate::{index, neuron};
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime};

pub fn run(root: &Path, min_use: u32, older_than: Option<u64>, dry_run: bool) -> Result<usize> {
    let mut idx = index::NeuronIndex::load_or_create(root)?;
    let now = SystemTime::now();
    let age_cutoff: Option<SystemTime> = older_than.map(|days| {
        now.checked_sub(Duration::from_secs(days * 86_400))
            .unwrap_or(SystemTime::UNIX_EPOCH)
    });

    let candidates: Vec<PathBuf> = idx
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
        let sidecar = neuron::meta_path(path);
        let _ = std::fs::remove_file(sidecar);
    }

    if count > 0 {
        // One single-pass rebuild after all evictions — O(n) not O(n²)
        idx.rebuild_derived_pub();
        idx.save()?;
    }

    Ok(count)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    #[test]
    fn prune_returns_zero_when_no_candidates() {
        let temp = TempDir::new().unwrap();
        let cortyx_dir = temp.path().join(".cortyx");
        fs::create_dir_all(&cortyx_dir).unwrap();

        // Create minimal valid index
        let index_json = cortyx_dir.join("index.json");
        fs::write(
            &index_json,
            r#"{"version":1,"neurons":{},"inverted_index":{}}"#,
        )
        .unwrap();

        let result = run(temp.path(), 5, None, true);
        assert!(result.is_ok());
        assert_eq!(
            result.unwrap(),
            0,
            "Should return 0 when no neurons to prune"
        );
    }

    #[test]
    fn prune_dry_run_does_not_modify_index() {
        let temp = TempDir::new().unwrap();
        let cortyx_dir = temp.path().join(".cortyx");
        fs::create_dir_all(&cortyx_dir).unwrap();

        let index_json = cortyx_dir.join("index.json");
        let index_content = r#"{"version":1,"neurons":{},"inverted_index":{}}"#;
        fs::write(&index_json, index_content).unwrap();

        let _ = run(temp.path(), 0, None, true);

        let after_content = fs::read_to_string(&index_json).unwrap();
        assert_eq!(
            after_content, index_content,
            "Dry run should not modify index"
        );
    }
}
