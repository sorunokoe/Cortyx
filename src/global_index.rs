/// D1: Global Concept Layer
///
/// A read-only shared neuron registry at `~/.cortyx/global/neurons/`.
/// Published concepts are immutable, project-agnostic neurons that describe
/// universal patterns (e.g., "JWT authentication", "BM25 retrieval", "event sourcing").
///
/// Workflow:
/// 1. `cortyx publish-concept <neuron_path>` copies the neuron to ~/.cortyx/global/neurons/
///    and registers it in ~/.cortyx/global/index.json (a minimal flat BM25 index).
/// 2. `cortyx_get_contexts` Phase 3 appends up to 2 global concept neurons when the
///    local index has <3 high-confidence results — filling gaps with global knowledge.
///
/// Design principles (TRIZ R14-D1):
/// - Global neurons are NEVER modified by local projects (read-only integration)
/// - No sync, no cloud, no daemon — purely local file system operations
/// - Zero performance cost when global index is absent (graceful fallback)
/// - Concept fingerprint deduplication (D2) prevents publishing redundant neurons

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use anyhow::Result;
use serde::{Deserialize, Serialize};

/// Default global concept directory.
pub fn global_dir() -> PathBuf {
    dirs_home().join(".cortyx").join("global")
}

/// Path to the global neuron storage directory.
pub fn global_neurons_dir() -> PathBuf {
    global_dir().join("neurons")
}

/// Path to the global index file.
pub fn global_index_path() -> PathBuf {
    global_dir().join("index.json")
}

fn dirs_home() -> PathBuf {
    std::env::var("HOME")
        .or_else(|_| std::env::var("USERPROFILE"))
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("."))
}

/// A minimal BM25 entry for global concept neurons.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GlobalEntry {
    /// Absolute path to the global neuron file.
    pub path: PathBuf,
    /// Source project root (for attribution only).
    pub source_project: String,
    /// Simplified term-frequency map (term → count, normalized).
    pub term_freq: HashMap<String, f32>,
    /// Total term count (for BM25 length normalization).
    pub term_count: usize,
    /// BLAKE3 fingerprint of the top-20 BM25 terms (for D2 deduplication).
    pub fingerprint: String,
}

/// The global index — serialized to `~/.cortyx/global/index.json`.
#[derive(Debug, Default, Serialize, Deserialize)]
pub struct GlobalIndex {
    pub version: u32,
    pub entries: Vec<GlobalEntry>,
}

impl GlobalIndex {
    pub const VERSION: u32 = 1;

    /// Load the global index from disk. Returns an empty index if file is absent.
    pub fn load() -> Self {
        let path = global_index_path();
        if !path.exists() {
            return Self { version: Self::VERSION, entries: Vec::new() };
        }
        let data = match std::fs::read_to_string(&path) {
            Ok(d) => d,
            Err(_) => return Self::default(),
        };
        serde_json::from_str(&data).unwrap_or_default()
    }

    /// Save the global index to disk, creating directories as needed.
    pub fn save(&self) -> Result<()> {
        let path = global_index_path();
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let json = serde_json::to_string_pretty(self)?;
        crate::neuron::atomic_write(&path, json.as_bytes())?;
        Ok(())
    }

    /// Query the global index for neurons relevant to `terms`.
    ///
    /// Returns up to `limit` global neuron paths, sorted by BM25 score.
    /// Called from Phase 3 of `get_contexts` when local results are sparse.
    pub fn query(&self, terms: &[String], limit: usize) -> Vec<PathBuf> {
        if self.entries.is_empty() || terms.is_empty() {
            return Vec::new();
        }

        let n = self.entries.len().max(1) as f32;
        let avg_len: f32 = if self.entries.is_empty() {
            1.0
        } else {
            self.entries.iter().map(|e| e.term_count as f32).sum::<f32>()
                / self.entries.len() as f32
        };

        // Build DF cache
        let mut df: HashMap<&str, usize> = HashMap::new();
        for entry in &self.entries {
            for t in entry.term_freq.keys() {
                *df.entry(t.as_str()).or_insert(0) += 1;
            }
        }

        let mut scored: Vec<(f32, &PathBuf)> = self.entries.iter().map(|entry| {
            let dl = entry.term_count as f32;
            let len_norm = 0.25 + 0.75 * (dl / avg_len.max(1.0));
            let k1 = 1.5_f32;
            let score: f32 = terms.iter().map(|t| {
                let tf = entry.term_freq.get(t.as_str()).copied().unwrap_or(0.0);
                if tf == 0.0 { return 0.0; }
                let df_val = df.get(t.as_str()).copied().unwrap_or(1) as f32;
                let idf = ((n - df_val + 0.5) / (df_val + 0.5) + 1.0).ln().max(0.0);
                idf * (tf * (k1 + 1.0)) / (tf + k1 * len_norm)
            }).sum();
            (score, &entry.path)
        }).collect();

        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        scored
            .into_iter()
            .filter(|(s, _)| *s > 0.0)
            .take(limit)
            .filter(|(_, p)| p.exists())
            .map(|(_, p)| p.clone())
            .collect()
    }

    /// Publish a neuron to the global concept library.
    ///
    /// 1. Copies the neuron file to `~/.cortyx/global/neurons/<name>`
    /// 2. Builds a GlobalEntry from the neuron content
    /// 3. Deduplicates by fingerprint (D2) — returns error if duplicate found
    /// 4. Appends to the global index and saves
    pub fn publish(&mut self, neuron_path: &Path, project_root: &Path) -> Result<PathBuf> {
        let name = neuron_path
            .file_name()
            .ok_or_else(|| anyhow::anyhow!("Invalid neuron path"))?;

        let dest_dir = global_neurons_dir();
        std::fs::create_dir_all(&dest_dir)?;
        let dest = dest_dir.join(name);

        let content = std::fs::read_to_string(neuron_path)?;
        let (tf, term_count) = build_term_freq(&content);

        // D2: Compute fingerprint from top-20 BM25 terms (sorted by frequency)
        let fingerprint = compute_fingerprint(&tf);

        // D2: Dedup check — reject if same fingerprint already published
        if self.entries.iter().any(|e| e.fingerprint == fingerprint) {
            anyhow::bail!(
                "Concept already published (fingerprint collision). \
                 Use a more unique/specialized neuron."
            );
        }

        // Copy the neuron file
        std::fs::copy(neuron_path, &dest)?;

        let source_project = project_root
            .file_name()
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_else(|| "unknown".to_string());

        self.entries.push(GlobalEntry {
            path: dest.clone(),
            source_project,
            term_freq: tf,
            term_count,
            fingerprint,
        });

        self.save()?;
        Ok(dest)
    }
}

/// Build a term-frequency map from neuron content.
fn build_term_freq(content: &str) -> (HashMap<String, f32>, usize) {
    let mut tf: HashMap<String, f32> = HashMap::new();
    let mut count = 0usize;
    for raw in content.split(|c: char| !c.is_alphanumeric() && c != '_') {
        if raw.len() < 3 {
            continue;
        }
        let t = raw.to_lowercase();
        *tf.entry(t).or_insert(0.0) += 1.0;
        count += 1;
    }
    (tf, count)
}

/// Compute a BLAKE3-like fingerprint from the top-20 terms by frequency.
///
/// Uses a simple deterministic hash (sorted term list joined) since BLAKE3
/// is not a dependency. Sufficient for D2 deduplication purposes.
fn compute_fingerprint(tf: &HashMap<String, f32>) -> String {
    let mut sorted: Vec<(&String, &f32)> = tf.iter().collect();
    sorted.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap_or(std::cmp::Ordering::Equal));
    let top20: Vec<&str> = sorted.iter().take(20).map(|(t, _)| t.as_str()).collect();
    // Deterministic hash: sorted top-20 terms joined
    let mut terms_sorted = top20.clone();
    terms_sorted.sort_unstable();
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    let mut hasher = DefaultHasher::new();
    terms_sorted.join("|").hash(&mut hasher);
    format!("{:x}", hasher.finish())
}

/// List all published global concepts.
pub fn list_global_concepts() -> Vec<(PathBuf, String)> {
    let idx = GlobalIndex::load();
    idx.entries
        .iter()
        .filter(|e| e.path.exists())
        .map(|e| (e.path.clone(), e.source_project.clone()))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn test_fingerprint_deterministic() {
        let mut tf = HashMap::new();
        tf.insert("auth".to_string(), 5.0);
        tf.insert("token".to_string(), 3.0);
        tf.insert("validate".to_string(), 2.0);
        let fp1 = compute_fingerprint(&tf);
        let fp2 = compute_fingerprint(&tf);
        assert_eq!(fp1, fp2);
    }

    #[test]
    fn test_fingerprint_different_for_different_terms() {
        let mut tf1 = HashMap::new();
        tf1.insert("auth".to_string(), 5.0);
        let mut tf2 = HashMap::new();
        tf2.insert("database".to_string(), 5.0);
        assert_ne!(compute_fingerprint(&tf1), compute_fingerprint(&tf2));
    }

    #[test]
    fn test_global_index_default_empty() {
        let idx = GlobalIndex::default();
        assert!(idx.entries.is_empty());
    }

    #[test]
    fn test_query_empty_index() {
        let idx = GlobalIndex::default();
        let result = idx.query(&["auth".to_string()], 3);
        assert!(result.is_empty());
    }
}
