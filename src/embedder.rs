/// Embedding backend — optional dense vector layer for hybrid retrieval.
///
/// Enabled via `--features embed` (fastembed crate).  When the feature is
/// absent every function is a no-op stub so the binary stays small (<6MB)
/// and the BM25-only path still compiles and passes all tests.
///
/// Model: all-MiniLM-L6-v2 (384-dim, same as MemPalace).
/// Cache: ~/.cache/cortyx/models/ (~80MB first download).
//
// The storage functions (save_embeddings, upsert_embedding, etc.) are part of
// the planned public API surface for the embed feature. They are compiled in
// all configurations so the binary format is stable, but are only called from
// user code once the embed feature is active.
#[cfg(feature = "embed")]
mod inner {
    use anyhow::Result;
    use fastembed::{EmbeddingModel, InitOptions, TextEmbedding, TextRerank, RerankerInitOptions, RerankerModel};
    use std::path::PathBuf;

    /// Returns the directory where model weights are cached.
    pub fn cache_dir() -> PathBuf {
        dirs::cache_dir()
            .unwrap_or_else(|| PathBuf::from(".cache"))
            .join("cortyx")
            .join("models")
    }

    /// Dense embedding backend using all-MiniLM-L6-v2 (384-dim).
    pub struct EmbeddingBackend {
        model: TextEmbedding,
    }

    impl EmbeddingBackend {
        /// Load the model (downloads on first use, ~80MB).
        pub fn new() -> Result<Self> {
            let dir = cache_dir();
            std::fs::create_dir_all(&dir)?;
            let model = TextEmbedding::try_new(
                InitOptions::new(EmbeddingModel::AllMiniLML6V2)
                    .with_cache_dir(dir),
            )?;
            Ok(Self { model })
        }

        /// Embed a batch of texts. Returns a list of 384-dim f32 vectors.
        pub fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
            Ok(self.model.embed(texts.to_vec(), None)?)
        }

        /// Embed a single query string.
        pub fn embed_query(&self, query: &str) -> Result<Vec<f32>> {
            let mut batch = self.embed_batch(&[query])?;
            batch.pop().ok_or_else(|| anyhow::anyhow!("Empty embedding result"))
        }
    }

    /// Optional cross-encoder reranker (ms-marco-MiniLM-L-6-v2).
    ///
    /// Used to reorder the top-20 hybrid candidates before returning top-5.
    /// Expected: +1-2% recall at ~40ms latency cost.
    pub struct RerankerBackend {
        model: TextRerank,
    }

    impl RerankerBackend {
        pub fn new() -> Result<Self> {
            let dir = cache_dir();
            std::fs::create_dir_all(&dir)?;
            let model = TextRerank::try_new(
                RerankerInitOptions::new(RerankerModel::MsMarcoMiniLML12V2)
                    .with_cache_dir(dir),
            )?;
            Ok(Self { model })
        }

        /// Rerank `candidates` (document strings) by relevance to `query`.
        /// Returns indices in order of descending relevance.
        pub fn rerank(&self, query: &str, candidates: &[&str]) -> Result<Vec<usize>> {
            let results = self.model.rerank(query, candidates.to_vec(), false, None)?;
            Ok(results.iter().map(|r| r.index).collect())
        }
    }

    /// Cosine similarity between two unit-norm vectors.
    pub fn cosine_sim(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
    }
}

// ─── Public re-exports ────────────────────────────────────────────────────────

#[cfg(feature = "embed")]
pub use inner::{EmbeddingBackend, RerankerBackend, cosine_sim, cache_dir};

// ─── No-op stubs (no `embed` feature) ────────────────────────────────────────

/// Cosine similarity (no-op f32 stub when embed feature absent).
#[cfg(not(feature = "embed"))]
#[allow(dead_code)]
pub fn cosine_sim(_a: &[f32], _b: &[f32]) -> f32 {
    0.0
}

// ─── Embedding storage (.cortyx/embeddings.bin) ───────────────────────────────
//
// Binary format:
//   magic:   u32  = 0xC07EEB
//   version: u32  = 1
//   dim:     u32  = 384
//   count:   u32  = N
//   entries: N × { path_len: u32, path: utf8 bytes, dim × f32 }
//
// All functions gated on the `embed` feature.

#[cfg(feature = "embed")]
use anyhow::Result;
#[cfg(feature = "embed")]
use std::{
    collections::HashMap,
    path::{Path, PathBuf},
};

#[cfg(feature = "embed")]
const MAGIC: u32 = 0xC07EEB;
#[cfg(feature = "embed")]
const EMBED_VERSION: u32 = 1;
#[cfg(feature = "embed")]
const EMBEDDING_DIM: usize = 384;

/// In-memory embedding store: neuron path → unit-norm 384-dim vector.
#[cfg(feature = "embed")]
pub type EmbeddingStore = HashMap<PathBuf, Vec<f32>>;

/// Load the embedding store from `.cortyx/embeddings.bin`.
///
/// Returns an empty map if the file is absent or malformed (BM25-only mode).
#[cfg(feature = "embed")]
pub fn load_embeddings(project_root: &Path) -> EmbeddingStore {
    let path = embeddings_path(project_root);
    if !path.exists() {
        return HashMap::new();
    }
    match read_embeddings(&path) {
        Ok(store) => store,
        Err(e) => {
            tracing::warn!("Failed to load embeddings.bin: {e} — falling back to BM25-only");
            HashMap::new()
        }
    }
}

/// Persist the embedding store to `.cortyx/embeddings.bin`.
#[cfg(feature = "embed")]
pub fn save_embeddings(project_root: &Path, store: &EmbeddingStore) -> Result<()> {
    let path = embeddings_path(project_root);
    std::fs::create_dir_all(path.parent().unwrap_or(Path::new(".")))?;
    anyhow::ensure!(
        store.len() <= u32::MAX as usize,
        "Embedding store too large to serialize ({} entries)", store.len()
    );
    let mut buf = Vec::new();
    write_u32(&mut buf, MAGIC);
    write_u32(&mut buf, EMBED_VERSION);
    write_u32(&mut buf, EMBEDDING_DIM as u32);
    write_u32(&mut buf, store.len() as u32);
    for (p, vec) in store {
        let path_bytes = p.to_string_lossy().into_owned().into_bytes();
        anyhow::ensure!(
            path_bytes.len() <= u32::MAX as usize,
            "Path too long to serialize: {} bytes", path_bytes.len()
        );
        write_u32(&mut buf, path_bytes.len() as u32);
        buf.extend_from_slice(&path_bytes);
        for &f in vec {
            buf.extend_from_slice(&f.to_le_bytes());
        }
    }
    // Atomic write: write to a temp file, then rename to avoid corruption on crash.
    let tmp = path.with_extension("bin.tmp");
    std::fs::write(&tmp, &buf)?;
    std::fs::rename(&tmp, &path)?;
    Ok(())
}

/// Insert or update a single embedding in the store file.
///
/// Loads, mutates, and saves — acceptable for single-entry updates.
#[cfg(feature = "embed")]
pub fn upsert_embedding(project_root: &Path, neuron_path: &Path, vector: Vec<f32>) -> Result<()> {
    let mut store = load_embeddings(project_root);
    store.insert(neuron_path.to_path_buf(), unit_norm(vector));
    save_embeddings(project_root, &store)
}

/// Reciprocal Rank Fusion score for hybrid ranking.
///
/// `score = 1/(RRF_K + rank)` for each ranking; combined by summing.
/// k=60 is the standard default (Cormack et al. 2009).
#[cfg(feature = "embed")]
pub const RRF_K: f32 = 60.0;

#[cfg(feature = "embed")]
pub fn rrf_score(bm25_rank: usize, cos_rank: usize) -> f32 {
    1.0 / (RRF_K + bm25_rank as f32) + 1.0 / (RRF_K + cos_rank as f32)
}

/// L2-normalize a vector to unit norm (for dot-product = cosine similarity).
#[cfg(feature = "embed")]
pub fn unit_norm(mut v: Vec<f32>) -> Vec<f32> {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for x in &mut v { *x /= norm; }
    }
    v
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

#[cfg(feature = "embed")]
fn embeddings_path(project_root: &Path) -> PathBuf {
    project_root.join(".cortyx").join("embeddings.bin")
}

#[cfg(feature = "embed")]
fn write_u32(buf: &mut Vec<u8>, v: u32) {
    buf.extend_from_slice(&v.to_le_bytes());
}

#[cfg(feature = "embed")]
fn read_u32(data: &[u8], offset: &mut usize) -> Result<u32> {
    if *offset + 4 > data.len() {
        anyhow::bail!("Unexpected EOF reading u32 at {}", offset);
    }
    let v = u32::from_le_bytes(data[*offset..*offset + 4].try_into()?);
    *offset += 4;
    Ok(v)
}

#[cfg(feature = "embed")]
fn read_embeddings(path: &Path) -> Result<EmbeddingStore> {
    // Reasonable upper bounds to guard against malicious or corrupted files.
    const MAX_ENTRIES: usize = 1_000_000;
    const MAX_DIM: usize = 10_000;
    const MAX_PATH_LEN: usize = 4_096;

    let data = std::fs::read(path)?;
    let mut off = 0usize;

    let magic = read_u32(&data, &mut off)?;
    anyhow::ensure!(magic == MAGIC, "Bad magic in embeddings.bin: {magic:#x}");
    let version = read_u32(&data, &mut off)?;
    anyhow::ensure!(
        version == EMBED_VERSION,
        "Unsupported embeddings.bin version: {version} (expected {EMBED_VERSION}). \
         Delete .cortyx/embeddings.bin and rerun `cortyx compile --features embed` to regenerate."
    );
    let dim = read_u32(&data, &mut off)? as usize;
    anyhow::ensure!(dim <= MAX_DIM, "dim too large in embeddings.bin: {dim}");
    let count = read_u32(&data, &mut off)? as usize;
    anyhow::ensure!(count <= MAX_ENTRIES, "entry count too large in embeddings.bin: {count}");

    let mut store = HashMap::with_capacity(count);
    for _ in 0..count {
        let path_len = read_u32(&data, &mut off)? as usize;
        anyhow::ensure!(path_len <= MAX_PATH_LEN, "path_len too large: {path_len}");
        let end = off.checked_add(path_len)
            .ok_or_else(|| anyhow::anyhow!("path_len overflow"))?;
        anyhow::ensure!(end <= data.len(), "Truncated path");
        let path_str = std::str::from_utf8(&data[off..end])?;
        let neuron_path = PathBuf::from(path_str);
        off = end;

        let vec_bytes = dim.checked_mul(4)
            .ok_or_else(|| anyhow::anyhow!("dim*4 overflow"))?;
        let vec_end = off.checked_add(vec_bytes)
            .ok_or_else(|| anyhow::anyhow!("vector offset overflow"))?;
        anyhow::ensure!(vec_end <= data.len(), "Truncated vector");
        let vec: Vec<f32> = (0..dim).map(|i| {
            // Safety: bounds checked above; slice is always exactly 4 bytes.
            f32::from_le_bytes(data[off + i * 4..off + i * 4 + 4].try_into().unwrap())
        }).collect();
        off = vec_end;
        store.insert(neuron_path, vec);
    }
    Ok(store)
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(all(test, feature = "embed"))]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn save_load_round_trip() {
        let dir = TempDir::new().unwrap();
        let mut store: EmbeddingStore = HashMap::new();
        let p1 = PathBuf::from(".cortyx/neurons/a.context.md");
        let p2 = PathBuf::from(".cortyx/neurons/b.context.md");
        let v1: Vec<f32> = (0..EMBEDDING_DIM).map(|i| i as f32 / EMBEDDING_DIM as f32).collect();
        let v2: Vec<f32> = (0..EMBEDDING_DIM).map(|i| (EMBEDDING_DIM - i) as f32 / EMBEDDING_DIM as f32).collect();
        store.insert(p1.clone(), unit_norm(v1.clone()));
        store.insert(p2.clone(), unit_norm(v2));

        save_embeddings(dir.path(), &store).unwrap();
        let loaded = load_embeddings(dir.path());

        assert_eq!(loaded.len(), 2);
        assert!(loaded.contains_key(&p1));
        let loaded_v1 = &loaded[&p1];
        let expected = unit_norm(v1);
        for (a, b) in loaded_v1.iter().zip(expected.iter()) {
            assert!((a - b).abs() < 1e-6, "Vector mismatch");
        }
    }

    #[test]
    fn load_missing_returns_empty() {
        let dir = TempDir::new().unwrap();
        let store = load_embeddings(dir.path());
        assert!(store.is_empty());
    }

    #[test]
    fn unit_norm_produces_unit_vector() {
        let v = vec![3.0f32, 4.0];
        let n = unit_norm(v);
        let len: f32 = n.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((len - 1.0).abs() < 1e-6);
    }

    #[test]
    fn unit_norm_zero_vector_safe() {
        let v = vec![0.0f32; 384];
        let n = unit_norm(v);
        assert!(n.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn rrf_score_correct() {
        // rank 0: 1/(60+0) + 1/(60+0) ≈ 0.0333
        let s = rrf_score(0, 0);
        assert!((s - 1.0 / 60.0 * 2.0).abs() < 1e-6);
    }

    #[test]
    fn rrf_score_decreases_with_rank() {
        let s0 = rrf_score(0, 0);
        let s1 = rrf_score(1, 1);
        let s5 = rrf_score(5, 5);
        assert!(s0 > s1);
        assert!(s1 > s5);
    }

    #[test]
    fn upsert_embedding_persists() {
        let dir = TempDir::new().unwrap();
        let p = PathBuf::from(".cortyx/neurons/x.context.md");
        let v: Vec<f32> = vec![1.0; EMBEDDING_DIM];
        upsert_embedding(dir.path(), &p, v).unwrap();
        let store = load_embeddings(dir.path());
        assert!(store.contains_key(&p));
    }
}
