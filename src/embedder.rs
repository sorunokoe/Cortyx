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
    use crate::error::Result;
    use fastembed::{
        EmbeddingModel, InitOptions, RerankInitOptions, RerankerModel, TextEmbedding, TextRerank,
    };
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
        model: std::sync::Mutex<TextEmbedding>,
    }

    impl EmbeddingBackend {
        /// Load the model (downloads on first use, ~80MB).
        ///
        /// Respects `CORTYX_NO_DOWNLOAD=1` (or any value): when set, returns an error
        /// immediately so the caller falls back to BM25-only without any network access.
        /// Use this in air-gapped environments, CI, or corporate proxies that block downloads.
        pub fn new() -> Result<Self> {
            if std::env::var("CORTYX_NO_DOWNLOAD").is_ok() {
                crate::cortyx_bail!(
                    "CORTYX_NO_DOWNLOAD is set — dense embedding model not loaded. \
                     Falling back to BM25-only retrieval."
                );
            }
            let dir = cache_dir();
            std::fs::create_dir_all(&dir)?;
            let model = TextEmbedding::try_new(
                InitOptions::new(EmbeddingModel::NomicEmbedTextV15).with_cache_dir(dir),
            )?;
            Ok(Self {
                model: std::sync::Mutex::new(model),
            })
        }

        /// Embed a batch of texts. Returns a list of 384-dim f32 vectors.
        pub fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
            let mut guard = self
                .model
                .lock()
                .map_err(|_| crate::cortyx_err!("embedding model mutex was poisoned"))?;
            Ok(guard.embed(texts.to_vec(), None)?)
        }

        /// Embed a single query string.
        pub fn embed_query(&self, query: &str) -> Result<Vec<f32>> {
            let mut batch = self.embed_batch(&[query])?;
            batch
                .pop()
                .ok_or_else(|| crate::cortyx_err!("Empty embedding result"))
        }
    }

    /// Optional cross-encoder reranker (BGE-reranker-base, the default fastembed v5 reranker).
    ///
    /// Used to reorder the top-20 hybrid candidates before returning top-5.
    /// Expected: +1-2% recall at ~40ms latency cost.
    pub struct RerankerBackend {
        model: TextRerank,
    }

    impl RerankerBackend {
        pub fn new() -> Result<Self> {
            if std::env::var("CORTYX_NO_DOWNLOAD").is_ok() {
                crate::cortyx_bail!(
                    "CORTYX_NO_DOWNLOAD is set — reranker model not loaded. \
                     Falling back to BM25+TF-IDF."
                );
            }
            let dir = cache_dir();
            std::fs::create_dir_all(&dir)?;
            let model = TextRerank::try_new(
                RerankInitOptions::new(RerankerModel::BGERerankerBase).with_cache_dir(dir),
            )?;
            Ok(Self { model })
        }

        /// Rerank `candidates` (document strings) by relevance to `query`.
        /// Returns indices in order of descending relevance.
        pub fn rerank(&mut self, query: &str, candidates: &[&str]) -> Result<Vec<usize>> {
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
pub use inner::{cache_dir, cosine_sim, EmbeddingBackend, RerankerBackend};

// ─── No-op stubs (no `embed` feature) ────────────────────────────────────────

/// Cosine similarity (no-op f32 stub when embed feature absent).
#[must_use]
#[cfg(not(feature = "embed"))]
#[allow(dead_code)]
pub fn cosine_sim(_a: &[f32], _b: &[f32]) -> f32 {
    0.0
}

// ─── Embedding storage (.cortyx/embeddings.bin + embeddings.tvim) ──────────────
//
// `embeddings.bin` remains the authoritative raw store because it preserves
// path strings and full-precision vectors. `embeddings.tvim` is the derived
// TurboVec ANN cache rebuilt from `embeddings.bin` when missing or stale.

#[cfg(feature = "embed")]
use crate::error::Result;
#[cfg(feature = "embed")]
use std::{
    collections::{HashMap, HashSet},
    path::{Path, PathBuf},
};
#[cfg(feature = "embed")]
use turbovec::IdMapIndex;

#[cfg(feature = "embed")]
const MAGIC: u32 = 0xC07EEB;
#[cfg(feature = "embed")]
const EMBED_VERSION: u32 = 2;
#[cfg(feature = "embed")]
const EMBEDDING_DIM: usize = 768;
#[cfg(feature = "embed")]
const EMBEDDING_BIT_WIDTH: usize = 4;

/// TurboVec-backed embedding store with stable path IDs.
#[cfg(feature = "embed")]
pub struct EmbeddingStore {
    index: IdMapIndex,
    path_to_id: HashMap<PathBuf, u64>,
    id_to_path: HashMap<u64, PathBuf>,
    raw_vectors: HashMap<PathBuf, Vec<f32>>,
}

#[cfg(feature = "embed")]
impl std::fmt::Debug for EmbeddingStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EmbeddingStore")
            .field("len", &self.len())
            .field("dim", &self.index.dim())
            .field("bit_width", &self.index.bit_width())
            .finish()
    }
}

#[cfg(feature = "embed")]
impl Default for EmbeddingStore {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(feature = "embed")]
impl EmbeddingStore {
    /// Create an empty embedding store.
    pub fn new() -> Self {
        Self {
            index: IdMapIndex::new(EMBEDDING_DIM, EMBEDDING_BIT_WIDTH),
            path_to_id: HashMap::new(),
            id_to_path: HashMap::new(),
            raw_vectors: HashMap::new(),
        }
    }

    /// Insert or replace the vector for `path`.
    pub fn insert(&mut self, path: PathBuf, vec: Vec<f32>) {
        self.try_insert(path, vec)
            .unwrap_or_else(|err| panic!("failed to insert embedding: {err}"));
    }

    /// Borrow the stored full-precision vector for `path`, if available.
    pub fn get(&self, path: &Path) -> Option<&Vec<f32>> {
        self.raw_vectors.get(path)
    }

    /// Return the stored full-precision vector for `path`, if available.
    pub fn get_vec(&self, path: &Path) -> Option<Vec<f32>> {
        self.get(path).cloned()
    }

    /// Return the stable external ID for `path`.
    pub fn get_id(&self, path: &Path) -> Option<u64> {
        self.path_to_id.get(path).copied()
    }

    /// Return `true` when an embedding exists for `path`.
    pub fn contains(&self, path: &Path) -> bool {
        self.path_to_id.contains_key(path)
    }

    /// Return `true` when the store has no embeddings.
    pub fn is_empty(&self) -> bool {
        self.path_to_id.is_empty()
    }

    /// Return the number of embeddings in the store.
    pub fn len(&self) -> usize {
        self.path_to_id.len()
    }

    /// Return the stable IDs for the provided paths, skipping paths not present in the store.
    pub fn ids_for_paths(&self, paths: &HashSet<PathBuf>) -> Vec<u64> {
        paths
            .iter()
            .filter_map(|path| self.get_id(path.as_path()))
            .collect()
    }

    /// Search the full corpus and return `(score, path)` pairs.
    pub fn search(&self, query: &[f32], k: usize) -> Vec<(f32, PathBuf)> {
        self.search_ids(query, k, None)
    }

    /// Search a path-restricted subset of the corpus and return `(score, path)` pairs.
    pub fn search_filtered(
        &self,
        query: &[f32],
        k: usize,
        allow_paths: &HashSet<PathBuf>,
    ) -> Vec<(f32, PathBuf)> {
        if allow_paths.is_empty() {
            return Vec::new();
        }
        let allow_ids = self.ids_for_paths(allow_paths);
        if allow_ids.is_empty() {
            return Vec::new();
        }
        self.search_ids(query, k, Some(&allow_ids))
    }

    /// Eagerly populate TurboVec's lazy SIMD caches.
    pub fn prepare(&self) {
        self.index.prepare();
    }

    /// Persist the raw authoritative store and the derived TurboVec index.
    pub fn save(&self, path: &Path) -> Result<()> {
        crate::cortyx_ensure!(
            self.index.len() == self.raw_vectors.len(),
            "embedding index/raw vector count mismatch (index={}, raw={})",
            self.index.len(),
            self.raw_vectors.len()
        );
        let raw_path = raw_embeddings_path(path);
        std::fs::create_dir_all(raw_path.parent().unwrap_or(Path::new(".")))?;
        write_raw_embeddings(&raw_path, &self.raw_vectors)?;
        write_index_atomically(path, &self.index)?;
        Ok(())
    }

    /// Load the raw authoritative store and the derived TurboVec index.
    pub fn load(path: &Path) -> Result<Self> {
        let raw_path = raw_embeddings_path(path);
        if !raw_path.exists() {
            if path.exists() {
                crate::cortyx_bail!(
                    "{} exists but {} is missing; cannot map ANN ids back to neuron paths.",
                    path.display(),
                    raw_path.display()
                );
            }
            return Ok(Self::new());
        }

        let raw_vectors = read_raw_embeddings(&raw_path)?;
        if !path.exists() {
            let store = Self::from_raw_vectors(raw_vectors)?;
            store.save(path)?;
            tracing::info!(
                raw = %raw_path.display(),
                index = %path.display(),
                "embed: migrated legacy embeddings.bin to TurboVec index"
            );
            store.prepare();
            return Ok(store);
        }

        match IdMapIndex::load(path) {
            Ok(index) => match Self::from_index_and_raw(index, raw_vectors) {
                Ok(store) => {
                    store.prepare();
                    Ok(store)
                },
                Err(err) => {
                    tracing::warn!(
                        index = %path.display(),
                        raw = %raw_path.display(),
                        "embed: invalid embeddings.tvim ({err}) — rebuilding from embeddings.bin"
                    );
                    let rebuilt = Self::from_raw_vectors(read_raw_embeddings(&raw_path)?)?;
                    rebuilt.save(path)?;
                    rebuilt.prepare();
                    Ok(rebuilt)
                },
            },
            Err(err) => {
                tracing::warn!(
                    index = %path.display(),
                    raw = %raw_path.display(),
                    "embed: failed to load embeddings.tvim ({err}) — rebuilding from embeddings.bin"
                );
                let rebuilt = Self::from_raw_vectors(raw_vectors)?;
                rebuilt.save(path)?;
                rebuilt.prepare();
                Ok(rebuilt)
            },
        }
    }

    fn search_ids(
        &self,
        query: &[f32],
        k: usize,
        allowlist: Option<&[u64]>,
    ) -> Vec<(f32, PathBuf)> {
        if self.is_empty() || k == 0 || query.len() != EMBEDDING_DIM {
            return Vec::new();
        }
        let normalized = unit_norm(query.to_vec());
        let (scores, ids) = match allowlist {
            Some(ids) => self.index.search_with_allowlist(&normalized, k, Some(ids)),
            None => self.index.search(&normalized, k),
        };
        scores
            .into_iter()
            .zip(ids)
            .filter_map(|(score, id)| self.id_to_path.get(&id).cloned().map(|path| (score, path)))
            .collect()
    }

    fn try_insert(&mut self, path: PathBuf, vec: Vec<f32>) -> Result<()> {
        crate::cortyx_ensure!(
            vec.len() == EMBEDDING_DIM,
            "Embedding dimension mismatch: got {}, expected {}",
            vec.len(),
            EMBEDDING_DIM
        );
        let normalized = unit_norm(vec);
        if let Some(old_id) = self.path_to_id.remove(&path) {
            self.index.remove(old_id);
            self.id_to_path.remove(&old_id);
        }

        let id = stable_path_id(path.as_path());
        if let Some(existing_path) = self.id_to_path.get(&id) {
            crate::cortyx_ensure!(
                existing_path == &path,
                "Embedding ID collision between {} and {}",
                existing_path.display(),
                path.display()
            );
        }

        self.index.add_with_ids(&normalized, &[id]);
        self.path_to_id.insert(path.clone(), id);
        self.id_to_path.insert(id, path.clone());
        self.raw_vectors.insert(path, normalized);
        Ok(())
    }

    fn from_raw_vectors(raw_vectors: HashMap<PathBuf, Vec<f32>>) -> Result<Self> {
        let mut store = Self::new();
        let mut items: Vec<_> = raw_vectors.into_iter().collect();
        items.sort_by(|a, b| a.0.cmp(&b.0));
        for (path, vec) in items {
            store.try_insert(path, vec)?;
        }
        Ok(store)
    }

    fn from_index_and_raw(
        index: IdMapIndex,
        raw_vectors: HashMap<PathBuf, Vec<f32>>,
    ) -> Result<Self> {
        crate::cortyx_ensure!(
            index.len() == raw_vectors.len(),
            "Embedding index/raw size mismatch (index={}, raw={})",
            index.len(),
            raw_vectors.len()
        );
        let mut path_to_id = HashMap::with_capacity(raw_vectors.len());
        let mut id_to_path = HashMap::with_capacity(raw_vectors.len());
        for path in raw_vectors.keys() {
            let id = stable_path_id(path.as_path());
            crate::cortyx_ensure!(
                index.contains(id),
                "Embedding index missing id for {}",
                path.display()
            );
            if let Some(existing_path) = id_to_path.insert(id, path.clone()) {
                crate::cortyx_bail!(
                    "Embedding ID collision between {} and {}",
                    existing_path.display(),
                    path.display()
                );
            }
            path_to_id.insert(path.clone(), id);
        }
        Ok(Self {
            index,
            path_to_id,
            id_to_path,
            raw_vectors,
        })
    }
}

/// Load the embedding store from `.cortyx/embeddings.tvim`.
#[cfg(feature = "embed")]
pub fn load_embeddings(project_root: &Path) -> EmbeddingStore {
    let path = embeddings_index_path(project_root);
    match EmbeddingStore::load(&path) {
        Ok(store) => store,
        Err(e) => {
            let needs_rebuild = e.to_string().contains("Unsupported embeddings");
            if needs_rebuild {
                tracing::warn!(
                    "Embedding cache is incompatible with this version of Cortyx: {e}\n\
                     → Delete .cortyx/embeddings.bin and .cortyx/embeddings.tvim, \
                     then rerun `cortyx compile --features embed` to rebuild.\n\
                     → Falling back to BM25-only retrieval until rebuilt."
                );
            } else {
                tracing::warn!(
                    "Failed to load embeddings cache: {e} — falling back to BM25-only retrieval"
                );
            }
            EmbeddingStore::new()
        },
    }
}

/// Persist the embedding store to `.cortyx/embeddings.bin` and `.cortyx/embeddings.tvim`.
#[cfg(feature = "embed")]
pub fn save_embeddings(project_root: &Path, store: &EmbeddingStore) -> Result<()> {
    store.save(&embeddings_index_path(project_root))
}

/// Insert or update a single embedding in the store file.
///
/// Loads, mutates, and saves — acceptable for single-entry updates.
#[cfg(feature = "embed")]
pub fn upsert_embedding(project_root: &Path, neuron_path: &Path, vector: Vec<f32>) -> Result<()> {
    let mut store = load_embeddings(project_root);
    store.insert(neuron_path.to_path_buf(), vector);
    save_embeddings(project_root, &store)
}

/// Reciprocal Rank Fusion score for hybrid ranking.
///
/// `score = 1/(RRF_K + rank)` for each ranking; combined by summing.
/// k=60 is the standard default (Cormack et al. 2009).
#[cfg(feature = "embed")]
pub const RRF_K: f32 = 60.0;

/// Compute Reciprocal Rank Fusion for BM25 and dense ranks.
#[cfg(feature = "embed")]
pub fn rrf_score(bm25_rank: usize, cos_rank: usize) -> f32 {
    1.0 / (RRF_K + bm25_rank as f32) + 1.0 / (RRF_K + cos_rank as f32)
}

/// L2-normalize a vector to unit norm (for dot-product = cosine similarity).
#[cfg(feature = "embed")]
pub fn unit_norm(mut v: Vec<f32>) -> Vec<f32> {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for x in &mut v {
            *x /= norm;
        }
    }
    v
}

/// No-op embedding store stub when the `embed` feature is disabled.
#[cfg(not(feature = "embed"))]
#[derive(Debug, Default, Clone)]
pub struct EmbeddingStore;

// ─── Helpers ─────────────────────────────────────────────────────────────────

#[cfg(feature = "embed")]
fn embeddings_index_path(project_root: &Path) -> PathBuf {
    project_root.join(".cortyx").join("embeddings.tvim")
}

#[cfg(feature = "embed")]
fn raw_embeddings_path(index_path: &Path) -> PathBuf {
    index_path.with_extension("bin")
}

#[cfg(feature = "embed")]
fn stable_path_id(path: &Path) -> u64 {
    let hash = blake3::hash(path.to_string_lossy().as_bytes());
    let mut bytes = [0u8; 8];
    bytes.copy_from_slice(&hash.as_bytes()[..8]);
    u64::from_le_bytes(bytes)
}

#[cfg(feature = "embed")]
fn write_index_atomically(path: &Path, index: &IdMapIndex) -> Result<()> {
    let tmp = path.with_extension("tvim.tmp");
    index.write(&tmp)?;
    std::fs::rename(&tmp, path)?;
    Ok(())
}

#[cfg(feature = "embed")]
fn write_u32(buf: &mut Vec<u8>, v: u32) {
    buf.extend_from_slice(&v.to_le_bytes());
}

#[cfg(feature = "embed")]
fn read_u32(data: &[u8], offset: &mut usize) -> Result<u32> {
    if *offset + 4 > data.len() {
        crate::cortyx_bail!("Unexpected EOF reading u32 at {}", offset);
    }
    let v = u32::from_le_bytes(data[*offset..*offset + 4].try_into()?);
    *offset += 4;
    Ok(v)
}

#[cfg(feature = "embed")]
fn write_raw_embeddings(path: &Path, store: &HashMap<PathBuf, Vec<f32>>) -> Result<()> {
    crate::cortyx_ensure!(
        store.len() <= u32::MAX as usize,
        "Embedding store too large to serialize ({} entries)",
        store.len()
    );
    let mut entries: Vec<_> = store.iter().collect();
    entries.sort_by(|a, b| a.0.cmp(b.0));
    let mut buf = Vec::new();
    write_u32(&mut buf, MAGIC);
    write_u32(&mut buf, EMBED_VERSION);
    write_u32(&mut buf, EMBEDDING_DIM as u32);
    write_u32(&mut buf, entries.len() as u32);
    for (path_buf, vec) in entries {
        crate::cortyx_ensure!(
            vec.len() == EMBEDDING_DIM,
            "Embedding dimension mismatch for {}: got {}, expected {}",
            path_buf.display(),
            vec.len(),
            EMBEDDING_DIM
        );
        let path_bytes = path_buf.to_string_lossy().into_owned().into_bytes();
        crate::cortyx_ensure!(
            path_bytes.len() <= u32::MAX as usize,
            "Path too long to serialize: {} bytes",
            path_bytes.len()
        );
        write_u32(&mut buf, path_bytes.len() as u32);
        buf.extend_from_slice(&path_bytes);
        for &f in vec {
            buf.extend_from_slice(&f.to_le_bytes());
        }
    }
    let tmp = path.with_extension("bin.tmp");
    std::fs::write(&tmp, &buf)?;
    std::fs::rename(&tmp, path)?;
    Ok(())
}

#[cfg(feature = "embed")]
fn read_raw_embeddings(path: &Path) -> Result<HashMap<PathBuf, Vec<f32>>> {
    // Reasonable upper bounds to guard against malicious or corrupted files.
    const MAX_ENTRIES: usize = 1_000_000;
    const MAX_DIM: usize = 10_000;
    const MAX_PATH_LEN: usize = 4_096;

    let data = std::fs::read(path)?;
    let mut off = 0usize;

    let magic = read_u32(&data, &mut off)?;
    crate::cortyx_ensure!(magic == MAGIC, "Bad magic in embeddings.bin: {magic:#x}");
    let version = read_u32(&data, &mut off)?;
    crate::cortyx_ensure!(
        version == EMBED_VERSION,
        "Unsupported embeddings.bin version: {version} (expected {EMBED_VERSION}). \
         Delete .cortyx/embeddings.bin and rerun `cortyx compile --features embed` to regenerate."
    );
    let dim = read_u32(&data, &mut off)? as usize;
    crate::cortyx_ensure!(dim <= MAX_DIM, "dim too large in embeddings.bin: {dim}");
    crate::cortyx_ensure!(
        dim == EMBEDDING_DIM,
        "Unsupported embedding dimension in embeddings.bin: {dim} (expected {EMBEDDING_DIM})"
    );
    let count = read_u32(&data, &mut off)? as usize;
    crate::cortyx_ensure!(
        count <= MAX_ENTRIES,
        "entry count too large in embeddings.bin: {count}"
    );

    let mut store = HashMap::with_capacity(count);
    for _ in 0..count {
        let path_len = read_u32(&data, &mut off)? as usize;
        crate::cortyx_ensure!(path_len <= MAX_PATH_LEN, "path_len too large: {path_len}");
        let end = off
            .checked_add(path_len)
            .ok_or_else(|| crate::cortyx_err!("path_len overflow"))?;
        crate::cortyx_ensure!(end <= data.len(), "Truncated path");
        let path_str = std::str::from_utf8(&data[off..end])?;
        let neuron_path = PathBuf::from(path_str);
        off = end;

        let vec_bytes = dim
            .checked_mul(4)
            .ok_or_else(|| crate::cortyx_err!("dim*4 overflow"))?;
        let vec_end = off
            .checked_add(vec_bytes)
            .ok_or_else(|| crate::cortyx_err!("vector offset overflow"))?;
        crate::cortyx_ensure!(vec_end <= data.len(), "Truncated vector");
        let vec: Vec<f32> = (0..dim)
            .map(|i| {
                let mut bytes = [0u8; 4];
                bytes.copy_from_slice(&data[off + i * 4..off + i * 4 + 4]);
                f32::from_le_bytes(bytes)
            })
            .collect();
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
        let mut store = EmbeddingStore::new();
        let p1 = PathBuf::from(".cortyx/neurons/a.context.md");
        let p2 = PathBuf::from(".cortyx/neurons/b.context.md");
        let v1: Vec<f32> = (0..EMBEDDING_DIM)
            .map(|i| i as f32 / EMBEDDING_DIM as f32)
            .collect();
        let v2: Vec<f32> = (0..EMBEDDING_DIM)
            .map(|i| (EMBEDDING_DIM - i) as f32 / EMBEDDING_DIM as f32)
            .collect();
        store.insert(p1.clone(), v1.clone());
        store.insert(p2.clone(), v2);

        save_embeddings(dir.path(), &store).unwrap();
        let loaded = load_embeddings(dir.path());

        assert_eq!(loaded.len(), 2);
        assert!(loaded.contains(p1.as_path()));
        let loaded_v1 = loaded.get_vec(p1.as_path()).unwrap();
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
        assert!(store.contains(p.as_path()));
    }
}
