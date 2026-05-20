//! Persistence helpers and serialized index types.

use super::*;

// ─── Write-ahead log ─────────────────────────────────────────────────────────
//
// The WAL provides crash-safety for pending index mutations that have not yet been
// written to the authoritative `index.json`. New WAL files use line-oriented CRC32
// records so one corrupted entry can be skipped while the rest still replay.
// Legacy `[32-byte BLAKE3 hash][JSON payload]` WAL files remain readable.
//
// # Crash-safety contract
// - **WAL present, CRC32-valid entries:** clean entries are applied on top of the
//   loaded `index.json`. Entries present in both are idempotently overwritten
//   (last-write wins by neuron path, so replaying a WAL into an already-complete
//   index is safe).
// - **WAL present, corrupted entry:** that entry is skipped with a warning and the
//   remaining clean entries are still replayed.
// - **WAL absent:** clean state; `index.json` is fully up-to-date.
//
// # Integrity note
// CRC32 detects accidental corruption and torn records, but it is not a
// cryptographic integrity check and does not protect against tampering.

const WAL_MAGIC: &str = "CORTYXWAL1";

#[derive(Serialize, Deserialize)]
struct WalHeader {
    base_count: usize,
}

/// A pending mutation buffered in the WAL before the next full `index.json` write.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[allow(clippy::large_enum_variant)]
enum WalEntry {
    /// A new or updated BM25 index entry.
    Upsert(BM25Entry),
    /// A neuron removed from the index (stale or evicted).
    Invalidate { path: PathBuf },
}

fn wal_path(cortyx_dir: &Path) -> PathBuf {
    cortyx_dir.join("index.wal")
}

fn index_checksum_path(cortyx_dir: &Path) -> PathBuf {
    cortyx_dir.join("index.checksum")
}

fn read_le_u32(bytes: &[u8]) -> Option<u32> {
    if bytes.len() != 4 {
        return None;
    }
    let mut raw = [0u8; 4];
    raw.copy_from_slice(bytes);
    Some(u32::from_le_bytes(raw))
}

fn write_crc32_sidecar(path: &Path, payload: &[u8]) -> Result<()> {
    let checksum = crc32fast::hash(payload).to_le_bytes();
    atomic_write(path, &checksum)
}

fn purge_persistence_artifacts(project_root: &Path, index_path: &Path) {
    let cortyx_dir = index_path.parent().unwrap_or_else(|| Path::new("."));
    for stale_path in [
        index_path.to_path_buf(),
        index_checksum_path(cortyx_dir),
        activation_cache_path(project_root),
        wal_path(cortyx_dir),
        cortyx_dir.join("index.delta.json"),
    ] {
        if let Err(err) = std::fs::remove_file(&stale_path) {
            if err.kind() != std::io::ErrorKind::NotFound {
                tracing::debug!(
                    path = %stale_path.display(),
                    "Failed to remove stale persistence artifact: {err}"
                );
            }
        }
    }
}

fn verify_index_checksum(project_root: &Path, index_path: &Path) -> bool {
    let cortyx_dir = index_path.parent().unwrap_or_else(|| Path::new("."));
    let checksum_path = index_checksum_path(cortyx_dir);
    if !checksum_path.exists() {
        return false;
    }
    let Ok(index_bytes) = std::fs::read(index_path) else {
        tracing::error!("index.json CRC32 mismatch — rebuilding from neuron files");
        purge_persistence_artifacts(project_root, index_path);
        return true;
    };
    let Ok(checksum_bytes) = std::fs::read(&checksum_path) else {
        tracing::error!("index.json CRC32 mismatch — rebuilding from neuron files");
        purge_persistence_artifacts(project_root, index_path);
        return true;
    };
    let Some(stored_checksum) = read_le_u32(&checksum_bytes) else {
        tracing::error!("index.json CRC32 mismatch — rebuilding from neuron files");
        purge_persistence_artifacts(project_root, index_path);
        return true;
    };
    if crc32fast::hash(&index_bytes) != stored_checksum {
        tracing::error!("index.json CRC32 mismatch — rebuilding from neuron files");
        purge_persistence_artifacts(project_root, index_path);
        return true;
    }
    false
}

fn append_crc32_wal_line(buf: &mut Vec<u8>, prefix: Option<&str>, payload: &[u8]) {
    if let Some(prefix) = prefix {
        buf.extend_from_slice(prefix.as_bytes());
        buf.push(b'\t');
    }
    let checksum = crc32fast::hash(payload);
    let checksum_hex = format!("{checksum:08x}");
    buf.extend_from_slice(checksum_hex.as_bytes());
    buf.push(b'\t');
    buf.extend_from_slice(payload);
    buf.push(b'\n');
}

/// Write `entries` to `path` atomically using versioned CRC32-framed records.
fn write_wal(path: &Path, base_count: usize, entries: &[WalEntry]) -> Result<()> {
    let header = serde_json::to_vec(&WalHeader { base_count })?;
    let mut bytes = Vec::with_capacity(header.len() + entries.len() * 128);
    append_crc32_wal_line(&mut bytes, Some(WAL_MAGIC), &header);
    for entry in entries {
        let payload = serde_json::to_vec(entry)?;
        append_crc32_wal_line(&mut bytes, None, &payload);
    }
    atomic_write(path, &bytes)
}

fn read_crc32_wal(path: &Path, data: &[u8]) -> Option<(usize, Vec<WalEntry>)> {
    let text = match std::str::from_utf8(data) {
        Ok(text) => text,
        Err(_) => {
            tracing::warn!(wal = %path.display(), "WAL is not valid UTF-8 — discarding");
            return None;
        },
    };
    let has_trailing_newline = data.ends_with(b"\n");
    let lines: Vec<&str> = text.split('\n').collect();
    let header_line = lines.first().copied().filter(|line| !line.is_empty())?;
    let mut header_parts = header_line.splitn(3, '\t');
    let magic = header_parts.next()?;
    let stored_crc = header_parts.next()?;
    let payload = header_parts.next()?;
    if magic != WAL_MAGIC {
        tracing::warn!(wal = %path.display(), "WAL header magic mismatch — discarding");
        return None;
    }
    let Ok(stored_crc) = u32::from_str_radix(stored_crc, 16) else {
        tracing::warn!(wal = %path.display(), "WAL header checksum is malformed — discarding");
        return None;
    };
    let actual_crc = crc32fast::hash(payload.as_bytes());
    if stored_crc != actual_crc {
        tracing::warn!(
            wal = %path.display(),
            "WAL header checksum mismatch — discarding (falling back to last full save)"
        );
        return None;
    }
    let header: WalHeader = match serde_json::from_slice(payload.as_bytes()) {
        Ok(header) => header,
        Err(_) => {
            tracing::warn!(wal = %path.display(), "WAL header is malformed — discarding");
            return None;
        },
    };

    let mut entries = Vec::new();
    for (idx, line) in lines.iter().skip(1).enumerate() {
        if line.is_empty() {
            continue;
        }
        if !has_trailing_newline && idx + 2 == lines.len() {
            tracing::warn!("WAL entry {} truncated — stopping replay", idx);
            break;
        }
        let mut parts = line.splitn(2, '\t');
        let Some(stored_crc) = parts.next() else {
            tracing::warn!("WAL entry {} malformed — skipping corrupted entry", idx);
            continue;
        };
        let Some(payload) = parts.next() else {
            tracing::warn!("WAL entry {} malformed — skipping corrupted entry", idx);
            continue;
        };
        let Ok(stored_crc) = u32::from_str_radix(stored_crc, 16) else {
            tracing::warn!("WAL entry {} malformed — skipping corrupted entry", idx);
            continue;
        };
        if crc32fast::hash(payload.as_bytes()) != stored_crc {
            tracing::warn!(
                "WAL entry {} checksum mismatch — skipping corrupted entry",
                idx
            );
            continue;
        }
        match serde_json::from_slice::<WalEntry>(payload.as_bytes()) {
            Ok(entry) => entries.push(entry),
            Err(_) => {
                tracing::warn!("WAL entry {} malformed — skipping corrupted entry", idx);
            },
        }
    }
    Some((header.base_count, entries))
}

fn read_legacy_wal(path: &Path, data: &[u8]) -> Option<(usize, Vec<WalEntry>)> {
    if data.len() < 32 {
        tracing::warn!(wal = %path.display(), "WAL file too short — discarding");
        return None;
    }
    let (checksum_bytes, payload) = data.split_at(32);
    let expected = blake3::hash(payload);
    if checksum_bytes != expected.as_bytes() {
        tracing::warn!(
            wal = %path.display(),
            "WAL checksum mismatch — discarding (falling back to last full save)"
        );
        return None;
    }
    let raw: serde_json::Value = serde_json::from_slice(payload).ok()?;
    let base_count = raw.get("base_count")?.as_u64()? as usize;
    let entries: Vec<WalEntry> = serde_json::from_value(raw.get("entries")?.clone()).ok()?;
    Some((base_count, entries))
}

/// Read and verify a WAL file.
fn read_wal(path: &Path) -> Option<(usize, Vec<WalEntry>)> {
    let data = std::fs::read(path).ok()?;
    if data.starts_with(WAL_MAGIC.as_bytes()) {
        return read_crc32_wal(path, &data);
    }
    read_legacy_wal(path, &data)
}

// ─── Persisted index wrapper ───────────────────────────────────────────────────

/// Borrowed view used for serialization — avoids cloning the entire entry vector
/// on every save() call (which would otherwise be O(n) allocation per MCP mutation).
#[derive(Serialize)]
pub(super) struct PersistedIndexRef<'a> {
    version: u32,
    cache_generation: u64,
    entries: &'a [BM25Entry],
    #[serde(skip_serializing_if = "<[[usize; 2]]>::is_empty")]
    session_utilization: &'a [[usize; 2]],
    #[serde(skip_serializing_if = "<[String]>::is_empty")]
    shards: &'a [String],
}

/// Binary activation cache persisted alongside index.json.
///
/// TRIZ P10 (Preliminary Action): precompute and persist the query-hot derived
/// structures at save time, so CLI startup does not have to rebuild them on
/// every `status` / `get-contexts` invocation.
#[derive(Serialize, Deserialize)]
pub(super) struct PersistedActivationCache {
    version: u32,
    index_generation: u64,
    entries: Vec<BM25Entry>,
    concept_clouds: Vec<Vec<String>>,
    summaries: Vec<String>,
    adjacency: HashMap<PathBuf, Vec<Synapse>>,
    path_index: HashMap<PathBuf, usize>,
    parent_index: HashMap<PathBuf, Vec<usize>>,
    df_cache: HashMap<String, usize>,
    posting_list: HashMap<String, Vec<usize>>,
    avg_doc_len: f32,
    avg_verbatim_doc_len: f32,
    module_index: HashMap<String, Vec<usize>>,
    vocab_bridge: HashMap<String, HashSet<String>>,
    morpheme_map: HashMap<String, Vec<String>>,
    session_utilization: Vec<[usize; 2]>,
    session_index: HashMap<String, Vec<usize>>,
    pmi_neighbors: HashMap<String, Vec<String>>,
    idf_n: usize,
    /// S4-delta: entries.len() at last full index.json write.
    #[serde(rename = "wal_base")]
    delta_base: usize,
}

#[derive(Serialize)]
pub(super) struct PersistedActivationCacheRef<'a> {
    version: u32,
    index_generation: u64,
    entries: &'a [BM25Entry],
    concept_clouds: Vec<&'a [String]>,
    summaries: Vec<&'a str>,
    adjacency: &'a HashMap<PathBuf, Vec<Synapse>>,
    path_index: &'a HashMap<PathBuf, usize>,
    parent_index: &'a HashMap<PathBuf, Vec<usize>>,
    df_cache: &'a HashMap<String, usize>,
    posting_list: &'a HashMap<String, Vec<usize>>,
    avg_doc_len: f32,
    avg_verbatim_doc_len: f32,
    module_index: &'a HashMap<String, Vec<usize>>,
    vocab_bridge: &'a HashMap<String, HashSet<String>>,
    morpheme_map: &'a HashMap<String, Vec<String>>,
    session_utilization: &'a [[usize; 2]],
    session_index: &'a HashMap<String, Vec<usize>>,
    pmi_neighbors: &'a HashMap<String, Vec<String>>,
    idf_n: usize,
    /// S4-delta: entries.len() at last full index.json write.
    #[serde(rename = "wal_base")]
    delta_base: usize,
}

// ─── Schema migrations ────────────────────────────────────────────────────────

/// Migration function signature: transforms a stored JSON value to be compatible
/// with the next schema version. Fields not touched by the migration are preserved
/// (use_count, hit_count, staleness_multiplier etc. survive every upgrade).
pub(super) type MigrationFn = fn(serde_json::Value) -> serde_json::Value;

/// v7 → v8: rename `lsh_fingerprint: u64` → `lsh_fingerprints: [u64; 16]` (interim 1024-bit LSH).
pub(super) fn migrate_v7_to_v8(mut entries_val: serde_json::Value) -> serde_json::Value {
    if let Some(arr) = entries_val.as_array_mut() {
        for entry in arr.iter_mut() {
            let old_fp = entry
                .get("lsh_fingerprint")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            let fps: Vec<serde_json::Value> =
                std::iter::once(serde_json::Value::Number(old_fp.into()))
                    .chain(std::iter::repeat_n(
                        serde_json::Value::Number(0u64.into()),
                        15,
                    ))
                    .collect();
            entry["lsh_fingerprints"] = serde_json::Value::Array(fps);
            if let Some(obj) = entry.as_object_mut() {
                obj.remove("lsh_fingerprint");
            }
        }
    }
    entries_val
}

/// v8 → v9: trim `lsh_fingerprints` from 16 elements down to 4.
///
/// v8 stored a `[u64; 16]` array where only the first 4 slots were non-zero.
/// v9 uses a `[u64; 4]` array — take the first 4 elements and discard the rest.
pub(super) fn migrate_v8_to_v9(mut entries_val: serde_json::Value) -> serde_json::Value {
    if let Some(arr) = entries_val.as_array_mut() {
        for entry in arr.iter_mut() {
            if let Some(fps) = entry.get("lsh_fingerprints").and_then(|v| v.as_array()) {
                let trimmed: Vec<serde_json::Value> = fps.iter().take(4).cloned().collect();
                entry["lsh_fingerprints"] = serde_json::Value::Array(trimmed);
            }
        }
    }
    entries_val
}

/// Chain of (from_version, to_version, migration_fn) applied in sequence when
/// the stored index is older than INDEX_VERSION.
///
/// Adding a new entry here (rather than bumping INDEX_VERSION and discarding) means
/// users never lose curated data from a routine `cargo install cortyx` upgrade.
const MIGRATIONS: &[(u32, u32, MigrationFn)] = &[
    // v5 → v6: no structural change (INDEX_VERSION bumped to introduce migration infra).
    (5, 6, |v| v),
    // v6 → v7: add concept_cloud field (populated by rebuild_derived; serde default=[]).
    // Existing entries load fine — serde fills concept_cloud with [].
    (6, 7, |v| v),
    // v7 → v8: rename lsh_fingerprint (u64) → lsh_fingerprints ([u64; 16]).
    (7, 8, migrate_v7_to_v8),
    // v8 → v9: trim lsh_fingerprints from [u64; 16] to [u64; 4] (first 4 slots were the only
    // non-zero entries; the remaining 12 were always zero padding).
    (8, 9, migrate_v8_to_v9),
];

/// Apply all migrations from `stored_version` to `INDEX_VERSION` in sequence.
/// Returns the migrated entries (deserialized from the final Value), or an error.
pub(super) fn migrate_entries(
    mut raw: serde_json::Value,
    stored_version: u32,
) -> Result<Vec<BM25Entry>> {
    let mut ver = stored_version;
    for &(from, to, migrate) in MIGRATIONS {
        if ver == from && ver < INDEX_VERSION {
            // Migrate the "entries" array within the persisted object.
            if let Some(entries_val) = raw.get("entries").cloned() {
                let migrated_entries = migrate(entries_val);
                raw["entries"] = migrated_entries;
            }
            raw["version"] = serde_json::Value::Number(to.into());
            ver = to;
        }
    }
    if ver != INDEX_VERSION {
        let oldest_supported = MIGRATIONS
            .first()
            .map(|(from, _, _)| *from)
            .unwrap_or(INDEX_VERSION);
        if stored_version < oldest_supported {
            crate::cortyx_bail!(
                "Index version {stored_version} predates supported migrations (oldest supported: v{oldest_supported}). \
                 Curated neuron markdown remains on disk; run `cortyx compile .` to rebuild only the search index."
            );
        }
        crate::cortyx_bail!(
            "No migration path from version {stored_version} to {INDEX_VERSION}; \
             run `cortyx compile .` to rebuild."
        );
    }
    let entries: Vec<BM25Entry> = serde_json::from_value(
        raw.get("entries")
            .cloned()
            .unwrap_or(serde_json::Value::Array(vec![])),
    )?;
    Ok(entries)
}

impl NeuronIndex {
    // ── Load / save ───────────────────────────────────────────────────────────

    fn rebuild_from_neuron_files(&mut self) -> usize {
        let neuron_root = neuron_dir(&self.persistence.project_root);
        if !neuron_root.exists() {
            return 0;
        }

        let mut recovered = 0usize;
        for entry in WalkDir::new(&neuron_root)
            .min_depth(1)
            .into_iter()
            .filter_map(|entry| entry.ok())
        {
            if !entry.file_type().is_file() {
                continue;
            }
            let neuron_path = entry.path();
            if !neuron_path.to_string_lossy().ends_with(".context.md") {
                continue;
            }
            let content = match std::fs::read_to_string(neuron_path) {
                Ok(content) => content,
                Err(err) => {
                    tracing::warn!(
                        neuron = %neuron_path.display(),
                        "Failed to read neuron during index rebuild: {err}"
                    );
                    continue;
                },
            };
            let meta_file = meta_path(neuron_path);
            let meta = match std::fs::read_to_string(&meta_file)
                .ok()
                .and_then(|data| serde_json::from_str::<NeuronMeta>(&data).ok())
            {
                Some(meta) => meta,
                None => {
                    tracing::warn!(
                        neuron = %neuron_path.display(),
                        "Failed to read neuron metadata during index rebuild — skipping neuron"
                    );
                    continue;
                },
            };
            self.index_neuron(neuron_path, &content, &meta);
            recovered += 1;
        }
        recovered
    }

    /// Load an existing index from `.cortyx/index.json`, or create an empty one.
    /// Also loads the embedding cache (`embeddings.tvim` + `embeddings.bin`) if present
    /// and warms TurboVec's SIMD search path.
    ///
    /// # Errors
    ///
    /// Returns an error if the underlying operation fails.
    pub fn load_or_create(project_root: &Path) -> Result<Self> {
        let path = index_path(project_root);
        let checksum_mismatch = verify_index_checksum(project_root, &path);

        if !checksum_mismatch {
            if let Some(mut idx) = Self::try_load_activation_cache(project_root, &path) {
                #[cfg(feature = "embed")]
                idx.retrieval.embeddings.prepare();
                idx.feedback.coactivation_counts = load_coactivation_counts(project_root);
                return Ok(idx);
            }
        }

        let mut idx = NeuronIndex::default();
        idx.persistence.project_root = project_root.to_path_buf();
        #[cfg(feature = "embed")]
        {
            idx.retrieval.embeddings = std::sync::Arc::new(load_embeddings(project_root));
        }
        #[cfg(feature = "embed")]
        idx.retrieval.embeddings.prepare();
        let mut activation_generation = 0u64;
        let mut persist_index = false;
        let mut rebuilt_from_neurons = false;

        if checksum_mismatch {
            let recovered = idx.rebuild_from_neuron_files();
            tracing::info!(recovered, "Rebuilt index from persisted neuron files");
            persist_index = true;
            rebuilt_from_neurons = true;
        } else if path.exists() {
            match std::fs::read_to_string(&path) {
                Ok(data) => match serde_json::from_str::<serde_json::Value>(&data) {
                    Ok(raw) => {
                        let stored_version = raw
                            .get("version")
                            .and_then(|v| v.as_u64())
                            .map(|v| v as u32)
                            .unwrap_or(0);
                        activation_generation = raw
                            .get("cache_generation")
                            .and_then(|v| v.as_u64())
                            .unwrap_or(0);

                        if stored_version == INDEX_VERSION {
                            // Current version — fast path: deserialize directly.
                            if let Ok(entries) = serde_json::from_value::<Vec<BM25Entry>>(
                                raw.get("entries").cloned().unwrap_or_default(),
                            ) {
                                idx.retrieval.entries = entries;
                            }
                            // Load session utilization history if present
                            if let Ok(util) = serde_json::from_value::<Vec<[usize; 2]>>(
                                raw.get("session_utilization").cloned().unwrap_or_default(),
                            ) {
                                idx.feedback.session_utilization = util;
                            }
                            if activation_generation == 0 {
                                persist_index = true;
                            }
                        } else if stored_version < INDEX_VERSION {
                            // Older version — apply migration chain to preserve curated data.
                            match migrate_entries(raw, stored_version) {
                                Ok(entries) => {
                                    tracing::info!(
                                        "Migrated index from v{stored_version} to v{INDEX_VERSION} \
                                         ({} entries preserved).",
                                        entries.len()
                                    );
                                    idx.retrieval.entries = entries;
                                    persist_index = true;
                                },
                                Err(e) => {
                                    tracing::warn!(
                                        "Migration from v{stored_version} failed ({e}): \
                                         starting fresh. Run `cortyx compile .` to rebuild."
                                    );
                                },
                            }
                        } else {
                            tracing::warn!(
                                "Index version is newer than binary (stored={stored_version}, \
                                 current={INDEX_VERSION}): starting fresh."
                            );
                        }
                    },
                    Err(e) => {
                        tracing::warn!(
                            "Failed to parse index.json (corrupted?): {e}. \
                             Starting with empty index — run `cortyx compile .` to rebuild."
                        );
                        eprintln!(
                            "⚠ Cortyx: index.json is corrupted ({e}). Run `cortyx compile .`"
                        );
                    },
                },
                Err(e) => {
                    tracing::warn!("Failed to read index.json: {e}. Starting with empty index.");
                },
            }
        }

        // WAL replay: apply any pending mutations written before the last full index.json
        // write. This recovers entries that were buffered in the WAL but not yet
        // committed to index.json when the process last exited.
        //
        // The base_count check ensures idempotency: if the WAL was already applied
        // (i.e. index.json already contains those entries), base_count < entries.len()
        // and the WAL is skipped rather than double-applied.
        let cortyx_dir = path.parent().unwrap_or_else(|| Path::new("."));
        let wal_file = wal_path(cortyx_dir);
        if !rebuilt_from_neurons {
            if let Some((base_count, wal_entries)) = read_wal(&wal_file) {
                if base_count == idx.retrieval.entries.len() {
                    let n = wal_entries.len();
                    for entry in wal_entries {
                        if let WalEntry::Upsert(e) = entry {
                            idx.retrieval.entries.push(e);
                        }
                        // WalEntry::Invalidate: staleness is already persisted in sidecar
                        // JSON files; no replay action needed here.
                    }
                    tracing::debug!(n, "Replayed WAL entries");
                } else {
                    tracing::debug!(
                        base_count,
                        loaded = idx.retrieval.entries.len(),
                        "WAL base_count mismatch — skipping stale WAL"
                    );
                }
            } else {
                // Backward compat: read the pre-WAL `index.delta.json` format (no checksum).
                // Removed once all installations have been upgraded past this version.
                let delta_path = cortyx_dir.join("index.delta.json");
                if let Ok(data) = std::fs::read_to_string(&delta_path) {
                    if let Ok(raw) = serde_json::from_str::<serde_json::Value>(&data) {
                        let base_count =
                            raw.get("base_count").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
                        if base_count == idx.retrieval.entries.len() {
                            if let Ok(delta_entries) = serde_json::from_value::<Vec<BM25Entry>>(
                                raw.get("entries").cloned().unwrap_or_default(),
                            ) {
                                tracing::debug!(
                                    n = delta_entries.len(),
                                    "Replaying legacy delta entries (upgrading to WAL)"
                                );
                                idx.retrieval.entries.extend(delta_entries);
                            }
                        }
                    }
                }
            }
        }

        idx.rebuild_derived();
        idx.feedback.coactivation_counts = load_coactivation_counts(project_root);
        if persist_index {
            if rebuilt_from_neurons {
                idx.persistence
                    .structural_artifacts_dirty
                    .store(true, Ordering::Relaxed);
            }
            if let Err(e) = idx.save() {
                tracing::warn!("Failed to persist upgraded index metadata: {e}");
            }
        } else if activation_generation > 0 {
            let index_bytes = std::fs::metadata(&path).map(|meta| meta.len()).unwrap_or(0);
            if let Err(e) = idx.save_activation_cache(activation_generation, index_bytes) {
                tracing::warn!("Failed to refresh activation cache after rebuild: {e}");
            }
            idx.persistence
                .structural_artifacts_dirty
                .store(false, Ordering::Relaxed);
        }
        Ok(idx)
    }

    /// Reload the embedding store from disk (called after `cortyx compile --embed`).
    #[cfg(feature = "embed")]
    pub fn reload_embeddings(&mut self) {
        self.retrieval.embeddings =
            std::sync::Arc::new(load_embeddings(&self.persistence.project_root));
        self.retrieval.embeddings.prepare();
    }

    /// Serialize the index to `.cortyx/index.json` atomically (write-then-rename).
    ///
    /// S-VI (R16): Also writes per-module shards to `.cortyx/index.{module}.json`
    /// for multi-agent safety — concurrent writes to different modules go to
    /// different files, eliminating the global-lock contention on `index.json`.
    /// The monolithic `index.json` is still written (backward compatibility);
    /// a shard registry field marks which shards are current so future binaries
    /// can fast-load specific modules without reading the full file. Stable
    /// module capsules are also regenerated here so `cortyx_get_contexts` can
    /// serve cache-friendly subsystem summaries without runtime synthesis.
    ///
    /// # Errors
    ///
    /// Returns an error if the underlying operation fails.
    pub fn save(&self) -> Result<()> {
        let path = index_path(&self.persistence.project_root);
        let cortyx_dir = path.parent().unwrap_or(Path::new("."));
        std::fs::create_dir_all(cortyx_dir)?;
        self.flush_dirty_sidecars();
        let structural_dirty = self
            .persistence
            .structural_artifacts_dirty
            .load(Ordering::Relaxed);
        let prior_generation = read_index_cache_generation(&path).unwrap_or(0);

        // WAL-append: determine whether this save can use the append-only path.
        //
        // When only new entries have been added (no in-place mutations), we skip
        // rewriting the monolithic index.json and instead write only the new entries
        // to a WAL file with per-record CRC32 checksums. This reduces serialisation
        // work from O(N+n) to O(n) for pure-append mine batches while still detecting
        // accidental corruption or torn records on next startup.
        //
        // On full saves (delta_dirty=true or threshold exceeded), the WAL is written
        // first as a pre-save checkpoint, then index.json is written, then the WAL
        // is deleted. A crash between WAL write and index.json write is recoverable;
        // a crash between index.json write and WAL deletion results in a benign
        // idempotent re-application of the WAL on next startup.
        let delta_path = cortyx_dir.join("index.delta.json");
        let wal_file = wal_path(cortyx_dir);
        let checksum_path = index_checksum_path(cortyx_dir);
        let delta_base = self.persistence.delta_base.load(Ordering::Relaxed);
        let delta_len = self.retrieval.entries.len().saturating_sub(delta_base);
        // Compact to a full write when the WAL would exceed 25% of the base — keeps
        // WAL replay fast and prevents unbounded WAL growth.
        let over_threshold = delta_base > 0 && delta_len > delta_base / 4;
        let in_wal_mode = delta_base > 0
            && !self.persistence.delta_dirty.load(Ordering::Relaxed)
            && !over_threshold
            && delta_len > 0;

        // In WAL-append mode the generation must stay unchanged so the activation cache
        // passes the index_generation check against the (unchanged) index.json on the next load.
        let cache_generation = if structural_dirty && !in_wal_mode {
            prior_generation.saturating_add(1)
        } else {
            prior_generation
        };

        // S-VI: group entries by module for shard files and stable module capsules.
        // Prefer the in-memory entry.module tag; fall back to the sidecar for
        // older persisted indices that may be missing the field.
        let mut modules: std::collections::HashMap<String, Vec<&BM25Entry>> =
            std::collections::HashMap::new();
        let mut path_modules: std::collections::HashMap<PathBuf, String> =
            std::collections::HashMap::new();
        for entry in &self.retrieval.entries {
            let module_name = entry
                .module
                .clone()
                .or_else(|| sidecar_module_for(&entry.neuron_path))
                .unwrap_or_else(|| "__global".to_string());
            path_modules.insert(entry.neuron_path.clone(), module_name.clone());
            modules.entry(module_name).or_default().push(entry);
        }

        let mut module_names: Vec<&String> = modules.keys().collect();
        module_names.sort();
        let shard_names: Vec<String> = module_names
            .iter()
            .map(|module| safe_module_name(module))
            .collect();

        if structural_dirty {
            for module in &module_names {
                let safe_name = safe_module_name(module);
                let shard_path = cortyx_dir.join(format!("index.{safe_name}.json"));
                let shard = serde_json::json!({
                    "version": INDEX_VERSION,
                    "module": module,
                    "entries": modules[*module],
                });
                if let Err(e) = atomic_write_json(&shard_path, &shard) {
                    tracing::warn!("S-VI: could not write shard for module '{module}': {e}");
                }
            }

            if let Err(e) = self.write_module_capsules(cortyx_dir, &modules, &path_modules) {
                tracing::warn!("Failed to refresh module capsules: {e}");
            }
        }

        // Write monolithic index.json (backward compat) with shard registry embedded,
        // or in WAL-append mode write only the pending entries to a checksummed WAL file.
        if in_wal_mode {
            // WAL-append mode: write only entries[delta_base..] to the WAL. index.json is unchanged.
            let wal_entries: Vec<WalEntry> = self.retrieval.entries[delta_base..]
                .iter()
                .cloned()
                .map(WalEntry::Upsert)
                .collect();
            write_wal(&wal_file, delta_base, &wal_entries)?;
            if !checksum_path.exists() && path.exists() {
                match std::fs::read(&path) {
                    Ok(index_bytes) => {
                        if let Err(e) = write_crc32_sidecar(&checksum_path, &index_bytes) {
                            tracing::warn!("Failed to write index checksum during WAL save: {e}");
                        }
                    },
                    Err(e) => {
                        tracing::warn!("Failed to read index.json for checksum upgrade: {e}");
                    },
                }
            }
            // Pass index_bytes=0 to bypass the size guard — the cache legitimately
            // contains more entries than the (unchanged) index.json.
            if let Err(e) = self.save_activation_cache(cache_generation, 0) {
                tracing::warn!("Failed to write activation cache (WAL-append mode): {e}");
            }
        } else {
            // Full save: rewrite index.json, refresh its checksum, then delete WAL and any legacy delta file.
            let persisted = PersistedIndexRef {
                version: INDEX_VERSION,
                cache_generation,
                entries: &self.retrieval.entries,
                session_utilization: &self.feedback.session_utilization,
                shards: &shard_names,
            };
            let json_bytes = serde_json::to_string_pretty(&persisted)?;
            let _ = std::fs::remove_file(&checksum_path);
            atomic_write(&path, json_bytes.as_bytes())?;
            write_crc32_sidecar(&checksum_path, json_bytes.as_bytes())?;
            let _ = std::fs::remove_file(&wal_file); // WAL committed into index.json
            let _ = std::fs::remove_file(&delta_path); // remove any legacy delta file
            self.persistence
                .delta_base
                .store(self.retrieval.entries.len(), Ordering::Relaxed);
            self.persistence.delta_dirty.store(false, Ordering::Relaxed);
            let index_bytes = json_bytes.len() as u64;
            if let Err(e) = self.save_activation_cache(cache_generation, index_bytes) {
                tracing::warn!("Failed to write activation cache: {e}");
            }
        }
        if let Err(e) = save_coactivation_counts(
            &self.persistence.project_root,
            &self.feedback.coactivation_counts,
        ) {
            tracing::warn!("Failed to write coactivation counts: {e}");
        }
        if structural_dirty {
            self.persistence
                .structural_artifacts_dirty
                .store(false, Ordering::Relaxed);
        }
        Ok(())
    }

    pub(in crate::index) fn write_module_capsules(
        &self,
        cortyx_dir: &Path,
        modules: &HashMap<String, Vec<&BM25Entry>>,
        path_modules: &HashMap<PathBuf, String>,
    ) -> Result<()> {
        let capsule_dir = cortyx_dir.join("capsules");
        std::fs::create_dir_all(&capsule_dir)?;

        let mut live_capsules = HashSet::new();
        let mut module_names: Vec<&String> = modules.keys().collect();
        module_names.sort();

        for module in module_names {
            let Some(content) =
                build_module_capsule_content(module, &modules[module], path_modules)
            else {
                continue;
            };

            let safe_name = safe_module_name(module);
            let capsule_path = capsule_dir.join(format!("{safe_name}.capsule.md"));
            if let Err(e) = atomic_write(&capsule_path, content.as_bytes()) {
                tracing::warn!("Failed to write module capsule for '{module}': {e}");
            } else {
                live_capsules.insert(safe_name);
            }
        }

        if let Ok(entries) = std::fs::read_dir(&capsule_dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                let Some(name) = path.file_name().and_then(|name| name.to_str()) else {
                    continue;
                };
                if !name.ends_with(".capsule.md") {
                    continue;
                }
                let stem = name.trim_end_matches(".capsule.md");
                if live_capsules.contains(stem) {
                    continue;
                }
                if let Err(e) = std::fs::remove_file(&path) {
                    tracing::warn!(
                        "Failed to remove stale module capsule {}: {e}",
                        path.display()
                    );
                }
            }
        }

        Ok(())
    }

    pub(in crate::index) fn try_load_activation_cache(
        project_root: &Path,
        index_path: &Path,
    ) -> Option<Self> {
        let index_generation = read_index_cache_generation(index_path)?;
        let cache_path = activation_cache_path(project_root);
        let index_bytes = std::fs::metadata(index_path).ok()?.len();
        let cache_bytes = std::fs::metadata(&cache_path).ok()?.len();
        // Allow the cache to be larger than index.json when a WAL file (or legacy
        // delta file) exists. In WAL-append mode the cache contains entries not yet
        // flushed to the full index.json.
        let cortyx_dir = index_path.parent().unwrap_or_else(|| Path::new("."));
        let has_pending =
            wal_path(cortyx_dir).exists() || cortyx_dir.join("index.delta.json").exists();
        if !has_pending && cache_bytes > index_bytes {
            tracing::debug!(
                cache = %cache_path.display(),
                index_bytes,
                cache_bytes,
                "Skipping activation cache because it is larger than index.json"
            );
            return None;
        }
        let bytes = std::fs::read(&cache_path).ok()?;
        if bytes.len() < 4 {
            tracing::debug!(
                cache = %cache_path.display(),
                "Activation cache missing CRC32 trailer — rebuilding"
            );
            return None;
        }
        let (payload, trailer) = bytes.split_at(bytes.len() - 4);
        let Some(stored_crc) = read_le_u32(trailer) else {
            tracing::debug!(
                cache = %cache_path.display(),
                "Activation cache has malformed CRC32 trailer — rebuilding"
            );
            return None;
        };
        if crc32fast::hash(payload) != stored_crc {
            tracing::debug!(
                cache = %cache_path.display(),
                "Activation cache CRC32 mismatch — rebuilding"
            );
            return None;
        }
        let cache: PersistedActivationCache = bincode::deserialize(payload).ok()?;
        if cache.version != INDEX_VERSION || cache.index_generation != index_generation {
            return None;
        }
        if cache.concept_clouds.len() != cache.entries.len()
            || cache.summaries.len() != cache.entries.len()
        {
            tracing::warn!(
                "Activation cache shape mismatch (entries={}, clouds={}, summaries={}) — rebuilding.",
                cache.entries.len(),
                cache.concept_clouds.len(),
                cache.summaries.len()
            );
            return None;
        }

        let mut entries = cache.entries;
        for (entry, cloud) in entries.iter_mut().zip(cache.concept_clouds) {
            entry.concept_cloud = cloud;
        }
        for (entry, summary) in entries.iter_mut().zip(cache.summaries) {
            entry.summary = summary;
        }

        tracing::debug!(
            entries = entries.len(),
            cache = %cache_path.display(),
            "Loaded activation cache"
        );

        let mut idx = NeuronIndex::default();
        idx.persistence.project_root = project_root.to_path_buf();
        idx.retrieval.entries = entries;
        idx.retrieval.adjacency = cache.adjacency;
        idx.retrieval.path_index = cache.path_index;
        idx.retrieval.parent_index = cache.parent_index;
        idx.retrieval.df_cache = cache.df_cache;
        idx.retrieval.posting_list = cache.posting_list;
        idx.retrieval.avg_doc_len = cache.avg_doc_len;
        idx.retrieval.avg_verbatim_doc_len = cache.avg_verbatim_doc_len;
        idx.retrieval.module_index = cache.module_index;
        idx.retrieval.vocab_bridge = cache.vocab_bridge;
        idx.retrieval.morpheme_map = cache.morpheme_map;
        idx.feedback.session_utilization = cache.session_utilization;
        idx.retrieval.session_index = cache.session_index;
        idx.retrieval.pmi_neighbors = cache.pmi_neighbors;
        idx.retrieval.idf_n = cache.idf_n;
        idx.persistence.delta_base = AtomicUsize::new(cache.delta_base);
        #[cfg(feature = "embed")]
        {
            idx.retrieval.embeddings = std::sync::Arc::new(load_embeddings(project_root));
        }
        Some(idx)
    }

    pub(in crate::index) fn save_activation_cache(
        &self,
        index_generation: u64,
        index_bytes: u64,
    ) -> Result<()> {
        let cache = PersistedActivationCacheRef {
            version: INDEX_VERSION,
            index_generation,
            entries: &self.retrieval.entries,
            concept_clouds: self
                .retrieval
                .entries
                .iter()
                .map(|entry| entry.concept_cloud.as_slice())
                .collect(),
            summaries: self
                .retrieval
                .entries
                .iter()
                .map(|entry| entry.summary.as_str())
                .collect(),
            adjacency: &self.retrieval.adjacency,
            path_index: &self.retrieval.path_index,
            parent_index: &self.retrieval.parent_index,
            df_cache: &self.retrieval.df_cache,
            posting_list: &self.retrieval.posting_list,
            avg_doc_len: self.retrieval.avg_doc_len,
            avg_verbatim_doc_len: self.retrieval.avg_verbatim_doc_len,
            module_index: &self.retrieval.module_index,
            vocab_bridge: &self.retrieval.vocab_bridge,
            morpheme_map: &self.retrieval.morpheme_map,
            session_utilization: &self.feedback.session_utilization,
            session_index: &self.retrieval.session_index,
            pmi_neighbors: &self.retrieval.pmi_neighbors,
            idf_n: self.retrieval.idf_n,
            delta_base: self.persistence.delta_base.load(Ordering::Relaxed),
        };
        let mut bytes = bincode::serialize(&cache)?;
        let cache_checksum = crc32fast::hash(&bytes).to_le_bytes();
        bytes.extend_from_slice(&cache_checksum);
        let cache_path = activation_cache_path(&self.persistence.project_root);
        if index_bytes > 0 && bytes.len() as u64 > index_bytes {
            if let Err(err) = std::fs::remove_file(&cache_path) {
                if err.kind() != std::io::ErrorKind::NotFound {
                    return Err(err.into());
                }
            }
            tracing::debug!(
                cache = %cache_path.display(),
                index_bytes,
                cache_bytes = bytes.len(),
                "Skipping activation cache write because it is larger than index.json"
            );
            return Ok(());
        }
        atomic_write(&cache_path, &bytes)
    }

    pub(in crate::index) fn index_compiled_files(
        &mut self,
        compiled: Vec<CompiledFile>,
        cascade_core_staleness: bool,
    ) -> usize {
        let new_count = compiled.len();
        for cf in compiled {
            let should_cascade = cascade_core_staleness && matches!(cf.meta.kind, NeuronKind::Core);
            let neuron_path = cf.neuron_path.clone();
            self.index_neuron(&cf.neuron_path, &cf.content, &cf.meta);
            if should_cascade {
                self.cascade_staleness(&neuron_path);
            }
        }
        new_count
    }

    pub(in crate::index) fn finalize_compile_pass(&mut self, root: &Path) -> Result<()> {
        self.apply_call_graph_synapses(root);
        self.apply_cochange_synapses(root);
        self.apply_rename_detection(root);
        self.rebuild_derived();
        self.save()
    }
}
