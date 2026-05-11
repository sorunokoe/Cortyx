//! Persistence helpers and serialized index types.

use super::*;

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
    /// S4-WAL: entries.len() at last full index.json write.
    wal_base: usize,
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
    session_utilization: &'a Vec<[usize; 2]>,
    session_index: &'a HashMap<String, Vec<usize>>,
    pmi_neighbors: &'a HashMap<String, Vec<String>>,
    idf_n: usize,
    /// S4-WAL: entries.len() at last full index.json write.
    wal_base: usize,
}

// ─── Schema migrations ────────────────────────────────────────────────────────

/// Migration function signature: transforms a stored JSON value to be compatible
/// with the next schema version. Fields not touched by the migration are preserved
/// (use_count, hit_count, staleness_multiplier etc. survive every upgrade).
pub(super) type MigrationFn = fn(serde_json::Value) -> serde_json::Value;

/// v7 → v8: rename `lsh_fingerprint: u64` → `lsh_fingerprints: [u64; 16]` (1024-bit LSH).
pub(super) fn migrate_v7_to_v8(mut entries_val: serde_json::Value) -> serde_json::Value {
    if let Some(arr) = entries_val.as_array_mut() {
        for entry in arr.iter_mut() {
            let old_fp = entry
                .get("lsh_fingerprint")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            let fps: Vec<serde_json::Value> =
                std::iter::once(serde_json::Value::Number(old_fp.into()))
                    .chain(std::iter::repeat_n(serde_json::Value::Number(0u64.into()), 15))
                    .collect();
            entry["lsh_fingerprints"] = serde_json::Value::Array(fps);
            if let Some(obj) = entry.as_object_mut() {
                obj.remove("lsh_fingerprint");
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

    /// Load an existing index from `.cortyx/index.json`, or create an empty one.
    /// Also loads embeddings.bin if present (BM25-only if absent).
    pub fn load_or_create(project_root: &Path) -> Result<Self> {
        let path = index_path(project_root);

        if let Some(mut idx) = Self::try_load_activation_cache(project_root, &path) {
            idx.coactivation_counts = load_coactivation_counts(project_root);
            return Ok(idx);
        }

        let mut idx = NeuronIndex {
            project_root: project_root.to_path_buf(),
            #[cfg(feature = "embed")]
            embeddings: load_embeddings(project_root),
            ..Default::default()
        };
        let mut activation_generation = 0u64;
        let mut persist_index = false;

        if path.exists() {
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
                                idx.entries = entries;
                            }
                            // Load session utilization history if present
                            if let Ok(util) = serde_json::from_value::<Vec<[usize; 2]>>(
                                raw.get("session_utilization").cloned().unwrap_or_default(),
                            ) {
                                idx.session_utilization = util;
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
                                    idx.entries = entries;
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

        // S4-WAL: replay any delta entries written by WAL-mode saves since the last
        // full index.json write.  Only applied when the base_count matches the number
        // of entries just loaded from index.json, preventing double-counting on
        // crash-between-write scenarios.
        let cortyx_dir = path.parent().unwrap_or_else(|| Path::new("."));
        let delta_path = cortyx_dir.join("index.delta.json");
        if let Ok(data) = std::fs::read_to_string(&delta_path) {
            if let Ok(raw) = serde_json::from_str::<serde_json::Value>(&data) {
                let base_count =
                    raw.get("base_count").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
                if base_count == idx.entries.len() {
                    if let Ok(delta_entries) = serde_json::from_value::<Vec<BM25Entry>>(
                        raw.get("entries").cloned().unwrap_or_default(),
                    ) {
                        tracing::debug!(n = delta_entries.len(), "Replaying WAL delta entries");
                        idx.entries.extend(delta_entries);
                    }
                } else {
                    tracing::debug!(
                        base_count,
                        loaded = idx.entries.len(),
                        "WAL delta base_count mismatch — skipping stale delta"
                    );
                }
            }
        }

        idx.rebuild_derived();
        idx.coactivation_counts = load_coactivation_counts(project_root);
        if persist_index {
            if let Err(e) = idx.save() {
                tracing::warn!("Failed to persist upgraded index metadata: {e}");
            }
        } else if activation_generation > 0 {
            let index_bytes = std::fs::metadata(&path).map(|meta| meta.len()).unwrap_or(0);
            if let Err(e) = idx.save_activation_cache(activation_generation, index_bytes) {
                tracing::warn!("Failed to refresh activation cache after rebuild: {e}");
            }
            idx.structural_artifacts_dirty
                .store(false, Ordering::Relaxed);
        }
        Ok(idx)
    }

    /// Reload the embedding store from disk (called after `cortyx compile --embed`).
    #[cfg(feature = "embed")]
    pub fn reload_embeddings(&mut self) {
        self.embeddings = load_embeddings(&self.project_root);
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
    pub fn save(&self) -> Result<()> {
        let path = index_path(&self.project_root);
        let cortyx_dir = path.parent().unwrap_or(Path::new("."));
        std::fs::create_dir_all(cortyx_dir)?;
        let structural_dirty = self.structural_artifacts_dirty.load(Ordering::Relaxed);
        let prior_generation = read_index_cache_generation(&path).unwrap_or(0);

        // S4-WAL: determine whether this save can use the delta (WAL) path.
        // WAL mode skips rewriting the monolithic index.json and instead appends
        // only new entries to a small delta file, reducing serialisation work from
        // O(N+n) to O(n) for pure-append mine batches.
        let delta_path = cortyx_dir.join("index.delta.json");
        let wal_base = self.wal_base.load(Ordering::Relaxed);
        let delta_len = self.entries.len().saturating_sub(wal_base);
        // Compact to a full write when delta exceeds 25% of the base — keeps the
        // fallback replay path fast and prevents unbounded delta file growth.
        let over_threshold = wal_base > 0 && delta_len > wal_base / 4;
        let in_wal_mode = wal_base > 0
            && !self.needs_full_save.load(Ordering::Relaxed)
            && !over_threshold
            && delta_len > 0;

        // In WAL mode the generation must stay unchanged so the activation cache passes
        // the index_generation check against the (unchanged) index.json on the next load.
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
        for entry in &self.entries {
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
        // or in WAL mode write only the delta entries to a small delta file.
        if in_wal_mode {
            // WAL mode: write only entries[wal_base..] to the delta file.
            let delta = serde_json::json!({
                "base_count": wal_base,
                "entries": &self.entries[wal_base..],
            });
            atomic_write_json(&delta_path, &delta)?;
            // Pass index_bytes=0 to bypass the size guard — the cache legitimately
            // contains more entries than the (unchanged) index.json.
            if let Err(e) = self.save_activation_cache(cache_generation, 0) {
                tracing::warn!("Failed to write activation cache (WAL mode): {e}");
            }
        } else {
            // Full save: rewrite index.json, clear any stale delta, update WAL baseline.
            let persisted = PersistedIndexRef {
                version: INDEX_VERSION,
                cache_generation,
                entries: &self.entries,
                session_utilization: &self.session_utilization,
                shards: &shard_names,
            };
            atomic_write_json(&path, &persisted)?;
            let _ = std::fs::remove_file(&delta_path); // clear stale delta on full write
            self.wal_base.store(self.entries.len(), Ordering::Relaxed);
            self.needs_full_save.store(false, Ordering::Relaxed);
            let index_bytes = std::fs::metadata(&path).map(|meta| meta.len()).unwrap_or(0);
            if let Err(e) = self.save_activation_cache(cache_generation, index_bytes) {
                tracing::warn!("Failed to write activation cache: {e}");
            }
        }
        if let Err(e) = save_coactivation_counts(&self.project_root, &self.coactivation_counts) {
            tracing::warn!("Failed to write coactivation counts: {e}");
        }
        if structural_dirty {
            self.structural_artifacts_dirty
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
        // S4-WAL: allow the cache to be larger than index.json when a delta file exists.
        // In WAL mode the cache contains entries not yet in the full index.json.
        let cortyx_dir = index_path.parent().unwrap_or_else(|| Path::new("."));
        let has_delta = cortyx_dir.join("index.delta.json").exists();
        if !has_delta && cache_bytes > index_bytes {
            tracing::debug!(
                cache = %cache_path.display(),
                index_bytes,
                cache_bytes,
                "Skipping activation cache because it is larger than index.json"
            );
            return None;
        }
        let bytes = std::fs::read(&cache_path).ok()?;
        let cache: PersistedActivationCache = bincode::deserialize(&bytes).ok()?;
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

        Some(NeuronIndex {
            project_root: project_root.to_path_buf(),
            entries,
            adjacency: cache.adjacency,
            path_index: cache.path_index,
            parent_index: cache.parent_index,
            df_cache: cache.df_cache,
            posting_list: cache.posting_list,
            avg_doc_len: cache.avg_doc_len,
            avg_verbatim_doc_len: cache.avg_verbatim_doc_len,
            module_index: cache.module_index,
            vocab_bridge: cache.vocab_bridge,
            morpheme_map: cache.morpheme_map,
            session_utilization: cache.session_utilization,
            session_index: cache.session_index,
            pmi_neighbors: cache.pmi_neighbors,
            idf_n: cache.idf_n,
            wal_base: AtomicUsize::new(cache.wal_base),
            #[cfg(feature = "embed")]
            embeddings: load_embeddings(project_root),
            ..Default::default()
        })
    }

    pub(in crate::index) fn save_activation_cache(
        &self,
        index_generation: u64,
        index_bytes: u64,
    ) -> Result<()> {
        let cache = PersistedActivationCacheRef {
            version: INDEX_VERSION,
            index_generation,
            entries: &self.entries,
            concept_clouds: self
                .entries
                .iter()
                .map(|entry| entry.concept_cloud.as_slice())
                .collect(),
            summaries: self
                .entries
                .iter()
                .map(|entry| entry.summary.as_str())
                .collect(),
            adjacency: &self.adjacency,
            path_index: &self.path_index,
            parent_index: &self.parent_index,
            df_cache: &self.df_cache,
            posting_list: &self.posting_list,
            avg_doc_len: self.avg_doc_len,
            avg_verbatim_doc_len: self.avg_verbatim_doc_len,
            module_index: &self.module_index,
            vocab_bridge: &self.vocab_bridge,
            morpheme_map: &self.morpheme_map,
            session_utilization: &self.session_utilization,
            session_index: &self.session_index,
            pmi_neighbors: &self.pmi_neighbors,
            idf_n: self.idf_n,
            wal_base: self.wal_base.load(Ordering::Relaxed),
        };
        let bytes = bincode::serialize(&cache)?;
        let cache_path = activation_cache_path(&self.project_root);
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
