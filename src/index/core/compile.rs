// This file is a submodule of `crate::index::core`.
// It contains `impl NeuronIndex` methods extracted from helpers.rs.
// All visibility is relative to `crate::index` (the parent of `core`).
use super::*;

impl NeuronIndex {
    // ── Compile ───────────────────────────────────────────────────────────────

    /// Walk the project tree, create stubs for new/changed source files.
    ///
    /// Idempotent: re-running on an unchanged project is a no-op (only the
    /// hash check causes any work). Returns the total number of neurons managed.
    ///
    /// Enhancements per compile pass:
    /// - **AST Bootstrap**: extracts function signatures + types from source at compile
    ///   time and pre-fills the `api` section of new stubs so BM25 has vocabulary
    ///   from day 1, before any LLM curation.
    /// - **Auto-Synapse**: parses import statements and creates `Imports`-typed synapse
    ///   edges automatically so the graph traversal works from day 1.
    /// - **Git Confidence**: queries `git ls-files` once to classify files as committed
    ///   (1.0), modified (0.9), or untracked (0.85) — applied as a mild BM25 multiplier.
    ///
    /// # Errors
    ///
    /// Returns an error if the underlying operation fails.
    pub fn compile(&mut self) -> Result<usize> {
        let root = self.persistence.project_root.clone();
        let ndir = neuron_dir(&root);
        std::fs::create_dir_all(&ndir)?;

        // Ensure the project neuron exists.
        self.ensure_project_neuron(&root)?;
        // S5 (R15 NE4): generate wake-up neurons from project metadata.
        self.ensure_wake_up_neurons(&root, &ndir)?;

        // Build git confidence map once (3 git commands, silent on non-git projects).
        let git_confidence = build_git_confidence_map(&root);

        // S4 — Parallel compile: Phase 1 collect files, Phase 2 process in parallel,
        // Phase 3 batch-insert sequentially.
        //
        // Each file's pipeline (hash-check → AST extract → stub write → meta write) is
        // fully data-parallel: no shared mutable state across files. Only the final
        // index_neuron() calls require &mut self and run sequentially in Phase 3.
        //
        // Expected speedup: 4–8× on a modern multi-core laptop for 1 000-file projects.

        // Phase 1: collect all source file paths (sequential WalkDir, fast).
        let files: Vec<PathBuf> = WalkDir::new(&root)
            .min_depth(1)
            .into_iter()
            .filter_map(|e| e.ok())
            .filter(|e| e.file_type().is_file())
            .map(|e| e.path().to_path_buf())
            .collect();
        let ignore_patterns = load_cortyxignore(&root);
        let files: Vec<PathBuf> = files
            .into_iter()
            .filter(|f| !is_cortyxignored(f, &root, &ignore_patterns))
            .collect();

        // Phase 2: hash-check + AST + stub/meta writes (parallel, I/O-bound).
        // process_source_file returns Vec<CompiledFile>: [Core] + any UseCase sub-neurons (S3).
        let compiled: Vec<CompiledFile> = files
            .par_iter()
            .flat_map(|abs| process_source_file(abs, &root, &git_confidence))
            .collect();

        // Phase 3: sequential batch insert into the in-memory index.
        let new_count = self.index_compiled_files(compiled, false);
        self.finalize_compile_pass(&root)?;
        Ok(new_count)
    }

    /// Incremental compile — processes only paths currently in the in-memory dirty set.
    ///
    /// The file watcher inserts changed source paths into `self.watcher.dirty_set` after each
    /// debounce batch.  On next server start (or `cortyx compile --incremental`), only
    /// those files are re-indexed instead of walking the entire tree — O(changed) not O(all).
    ///
    /// Migration: if the dirty set is empty and a legacy `.cortyx/dirty.json` file exists,
    /// its paths are loaded into the dirty set first (one-time on-disk → in-memory upgrade).
    ///
    /// Falls back to a full `compile()` when the dirty set is empty and no legacy file exists.
    ///
    /// # Errors
    ///
    /// Returns an error if the underlying operation fails.
    pub fn compile_dirty(&mut self) -> Result<usize> {
        // Migration: pull any legacy dirty.json paths into the in-memory set before draining.
        let dirty_file = dirty_path(&self.persistence.project_root);
        if dirty_file.exists() {
            if let Ok(raw) = std::fs::read_to_string(&dirty_file) {
                if let Ok(paths) = serde_json::from_str::<Vec<PathBuf>>(&raw) {
                    if !paths.is_empty() {
                        self.watcher.extend(paths);
                    }
                }
            }
            // Remove the legacy file regardless — the in-memory set is now authoritative.
            let _ = std::fs::remove_file(&dirty_file);
        }

        // Atomically drain the dirty set (swap with an empty one so the watcher can
        // continue inserting into the new empty set while we compile).
        let dirty_paths = self.watcher.drain_paths();

        if dirty_paths.is_empty() {
            tracing::debug!("dirty_set empty — falling back to full compile.");
            return self.compile();
        }

        tracing::info!(
            "Incremental compile: processing {} dirty file(s).",
            dirty_paths.len()
        );

        let root = self.persistence.project_root.clone();
        let git_confidence = build_git_confidence_map(&root);
        let ignore_patterns = load_cortyxignore(&root);
        let dirty_paths: Vec<PathBuf> = dirty_paths
            .into_iter()
            .filter(|path| !is_cortyxignored(path, &root, &ignore_patterns))
            .collect();
        let compiled: Vec<CompiledFile> = dirty_paths
            .par_iter()
            .flat_map(|abs| process_source_file(abs, &root, &git_confidence))
            .collect();

        let new_count = self.index_compiled_files(compiled, true);
        self.finalize_compile_pass(&root)?;
        Ok(new_count)
    }

    /// Returns a cloned handle to the in-memory dirty set.
    ///
    /// The file watcher holds this handle to insert changed paths without going through
    /// the index write lock.  This breaks the previous "lock held during compile" pattern.
    pub fn dirty_set_handle(&self) -> std::sync::Arc<std::sync::Mutex<HashSet<PathBuf>>> {
        self.watcher.handle()
    }

    /// Add/update a single entry in the in-memory index without rebuilding derived structures.
    ///
    /// Use this in tight loops (e.g. bulk mining) and call `commit()` once at the end.
    pub fn stage(&mut self, neuron_path: &Path, content: &str, meta: &NeuronMeta) {
        self.index_neuron(neuron_path, content, meta);
    }

    /// Rebuild all derived structures and persist the index.
    ///
    /// Call after a batch of `stage()` calls to apply changes in a single pass.
    ///
    /// # Errors
    ///
    /// Returns an error if the underlying operation fails.
    pub fn commit(&mut self) -> Result<()> {
        self.rebuild_derived();
        self.save()
    }

    /// Add/update a single neuron in the index (called by MCP tools).
    ///
    /// Persists the index to disk after every mutation so MCP changes
    /// survive a server restart.
    ///
    /// # Errors
    ///
    /// Returns an error if the underlying operation fails.
    pub fn upsert_neuron(
        &mut self,
        neuron_path: &Path,
        content: &str,
        meta: &NeuronMeta,
    ) -> Result<()> {
        self.stage(neuron_path, content, meta);
        self.commit()
    }
}

fn load_cortyxignore(root: &Path) -> Vec<String> {
    let path = root.join(".cortyxignore");
    match std::fs::read_to_string(&path) {
        Ok(contents) => contents
            .lines()
            .map(str::trim)
            .filter(|line| !line.is_empty() && !line.starts_with('#'))
            .map(String::from)
            .collect(),
        Err(_) => Vec::new(),
    }
}

fn is_cortyxignored(file: &Path, root: &Path, patterns: &[String]) -> bool {
    let rel = match file.strip_prefix(root) {
        Ok(rel) => rel,
        Err(_) => return false,
    };
    let rel_str = rel.to_string_lossy();

    for pattern in patterns {
        if rel_str == pattern.as_str() {
            return true;
        }
        if pattern.ends_with('/') && rel_str.starts_with(pattern.as_str()) {
            return true;
        }
        if let Some(suffix) = pattern.strip_prefix("**/") {
            if rel_str.ends_with(suffix) || rel_str.contains(&format!("/{suffix}")) {
                return true;
            }
        }
        if let Some(ext) = pattern.strip_prefix("*.") {
            if file.extension().and_then(|e| e.to_str()) == Some(ext) {
                return true;
            }
        }
    }

    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cortyxignore_matches_directory_prefixes() {
        let root = Path::new("/project");
        let file = root.join("secrets/token.txt");
        assert!(is_cortyxignored(&file, root, &["secrets/".to_string()]));
    }

    #[test]
    fn cortyxignore_matches_glob_suffix_and_extension() {
        let root = Path::new("/project");
        assert!(is_cortyxignored(
            &root.join("docs/private.md"),
            root,
            &["**/private.md".to_string()]
        ));
        assert!(is_cortyxignored(
            &root.join("notes/credentials.secret"),
            root,
            &["*.secret".to_string()]
        ));
    }
}
