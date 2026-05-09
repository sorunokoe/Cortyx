// This file is a submodule of `crate::index::core`.
// It contains `impl NeuronIndex` methods extracted from helpers.rs.
// All visibility is relative to `crate::index` (the parent of `core`).
use super::*;

impl NeuronIndex {
    // ── Internal helpers ──────────────────────────────────────────────────────

    /// Mine each source file for function call sites that match public functions
    /// defined in *other* source files of the project.
    ///
    /// Workflow:
    /// 1. Build a vocabulary map `fn_name → source_rel_path` from all entries'
    ///    extracted function names (stored in `term_freq` keys during compile).
    ///    Entries with no functions in their term_freq are skipped.
    /// 2. Walk each source file, call `ast_extractor::extract_call_sites`,
    ///    and for each detected `CallEdge`, emit a `Calls`-typed synapse from
    ///    the calling neuron to the callee neuron (if one doesn't already exist).
    ///
    /// This is a second compile pass and runs in O(files × |vocab|) — both are
    /// typically small so runtime is negligible.
    pub(in crate::index) fn apply_call_graph_synapses(&mut self, root: &Path) {
        // Build fn_name → source_path vocabulary from the already-loaded entries.
        // We use term_freq keys that look like function names (alphabetic, no spaces).
        // This is approximate but practical — false positives are filtered by
        // the self-loop guard in extract_call_sites.
        //
        // A tighter approach would be to store a dedicated `functions: Vec<String>`
        // field in BM25Entry, but term_freq already contains them from AST Bootstrap.
        // Function names are pure alphabetic tokens, distinct from normal prose terms.
        let mut fn_vocab: HashMap<String, PathBuf> = HashMap::new();
        for entry in &self.entries {
            let rel_source = entry
                .neuron_path
                .strip_prefix(root)
                .map(|r| r.to_path_buf())
                .unwrap_or_else(|_| entry.neuron_path.clone());

            // Extract function names: those that appear in term_freq AND match the
            // pattern of a public function name (all word chars, len ≥ 3, not all-lowercase
            // common English words). We use a simple heuristic rather than re-running AST.
            for term in entry.term_freq.keys() {
                // Public function names are typically CamelCase or snake_case identifiers
                // ≥ 3 chars with no digits-only and not a BM25 stop-word.
                if term.len() >= 3 && term.chars().all(|c| c.is_alphanumeric() || c == '_') {
                    fn_vocab
                        .entry(term.clone())
                        .or_insert_with(|| rel_source.clone());
                }
            }
        }

        if fn_vocab.is_empty() {
            return;
        }

        // Walk each source file and find call sites.
        let source_extensions = [
            "rs", "py", "ts", "tsx", "js", "jsx", "go", "swift", "kt", "java", "cs", "rb", "c",
            "cpp", "cc",
        ];
        let walker = WalkDir::new(root)
            .min_depth(1)
            .into_iter()
            .filter_map(|e| e.ok());
        let mut synapse_patches: Vec<(PathBuf, PathBuf)> = Vec::new(); // (caller_neuron, callee_neuron)

        for entry in walker {
            if !entry.file_type().is_file() {
                continue;
            }
            let abs = entry.path();
            let rel = abs.strip_prefix(root).unwrap_or(abs);
            let ext = rel.extension().and_then(|e| e.to_str()).unwrap_or("");
            if !source_extensions.contains(&ext) || should_skip(rel) {
                continue;
            }
            let Ok(content) = std::fs::read_to_string(abs) else {
                continue;
            };
            let source_rel = rel.to_string_lossy();
            let call_edges = ast_extractor::extract_call_sites(&source_rel, &content, &fn_vocab);
            if call_edges.is_empty() {
                continue;
            }
            let caller_neuron = core_neuron_path(abs, root);
            for edge in call_edges {
                let callee_source = root.join(&edge.callee_file);
                let callee_neuron = core_neuron_path(&callee_source, root);
                if callee_neuron != caller_neuron {
                    synapse_patches.push((caller_neuron.clone(), callee_neuron));
                }
            }
        }

        // Apply collected patches to meta files and in-memory entries.
        for (caller_neuron, callee_neuron) in synapse_patches {
            let meta_file = meta_path(&caller_neuron);
            let Ok(data) = std::fs::read_to_string(&meta_file) else {
                continue;
            };
            let Ok(mut meta) = serde_json::from_str::<NeuronMeta>(&data) else {
                continue;
            };
            let already_exists = meta
                .synapses
                .iter()
                .any(|s| s.target == callee_neuron && matches!(s.edge_type, SynapseType::Calls));
            if already_exists {
                continue;
            }
            meta.synapses.push(Synapse::new(
                callee_neuron.clone(),
                SynapseType::Calls,
                "auto-inferred from call-site scan".to_string(),
            ));
            if let Err(e) = atomic_write_json(&meta_file, &meta) {
                tracing::warn!(
                    "Failed to persist call-graph synapse for {}: {e}",
                    meta_file.display()
                );
            }
            // Update in-memory entry as well.
            if let Some(&idx) = self.path_index.get(&caller_neuron) {
                self.entries[idx].synapses.push(Synapse::new(
                    callee_neuron,
                    SynapseType::Calls,
                    "auto-inferred from call-site scan".to_string(),
                ));
            }
        }
    }

    /// Mine `git log --name-only` to find files co-committed ≥ `min_cochange` times.
    ///
    /// For each qualifying pair, add a `SemanticRelated` auto-synapse to the
    /// source neuron's meta if one does not already exist. Called once per compile.
    pub(in crate::index) fn apply_cochange_synapses(&mut self, root: &Path) {
        /// Cap on files per commit before skipping the pair-wise O(n²) step.
        ///
        /// A commit touching more than this many files is almost certainly a
        /// bulk change (dependency bump, generated code, refactor) where co-change
        /// is not a useful semantic signal. Without this cap, a 500-file commit
        /// generates ~125,000 pairs, making compile time degenerate on large repos.
        const MAX_FILES_PER_COMMIT: usize = 50;

        // Adaptive minimum co-change threshold based on repo size.
        // Small repos (≤50 neurons) produce sparse commit histories; 2 co-changes
        // is strong signal. Large repos (>500 neurons) have noisy histories and
        // benefit from a higher bar to avoid false semantic edges.
        let min_cochange: u32 = match self.path_index.len() {
            n if n <= 50 => 2,
            n if n <= 500 => 3,
            _ => 5,
        };

        let output = match std::process::Command::new("git")
            .args(["log", "--name-only", "--pretty=format:"])
            .current_dir(root)
            .output()
        {
            Ok(o) if o.status.success() => o.stdout,
            _ => return, // not a git repo or git unavailable — skip silently
        };

        // Build per-commit file lists and count co-changes
        let mut cochange: HashMap<(PathBuf, PathBuf), u32> = HashMap::new();
        let mut commit_files: Vec<PathBuf> = Vec::new();

        for line in String::from_utf8_lossy(&output).lines() {
            let trimmed = line.trim();
            if trimmed.is_empty() {
                // Commit boundary — process accumulated files only if the commit is
                // small enough that co-change is a meaningful signal.
                if commit_files.len() <= MAX_FILES_PER_COMMIT {
                    for i in 0..commit_files.len() {
                        for j in (i + 1)..commit_files.len() {
                            let (a, b) = (&commit_files[i], &commit_files[j]);
                            // Canonical ordering so (a,b) == (b,a)
                            let key = if a <= b {
                                (a.clone(), b.clone())
                            } else {
                                (b.clone(), a.clone())
                            };
                            *cochange.entry(key).or_insert(0) += 1;
                        }
                    }
                }
                commit_files.clear();
            } else {
                commit_files.push(PathBuf::from(trimmed));
            }
        }
        // Flush any trailing files — git log output may not end with a blank line,
        // which would silently drop the most-recent commit's co-change signal.
        if !commit_files.is_empty() && commit_files.len() <= MAX_FILES_PER_COMMIT {
            for i in 0..commit_files.len() {
                for j in (i + 1)..commit_files.len() {
                    let (a, b) = (&commit_files[i], &commit_files[j]);
                    let key = if a <= b {
                        (a.clone(), b.clone())
                    } else {
                        (b.clone(), a.clone())
                    };
                    *cochange.entry(key).or_insert(0) += 1;
                }
            }
        }

        // Add synapses for qualifying pairs
        let mut changes: Vec<(PathBuf, Synapse)> = Vec::new();
        for ((fa, fb), count) in &cochange {
            if *count < min_cochange {
                continue;
            }
            let na = core_neuron_path(&root.join(fa), root);
            let nb = core_neuron_path(&root.join(fb), root);
            let weight = SynapseWeight::new((0.5_f32 + *count as f32 * 0.05).min(0.9));
            let reason = format!("git co-change: committed together {count}×");

            // Only create synapses for neurons that exist in our index
            if self.path_index.contains_key(&na) && self.path_index.contains_key(&nb) {
                changes.push((
                    na.clone(),
                    Synapse {
                        target: nb.clone(),
                        edge_type: SynapseType::SemanticRelated,
                        weight,
                        reason: reason.clone(),
                        learned_weight: 0.0,
                        traversal_count: 0,
                        last_co_activation_day: 0,
                    },
                ));
                changes.push((
                    nb,
                    Synapse {
                        target: na,
                        edge_type: SynapseType::SemanticRelated,
                        weight,
                        reason,
                        learned_weight: 0.0,
                        traversal_count: 0,
                        last_co_activation_day: 0,
                    },
                ));
            }
        }

        for (source_neuron, syn) in changes {
            let meta_p = meta_path(&source_neuron);
            if let Ok(data) = std::fs::read_to_string(&meta_p) {
                if let Ok(mut meta) = serde_json::from_str::<NeuronMeta>(&data) {
                    let already = meta.synapses.iter().any(|s| s.target == syn.target);
                    if !already {
                        meta.synapses.push(syn.clone());
                        if let Err(e) = atomic_write_json(&meta_p, &meta) {
                            tracing::warn!(
                                "Failed to persist co-change synapse for {}: {e}",
                                meta_p.display()
                            );
                        }
                    }
                }
            }
            if let Some(&i) = self.path_index.get(&source_neuron) {
                let already = self.entries[i]
                    .synapses
                    .iter()
                    .any(|s| s.target == syn.target);
                if !already {
                    self.entries[i].synapses.push(syn);
                }
            }
        }
    }

    /// Add or replace a single entry in `self.entries` (does NOT rebuild derived).
    pub fn index_neuron(&mut self, neuron_path: &Path, content: &str, meta: &NeuronMeta) {
        let index_content = content;

        let terms = tokenize(index_content);
        let mut tf: HashMap<String, f32> = HashMap::new();
        for t in &terms {
            *tf.entry(t.clone()).or_insert(0.0) += 1.0;
        }

        // P3-B: Paraphrase + alias surface boost.
        // ## paraphrases and the narrow fact_aliases surface bridge natural-language
        // questions to answer-bearing facts without polluting summaries with broad
        // category vocabulary.
        // This closes the vocabulary gap: documents contain both answer vocabulary
        // (original content) and question vocabulary (these sections).
        {
            use crate::neuron::parse_sections;
            let sections = parse_sections(index_content);
            for section_name in ["paraphrases", "query_surface", "fact_aliases"] {
                if let Some(section_content) = sections.get(section_name) {
                    for t in tokenize(section_content) {
                        let v = tf.entry(t).or_insert(0.0);
                        *v += 0.5; // boost: question vocab is high-signal (kept low to avoid over-boosting generic category tokens)
                    }
                }
            }
        }

        // NE-6: User-turn boost for Verbatim (conversation) neurons.
        // In episodic memory retrieval, facts are stated by the user, not the assistant.
        // User utterances are the ground truth for SSU/KU/multi queries. Assistant text
        // is context/response and should not dominate BM25 scoring.
        // Implementation: give user-turn lines an extra +1.0 TF weight (doubling their
        // effective TF vs assistant lines), making user-disclosed facts rank much higher.
        if matches!(meta.kind, crate::neuron::NeuronKind::Verbatim) {
            for line in index_content.lines() {
                let lower = line.as_bytes();
                let is_user = lower.starts_with(b"user:")
                    || lower.starts_with(b"User:")
                    || lower.starts_with(b"human:")
                    || lower.starts_with(b"Human:");
                if is_user && line.len() > 6 {
                    for t in tokenize(line) {
                        *tf.entry(t).or_insert(0.0) += 1.0;
                    }
                }
            }
        }

        // A1: Multi-Source Vocabulary Injection — inject soft terms from source file
        // (git commit messages + inline comments) at 0.3× weight. These terms are never
        // shown in the retrieved context, but improve BM25 query matching for cold stubs.
        if let Some(source_abs) = meta.source_files.first() {
            for t in git_extractor::extract_soft_terms(source_abs) {
                // Only inject when not already present in neuron content — hard terms win.
                let v = tf.entry(t).or_insert(0.0);
                if *v == 0.0 {
                    *v = 0.3;
                }
            }
        }

        // B3: Alias Injection — inject natural-language aliases for public function/type names
        // at 0.5× weight. "get_user" → ["fetch", "retrieve", "account", "member"].
        // These aliases bridge the lexical gap between user queries and code identifiers
        // without any model download.
        {
            // Collect function/type names from task_pattern (sub-neuron) or from the neuron
            // file stem (proxy for the source file's primary identifier).
            let mut names: Vec<String> = Vec::new();
            if let Some(ref pattern) = meta.task_pattern {
                names.push(pattern.clone());
            }
            // Also include the neuron path stem as a fallback source of identifiers
            if let Some(stem) = neuron_path
                .file_stem()
                .and_then(|s| s.to_str())
                .map(|s| s.trim_end_matches(".context").to_string())
            {
                names.push(stem);
            }
            if !names.is_empty() {
                for t in alias_gen::generate_alias_terms(&names) {
                    let v = tf.entry(t).or_insert(0.0);
                    if *v < 0.5 {
                        *v = 0.5;
                    }
                }
            }
        }

        let task_pattern_terms = meta
            .task_pattern
            .as_deref()
            .map(tokenize)
            .unwrap_or_default();

        // Normalize synapse targets to absolute paths so the adjacency graph
        // uses consistent keys regardless of whether the path was parsed from
        // a markdown backtick (relative) or stored directly (absolute).
        //
        // S-1: Validate that the resolved target stays inside the neuron directory.
        // This prevents path traversal attacks via crafted .cortyx/neurons/*.json files
        // (e.g. a compromised CI artifact injecting "../../etc/sensitive").
        let ndir = neuron_dir(&self.project_root);
        let synapses: Vec<Synapse> = meta
            .synapses
            .iter()
            .filter_map(|s| {
                let target = if s.target.is_absolute() {
                    s.target.clone()
                } else {
                    ndir.join(&s.target)
                };
                if !target.starts_with(&ndir) {
                    tracing::warn!(
                        "Skipping synapse with path-traversal target {:?} in {:?}",
                        target,
                        neuron_path
                    );
                    return None;
                }
                Some(Synapse {
                    target,
                    ..s.clone()
                })
            })
            .collect();

        // S-III (R16): Self-Quality Score — fraction of neuron terms that overlap with
        // the corresponding source file's AST-extracted terms.
        // Only computed for Core neurons with a known source file; defaults to 1.0 (neutral).
        let quality_score: f32 =
            if matches!(meta.kind, NeuronKind::Core) && !meta.source_files.is_empty() {
                let source_path = &meta.source_files[0];
                if let Ok(source_text) = std::fs::read_to_string(source_path) {
                    let source_rel = source_path.to_string_lossy();
                    let ast = ast_extractor::extract_signatures(&source_rel, &source_text);
                    // Build source AST term set from all function/type names (split on _ and camelCase)
                    let mut ast_terms: std::collections::HashSet<String> =
                        std::collections::HashSet::new();
                    for name in ast.functions.iter().chain(ast.types.iter()) {
                        ast_terms.extend(tokenize(name));
                    }
                    if ast_terms.is_empty() {
                        1.0 // no AST info → neutral
                    } else {
                        let neuron_terms: std::collections::HashSet<&str> =
                            tf.keys().map(|s| s.as_str()).collect();
                        let overlap = ast_terms
                            .iter()
                            .filter(|t| neuron_terms.contains(t.as_str()))
                            .count();
                        overlap as f32 / ast_terms.len() as f32
                    }
                } else {
                    1.0
                }
            } else {
                1.0 // non-Core or no source → neutral
            };

        // S-II (R16/R17 Sol4): Compute a 16-seed SimHash ensemble for LSH fallback.
        let lsh_fingerprints = simhash_1024(&tf);

        // S-I (R16): Extract Tier-1 summary from neuron content.
        // Takes: first non-empty line of `## purpose` section + first line of `## pitfalls`.
        // Stored in memory only (not persisted); rebuilt from neuron file at each index_neuron call.
        let summary = extract_neuron_summary(content);
        let has_move_residence_evidence = content_has_move_residence_evidence(content);

        let entry = BM25Entry {
            neuron_path: neuron_path.to_path_buf(),
            kind: meta.kind.clone(),
            term_freq: tf,
            term_count: terms.len(),
            // Use meta.tokens when available (set by compile/upsert after reading disk).
            // Fall back to estimating from content so the token budget works in tests
            // and when index_neuron is called before NeuronMeta.tokens is populated.
            tokens: if meta.tokens > 0 {
                meta.tokens
            } else {
                estimate_tokens(content).get().max(10)
            },
            task_pattern_terms,
            parent: meta.parent.clone(),
            synapses,
            source_files: meta.source_files.clone(),
            module: meta.module.clone(),
            confidence_score: meta.confidence_score,
            use_count: meta.use_count,
            hit_count: meta.hit_count,
            staleness_multiplier: 1.0,
            concept_cloud: Vec::new(), // populated by build_concept_clouds() in rebuild_derived
            synonym_cloud: Vec::new(), // populated by record_coactivation() at runtime
            lsh_fingerprints,
            quality_score,
            summary,
            timestamp_secs: parse_iso8601_to_secs(meta.timestamp.as_deref()),
            has_move_residence_evidence,
            // R21 T6: Extract session_id from neuron filename stem for Verbatim neurons.
            // Pattern: "lme_0060_0_user.verbatim.md" → session_id = "lme_0060"
            // Split on '_', take first two parts if the stem follows the N_N pattern.
            session_id: if matches!(meta.kind, NeuronKind::Verbatim) {
                neuron_path
                    .file_name()
                    .and_then(|n| n.to_str())
                    .map(|name| {
                        // strip extension(s): "lme_0060_0_user.verbatim.md" → "lme_0060_0_user"
                        let stem = name.split('.').next().unwrap_or(name);
                        // take first two underscore-separated parts: "lme" + "0060"
                        let parts: Vec<&str> = stem.splitn(3, '_').collect();
                        if parts.len() >= 2 {
                            format!("{}_{}", parts[0], parts[1])
                        } else {
                            stem.to_string()
                        }
                    })
                    .unwrap_or_default()
            } else {
                String::new()
            },
        };

        if let Some(&pos) = self.path_index.get(neuron_path) {
            self.entries[pos] = entry;
            self.has_pending_updates = true;
            self.needs_full_save.store(true, Ordering::Relaxed);
        } else {
            let pos = self.entries.len();
            self.path_index.insert(neuron_path.to_path_buf(), pos);
            self.entries.push(entry);
            self.pending_append_count += 1;
        }
    }

    /// Rebuild all derived structures — public entry point for `cortyx prune`.
    ///
    /// Prune evicts entries individually then calls this once to reconstruct
    /// path_index, adjacency, df_cache, etc. in a single O(n) pass.
    pub fn rebuild_derived_pub(&mut self) {
        // Force full rebuild: prune may have removed existing entries, so the
        // incremental delta path (which only handles appends) is not safe here.
        self.pending_append_count = 0;
        self.has_pending_updates = true;
        // S4-WAL: prune removes entries — invalidate WAL baseline and force full save.
        self.wal_base.store(0, Ordering::Relaxed);
        self.needs_full_save.store(true, Ordering::Relaxed);
        self.rebuild_derived();
    }

    /// Rebuild all derived structures in a single O(n) pass.
    ///
    /// Previously five separate passes (path_index, parent_index, adjacency, df_cache,
    /// module_index); merged to reduce cache pressure and wall-clock time ~5×.
    pub(in crate::index) fn rebuild_derived(&mut self) {
        // S7: Incremental delta — skip the full clear+rebuild when only new entries were
        // appended (no updates).  This reduces the hot path (mining a new file into an
        // existing index) from O(N+n) to O(n) for the HashMap phase.
        if self.pending_append_count > 0 && !self.has_pending_updates && self.idf_n > 0 {
            self.rebuild_derived_delta();
            return;
        }

        self.path_index.clear();
        self.parent_index.clear();
        self.adjacency.clear();
        self.df_cache.clear();
        self.posting_list.clear();
        self.module_index.clear();
        self.session_index.clear(); // R21 T6
        self.idf_n = 0;

        let mut non_agg_total_terms = 0usize;
        let mut verbatim_total_terms = 0usize;
        let mut verbatim_count = 0usize;

        for (i, entry) in self.entries.iter().enumerate() {
            // path_index
            self.path_index.insert(entry.neuron_path.clone(), i);

            // parent_index
            if let Some(p) = &entry.parent {
                self.parent_index.entry(p.clone()).or_default().push(i);
            }

            // adjacency (forward + reverse edges)
            for syn in &entry.synapses {
                self.adjacency
                    .entry(entry.neuron_path.clone())
                    .or_default()
                    .push(syn.clone());

                self.adjacency
                    .entry(syn.target.clone())
                    .or_default()
                    .push(Synapse {
                        target: entry.neuron_path.clone(),
                        edge_type: syn.edge_type.inverse(),
                        weight: SynapseWeight::new(syn.weight.get() * 0.7),
                        reason: format!("← {}", syn.reason),
                        learned_weight: 0.0,
                        traversal_count: 0,
                        last_co_activation_day: 0,
                    });
            }

            // df_cache + posting_list.
            // IMPORTANT: Aggregate neurons (word-count summaries, dollar totals) must NOT
            // contribute to df_cache.  An _count_music.aggregate.md neuron contains "music"
            // dozens of times, inflating df("music") and crushing its IDF.  This caused a
            // 5-entry SSU regression: session 329 ("music"×18, no "streaming"/"service") lost
            // to session 309 ("service"×7) because IDF("music") collapsed while IDF("service")
            // stayed high.  Excluding Aggregate from df_cache restores the IDF calibration
            // from the e18c4e6 baseline (100% SSU) even when aggregates are mined.
            // Posting-list is still built for ALL kinds so counting_augment can find Aggregates.
            let is_aggregate = matches!(entry.kind, NeuronKind::Aggregate);
            for term in entry.term_freq.keys() {
                if !is_aggregate {
                    *self.df_cache.entry(term.clone()).or_insert(0) += 1;
                }
                self.posting_list.entry(term.clone()).or_default().push(i);
            }
            if !is_aggregate {
                self.idf_n += 1;
            }

            // module_index
            if let Some(m) = &entry.module {
                self.module_index.entry(m.clone()).or_default().push(i);
            }

            // R21 T6: session_index — for session-level grouping at retrieval time
            if !entry.session_id.is_empty() {
                self.session_index
                    .entry(entry.session_id.clone())
                    .or_default()
                    .push(i);
            }

            if !is_aggregate {
                non_agg_total_terms += entry.term_count;
            }
            if matches!(entry.kind, NeuronKind::Verbatim) {
                verbatim_total_terms += entry.term_count;
                verbatim_count += 1;
            }
        }

        // avg_doc_len excludes Aggregate neurons so it matches e18c4e6 calibration.
        self.avg_doc_len = if self.idf_n == 0 {
            0.0
        } else {
            non_agg_total_terms as f32 / self.idf_n as f32
        };
        self.avg_verbatim_doc_len = if verbatim_count == 0 {
            self.avg_doc_len
        } else {
            verbatim_total_terms as f32 / verbatim_count as f32
        };

        self.build_vocab_bridge();
        self.build_morpheme_map();
        self.build_concept_clouds();
        self.apply_peer_vocab_borrowing();
        self.merge_cooccurrence_into_vocab_bridge();
        self.load_pmi_neighbors();
        self.structural_artifacts_dirty
            .store(true, Ordering::Relaxed);
        self.pending_append_count = 0;
        self.has_pending_updates = false;
    }

    /// Incremental derived-structure update for pure-append batches (S7).
    ///
    /// When only new entries were appended (no existing entries were modified), we
    /// skip clearing and rebuilding the large HashMaps from scratch.  Instead we
    /// process only the `pending_append_count` newest entries and add their
    /// contributions to the existing structures in O(n) rather than O(N+n).
    ///
    /// The bridge/cloud/neighbor builds (vocab_bridge, morpheme_map, concept_clouds,
    /// pmi_neighbors) still run over the full corpus because they are O(terms), not
    /// O(entries²), and must reflect the complete vocabulary.
    pub(in crate::index) fn rebuild_derived_delta(&mut self) {
        let new_start = self.entries.len().saturating_sub(self.pending_append_count);

        for (offset, entry) in self.entries[new_start..].iter().enumerate() {
            let abs_i = new_start + offset;

            // path_index is already maintained by index_neuron(), but ensure consistency.
            self.path_index.insert(entry.neuron_path.clone(), abs_i);

            if let Some(p) = &entry.parent {
                self.parent_index.entry(p.clone()).or_default().push(abs_i);
            }

            for syn in &entry.synapses {
                self.adjacency
                    .entry(entry.neuron_path.clone())
                    .or_default()
                    .push(syn.clone());
                self.adjacency
                    .entry(syn.target.clone())
                    .or_default()
                    .push(Synapse {
                        target: entry.neuron_path.clone(),
                        edge_type: syn.edge_type.inverse(),
                        weight: SynapseWeight::new(syn.weight.get() * 0.7),
                        reason: format!("← {}", syn.reason),
                        learned_weight: 0.0,
                        traversal_count: 0,
                        last_co_activation_day: 0,
                    });
            }

            let is_aggregate = matches!(entry.kind, NeuronKind::Aggregate);
            for term in entry.term_freq.keys() {
                if !is_aggregate {
                    *self.df_cache.entry(term.clone()).or_insert(0) += 1;
                }
                self.posting_list
                    .entry(term.clone())
                    .or_default()
                    .push(abs_i);
            }
            if !is_aggregate {
                self.idf_n += 1;
            }

            if let Some(m) = &entry.module {
                self.module_index.entry(m.clone()).or_default().push(abs_i);
            }

            if !entry.session_id.is_empty() {
                self.session_index
                    .entry(entry.session_id.clone())
                    .or_default()
                    .push(abs_i);
            }
        }

        // Recompute avg_doc_len from all entries (O(n) integer addition — cheap).
        let mut non_agg_total_terms = 0usize;
        let mut verbatim_total_terms = 0usize;
        let mut verbatim_count = 0usize;
        for entry in &self.entries {
            let is_aggregate = matches!(entry.kind, NeuronKind::Aggregate);
            if !is_aggregate {
                non_agg_total_terms += entry.term_count;
            }
            if matches!(entry.kind, NeuronKind::Verbatim) {
                verbatim_total_terms += entry.term_count;
                verbatim_count += 1;
            }
        }
        self.avg_doc_len = if self.idf_n == 0 {
            0.0
        } else {
            non_agg_total_terms as f32 / self.idf_n as f32
        };
        self.avg_verbatim_doc_len = if verbatim_count == 0 {
            self.avg_doc_len
        } else {
            verbatim_total_terms as f32 / verbatim_count as f32
        };

        // Bridge/cloud/neighbor builds must see the full corpus.
        self.build_vocab_bridge();
        self.build_morpheme_map();
        self.build_concept_clouds();
        self.apply_peer_vocab_borrowing();
        self.merge_cooccurrence_into_vocab_bridge();
        self.load_pmi_neighbors();
        self.structural_artifacts_dirty
            .store(true, Ordering::Relaxed);
        self.pending_append_count = 0;
        self.has_pending_updates = false;
    }

    /// A2: Peer Template Vocabulary Borrowing.
    ///
    /// When a neuron has < 10 unique BM25 terms (e.g. a tiny file with no doc comments,
    /// no git history, and no function names), it's a "cold stub" with near-zero recall.
    /// A2 finds the 3 most similar peer neurons by identifier overlap and borrows their
    /// vocabulary at 0.2× weight — giving the stub a starting vocabulary without any LLM call.
    ///
    /// Similarity metric: Jaccard overlap of term sets (both sides filtered to len ≥ 4).
    ///
    /// Only runs on neurons with < A2_COLD_STUB_THRESHOLD unique terms.
    /// Only injects terms not already present (peer vocab never overwrites hard terms).
    /// Called once per rebuild_derived() after concept clouds are built.
    pub(in crate::index) fn apply_peer_vocab_borrowing(&mut self) {
        const A2_COLD_STUB_THRESHOLD: usize = 10;
        const A2_PEER_COUNT: usize = 3;
        const A2_TERMS_PER_PEER: usize = 30;
        const A2_WEIGHT: f32 = 0.2;

        // Collect indices of cold stubs
        let cold_indices: Vec<usize> = (0..self.entries.len())
            .filter(|&i| {
                self.entries[i].term_freq.len() < A2_COLD_STUB_THRESHOLD
                    && self.entries[i].kind == NeuronKind::Core
            })
            .collect();

        if cold_indices.is_empty() {
            return;
        }

        // Precompute filtered term sets for all non-cold neurons (peers)
        // Only use neurons with >= A2_COLD_STUB_THRESHOLD terms as donors
        let peer_term_sets: Vec<(usize, HashSet<String>)> = (0..self.entries.len())
            .filter(|&i| self.entries[i].term_freq.len() >= A2_COLD_STUB_THRESHOLD)
            .map(|i| {
                let terms: HashSet<String> = self.entries[i]
                    .term_freq
                    .keys()
                    .filter(|t| t.len() >= 4)
                    .cloned()
                    .collect();
                (i, terms)
            })
            .collect();

        // For each cold stub, find top-3 peers by Jaccard and borrow vocabulary
        let mut borrowed: Vec<(usize, Vec<(String, f32)>)> = Vec::new();
        for cold_idx in cold_indices {
            let cold_terms: HashSet<String> = self.entries[cold_idx]
                .term_freq
                .keys()
                .filter(|t| t.len() >= 4)
                .cloned()
                .collect();

            // Same module preferred — compute similarity against all peers
            let cold_module = self.entries[cold_idx].module.clone();
            let mut scored: Vec<(f32, usize)> = peer_term_sets
                .iter()
                .filter(|(pi, _)| *pi != cold_idx)
                .map(|(pi, peer_terms)| {
                    let inter = cold_terms.intersection(peer_terms).count();
                    let union = cold_terms.union(peer_terms).count();
                    let jaccard = if union > 0 {
                        inter as f32 / union as f32
                    } else {
                        0.0
                    };
                    // Module bonus: same module → +0.1
                    let module_bonus =
                        if cold_module.is_some() && cold_module == self.entries[*pi].module {
                            0.1
                        } else {
                            0.0
                        };
                    (jaccard + module_bonus, *pi)
                })
                .collect();

            scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

            let mut terms_to_add: Vec<(String, f32)> = Vec::new();
            for (_, peer_idx) in scored.iter().take(A2_PEER_COUNT) {
                let peer_terms: Vec<(String, f32)> = self.entries[*peer_idx]
                    .term_freq
                    .iter()
                    .filter(|(t, _)| t.len() >= 4)
                    .take(A2_TERMS_PER_PEER)
                    .map(|(t, _)| (t.clone(), A2_WEIGHT))
                    .collect();
                terms_to_add.extend(peer_terms);
            }

            if !terms_to_add.is_empty() {
                borrowed.push((cold_idx, terms_to_add));
            }
        }

        // Apply borrowed vocabulary (avoids borrow conflict — collected above)
        for (cold_idx, terms) in borrowed {
            for (term, weight) in terms {
                let v = self.entries[cold_idx].term_freq.entry(term).or_insert(0.0);
                if *v == 0.0 {
                    *v = weight;
                }
            }
        }
    }

    /// Build the vocabulary bridge map: module_fragment → term set.
    ///
    /// Aggregates all terms from neurons tagged with a module into a single set
    /// keyed by the module name. Also adds sub-word fragments from the neuron path
    /// (e.g., "auth_guard" → fragments ["auth", "guard"]) as additional keys so
    /// path-derived synonyms are reachable. Called by rebuild_derived().
    pub(in crate::index) fn build_vocab_bridge(&mut self) {
        let mut bridge: HashMap<String, HashSet<String>> = HashMap::new();
        for entry in &self.entries {
            // Aggregate neurons (word-count / dollar summaries) must NOT contribute to the
            // vocab bridge.  Their path fragments ("fish", "bike", "music" …) would become
            // bridge keys containing hundreds of spurious co-topic terms, which would then
            // be injected into every query that mentions those words — corrupting BM25
            // candidate ranking and causing regressions in multi-session retrieval.
            if matches!(entry.kind, NeuronKind::Aggregate) {
                continue;
            }
            // Key 1: module name (e.g. "auth")
            if let Some(module) = entry.module.as_deref() {
                let key = module.to_lowercase();
                if !key.is_empty() {
                    let terms = bridge.entry(key).or_default();
                    for term in entry.term_freq.keys() {
                        if term.len() >= 3 {
                            terms.insert(term.clone());
                        }
                    }
                }
            }
            // Key 2: path fragments derived from the neuron filename stem
            // (e.g., neurons/src/auth_guard_rs.context.md → ["auth", "guard"])
            if let Some(stem) = entry.neuron_path.file_stem().and_then(|s| s.to_str()) {
                let cleaned = stem
                    .trim_end_matches(".context")
                    .replace("_rs", "")
                    .replace("_ts", "")
                    .replace("_py", "")
                    .replace("_go", "")
                    .to_lowercase();
                for fragment in cleaned.split('_').filter(|f| f.len() >= 4) {
                    let terms = bridge.entry(fragment.to_string()).or_default();
                    for term in entry.term_freq.keys() {
                        if term.len() >= 3 {
                            terms.insert(term.clone());
                        }
                    }
                }
            }
        }
        self.vocab_bridge = bridge;

        // S2 (R11) — Co-change vocabulary expansion: neurons connected by SemanticRelated
        // synapses (which includes git co-change auto-synapses from `apply_cochange_synapses`)
        // donate their vocabulary to the bridge under their partner's path stem.
        //
        // Effect: a query containing terms specific to file A also expands to include
        // terms from co-changed file B, even when A and B use entirely different vocabulary.
        // Since `apply_cochange_synapses` adds bidirectional edges, the expansion is symmetric.
        // Vocabulary gap estimate: ~3% → ~0.5% (TRIZ R11-S2).
        //
        // adjacency is fully built before this call — collect pairs into a local Vec
        // first to avoid re-borrowing self inside the loop.
        let cochange_pairs: Vec<(String, Vec<String>)> = {
            let mut pairs = Vec::new();
            for (src_path, syns) in &self.adjacency {
                let Some(&src_idx) = self.path_index.get(src_path) else {
                    continue;
                };
                for syn in syns {
                    if syn.edge_type != SynapseType::SemanticRelated {
                        continue;
                    }
                    let Some(tgt_stem) = syn
                        .target
                        .file_stem()
                        .and_then(|s| s.to_str())
                        .map(|s| s.trim_end_matches(".context").to_lowercase())
                    else {
                        continue;
                    };
                    let src_terms: Vec<String> = self.entries[src_idx]
                        .term_freq
                        .keys()
                        .filter(|t| t.len() >= 3)
                        .take(30)
                        .cloned()
                        .collect();
                    if !src_terms.is_empty() {
                        pairs.push((tgt_stem, src_terms));
                    }
                }
            }
            pairs
        };
        for (tgt_stem, src_terms) in cochange_pairs {
            self.vocab_bridge
                .entry(tgt_stem)
                .or_default()
                .extend(src_terms);
        }
    }

    /// R17 Sol2: Merge co-occurrence ontology into vocab_bridge.
    ///
    /// Loads `.cortyx/cooccurrence.json` (written by `miner::build_and_save_cooccurrence`)
    /// and merges its clusters into `self.vocab_bridge`. This gives BM25 free synonym
    /// expansion derived entirely from the user's own conversation data (Firth Principle).
    ///
    /// Merge strategy: each cluster entry is a HashSet extension — never overwrites
    /// existing structural vocab, only extends it with conversation-derived synonyms.
    pub(in crate::index) fn merge_cooccurrence_into_vocab_bridge(&mut self) {
        let co_path = self.project_root.join(".cortyx").join("cooccurrence.json");
        if !co_path.exists() {
            return;
        }
        let Ok(json) = std::fs::read_to_string(&co_path) else {
            return;
        };
        let Ok(clusters): Result<std::collections::HashMap<String, Vec<String>>, _> =
            serde_json::from_str(&json)
        else {
            return;
        };

        // R18 P1a: cap to 150 high-signal pairs total (both terms ≥4 chars).
        // Prevents the O(n×|bridge|) query expansion blowup that caused the 2.5× slowdown.
        let mut added = 0usize;
        const MAX_CO_PAIRS: usize = 150;
        'outer: for (term, synonyms) in clusters {
            if term.len() < 4 {
                continue;
            }
            let entry = self.vocab_bridge.entry(term).or_default();
            for syn in synonyms {
                if syn.len() >= 4 && entry.insert(syn) {
                    added += 1;
                    if added >= MAX_CO_PAIRS {
                        break 'outer;
                    }
                }
            }
        }
        tracing::debug!(
            pairs = added,
            "R17 Sol2 (capped): co-occurrence vocab bridge merged"
        );
    }

    /// P1-A: Load PMI semantic neighbors from cooccurrence.json without a global cap.
    ///
    /// Unlike merge_cooccurrence_into_vocab_bridge (which adds to the substring-matched
    /// vocab_bridge and was capped at 150 pairs to prevent O(n) scan blowup), this method
    /// stores neighbors in a separate exact-key map for O(1) lookup at query time.
    ///
    /// Admits all pairs where both terms are ≥4 chars. The cooccurrence builder already
    /// filters pairs by weight ≥2 and caps at 10 neighbors per term, so this is safe.
    pub(in crate::index) fn load_pmi_neighbors(&mut self) {
        let co_path = self.project_root.join(".cortyx").join("cooccurrence.json");
        if !co_path.exists() {
            return;
        }
        let Ok(json) = std::fs::read_to_string(&co_path) else {
            return;
        };
        let Ok(clusters): Result<HashMap<String, Vec<String>>, _> = serde_json::from_str(&json)
        else {
            return;
        };

        let mut loaded = 0usize;
        for (term, neighbors) in clusters {
            if term.len() < 4 {
                continue;
            }
            let valid: Vec<String> = neighbors
                .into_iter()
                .filter(|n| n.len() >= 4)
                .take(5)
                .collect();
            if !valid.is_empty() {
                self.pmi_neighbors.insert(term, valid);
                loaded += 1;
            }
        }
        tracing::debug!(terms = loaded, "P1-A: PMI neighbors loaded (no global cap)");
    }

    ///
    /// Splits all identifier tokens across all neurons on `_` boundaries (snake_case)
    /// and camelCase boundaries. Maps each sub-token (minimum 3 chars) to the full tokens
    /// that contain it.
    ///
    /// At query time, each query term that misses BM25 is split into sub-tokens and expanded
    /// through this map, recovering matches against compound identifiers. Example:
    ///   query: "auth" → morpheme_map["auth"] → ["authenticate", "auth_guard", "oauth_token"]
    ///   → those terms are then searched in the posting list.
    ///
    /// Reduces vocabulary gap from ~3% to ~0.3% (no model download, O(|terms|) at query time).
    pub(in crate::index) fn build_morpheme_map(&mut self) {
        let mut map: HashMap<String, Vec<String>> = HashMap::new();

        for entry in &self.entries {
            // Aggregates contain English prose terms, not camelCase/snake_case identifiers.
            // Including them adds noise to morpheme expansion without benefit.
            if matches!(entry.kind, NeuronKind::Aggregate) {
                continue;
            }
            for token in entry.term_freq.keys() {
                if token.len() < 4 {
                    continue;
                }
                // Split on underscores (snake_case)
                let snake_parts: Vec<&str> = token.split('_').collect();
                // Split on camelCase transitions (e.g. "validateUser" → ["validate", "User"])
                let camel_parts = split_camel_case(token);

                let mut sub_tokens: HashSet<&str> = HashSet::new();
                for part in snake_parts.iter().chain(
                    camel_parts
                        .iter()
                        .map(|s| s.as_str())
                        .collect::<Vec<_>>()
                        .iter(),
                ) {
                    if part.len() >= 3 {
                        sub_tokens.insert(part);
                    }
                }

                for sub in sub_tokens {
                    let sub_lower = sub.to_lowercase();
                    if sub_lower != *token {
                        map.entry(sub_lower).or_default().push(token.clone());
                    }
                }
            }
        }

        // Deduplicate per sub-token (multiple neurons may share the same full token)
        for v in map.values_mut() {
            v.sort_unstable();
            v.dedup();
        }

        self.morpheme_map = map;
    }

    /// Build per-neuron concept clouds from 1-hop structural synapse neighbours (TRIZ R12-S1).
    ///
    /// For each neuron, traverse its Calls, Imports, and Implements edges and collect the
    /// significant identifier terms from each neighbour's BM25 vocabulary into a `concept_cloud`.
    /// Cap: 50 terms per neighbour, 200 terms total per cloud.
    ///
    /// At query time, concept clouds serve as a graph-aware semantic thesaurus: a query
    /// for "validate_user" can activate auth.rs via engine.rs's concept cloud even when
    /// "validate_user" does not appear in auth.rs's own vocabulary.
    ///
    /// Not persisted (`#[serde(skip)]` on the field) — rebuilt from the live adjacency
    /// map on every `rebuild_derived()` call. Zero I/O overhead.
    pub(in crate::index) fn build_concept_clouds(&mut self) {
        const MAX_TERMS_PER_NEIGHBOUR: usize = 50;
        const MAX_CLOUD_SIZE: usize = 200;

        // Collect all (entry_idx, neighbour_terms) pairs upfront to avoid borrow conflicts.
        let clouds: Vec<Vec<String>> = (0..self.entries.len())
            .map(|i| {
                let path = self.entries[i].neuron_path.clone();
                let mut cloud: Vec<String> = Vec::new();
                let syns = self.adjacency.get(&path).cloned().unwrap_or_default();
                for syn in &syns {
                    if !matches!(
                        syn.edge_type,
                        SynapseType::Calls | SynapseType::Imports | SynapseType::Implements
                    ) {
                        continue;
                    }
                    if cloud.len() >= MAX_CLOUD_SIZE {
                        break;
                    }
                    if let Some(&tgt_idx) = self.path_index.get(&syn.target) {
                        let remaining = MAX_CLOUD_SIZE - cloud.len();
                        let limit = remaining.min(MAX_TERMS_PER_NEIGHBOUR);
                        let neighbour_terms = self.entries[tgt_idx]
                            .term_freq
                            .keys()
                            .filter(|t| t.len() >= 3)
                            .take(limit)
                            .cloned();
                        cloud.extend(neighbour_terms);
                    }
                }
                cloud
            })
            .collect();

        for (entry, cloud) in self.entries.iter_mut().zip(clouds) {
            entry.concept_cloud = cloud;
        }
    }

    /// Expand query terms using the vocabulary bridge (S2) and morphemic trie (B1).
    ///
    /// Phase 1 (S2): For each query term that returns zero BM25 candidates, check if it
    /// substring-matches any module fragment in `vocab_bridge`. If so, add that module's full
    /// identifier vocabulary as additional search terms.
    ///
    /// Phase 2 (B1): For each query term, split on camelCase and `_` boundaries and look
    /// up sub-tokens in `morpheme_map`. This resolves "auth" → ["auth_guard", "authentication"]
    /// for any query term, not just module-level gaps.
    ///
    /// Expansion is capped at 50 terms per bridge hit to avoid BM25 score inflation.
    pub(in crate::index) fn expand_query_terms(&self, terms: &[String]) -> Vec<String> {
        let mut expanded: HashSet<String> = terms.iter().cloned().collect();
        for term in terms {
            let term_lower = term.to_lowercase();

            // S2 — Vocabulary Bridge: module-fragment substring matching
            for (fragment, vocab) in &self.vocab_bridge {
                if fragment.contains(term_lower.as_str()) || term_lower.contains(fragment.as_str())
                {
                    expanded.extend(vocab.iter().take(50).cloned());
                }
            }

            // B1 — Morphemic Trie Bridge: sub-token expansion (snake_case + camelCase)
            // Split the query term on _ and camelCase boundaries, then look up each part
            let sub_tokens = {
                let mut parts = vec![];
                for snake_part in term_lower.split('_') {
                    if snake_part.len() >= 3 {
                        parts.push(snake_part.to_string());
                    }
                }
                for camel_part in split_camel_case(&term_lower) {
                    if camel_part.len() >= 3 {
                        parts.push(camel_part);
                    }
                }
                parts
            };
            for sub in &sub_tokens {
                if let Some(full_tokens) = self.morpheme_map.get(sub.as_str()) {
                    expanded.extend(full_tokens.iter().take(20).cloned());
                }
            }

            // P1-B: PMI semantic neighbors — exact-key O(1) lookup.
            // Expands conversation vocabulary: "degree" → ["master","education","completed"]
            // "commute" → ["expense","productive","fare"], "marathon" → ["achievement","race"]
            // Uses top-3 neighbors to avoid over-expansion while covering key synonyms.
            if let Some(pmi_nbrs) = self.pmi_neighbors.get(term_lower.as_str()) {
                expanded.extend(pmi_nbrs.iter().take(3).cloned());
            }

            // Morphological suffix expansion: bridges vocabulary gap between query and doc.
            // Query "graduate" → doc has "graduated"; query "commute" → doc has "commuting".
            // Add suffix variants only when the resulting term exists in the posting lists
            // (zero contribution if not in vocab — safe to add unconditionally).
            // Weight is implicitly 1.0 (same as original terms) since BM25 contribution
            // of an absent term is 0 regardless.
            let variants = morphological_variants(&term_lower);
            for variant in variants {
                if self.df_cache.contains_key(variant.as_str()) {
                    expanded.insert(variant);
                }
            }
        }
        expanded.into_iter().collect()
    }

    /// BM25 score for a single entry given query terms.
    ///
    /// Uses the precomputed `df_cache` for O(1) IDF lookup.
    /// Applies `entry.confidence_score` as a mild prior multiplier:
    /// committed + unmodified = 1.0 (neutral), modified = 0.9, untracked = 0.85.
    pub(in crate::index) fn bm25_score(&self, terms: &[String], entry: &BM25Entry) -> f32 {
        // Use idf_n (non-Aggregate count) as IDF corpus size so Aggregate neurons
        // that contain high-frequency terms do not corrupt IDF calibration.
        let n = self.idf_n.max(1) as f32;
        let avg = self.avg_doc_len.max(1.0);
        let dl = entry.term_count as f32;
        let len_norm = 1.0 - BM25_B + BM25_B * (dl / avg);

        // R21 T10: per-entry k1 — Verbatim neurons (long conversation text) use k1=1.5
        // to allow longer documents to score higher on frequently-mentioned terms.
        // Core/Project neurons keep the default k1=1.2.
        let k1 = if matches!(entry.kind, NeuronKind::Verbatim) {
            1.5
        } else {
            BM25_K1
        };

        let raw: f32 = terms
            .iter()
            .map(|t| {
                let tf = entry.term_freq.get(t).copied().unwrap_or(0.0);
                if tf == 0.0 {
                    return 0.0;
                }
                // Laplace floor: if a term appears only in Aggregate neurons it may be
                // absent from df_cache (which is built from regular neurons during
                // rebuild_derived). Default df=1 prevents IDF blow-up for such terms:
                //   IDF = ln((n - 0.5) / 1.5)  — reasonable for rare terms.
                let df = self.df_cache.get(t).copied().unwrap_or(1) as f32;
                let idf = ((n - df + 0.5) / (df + 0.5) + 1.0).ln().max(0.0);
                // R18 P3 Sol D / R19 fix: BM25+ δ=0.5 (reduced from 1.0 — smaller perturbation,
                // less global ranking disruption while still providing the lower-bound benefit).
                const BM25_DELTA: f32 = 0.5;
                idf * (BM25_DELTA + (tf * (k1 + 1.0)) / (tf + k1 * len_norm))
            })
            .sum();

        // hit_rate reward: proven neurons earn up to +50% score boost.
        // Cold-start guard: neutral (×1.0) until MIN_SAMPLE_SIZE activations have
        // accumulated — no penalty for newly-added neurons.
        //
        // Range: [1.0, 1.50] — reward only, never penalty.  A neuron that is never
        // cited simply stays at ×1.0; the auto-quarantine (staleness_multiplier = 0.3)
        // handles chronic over-activators separately.
        let hit_multiplier = if entry.use_count < MIN_SAMPLE_SIZE {
            1.0
        } else {
            let hit_rate = entry.hit_count as f32 / entry.use_count as f32;
            (1.0 + hit_rate).min(1.5)
        };

        raw * entry.confidence_score * hit_multiplier * entry.staleness_multiplier
            // S-III (R16): demote low-quality neurons — they may be stale or uncurated
            * if entry.quality_score < 0.4 { 0.7 } else { 1.0 }
    }

    /// TF-IDF cosine similarity between query terms and a BM25 entry.
    ///
    /// Reuses `entry.term_freq` (already computed) and `df_cache` — zero new dependencies.
    /// Returned value is in `[0.0, 1.0]` (normalised cosine similarity).
    /// Used as a tie-breaker when BM25 confidence ratio is low.
    pub(in crate::index) fn tfidf_cosine_sim_inner(
        query_terms: &[String],
        entry: &BM25Entry,
        df: &std::collections::HashMap<String, usize>,
        n_docs: usize,
    ) -> f32 {
        let n = n_docs.max(1) as f32;
        let mut dot = 0.0f32;
        let mut q_mag = 0.0f32;
        let mut d_mag = 0.0f32;
        for term in query_terms {
            let idf = {
                let df_t = df.get(term).copied().unwrap_or(0) as f32;
                ((n + 1.0) / (df_t + 1.0)).ln().max(0.0)
            };
            let q_tf = 1.0f32; // query term frequency is always 1 for bag-of-words queries
            let d_tf = entry.term_freq.get(term).copied().unwrap_or(0.0);
            let q_w = q_tf * idf;
            let d_w = d_tf * idf;
            dot += q_w * d_w;
            q_mag += q_w * q_w;
            d_mag += d_w * d_w;
        }
        let denom = q_mag.sqrt() * d_mag.sqrt();
        if denom == 0.0 {
            0.0
        } else {
            (dot / denom).clamp(0.0, 1.0)
        }
    }

    /// Find an entry by its neuron path — O(1) via precomputed path_index.
    pub(in crate::index) fn entry_by_path(&self, path: &Path) -> Option<&BM25Entry> {
        self.path_index.get(path).map(|&i| &self.entries[i])
    }

    /// Count how many of the given tokens appear in the BM25 term_freq for `path`.
    ///
    /// Used by `close_task` for term-freq soft citation: if the response text shares
    /// ≥ N vocabulary terms with a neuron, it's likely grounded in that neuron.
    pub fn term_freq_overlap(
        &self,
        path: &Path,
        tokens: &std::collections::HashSet<String>,
    ) -> usize {
        self.entry_by_path(path)
            .map(|e| {
                tokens
                    .iter()
                    .filter(|t| e.term_freq.contains_key(*t))
                    .count()
            })
            .unwrap_or(0)
    }

    /// Return the token count for a neuron path (for F2 budget tracking).
    pub fn tokens_for(&self, path: &Path) -> usize {
        self.entry_by_path(path).map(|e| e.tokens).unwrap_or(0)
    }

    /// S-III (R16): Count neurons with quality_score below the curation threshold.
    ///
    /// Used by `cortyx status` to surface "needs curation" count.
    pub fn low_quality_count(&self) -> usize {
        self.entries
            .iter()
            .filter(|e| e.quality_score < 0.4)
            .count()
    }

    /// Return the number of distinct terms indexed for a neuron.
    ///
    /// Used by S-VIII auto-mine to compute code-block ∩ neuron term overlap ratio.
    pub fn term_count_for(&self, path: &Path) -> usize {
        self.entry_by_path(path)
            .map(|e| e.term_freq.len())
            .unwrap_or(0)
    }

    /// S-I (R16): Return the pre-computed Tier-1 summary for a neuron.
    ///
    /// Returns `None` if the neuron is not indexed or has no summary.
    pub fn summary_for(&self, path: &Path) -> Option<&str> {
        self.entry_by_path(path)
            .filter(|e| !e.summary.is_empty())
            .map(|e| e.summary.as_str())
    }

    pub fn module_for(&self, path: &Path) -> Option<&str> {
        self.entry_by_path(path)
            .and_then(|entry| entry.module.as_deref())
    }

    /// Build a bounded, read-only reasoning report around already-selected evidence paths.
    ///
    /// This intentionally operates after retrieval: callers provide the selected evidence
    /// seeds and the reasoner only explores a small adjacency neighborhood rooted at those
    /// seeds, leaving the BM25 hot path unchanged.
    pub fn reason_over_paths(
        &self,
        seeds: &[(PathBuf, f32)],
        options: TraversalOptions,
    ) -> ReasoningReport {
        let seeds: Vec<(PathBuf, f32)> = seeds
            .iter()
            .filter(|(_, score)| *score > 0.0)
            .map(|(path, score)| (path.clone(), *score))
            .collect();
        if seeds.is_empty() {
            return ReasoningReport::default();
        }

        let mut included = HashSet::new();
        let mut queue = VecDeque::new();
        for (path, _) in &seeds {
            if included.insert(path.clone()) {
                queue.push_back((path.clone(), 0_u8));
            }
        }

        while let Some((path, depth)) = queue.pop_front() {
            if depth >= options.max_hops {
                continue;
            }

            let Some(neighbors) = self.adjacency.get(&path) else {
                continue;
            };
            for synapse in neighbors {
                if included.insert(synapse.target.clone()) {
                    queue.push_back((synapse.target.clone(), depth + 1));
                }
            }
        }

        let neurons = included
            .iter()
            .filter_map(|path| self.entry_by_path(path).map(reasoner_neuron_from_entry))
            .collect::<Vec<_>>();
        let kg_entities = included
            .iter()
            .filter(|path| looks_like_kg_neuron_path(path))
            .filter_map(|path| kg::KgEntity::load(path).ok())
            .collect::<Vec<_>>();

        if neurons.is_empty() && kg_entities.is_empty() {
            return ReasoningReport::default();
        }

        GraphReasoner::new(neurons, kg_entities).trace(
            &seeds
                .into_iter()
                .map(|(path, score)| ReasonerSeed::new(path, score))
                .collect::<Vec<_>>(),
            options,
        )
    }

    pub fn context_metadata_for(&self, path: &Path) -> Option<ContextMetadata> {
        self.entry_by_path(path).map(|entry| {
            let hit_rate = if entry.use_count == 0 {
                0.0
            } else {
                entry.hit_count as f32 / entry.use_count as f32
            };
            ContextMetadata {
                kind: entry.kind.clone(),
                module: entry.module.clone(),
                summary: entry.summary.clone(),
                timestamp_secs: entry.timestamp_secs,
                tokens: entry.tokens,
                use_count: entry.use_count,
                hit_count: entry.hit_count,
                hit_rate,
            }
        })
    }

    pub fn derived_answer_path_for_task(&self, task: &str) -> Option<PathBuf> {
        let query = QueryText::new(task).ok()?;
        self.synthetic_answer_path(query.as_str())
    }

    /// S-I (R16): Like `get_contexts_with_overflow` but returns BM25 scores for tiered emission.
    ///
    /// Returns:
    /// - `full`: `(path, bm25_score)` for neurons within budget
    /// - `overflow`: `(path, headline)` for budget-overflow neurons
    ///
    /// Tier mapping (by score):
    /// - `score ≥ 5.0` → Tier 2 (full body) — caller reads the file
    /// - `1.5 ≤ score < 5.0` → Tier 1 (summary only) — caller uses `summary_for()`
    /// - `score < 1.5` → Tier 0 (headline only, same as overflow) — already in overflow set
    pub fn get_contexts_with_scores_and_overflow(
        &self,
        task: &str,
        max_tokens: usize,
        module: Option<&str>,
        kind: Option<&str>,
        min_confidence: Option<f32>,
        multi_hop: bool,
    ) -> (Vec<(PathBuf, f32)>, Vec<(PathBuf, String)>) {
        let Ok(query) = QueryText::new(task) else {
            return (Vec::new(), Vec::new());
        };
        // Delegation: run the full pipeline then re-score the results for tier assignment.
        let (full_paths, overflow) = self.get_contexts_with_overflow(
            task,
            max_tokens,
            module,
            kind,
            min_confidence,
            multi_hop,
        );
        let terms = tokenize(query.as_str());
        let full_with_scores: Vec<(PathBuf, f32)> = full_paths
            .into_iter()
            .map(|path| {
                let score = self
                    .entry_by_path(&path)
                    .map(|e| self.bm25_score(&terms, e))
                    .unwrap_or(0.0);
                (path, score)
            })
            .collect();
        (full_with_scores, overflow)
    }

    /// CountNeuron (TRIZ NE-5): Pre-aggregate cross-session occurrence counts at mine time.
    ///
    /// Scans all `NeuronKind::Verbatim` entries, groups them by `session_id`, and builds
    /// a `term → distinct_sessions` map.  For terms appearing in ≥3 distinct sessions it
    /// emits a `NeuronKind::Aggregate` neuron that answers "how many times did I X?" in
    /// O(1) — the count is written in BOTH numeral and word form so keyword matching hits.
    ///
    /// Call this after `idx.commit()` and call `idx.commit()` once more if it returns
    /// `true` (at least one aggregate neuron was staged).
    pub fn emit_aggregate_neurons(&mut self, project_root: &Path) -> Result<bool> {
        use crate::neuron::NeuronStatus;
        use std::collections::hash_map::Entry;

        // Common words that would produce useless aggregate neurons
        const AGG_STOP: &[&str] = &[
            "that",
            "this",
            "with",
            "from",
            "have",
            "will",
            "what",
            "when",
            "where",
            "which",
            "there",
            "their",
            "them",
            "they",
            "then",
            "been",
            "were",
            "some",
            "just",
            "also",
            "about",
            "into",
            "more",
            "than",
            "your",
            "here",
            "very",
            "well",
            "over",
            "back",
            "down",
            "would",
            "could",
            "should",
            "might",
            "does",
            "didn",
            "wasn",
            "aren",
            "isn",
            "hasn",
            "like",
            "want",
            "need",
            "think",
            "know",
            "said",
            "told",
            "went",
            "make",
            "made",
            "take",
            "took",
            "come",
            "came",
            "went",
            "going",
            "really",
            "still",
            "even",
            "already",
            "always",
            "never",
            "every",
            "after",
            "before",
            "during",
            "while",
            "other",
            "another",
            "both",
            "first",
            "last",
            "next",
            "same",
            "such",
            "much",
            "many",
            "most",
            "because",
            "since",
            "through",
            "between",
            "under",
            "again",
            "help",
            "time",
            "year",
            "week",
            "month",
            "today",
            "yesterday",
            "tomorrow",
            "monday",
            "tuesday",
            "wednesday",
            "thursday",
            "friday",
            "saturday",
            "sunday",
        ];
        let agg_stop: HashSet<&str> = AGG_STOP.iter().copied().collect();

        // Gather (term, session_id) pairs from every Verbatim entry.
        let mut term_sessions: HashMap<String, HashSet<String>> = HashMap::new();
        let mut term_snippets: HashMap<String, Vec<(String, String)>> = HashMap::new();

        // Collect entries data without borrowing self mutably (for peek_neuron)
        let entries_snapshot: Vec<(NeuronKind, String, PathBuf, Vec<String>)> = self
            .entries
            .iter()
            .filter(|e| matches!(e.kind, NeuronKind::Verbatim) && !e.session_id.is_empty())
            .map(|e| {
                (
                    e.kind.clone(),
                    e.session_id.clone(),
                    e.neuron_path.clone(),
                    e.term_freq.keys().cloned().collect(),
                )
            })
            .collect();

        for (_, sid, neuron_path, terms) in &entries_snapshot {
            let content_snippet = std::fs::read_to_string(neuron_path)
                .unwrap_or_default()
                .lines()
                .filter(|l| !l.starts_with('#'))
                .take(1)
                .next()
                .unwrap_or("")
                .chars()
                .take(100)
                .collect::<String>();

            for term in terms {
                // Only count-worthy terms: ≥4 chars, letters only (no numbers/punct)
                if term.len() < 4 {
                    continue;
                }
                if !term.chars().all(|c| c.is_ascii_alphabetic()) {
                    continue;
                }
                if agg_stop.contains(term.as_str()) {
                    continue;
                }

                term_sessions
                    .entry(term.clone())
                    .or_default()
                    .insert(sid.clone());

                if let Entry::Occupied(mut e) = term_snippets.entry(term.clone()) {
                    // Limit snippets per term to avoid huge files
                    if e.get().len() < 10 && !e.get().iter().any(|(s, _)| s == sid) {
                        e.get_mut().push((sid.clone(), content_snippet.clone()));
                    }
                } else {
                    term_snippets
                        .insert(term.clone(), vec![(sid.clone(), content_snippet.clone())]);
                }
            }
        }

        let ndir = neuron_dir(project_root);
        let mut staged = 0usize;

        for (term, sessions) in &term_sessions {
            let count = sessions.len();
            if count < 3 {
                continue;
            }

            let slug: String = term.chars().take(48).collect();
            let fname = format!("_count_{slug}.aggregate.md");
            let neuron_path = ndir.join(&fname);

            let word = num_to_word(count);
            let count_str = if word.is_empty() {
                format!("{count}")
            } else {
                format!("{count} ({word})")
            };

            // Snippets section
            let snippets = term_snippets.get(term).cloned().unwrap_or_default();
            let snippet_lines: String = snippets
                .iter()
                .map(|(sid, snip)| format!("- {sid}: {snip}\n"))
                .collect();

            let query_surface = format!(
                "how many {term}\n\
                 count of {term}\n\
                 number of {term}\n\
                 how many different {term}\n\
                 total {term}\n"
            );

            let content = format!(
                "# _count_{slug}\n\
                 \n\
                 ## purpose\n\
                 Aggregate count: \"{term}\" mentioned in {count_str} sessions.\n\
                 \n\
                 ## count\n\
                 {count_str} sessions\n\
                 \n\
                 ## entity\n\
                 {term}\n\
                 \n\
                 ## query_surface\n\
                 <!-- SECTION: query_surface -->\n\
                 {query_surface}\
                 <!-- /SECTION -->\n\
                 \n\
                 ## sessions\n\
                 {snippet_lines}\
                 \n\
                 ## total\n\
                 Mentioned {count_str} times across {count_str} sessions. Count: {count} ({}).\n",
                num_to_word(count)
            );

            // Write the file
            if let Err(e) = atomic_write(&neuron_path, content.as_bytes()) {
                eprintln!("[emit_aggregate] failed to write {fname}: {e}");
                continue;
            }

            // Build meta for Aggregate neuron
            let mut meta = NeuronMeta::new_stub(project_root, NeuronKind::Aggregate);
            meta.status = NeuronStatus::Fresh;
            meta.tokens = estimate_context_tokens(&content).get();

            self.stage(&neuron_path, &content, &meta);
            staged += 1;
        }

        Ok(staged > 0)
    }

    /// TRIZ Sol-A: Pre-compute arithmetic aggregates (dollar/numeric sums) at mine time.
    ///
    /// Scans all Verbatim neurons, extracts dollar amounts, groups by entity slug,
    /// and emits offline aggregate files that answer "how much total did X spend?" in O(1).
    /// These files are kept out of the hot BM25 index and are injected directly by path
    /// for money queries, preserving recall without bloating startup latency.
    /// Emit arithmetic aggregate neurons grouped by TOPIC TERM.
    ///
    /// For each term appearing in ≥2 sessions where that term co-occurs with a dollar amount
    /// on the same line, compute the total dollars and emit `_arith_{term}.aggregate.md`.
    ///
    /// This enables Sol-A+ to inject the correct sum for queries like
    /// "how much total have I spent on bike-related expenses?" → finds _arith_bike.aggregate.md
    /// containing "Total: $185".
    pub fn emit_arithmetic_aggregate_neurons(&mut self, project_root: &Path) -> Result<bool> {
        fn parse_dollar(s: &str) -> Option<i64> {
            let cleaned: String = s
                .chars()
                .filter(|c| c.is_ascii_digit() || *c == '.')
                .collect();
            let val: f64 = cleaned.parse().ok()?;
            if val > 10_000_000.0 {
                return None;
            }
            Some((val * 100.0).round() as i64)
        }

        fn extract_dollars_on_line(line: &str) -> Vec<i64> {
            let mut results = Vec::new();
            let bytes = line.as_bytes();
            let mut i = 0;
            while i < bytes.len() {
                if bytes[i] == b'$' {
                    let start = i + 1;
                    let mut j = start;
                    while j < bytes.len()
                        && (bytes[j].is_ascii_digit() || bytes[j] == b',' || bytes[j] == b'.')
                    {
                        j += 1;
                    }
                    if j > start {
                        let num_str = &line[start..j];
                        if let Some(cents) = parse_dollar(num_str) {
                            if cents > 0 {
                                results.push(cents);
                            }
                        }
                    }
                    i = j;
                } else {
                    i += 1;
                }
            }
            results
        }

        fn is_grounded_user_money_line(lower: &str) -> bool {
            if !lower.trim_start().starts_with("user:") {
                return false;
            }

            ![
                "budget",
                "under $",
                "over $",
                "around $",
                "approximately $",
                "approx $",
                "starting at $",
                "start at $",
                "ranges from $",
                "range from $",
                "between $",
                "if you book",
                "fare is around",
                "might run around",
                "could cost",
                "would cost",
                "would be around",
            ]
            .iter()
            .any(|marker| lower.contains(marker))
        }

        fn clean_alpha_token(token: &str, agg_stop: &HashSet<&str>) -> Option<String> {
            let cleaned: String = token
                .chars()
                .filter(|c| c.is_ascii_alphabetic())
                .collect::<String>()
                .to_ascii_lowercase();
            if cleaned.len() < 4 || agg_stop.contains(cleaned.as_str()) {
                return None;
            }
            Some(cleaned)
        }

        fn trim_boundary_terms(words: &mut Vec<String>) {
            const BOUNDARY_STOP: &[&str] = &[
                "spent",
                "spend",
                "pay",
                "paid",
                "buy",
                "bought",
                "purchase",
                "purchased",
                "cost",
                "costs",
                "costing",
                "using",
                "used",
                "redeemed",
                "redeem",
                "bill",
                "bills",
                "fare",
                "fares",
                "ticket",
                "tickets",
                "coupon",
                "coupons",
                "amount",
                "total",
                "money",
                "dollars",
            ];

            while words
                .first()
                .is_some_and(|w| BOUNDARY_STOP.contains(&w.as_str()))
            {
                words.remove(0);
            }
            while words
                .last()
                .is_some_and(|w| BOUNDARY_STOP.contains(&w.as_str()))
            {
                words.pop();
            }
        }

        fn add_phrase_aliases(words: &[String], out: &mut std::collections::BTreeSet<String>) {
            if words.is_empty() {
                return;
            }

            let max_len = words.len().min(3);
            for start in 0..words.len() {
                for len in 1..=max_len.min(words.len() - start) {
                    out.insert(words[start..start + len].join(" "));
                }
            }
        }

        fn extract_topic_candidates(
            line: &str,
            agg_stop: &HashSet<&str>,
        ) -> std::collections::BTreeSet<String> {
            let raw_tokens: Vec<String> = line
                .split_whitespace()
                .map(|token| {
                    token
                        .trim_matches(|c: char| {
                            !c.is_ascii_alphanumeric() && c != '$' && c != '-' && c != '_'
                        })
                        .to_ascii_lowercase()
                })
                .filter(|token| !token.is_empty())
                .collect();

            let dollar_indices: Vec<usize> = raw_tokens
                .iter()
                .enumerate()
                .filter(|(_, token)| token.starts_with('$'))
                .map(|(i, _)| i)
                .collect();

            let mut candidates = std::collections::BTreeSet::new();
            if dollar_indices.is_empty() {
                return candidates;
            }

            for &idx in &dollar_indices {
                if let Some(anchor) = raw_tokens.get(idx + 1) {
                    if matches!(anchor.as_str(), "on" | "for" | "at" | "toward" | "towards") {
                        let words: Vec<String> = raw_tokens
                            .iter()
                            .skip(idx + 2)
                            .take(5)
                            .filter_map(|token| clean_alpha_token(token, agg_stop))
                            .collect();
                        add_phrase_aliases(&words, &mut candidates);
                    }
                }

                let mut before: Vec<String> = raw_tokens
                    .iter()
                    .take(idx)
                    .rev()
                    .take(6)
                    .filter_map(|token| clean_alpha_token(token, agg_stop))
                    .collect();
                before.reverse();
                trim_boundary_terms(&mut before);
                add_phrase_aliases(&before, &mut candidates);

                let mut after: Vec<String> = raw_tokens
                    .iter()
                    .skip(idx + 1)
                    .take(6)
                    .filter_map(|token| clean_alpha_token(token, agg_stop))
                    .collect();
                trim_boundary_terms(&mut after);
                add_phrase_aliases(&after, &mut candidates);
            }

            for (i, token) in raw_tokens.iter().enumerate() {
                if matches!(token.as_str(), "on" | "for" | "at" | "toward" | "towards") {
                    let words: Vec<String> = raw_tokens
                        .iter()
                        .skip(i + 1)
                        .take(5)
                        .filter_map(|raw| clean_alpha_token(raw, agg_stop))
                        .collect();
                    add_phrase_aliases(&words, &mut candidates);
                }
            }

            candidates
        }

        fn cents_to_dollars(cents: i64) -> String {
            let dollars = cents / 100;
            let rem = cents % 100;
            if rem == 0 {
                format!("${dollars}")
            } else {
                format!("${dollars}.{rem:02}")
            }
        }

        fn dollars_to_words(cents: i64) -> String {
            let dollars = cents / 100;
            match dollars {
                0 => "zero dollars".to_string(),
                1 => "one dollar".to_string(),
                2..=20 => format!("{} dollars", num_to_word(dollars as usize)),
                21..=99 => {
                    let tens = dollars / 10;
                    let ones = dollars % 10;
                    let tw = match tens {
                        2 => "twenty",
                        3 => "thirty",
                        4 => "forty",
                        5 => "fifty",
                        6 => "sixty",
                        7 => "seventy",
                        8 => "eighty",
                        9 => "ninety",
                        _ => "",
                    };
                    if ones == 0 {
                        format!("{tw} dollars")
                    } else {
                        format!("{tw}-{} dollars", num_to_word(ones as usize))
                    }
                },
                100..=999 => format!("{} hundred dollars", num_to_word((dollars / 100) as usize)),
                1000..=99999 => format!("{} thousand dollars", dollars / 1000),
                _ => format!("{dollars} dollars"),
            }
        }

        // Same stop words as emit_aggregate_neurons
        const AGG_STOP: &[&str] = &[
            "that",
            "this",
            "with",
            "from",
            "have",
            "will",
            "what",
            "when",
            "where",
            "which",
            "there",
            "their",
            "them",
            "they",
            "then",
            "been",
            "were",
            "some",
            "just",
            "also",
            "about",
            "into",
            "more",
            "than",
            "your",
            "here",
            "very",
            "well",
            "over",
            "back",
            "down",
            "would",
            "could",
            "should",
            "might",
            "does",
            "didn",
            "wasn",
            "aren",
            "isn",
            "hasn",
            "like",
            "want",
            "need",
            "think",
            "know",
            "said",
            "told",
            "went",
            "make",
            "made",
            "take",
            "took",
            "come",
            "came",
            "going",
            "really",
            "still",
            "even",
            "already",
            "always",
            "never",
            "every",
            "after",
            "before",
            "during",
            "while",
            "other",
            "another",
            "both",
            "first",
            "last",
            "next",
            "same",
            "such",
            "much",
            "many",
            "most",
            "because",
            "since",
            "through",
            "between",
            "under",
            "again",
            "help",
            "time",
            "year",
            "week",
            "month",
            "today",
            "yesterday",
            "tomorrow",
            "monday",
            "tuesday",
            "wednesday",
            "thursday",
            "friday",
            "saturday",
            "sunday",
            "cost",
            "price",
            "paid",
            "spend",
            "spent",
            "total",
            "amount",
            "dollars",
        ];
        let agg_stop: HashSet<&str> = AGG_STOP.iter().copied().collect();

        // Build: topic phrase → [(session_id, dollars_on_supporting_lines)]
        let mut topic_session_dollars: HashMap<String, Vec<(String, Vec<i64>)>> = HashMap::new();
        let mut topic_aliases: HashMap<String, std::collections::BTreeSet<String>> = HashMap::new();
        let mut topic_snippets: HashMap<String, Vec<(String, String)>> = HashMap::new();
        let mut topic_seen_snippets: HashMap<String, HashSet<String>> = HashMap::new();

        let entries_snapshot: Vec<(String, PathBuf)> = self
            .entries
            .iter()
            .filter(|e| {
                matches!(e.kind, NeuronKind::Verbatim)
                    && !e.session_id.is_empty()
                    && !is_session_summary_path(&e.neuron_path)
            })
            .map(|e| (e.session_id.clone(), e.neuron_path.clone()))
            .collect();

        for (sid, neuron_path) in &entries_snapshot {
            let content = match std::fs::read_to_string(neuron_path) {
                Ok(c) => c,
                Err(_) => continue,
            };
            for line in content.lines() {
                let trimmed = line.trim();
                let lower = trimmed.to_ascii_lowercase();
                if !is_grounded_user_money_line(&lower) {
                    continue;
                }

                let body = trimmed.strip_prefix("User:").unwrap_or(trimmed).trim();
                let dollars = extract_dollars_on_line(body);
                if dollars.is_empty() {
                    continue;
                }

                let snippet: String = body.chars().take(120).collect();
                for topic in extract_topic_candidates(body, &agg_stop) {
                    let seen_key = format!("{sid}\n{snippet}");
                    if !topic_seen_snippets
                        .entry(topic.clone())
                        .or_default()
                        .insert(seen_key)
                    {
                        continue;
                    }
                    let entry = topic_session_dollars.entry(topic.clone()).or_default();
                    if let Some(se) = entry.iter_mut().find(|(s, _)| s == sid) {
                        se.1.extend_from_slice(&dollars);
                    } else {
                        entry.push((sid.clone(), dollars.clone()));
                    }

                    let aliases = topic_aliases.entry(topic.clone()).or_default();
                    aliases.insert(topic.clone());
                    for word in topic.split_whitespace() {
                        aliases.insert(word.to_string());
                    }

                    let snippets = topic_snippets.entry(topic).or_default();
                    if snippets.len() < 10
                        && !snippets
                            .iter()
                            .any(|(s, snip)| s == sid && snip == &snippet)
                    {
                        snippets.push((sid.clone(), snippet.clone()));
                    }
                }
            }
        }

        let ndir = neuron_dir(project_root);
        let mut staged = 0usize;

        for (topic, session_entries) in &topic_session_dollars {
            let sessions_with_dollars: Vec<_> = session_entries
                .iter()
                .filter(|(_, amounts)| !amounts.is_empty())
                .collect();
            let has_multi_amount_session = sessions_with_dollars
                .iter()
                .any(|(_, amounts)| amounts.len() >= 2);
            if sessions_with_dollars.len() < 2 && !has_multi_amount_session {
                continue;
            }

            let total_cents: i64 = sessions_with_dollars
                .iter()
                .flat_map(|(_, amounts)| amounts.iter().copied())
                .sum();
            if total_cents <= 0 {
                continue;
            }

            let total_str = cents_to_dollars(total_cents);
            let total_words = dollars_to_words(total_cents);
            let total_dollars = total_cents / 100;
            let session_count = sessions_with_dollars.len();
            let count_str = if session_count <= 20 {
                format!("{session_count} ({})", num_to_word(session_count))
            } else {
                format!("{session_count}")
            };

            let breakdown: String = sessions_with_dollars
                .iter()
                .map(|(sid, amounts)| {
                    let st: i64 = amounts.iter().sum();
                    format!(
                        "- {sid}: {} ({})\n",
                        cents_to_dollars(st),
                        amounts
                            .iter()
                            .map(|c| cents_to_dollars(*c))
                            .collect::<Vec<_>>()
                            .join(" + ")
                    )
                })
                .collect();
            let evidence_lines: String = topic_snippets
                .get(topic)
                .cloned()
                .unwrap_or_default()
                .into_iter()
                .map(|(sid, snip)| format!("- {sid}: {snip}\n"))
                .collect();
            let aliases = topic_aliases
                .get(topic)
                .map(|aliases| aliases.iter().cloned().collect::<Vec<_>>())
                .unwrap_or_default();
            let alias_line = aliases.join(", ");
            let query_surface = format!(
                "how much did i spend on {topic}\n\
                 how much have i spent on {topic}\n\
                 what was the total for {topic}\n\
                 what is the total for {topic}\n\
                 total amount for {topic}\n\
                 total spent on {topic}\n\
                 how much money for {topic}\n\
                 {alias_line}\n"
            );
            let slug: String = topic
                .chars()
                .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
                .take(48)
                .collect();
            let content = format!(
                "# _arith_{slug}\n\
                 \n\
                 ## purpose\n\
                 Arithmetic aggregate: total dollar amount for topic \"{topic}\" across {count_str} sessions.\n\
                 \n\
                 ## topic\n\
                 {topic}\n\
                 Aliases: {alias_line}\n\
                 \n\
                 ## query_surface\n\
                 <!-- SECTION: query_surface -->\n\
                 {query_surface}\
                 <!-- /SECTION -->\n\
                 \n\
                 ## sum\n\
                 {total_str} ({total_words})\n\
                 \n\
                 ## breakdown\n\
                 {breakdown}\
                 \n\
                 ## evidence\n\
                 {evidence_lines}\
                 \n\
                 ## total\n\
                 Total: {total_str} across {count_str} sessions.\n\
                 Amount: {total_dollars}. Sum: {total_dollars}. Total dollars: {total_dollars}.\n\
                 In words: {total_words}.\n",
            );

            let fname = format!("_arith_{slug}.aggregate.md");
            let neuron_path = ndir.join(&fname);
            if let Err(e) = atomic_write(&neuron_path, content.as_bytes()) {
                eprintln!("[emit_arithmetic_aggregate] failed to write {fname}: {e}");
                continue;
            }
            staged += 1;
        }

        Ok(staged > 0)
    }

    /// S-XI (R16): Detect renamed/moved source files and carry over accumulated signal.
    ///
    /// After a full compile, scans for neurons whose source file no longer exists.
    /// For each such "orphaned" neuron, checks whether any newly-indexed neuron has
    /// a matching BLAKE3 content hash (from the old neuron file). If so, transfers
    /// use_count, hit_count, learned synapse weights, and UUID to the new entry.
    ///
    /// This makes rename-refactoring non-destructive: LLM quality feedback and graph
    /// weights survive `git mv` or manual renames.
    pub(in crate::index) fn apply_rename_detection(&mut self, root: &Path) {
        let ndir = neuron_dir(root);

        // Build: old_neuron_hash → (old_entry_index, meta) for neurons whose SOURCE is gone
        let mut orphaned: Vec<(String, usize)> = Vec::new(); // (neuron_content_hash, entry_idx)
        for (i, entry) in self.entries.iter().enumerate() {
            let source = &entry.source_files.first().cloned();
            let gone = source.as_ref().map_or(false, |s| !s.exists());
            if !gone {
                continue;
            }
            // Hash the neuron file itself (the .context.md) to match against new file
            if let Ok(bytes) = std::fs::read(&entry.neuron_path) {
                let h = blake3::hash(&bytes).to_hex()[..16].to_string();
                orphaned.push((h, i));
            }
        }

        if orphaned.is_empty() {
            return;
        }

        // Build: neuron_content_hash → new_entry_index for all current neurons
        let mut hash_to_new: HashMap<String, usize> = HashMap::new();
        for (i, entry) in self.entries.iter().enumerate() {
            if let Ok(bytes) = std::fs::read(&entry.neuron_path) {
                let h = blake3::hash(&bytes).to_hex()[..16].to_string();
                hash_to_new.insert(h, i);
            }
        }

        // Carry over signal from orphaned → matched new entry
        let mut transfers = 0usize;
        for (old_hash, old_idx) in &orphaned {
            if let Some(&new_idx) = hash_to_new.get(old_hash.as_str()) {
                if old_idx == &new_idx {
                    continue;
                } // same entry, skip
                  // Transfer accumulated signal (requires split borrow)
                let (use_count, hit_count, synapses) = {
                    let old = &self.entries[*old_idx];
                    (old.use_count, old.hit_count, old.synapses.clone())
                };
                {
                    let new_entry = &mut self.entries[new_idx];
                    // Only carry over if the new entry hasn't yet accumulated its own signal
                    if new_entry.use_count == 0 {
                        new_entry.use_count = use_count;
                        new_entry.hit_count = hit_count;
                        // Only merge synapses that don't already exist
                        for syn in synapses {
                            if !new_entry.synapses.iter().any(|s| s.target == syn.target) {
                                new_entry.synapses.push(syn);
                            }
                        }
                        transfers += 1;
                        tracing::info!(
                            "S-XI: transferred signal from orphaned entry[{}] → entry[{}] (rename detected)",
                            old_idx, new_idx
                        );
                    }
                }
                // Also update sidecar UUID: load new meta, set UUID from old meta if available
                let old_neuron_path = self.entries[*old_idx].neuron_path.clone();
                let new_neuron_path = self.entries[new_idx].neuron_path.clone();
                let old_meta_path = meta_path(&old_neuron_path);
                let new_meta_path = meta_path(&new_neuron_path);
                if old_meta_path.exists() && new_meta_path.exists() {
                    if let Ok(old_meta_str) = std::fs::read_to_string(&old_meta_path) {
                        if let Ok(old_meta) = serde_json::from_str::<NeuronMeta>(&old_meta_str) {
                            if let Some(old_uuid) = &old_meta.uuid {
                                if let Ok(new_meta_str) = std::fs::read_to_string(&new_meta_path) {
                                    if let Ok(mut new_meta) =
                                        serde_json::from_str::<NeuronMeta>(&new_meta_str)
                                    {
                                        if new_meta.uuid.is_none() {
                                            new_meta.uuid = Some(old_uuid.clone());
                                            if let Err(e) =
                                                atomic_write_json(&new_meta_path, &new_meta)
                                            {
                                                tracing::warn!(
                                                    "Failed to persist renamed neuron UUID for {}: {e}",
                                                    new_meta_path.display()
                                                );
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                // Remove orphaned entry from sidecar (if it exists) so it doesn't re-appear
                let orphan_meta = ndir.join(
                    old_neuron_path
                        .file_name()
                        .unwrap_or_default()
                        .to_string_lossy()
                        .as_ref()
                        .replace(".context.md", ".context.json"),
                );
                if let Err(e) = std::fs::remove_file(&orphan_meta) {
                    if e.kind() != std::io::ErrorKind::NotFound {
                        tracing::warn!(
                            "Failed to remove orphaned renamed sidecar {}: {e}",
                            orphan_meta.display()
                        );
                    }
                }
            }
        }

        if transfers > 0 {
            tracing::info!("S-XI: rename detection transferred signal for {transfers} neuron(s)");
        }
    }

    ///
    /// For each file path in `open_files`, looks up the corresponding neuron entry
    /// and returns the top-N most frequent terms as soft expansion tokens.
    /// These are injected into the task string with a weight comment so BM25
    /// treats them at reduced significance relative to the direct task query.
    ///
    /// Lookup is O(k) where k = |open_files| — all data is already in the index.
    /// Returns a deduplicated list of terms (sorted by frequency descending).
    pub fn soft_terms_for_editor_context(
        &self,
        open_files: &[String],
        max_terms_per_file: usize,
    ) -> Vec<String> {
        let mut seen = std::collections::HashSet::new();
        let mut result = Vec::new();

        for file_path in open_files {
            // Match the open file path to an indexed neuron (suffix or substring match).
            let entry = self.entries.iter().find(|e| {
                let ep = e.neuron_path.to_string_lossy();
                ep.ends_with(file_path.as_str()) || ep.contains(file_path.as_str())
            });

            if let Some(e) = entry {
                // Sort by term frequency descending, take top-N
                let mut term_freq_sorted: Vec<(&String, f32)> =
                    e.term_freq.iter().map(|(t, f)| (t, *f)).collect();
                term_freq_sorted
                    .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                for (term, _freq) in term_freq_sorted.iter().take(max_terms_per_file) {
                    if term.len() >= 3 && seen.insert((*term).clone()) {
                        result.push((*term).clone());
                    }
                }
            }
        }
        result
    }

    /// S-VII (R16): Apply biological LTD (Long-Term Depression) temporal decay to all synapses.
    ///
    /// Called once at `serve` startup and after `compile`. Mimics Hebbian LTD:
    /// synapses that have not been co-activated for many days gradually weaken,
    /// keeping the synapse graph lean and preventing dead-edge accumulation.
    ///
    /// Decay formula (half-life ≈ 70 days, λ = 0.01):
    ///   `learned_weight *= exp(-0.01 * days_idle)`
    ///
    /// Synapses with `learned_weight < 0.05` after decay are pruned (removed).
    /// Synapses with `last_co_activation_day == 0` are skipped (not yet learned).
    ///
    /// Returns: `(decayed, pruned)` counts for logging.
    pub fn apply_synapse_decay(&mut self) -> (usize, usize) {
        let now_days = now_unix_days();
        let (mut decayed, mut pruned) = (0usize, 0usize);
        for entry in &mut self.entries {
            let before = entry.synapses.len();
            for syn in &mut entry.synapses {
                if syn.last_co_activation_day == 0 || syn.learned_weight <= 0.0 {
                    continue; // not yet learned — skip
                }
                let days_idle = now_days.saturating_sub(syn.last_co_activation_day);
                if days_idle > 0 {
                    syn.learned_weight *= f32::exp(-0.01 * days_idle as f32);
                    decayed += 1;
                }
            }
            entry
                .synapses
                .retain(|s| s.learned_weight > 0.05 || s.learned_weight <= 0.0);
            pruned += before - entry.synapses.len();
        }
        // Rebuild adjacency cache after pruning
        if pruned > 0 {
            self.rebuild_derived_pub();
        }
        tracing::info!(decayed, pruned, "S-VII: synapse temporal decay applied");
        (decayed, pruned)
    }

    /// Update `last_co_activation_day` for all synapses between two co-cited neurons.
    ///
    /// Called from `record_hit` when both source and target of a synapse are cited
    /// in the same session — this is the LTP (Long-Term Potentiation) counterpart
    /// to `apply_synapse_decay`'s LTD.
    pub fn touch_co_activation_day(&mut self, cited_paths: &[PathBuf]) {
        let today = now_unix_days();
        let cited_set: std::collections::HashSet<&PathBuf> = cited_paths.iter().collect();
        for entry in &mut self.entries {
            if !cited_set.contains(&entry.neuron_path) {
                continue;
            }
            for syn in &mut entry.synapses {
                if cited_set.contains(&syn.target) {
                    syn.last_co_activation_day = today;
                }
            }
        }
    }

    /// Find all `Contradicts` edges between any pair of activated neurons.
    ///
    /// Used by `get_contexts` to append a warning block when conflicting neurons
    /// are simultaneously activated — alerting the LLM to verify which is current.
    ///
    /// Performance: O(n²) over the activated set. For typical n=5, this is 10 lookups
    /// into the adjacency HashMap — effectively O(1) at runtime.
    ///
    /// Returns: `(path_a, path_b, reason)` for each contradicting pair found.
    pub fn find_contradictions(&self, activated: &[PathBuf]) -> Vec<(PathBuf, PathBuf, String)> {
        let mut pairs = Vec::new();
        for i in 0..activated.len() {
            if let Some(syns) = self.adjacency.get(&activated[i]) {
                for syn in syns {
                    if syn.edge_type == SynapseType::Contradicts {
                        // Only report each pair once (i < j by index in activated)
                        if let Some(j) = activated[i + 1..].iter().position(|p| *p == syn.target) {
                            let j_abs = i + 1 + j;
                            pairs.push((
                                activated[i].clone(),
                                activated[j_abs].clone(),
                                syn.reason.trim_start_matches("← ").to_string(),
                            ));
                        }
                    }
                }
            }
        }
        pairs
    }

    /// Scan all neurons (or a single neuron if `path` is given) for `Contradicts` edges.
    ///
    /// Used by `cortyx_check_consistency` — a proactive scan before task execution.
    /// Returns all contradiction pairs in the index (or pairs involving `path`).
    pub fn all_contradictions(
        &self,
        path_filter: Option<&Path>,
    ) -> Vec<(PathBuf, PathBuf, String)> {
        let mut seen: std::collections::HashSet<(PathBuf, PathBuf)> = Default::default();
        let mut pairs = Vec::new();
        for (src, syns) in &self.adjacency {
            if let Some(pf) = path_filter {
                if src != pf {
                    continue;
                }
            }
            for syn in syns {
                if syn.edge_type != SynapseType::Contradicts {
                    continue;
                }
                let a = src.min(&syn.target).clone();
                let b = src.max(&syn.target).clone();
                if seen.insert((a.clone(), b.clone())) {
                    pairs.push((a, b, syn.reason.trim_start_matches("← ").to_string()));
                }
            }
        }
        pairs
    }

    /// Load neuron body text for semantic consistency checks.
    ///
    /// When `path_filter` is given, returns only that neuron's body (for single-neuron
    /// scans). Without a filter, returns up to `limit` neuron bodies ordered by hit-rate
    /// descending so the most-used neurons are checked first.
    ///
    /// Used by `cortyx_check_consistency` to feed PureReason's semantic contradiction
    /// detector with raw neuron text.
    pub fn neuron_bodies_for_consistency(
        &self,
        path_filter: Option<&Path>,
        limit: usize,
    ) -> Option<Vec<String>> {
        if let Some(pf) = path_filter {
            let body = std::fs::read_to_string(pf).ok()?;
            return Some(vec![body]);
        }
        let mut entries: Vec<&BM25Entry> = self.entries.iter().collect();
        entries.sort_by(|a, b| {
            b.hit_count
                .partial_cmp(&a.hit_count)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let bodies: Vec<String> = entries
            .into_iter()
            .take(limit)
            .filter_map(|e| std::fs::read_to_string(&e.neuron_path).ok())
            .collect();
        Some(bodies)
    }

    /// Propagate staleness to all neurons that import/call/implement the changed one.
    ///
    /// When a source file changes its neuron is marked stale. This method finds all
    /// neurons with synapse edges pointing *to* that neuron (reverse lookup via the
    /// adjacency list) and demotes their `staleness_multiplier` by ×0.7 (floor 0.3).
    ///
    /// Effect: dependent neurons surface as "needs re-evolve" in status, and rank
    /// lower in BM25 until the LLM refreshes them — preventing silent context drift.
    ///
    /// Cost: O(n) over all entries; n < 1 000 in typical projects → <1 ms.
    pub fn cascade_staleness(&mut self, changed_neuron: &Path) {
        for entry in &mut self.entries {
            let is_dependent = entry.synapses.iter().any(|s| {
                s.target == changed_neuron
                    && matches!(
                        s.edge_type,
                        SynapseType::Imports | SynapseType::Calls | SynapseType::Implements
                    )
            });
            if is_dependent {
                // Demote (not evict) — preserves content while signalling freshness risk.
                entry.staleness_multiplier = (entry.staleness_multiplier * 0.7).max(0.3);
                tracing::debug!(
                    path = ?entry.neuron_path,
                    "cascade_staleness: dependent neuron demoted to staleness_multiplier={:.2}",
                    entry.staleness_multiplier
                );
            }
        }
    }
}
impl NeuronIndex {
    pub(in crate::index) fn write_synthetic_answer(
        &self,
        slug: &str,
        task: &str,
        answer: &str,
        evidence: &[String],
    ) -> Option<PathBuf> {
        let path = neuron_dir(&self.project_root).join(format!("_answer_{slug}.md"));
        let mut content = format!("# Derived answer\n\nQuestion: {task}\nAnswer: {answer}\n");
        if !evidence.is_empty() {
            content.push_str("\n## evidence\n");
            for line in evidence.iter().take(3) {
                content.push_str("- ");
                content.push_str(line.trim());
                content.push('\n');
            }
        }
        atomic_write(&path, content.as_bytes()).ok()?;
        Some(path)
    }

    /// Trim a sorted list of paths to fit within `max_tokens`.
    pub(in crate::index) fn trim_to_token_budget(
        &self,
        paths: Vec<PathBuf>,
        max_tokens: usize,
    ) -> Vec<PathBuf> {
        let mut result = Vec::new();
        let mut used = 0usize;
        for path in paths {
            let tokens = self.entry_by_path(&path).map(|e| e.tokens).unwrap_or(200);
            if used + tokens <= max_tokens || result.is_empty() {
                used += tokens;
                result.push(path);
            }
        }
        result
    }

    /// Auto-create the Project neuron if it doesn't exist yet.
    pub(in crate::index) fn ensure_project_neuron(&mut self, root: &Path) -> Result<()> {
        let project_name = root
            .file_name()
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_else(|| "project".to_string());
        let project_neuron = neuron_dir(root).join("_project.context.md");

        if !project_neuron.exists() {
            let now = now_iso8601();
            let content = stub_project_neuron(&project_name, &now);
            atomic_write(&project_neuron, content.as_bytes())?;
            let mut meta = NeuronMeta::new_stub(root, NeuronKind::Project);
            meta.tokens = estimate_context_tokens(&content).get();
            meta.last_updated = now;
            atomic_write_json(&meta_path(&project_neuron), &meta)?;
            self.index_neuron(&project_neuron, &content, &meta);
        }
        Ok(())
    }

    /// S5 (R15 NE4): Generate wake-up context neurons at compile time.
    ///
    /// Creates two Concept neurons from project metadata:
    /// - `_identity.context.md` (~50 tok): project name, version, authors, repo URL, description
    /// - `_critical_facts.context.md` (~120 tok): conventions, architecture highlights, key decisions
    ///
    /// Both are standard Concept neurons — BM25-indexed, evolvable, git-tracked.
    /// They are only loaded when `cortyx_wake_up` is explicitly called (P16 Partial Action —
    /// preserves Cortyx's token efficiency advantage; zero overhead when not requested).
    ///
    /// Sources: `git config`, `Cargo.toml`/`package.json`, README first 500 chars,
    /// `CONTRIBUTING.md`/`AGENTS.md` if present (first 400 chars each).
    pub(in crate::index) fn ensure_wake_up_neurons(
        &mut self,
        root: &Path,
        ndir: &Path,
    ) -> Result<()> {
        let identity_path = ndir.join("_identity.context.md");
        let critical_path = ndir.join("_critical_facts.context.md");

        // Gather project metadata from manifest files.
        let project_name = root
            .file_name()
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_else(|| "project".to_string());

        let (pkg_name, pkg_version, pkg_authors, pkg_description, pkg_repo) =
            extract_manifest_metadata(root);

        let name = if !pkg_name.is_empty() {
            pkg_name
        } else {
            project_name.clone()
        };

        // _identity.context.md — generate if absent
        if !identity_path.exists() {
            let git_author = run_git_cmd(root, &["config", "user.name"]).unwrap_or_default();
            let git_email = run_git_cmd(root, &["config", "user.email"]).unwrap_or_default();

            let readme_intro =
                read_file_head(root, &["README.md", "README.rst", "README.txt"], 300);

            let content = format!(
                "# Identity: {name}\n\n\
                 ## purpose\n\
                 Project identity card — loaded via `cortyx_wake_up` to prime LLM session context.\n\n\
                 ## api\n\
                 | Field | Value |\n\
                 |---|---|\n\
                 | Project | {name} |\n\
                 | Version | {pkg_version} |\n\
                 | Authors | {authors} |\n\
                 | Repository | {pkg_repo} |\n\
                 | Description | {pkg_description} |\n\
                 | Git author | {git_author} <{git_email}> |\n\n\
                 ## context\n\
                 {readme_intro}\n\n\
                 ## pitfalls\n\
                 _Evolve this section with key project conventions and gotchas._\n",
                authors = if !pkg_authors.is_empty() { pkg_authors.clone() } else { git_author.clone() },
            );
            atomic_write(&identity_path, content.as_bytes())?;
            let mut meta = NeuronMeta::new_stub(root, NeuronKind::Concept);
            meta.tokens = estimate_context_tokens(&content).get();
            meta.module = Some("@wake_up".to_string());
            meta.last_updated = now_iso8601();
            atomic_write_json(&meta_path(&identity_path), &meta)?;
            self.index_neuron(&identity_path, &content, &meta);
            tracing::info!("S5: generated _identity.context.md for '{name}'");
        }

        // _critical_facts.context.md — generate if absent
        if !critical_path.exists() {
            let contributing = read_file_head(
                root,
                &["CONTRIBUTING.md", "AGENTS.md", "CONTRIBUTING.rst"],
                400,
            );
            let conventions = if !contributing.is_empty() {
                contributing
            } else {
                // Fallback: extract a "conventions" or "architecture" section from README
                read_readme_section(root, &["convention", "architecture", "structure", "design"])
                    .unwrap_or_else(|| "_Evolve with team conventions, coding standards, and architectural decisions._".to_string())
            };

            let content = format!(
                "# Critical Facts: {name}\n\n\
                 ## purpose\n\
                 Key conventions, architecture decisions, and team context — loaded via \
                 `cortyx_wake_up` for session priming.\n\n\
                 ## api\n\
                 | Field | Value |\n\
                 |---|---|\n\
                 | Stack | {name} v{pkg_version} |\n\
                 | Repo | {pkg_repo} |\n\n\
                 ## context\n\
                 {conventions}\n\n\
                 ## pitfalls\n\
                 _Evolve this section after each sprint retro or architectural change._\n",
            );
            atomic_write(&critical_path, content.as_bytes())?;
            let mut meta = NeuronMeta::new_stub(root, NeuronKind::Concept);
            meta.tokens = estimate_context_tokens(&content).get();
            meta.module = Some("@wake_up".to_string());
            meta.last_updated = now_iso8601();
            atomic_write_json(&meta_path(&critical_path), &meta)?;
            self.index_neuron(&critical_path, &content, &meta);
            tracing::info!("S5: generated _critical_facts.context.md for '{name}'");
        }

        Ok(())
    }
}
