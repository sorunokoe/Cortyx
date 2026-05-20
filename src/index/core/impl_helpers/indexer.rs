use super::*;

impl NeuronIndex {
    /// Add or replace a single entry in `self.retrieval.entries` (does NOT rebuild derived).
    pub fn index_neuron(&mut self, neuron_path: &Path, content: &str, meta: &NeuronMeta) {
        let index_content = content;

        let terms = tokenize(index_content);
        let mut tf: HashMap<String, TermFrequency> = HashMap::new();
        for t in &terms {
            *tf.entry(t.clone()).or_insert(TermFrequency::ZERO) += 1.0;
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
                        let v = tf.entry(t).or_insert(TermFrequency::ZERO);
                        *v += 0.5;
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
                        *tf.entry(t).or_insert(TermFrequency::ZERO) += 1.0;
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
                let v = tf.entry(t).or_insert(TermFrequency::ZERO);
                if v.is_zero() {
                    *v = TermFrequency::new(0.3);
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
                    let v = tf.entry(t).or_insert(TermFrequency::ZERO);
                    if v.get() < 0.5 {
                        *v = TermFrequency::new(0.5);
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
        let ndir = neuron_dir(&self.persistence.project_root);
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
        let lsh_fingerprints = simhash_256(&tf);

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

        if let Some(&pos) = self.retrieval.path_index.get(neuron_path) {
            self.retrieval.entries[pos] = entry;
            self.persistence
                .has_pending_updates
                .store(true, Ordering::Release);
            self.persistence.delta_dirty.store(true, Ordering::Relaxed);
        } else {
            let pos = self.retrieval.entries.len();
            self.retrieval
                .path_index
                .insert(neuron_path.to_path_buf(), pos);
            self.retrieval.entries.push(entry);
            self.persistence.pending_append_count += 1;
        }
    }
}
