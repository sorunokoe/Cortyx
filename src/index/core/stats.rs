// This file is a submodule of `crate::index::core`.
// It contains `impl NeuronIndex` methods extracted from helpers.rs.
// All visibility is relative to `crate::index` (the parent of `core`).
use super::*;

impl NeuronIndex {
    // ── Stats ─────────────────────────────────────────────────────────────────

    pub fn neuron_count(&self) -> usize {
        self.entries.len()
    }

    pub fn synapse_count(&self) -> usize {
        // Count the forward synapses defined on each entry (not the reverse copies in adjacency).
        self.entries.iter().map(|e| e.synapses.len()).sum()
    }

    /// Status counts for doctor: (fresh, stale, stub)
    pub fn status_counts(&self) -> (usize, usize, usize) {
        let ndir = neuron_dir(&self.project_root);
        let mut fresh = 0usize;
        let mut stale = 0usize;
        let mut stub = 0usize;
        for entry in &self.entries {
            let meta_p = meta_path(&entry.neuron_path);
            let status = std::fs::read_to_string(&meta_p)
                .ok()
                .and_then(|d| serde_json::from_str::<NeuronMeta>(&d).ok())
                .map(|m| m.status)
                .unwrap_or(NeuronStatus::Stub);
            // If .context.md is in the ndir, it's a real neuron (avoid counting adjacency copies)
            if !entry.neuron_path.starts_with(&ndir) {
                continue;
            }
            match status {
                NeuronStatus::Fresh => fresh += 1,
                NeuronStatus::Stale => stale += 1,
                NeuronStatus::Stub => stub += 1,
            }
        }
        (fresh, stale, stub)
    }

    /// Return the use_count for a neuron (for display purposes).
    pub fn use_count_for(&self, path: &Path) -> u32 {
        self.path_index
            .get(path)
            .map(|&i| self.entries[i].use_count)
            .unwrap_or(0)
    }

    /// Increment `use_count` for each neuron in `paths` and persist their metadata.
    ///
    /// Also applies auto-quarantine: if a neuron has ≥ MIN_SAMPLE_SIZE activations
    /// but its hit_rate is below QUARANTINE_THRESHOLD (10%), it's a chronic
    /// over-activator — retrieved often but rarely cited. Its staleness_multiplier
    /// is reduced to 0.3, effectively deprioritising it without deletion.
    /// The quarantine lifts automatically when the neuron is re-evolved.
    pub fn record_activation(&mut self, paths: &[std::path::PathBuf]) {
        for path in paths {
            if let Some(&i) = self.path_index.get(path) {
                self.entries[i].use_count = self.entries[i].use_count.saturating_add(1);

                // Bayesian quarantine with adaptive confidence intervals (TRIZ S4 R11).
                //
                // Adaptive tiers:
                //   use_count <  5  → withhold judgment (too few samples)
                //   use_count  5–19 → z=1.0,   threshold=0.02 (react fast to obvious noise)
                //   use_count 20–99 → z=1.645, threshold=0.05 (90% CI — standard behaviour)
                //   use_count ≥100  → z=1.96,  threshold=0.08 (strict for mature neurons)
                // Quarantine is reversible: lower bound > QUARANTINE_RECOVERY_THRESHOLD → restore.
                let uc = self.entries[i].use_count;
                let hc = self.entries[i].hit_count;
                if let Some((z, threshold)) = adaptive_quarantine_params(uc) {
                    let lower = wilson_lower_bound_z(hc, uc, z);
                    let currently_quarantined = self.entries[i].staleness_multiplier <= 0.3;
                    if !currently_quarantined && lower < threshold {
                        self.entries[i].staleness_multiplier = 0.3;
                        tracing::debug!(
                            path = %path.display(),
                            wilson_lower_bound = lower,
                            use_count = uc,
                            hit_count = hc,
                            z = z,
                            threshold = threshold,
                            "Auto-quarantined: Wilson CI lower bound {lower:.3} < {threshold}"
                        );
                    } else if currently_quarantined && lower > QUARANTINE_RECOVERY_THRESHOLD {
                        self.entries[i].staleness_multiplier = 0.7;
                        tracing::debug!(
                            path = %path.display(),
                            wilson_lower_bound = lower,
                            "Quarantine lifted: Wilson CI lower bound {lower:.3} > {QUARANTINE_RECOVERY_THRESHOLD}"
                        );
                    }
                }

                // Persist the updated use_count to the sidecar JSON so it survives restarts.
                let meta_p = meta_path(path);
                if let Ok(data) = std::fs::read_to_string(&meta_p) {
                    if let Ok(mut meta) = serde_json::from_str::<NeuronMeta>(&data) {
                        meta.use_count = self.entries[i].use_count;
                        if let Err(e) = atomic_write_json(&meta_p, &meta) {
                            tracing::warn!(
                                "Failed to persist updated use_count for {}: {e}",
                                meta_p.display()
                            );
                        }
                    }
                }
            }
        }
    }

    /// Increment `hit_count` for a neuron when the LLM confirms it was cited.
    ///
    /// Returns the updated hit_rate = hit_count / use_count.max(1).
    pub fn record_hit(&mut self, neuron_path: &Path, was_cited: bool) -> f32 {
        if let Some(&i) = self.path_index.get(neuron_path) {
            if was_cited {
                self.entries[i].hit_count = self.entries[i].hit_count.saturating_add(1);
            }
            // Always increment use_count on explicit feedback (in case get_contexts missed it)
            self.entries[i].use_count = self.entries[i].use_count.saturating_add(1);

            let hit_rate =
                self.entries[i].hit_count as f32 / self.entries[i].use_count.max(1) as f32;

            // Persist both counters
            let meta_p = meta_path(neuron_path);
            if let Ok(data) = std::fs::read_to_string(&meta_p) {
                if let Ok(mut meta) = serde_json::from_str::<NeuronMeta>(&data) {
                    meta.use_count = self.entries[i].use_count;
                    meta.hit_count = self.entries[i].hit_count;
                    if let Err(e) = atomic_write_json(&meta_p, &meta) {
                        tracing::warn!(
                            "Failed to persist hit feedback for {}: {e}",
                            meta_p.display()
                        );
                    }
                }
            }

            // Adaptive synapse EMA: update learned_weight for all synapses that
            // point to this neuron, reinforcing or downweighting the traversal path.
            self.update_synapse_ema(neuron_path, was_cited);

            hit_rate
        } else {
            0.0
        }
    }

    /// B2: Record query term co-activations for a neuron.
    ///
    /// Called from `get_contexts` for each activated neuron with the query terms.
    /// After ≥30 co-activations, a term is promoted to the neuron's `synonym_cloud`.
    /// The synonym cloud is persisted to the BM25Entry and used at query time for
    /// vocabulary expansion before BM25 scoring.
    pub fn record_coactivation(&mut self, neuron_path: &Path, query_terms: &[String]) {
        const SYNONYM_THRESHOLD: u32 = 30;

        let Some(&entry_idx) = self.path_index.get(neuron_path) else {
            return;
        };

        let counts = self
            .coactivation_counts
            .entry(neuron_path.to_path_buf())
            .or_default();

        let mut promoted = Vec::new();
        for term in query_terms {
            if term.len() < 3 {
                continue;
            }
            let count = counts.entry(term.clone()).or_insert(0);
            *count += 1;
            if *count == SYNONYM_THRESHOLD {
                promoted.push(term.clone());
            }
        }

        if !promoted.is_empty() {
            let cloud = &mut self.entries[entry_idx].synonym_cloud;
            for term in &promoted {
                if !cloud.contains(term) {
                    cloud.push(term.clone());
                    tracing::debug!(
                        neuron = %neuron_path.display(),
                        term,
                        "B2: promoted term to synonym cloud"
                    );
                }
            }
        }

        // R20 C-2: Drain any pending Hebbian synapse creations.
        //
        // `get_contexts()` (a &self method) accumulates co-return counts in a Mutex.
        // Once a pair crosses HEBBIAN_THRESHOLD (10 co-returns), it's flagged there but
        // can't mutate adjacency. Here, in the first subsequent &mut self call, we drain
        // the flagged pairs and create bidirectional SemanticRelated synapses.
        self.apply_pending_hebbian_synapses();
    }

    /// Drain pending Hebbian synapse pairs and create SemanticRelated edges in adjacency.
    pub(in crate::index) fn apply_pending_hebbian_synapses(&mut self) {
        const HEBBIAN_THRESHOLD: u32 = 10;
        let pairs_to_wire: Vec<(PathBuf, PathBuf)> = {
            let Ok(counts) = self.co_return_counts.lock() else {
                return;
            };
            counts
                .iter()
                .filter(|(_, &c)| c == HEBBIAN_THRESHOLD) // exactly at threshold — fire once
                .map(|(k, _)| k.clone())
                .collect()
        };

        for (a, b) in pairs_to_wire {
            // Mark as wired (sentinel = HEBBIAN_THRESHOLD + 1) so we don't re-fire on future calls
            if let Ok(mut counts) = self.co_return_counts.lock() {
                if let Some(c) = counts.get_mut(&(a.clone(), b.clone())) {
                    *c = HEBBIAN_THRESHOLD + 1;
                }
            }

            let already_exists = self.adjacency.get(&a).map_or(false, |syns| {
                syns.iter()
                    .any(|s| s.target == b && s.edge_type == SynapseType::SemanticRelated)
            });
            if already_exists {
                continue;
            }

            let syn_ab = Synapse::new(
                b.clone(),
                SynapseType::SemanticRelated,
                "hebbian:co-return".to_string(),
            );
            let syn_ba = Synapse::new(
                a.clone(),
                SynapseType::SemanticRelated,
                "hebbian:co-return".to_string(),
            );
            self.adjacency.entry(a.clone()).or_default().push(syn_ab);
            self.adjacency.entry(b.clone()).or_default().push(syn_ba);
            tracing::debug!(
                a = %a.display(),
                b = %b.display(),
                "C-2 Hebbian: SemanticRelated synapse created from co-return signal"
            );
        }
    }

    /// B2: Expand query terms through per-neuron synonym clouds.
    ///
    /// For each activated neuron path, return any synonym-cloud terms that appear
    /// in the query — as augmented expansion terms for the next retrieval pass.
    /// Used during `get_contexts` vocabulary expansion phase.
    /// Return the highest raw BM25 score for `task` across all indexed neurons.
    ///
    /// Runs Phase 1 posting-list lookup + BM25 scoring only (no synapse traversal,
    /// no TF-IDF, no dense re-rank).  Used by `get_contexts_with_overflow` to
    /// implement the abstention signal: if the top score is below `min_confidence`,
    /// no neurons are returned and the caller prints a "no relevant memory" message.
    ///
    /// Complexity: O(|candidates|) — same as the fast path in `get_contexts`.
    pub fn peek_max_bm25_score(&self, task: &str) -> f32 {
        let Ok(query) = QueryText::new(task) else {
            return 0.0;
        };
        let terms = tokenize(query.as_str());
        let mut max_score = 0.0f32;
        for term in &terms {
            if let Some(idxs) = self.posting_list.get(term) {
                for &i in idxs {
                    let s = self.bm25_score(&terms, &self.entries[i]);
                    if s > max_score {
                        max_score = s;
                    }
                }
            }
        }
        max_score
    }

    /// Knowledge-update supersession: demote old Verbatim neurons whose content is
    /// substantially overlapped by a newer neuron in the same module/person scope.
    ///
    /// Called by `write_verbatim_neurons` after staging each new Verbatim neuron. When a
    /// newly-ingested turn has ≥60% term overlap with an older turn in the same module AND
    /// the older turn's timestamp pre-dates the new one, the old neuron's
    /// `staleness_multiplier` is halved (→ 0.5×BM25 score). This surfaces the most
    /// current fact for LME-500 knowledge-update questions without evicting history.
    ///
    /// Only applies to Verbatim neurons — code neurons are unaffected.
    pub fn detect_and_mark_supersessions(&mut self, new_path: &Path) {
        const OVERLAP_THRESHOLD: f32 = 0.60;
        const MIN_TERMS: usize = 4;

        let Some(&new_idx) = self.path_index.get(new_path) else {
            return;
        };

        // Snapshot new-entry data to avoid borrow conflicts below.
        let (new_module, new_ts, new_terms) = {
            let e = &self.entries[new_idx];
            if !matches!(e.kind, NeuronKind::Verbatim) {
                return;
            }
            let terms: HashSet<String> = e
                .term_freq
                .keys()
                .filter(|t| t.len() >= MIN_TERMS)
                .cloned()
                .collect();
            (e.module.clone(), e.timestamp_secs, terms)
        };

        if new_terms.is_empty() {
            return;
        }
        let new_ts_val = new_ts.unwrap_or(i64::MAX);

        for i in 0..self.entries.len() {
            if i == new_idx {
                continue;
            }
            let e = &self.entries[i];
            if !matches!(e.kind, NeuronKind::Verbatim) {
                continue;
            }
            if e.module != new_module {
                continue;
            }
            let old_ts = e.timestamp_secs.unwrap_or(0);
            // Only demote OLDER neurons — if old_ts ≥ new_ts, the "old" entry is newer
            // or simultaneous; skip it to avoid mutual demotion within a batch.
            if old_ts >= new_ts_val {
                continue;
            }

            let old_terms: HashSet<&str> = e
                .term_freq
                .keys()
                .filter(|t| t.len() >= MIN_TERMS)
                .map(|s| s.as_str())
                .collect();
            if old_terms.len() < MIN_TERMS {
                continue;
            }

            let overlap = new_terms
                .iter()
                .filter(|t| old_terms.contains(t.as_str()))
                .count();
            let ratio = overlap as f32 / old_terms.len() as f32;

            if ratio >= OVERLAP_THRESHOLD {
                self.entries[i].staleness_multiplier =
                    (self.entries[i].staleness_multiplier * 0.5).max(0.1);
                tracing::debug!(
                    old = ?self.entries[i].neuron_path,
                    new = ?new_path,
                    overlap_ratio = ratio,
                    "Knowledge-update supersession: demoted older neuron"
                );
            }
        }
    }

    pub fn synonym_cloud_expansion(&self, query_terms: &[String]) -> Vec<String> {
        let query_set: HashSet<&String> = query_terms.iter().collect();
        let mut expansion: HashSet<String> = HashSet::new();

        for entry in &self.entries {
            // For each neuron: check if any query term matches an entry term
            let neuron_has_query_term = entry.term_freq.keys().any(|t| query_set.contains(t));
            if neuron_has_query_term {
                // Expand with this neuron's synonym cloud
                for syn_term in &entry.synonym_cloud {
                    expansion.insert(syn_term.clone());
                }
            }
        }

        // Remove terms already in the query to avoid re-adding them
        for t in query_terms {
            expansion.remove(t);
        }

        expansion.into_iter().collect()
    }

    /// F2: Record session token utilization for budget adaptation.
    ///
    /// Call at the end of each session (close_task) with the tokens used and the budget.
    /// Keeps the last 5 sessions' data. The next call to `adaptive_budget_scale()` uses
    /// this history to adjust max_tokens up or down.
    pub fn record_session_utilization(&mut self, tokens_used: usize, tokens_budget: usize) {
        const MAX_HISTORY: usize = 5;
        self.session_utilization.push([tokens_used, tokens_budget]);
        if self.session_utilization.len() > MAX_HISTORY {
            self.session_utilization.remove(0);
        }
    }

    /// F2: Compute the budget scale factor from session history.
    ///
    /// - If last 5 sessions used < 40% of budget → scale down by 20% (too much headroom)
    /// - If ≥3 of last 5 sessions hit 100% of budget (overflow) → scale up by 20%
    /// - Otherwise: no change (scale = 1.0)
    ///
    /// Returns a multiplier [0.8, 1.2] to apply to max_tokens.
    /// Capped post-multiplication at [512, 8192] by the caller.
    pub fn adaptive_budget_scale(&self) -> f32 {
        let history = &self.session_utilization;
        if history.len() < 2 {
            return 1.0; // not enough data
        }

        let underused = history
            .iter()
            .filter(|[used, budget]| *budget > 0 && (*used as f32 / *budget as f32) < 0.4)
            .count();

        let overflowed = history
            .iter()
            .filter(|[used, budget]| *used >= *budget)
            .count();

        if underused == history.len() {
            0.8 // all sessions underused → shrink
        } else if overflowed >= 3 {
            1.2 // ≥3/5 sessions overflowed → grow
        } else {
            1.0 // normal
        }
    }

    /// `cited = true` → signal = 1.0 (this synapse helped); `false` → 0.0.
    ///
    /// EMA rule: `learned_weight ← α × signal + (1 − α) × learned_weight`  (α = 0.1)
    ///
    /// Cold-start: when `learned_weight == 0.0`, it is initialised to the type
    /// multiplier before the first update so the decay doesn't start from zero.
    ///
    /// Only in-memory entries are updated; `save()` persists them to `index.json`.
    /// NeuronMeta sidecar files are NOT updated (they are the source-of-truth for
    /// compile-time synapse topology, not runtime weights).
    pub fn update_synapse_ema(&mut self, target_path: &Path, cited: bool) {
        const ALPHA: f32 = 0.1;
        let signal = if cited { 1.0_f32 } else { 0.0_f32 };

        for entry in &mut self.entries {
            for syn in &mut entry.synapses {
                if syn.target == target_path {
                    // Cold-start init: seed from type multiplier so EMA starts at a
                    // sensible baseline rather than decaying from 0.
                    if syn.learned_weight <= 0.0 {
                        syn.learned_weight = syn.edge_type.type_multiplier();
                    }
                    syn.learned_weight = ALPHA * signal + (1.0 - ALPHA) * syn.learned_weight;
                    syn.traversal_count = syn.traversal_count.saturating_add(1);
                }
            }
        }
    }

    pub fn print_status(&self) {
        let mut cores = 0usize;
        let mut usecases = 0usize;
        let mut verbatim = 0usize;
        let mut concepts = 0usize;
        let mut stubs = 0usize;
        for e in &self.entries {
            match e.kind {
                NeuronKind::Core | NeuronKind::Project => {
                    cores += 1;
                    if e.term_count == 0 || e.term_freq.is_empty() {
                        stubs += 1;
                    }
                },
                NeuronKind::UseCase => usecases += 1,
                NeuronKind::Verbatim => verbatim += 1,
                NeuronKind::Concept | NeuronKind::Aggregate => concepts += 1,
            }
        }
        println!("Cortyx Index");
        println!("============");
        println!("  Core neurons:         {cores}  ({stubs} stubs — run cortyx_evolve_context)");
        println!("  Use-case neurons:     {usecases}");
        println!("  Verbatim chunks:      {verbatim}");
        println!("  Concept neurons:      {concepts}");
        println!("  Synapses:             {}", self.synapse_count());
        println!("  Modules indexed:      {}", self.module_index.len());
        println!("  Avg doc length:       {:.0} terms", self.avg_doc_len);
    }
}
