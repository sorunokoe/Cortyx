// This file is a submodule of `crate::index::core`.
// It contains `impl NeuronIndex` methods extracted from helpers.rs.
// All visibility is relative to `crate::index` (the parent of `core`).
use super::*;
use crate::types::{QueryText, SynapseWeight};

type ScoredContextPaths = Vec<(PathBuf, f32)>;
type OverflowContextPaths = Vec<(PathBuf, String)>;
type RankedContextResults = (ScoredContextPaths, OverflowContextPaths);

impl NeuronIndex {
    // ── Stats ─────────────────────────────────────────────────────────────────

    pub fn neuron_count(&self) -> usize {
        self.retrieval.entries.len()
    }

    pub fn synapse_count(&self) -> usize {
        // Count the forward synapses defined on each entry (not the reverse copies in adjacency).
        self.retrieval
            .entries
            .iter()
            .map(|e| e.synapses.len())
            .sum()
    }

    /// Status counts for doctor: (fresh, stale, stub)
    pub fn status_counts(&self) -> (usize, usize, usize) {
        let ndir = neuron_dir(&self.persistence.project_root);
        let mut fresh = 0usize;
        let mut stale = 0usize;
        let mut stub = 0usize;
        for entry in &self.retrieval.entries {
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
        self.retrieval
            .path_index
            .get(path)
            .map(|&i| {
                self.retrieval.entries[i]
                    .use_count
                    .load(std::sync::atomic::Ordering::Relaxed)
            })
            .unwrap_or(0)
    }

    pub(in crate::index) fn total_activations(&self) -> u64 {
        self.feedback
            .total_activations
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    fn mark_sidecar_dirty(&self, path: &Path) {
        self.persistence.mark_sidecar_dirty(path);
    }

    #[allow(dead_code)]
    fn persist_feedback_sidecar(&self, neuron_path: &Path) -> bool {
        self.persistence
            .persist_feedback_sidecar(neuron_path, &self.retrieval)
    }

    pub(in crate::index) fn flush_dirty_sidecars(&self) {
        self.persistence.flush_dirty_sidecars(&self.retrieval);
    }

    /// Increment `use_count` for each neuron in `paths` and defer sidecar persistence until save().
    ///
    /// Also applies auto-quarantine: if a neuron has ≥ MIN_SAMPLE_SIZE activations
    /// but its hit_rate is below QUARANTINE_THRESHOLD (10%), it's a chronic
    /// over-activator — retrieved often but rarely cited. Its staleness_multiplier
    /// is reduced to 0.3, effectively deprioritising it without deletion.
    /// The quarantine lifts automatically when the neuron is re-evolved.
    pub fn record_activation(&mut self, paths: &[std::path::PathBuf]) {
        for path in paths {
            if let Some(&i) = self.retrieval.path_index.get(path) {
                let uc = self.retrieval.entries[i].increment_use_count();
                self.feedback
                    .total_activations
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);

                // Bayesian quarantine with adaptive confidence intervals (TRIZ S4 R11).
                //
                // Adaptive tiers:
                //   use_count <  5  → withhold judgment (too few samples)
                //   use_count  5–19 → z=1.0,   threshold=0.02 (react fast to obvious noise)
                //   use_count 20–99 → z=1.645, threshold=0.05 (90% CI — standard behaviour)
                //   use_count ≥100  → z=1.96,  threshold=0.08 (strict for mature neurons)
                // Quarantine is reversible: lower bound > QUARANTINE_RECOVERY_THRESHOLD → restore.
                let hc = self.retrieval.entries[i].hit_count;
                if let Some((z, threshold)) = adaptive_quarantine_params(uc) {
                    let lower = wilson_lower_bound_z(hc, uc, z);
                    let currently_quarantined =
                        self.retrieval.entries[i].staleness_multiplier <= 0.3;
                    let has_quarantine_signal = hc > 0 || uc >= QUARANTINE_MIN_SAMPLES * 3;
                    if !currently_quarantined && has_quarantine_signal && lower < threshold {
                        self.retrieval.entries[i].staleness_multiplier = 0.3;
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
                        self.retrieval.entries[i].staleness_multiplier = 0.7;
                        tracing::debug!(
                            path = %path.display(),
                            wilson_lower_bound = lower,
                            "Quarantine lifted: Wilson CI lower bound {lower:.3} > {QUARANTINE_RECOVERY_THRESHOLD}"
                        );
                    }
                }

                self.mark_sidecar_dirty(path);
            }
        }
    }

    /// Increment `hit_count` for a neuron when the LLM confirms it was cited.
    ///
    /// Feedback sidecars are flushed on the next `save()` instead of synchronously here.
    /// Returns the updated hit_rate = hit_count / use_count.max(1).
    pub fn record_hit(&mut self, neuron_path: &Path, was_cited: bool) -> f32 {
        if let Some(&i) = self.retrieval.path_index.get(neuron_path) {
            if was_cited {
                self.retrieval.entries[i].hit_count =
                    self.retrieval.entries[i].hit_count.saturating_add(1);
            }
            // Always increment use_count on explicit feedback (in case get_contexts missed it)
            let use_count = self.retrieval.entries[i].increment_use_count();
            self.feedback
                .total_activations
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);

            let hit_rate = self.retrieval.entries[i].hit_count as f32 / use_count.max(1) as f32;

            self.mark_sidecar_dirty(neuron_path);

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
        self.feedback
            .record_coactivation(neuron_path, query_terms, &mut self.retrieval);
    }

    /// Drain pending co-return synapse pairs and create SemanticRelated edges in adjacency.
    ///
    /// # "Hebbian" as metaphor
    /// This mechanism borrows Hebb's rule ("neurons that fire together wire together") as an
    /// analogy: *neurons that are **returned** together wire together*. The underlying signal is
    /// co-return co-occurrence — two neurons appearing in the same query result ≥3 times —
    /// not simultaneous neural activation in the biological sense. The label is evocative, not
    /// mechanistically literal.
    ///
    /// # Wilson-Adaptive Threshold (TRIZ C6)
    /// Instead of a fixed threshold of 10 co-returns, synapse formation is governed by a Wilson
    /// score lower bound on the co-return rate:
    ///
    ///   rate = co_return_count / min(use_a, use_b)
    ///
    /// where `min(use_a, use_b)` is the tightest upper bound on co-occurrence opportunities
    /// (the rarer neuron's activation count). A Wilson lower bound ≥ 0.10 at 68% CI is
    /// required before a synapse fires. This means:
    ///
    /// - Strong pairs (always co-returned): synapse forms at count ≈ 3–5 (fast)
    /// - Weak pairs (10/100 activations): synapse never forms, noise-resistant
    ///
    /// `HEBBIAN_WIRED = u32::MAX` is used as the sentinel (not THRESHOLD+1) to prevent
    /// the counting window from being accidentally overshot by concurrent increments.
    #[allow(dead_code)]
    pub(in crate::index) fn apply_pending_hebbian_synapses(&mut self) {
        self.feedback
            .apply_pending_hebbian_synapses(&mut self.retrieval);
    }

    pub(crate) fn rerank_contexts_with_session_tf(
        &self,
        task: &str,
        max_tokens: usize,
        results: RankedContextResults,
        session_tf: &HashMap<String, f32>,
    ) -> RankedContextResults {
        const SESSION_TF_MIN_COUNT: f32 = 3.0;
        const SESSION_TF_WEIGHT: f32 = 0.2;

        let (full, overflow) = results;
        let Ok(query) = QueryText::new(task) else {
            return (full, overflow);
        };
        let mut hot_terms: Vec<(&str, f32)> = session_tf
            .iter()
            .filter(|(_, count)| **count >= SESSION_TF_MIN_COUNT)
            .map(|(term, count)| (term.as_str(), *count))
            .collect();
        if hot_terms.is_empty() {
            return (full, overflow);
        }
        hot_terms.sort_unstable_by(|a, b| a.0.cmp(b.0));

        let max_session_count = hot_terms
            .iter()
            .map(|(_, count)| *count)
            .fold(SESSION_TF_MIN_COUNT, f32::max);
        let total_session_weight: f32 = hot_terms
            .iter()
            .map(|(_, count)| *count / max_session_count)
            .sum();
        let boost_score = |entry: &BM25Entry, base_score: f32| {
            let matched_weight: f32 = hot_terms
                .iter()
                .filter(|(term, _)| entry.term_freq.contains_key(*term))
                .map(|(_, count)| *count / max_session_count)
                .sum();
            if matched_weight <= 0.0 {
                base_score
            } else {
                let session_tf_score = matched_weight / total_session_weight.max(f32::EPSILON);
                base_score * (1.0 + SESSION_TF_WEIGHT * session_tf_score)
            }
        };

        let terms = tokenize(query.as_str());
        let complexity = self.compute_task_complexity(&terms);
        let history_scale = self.adaptive_budget_scale();
        let adjusted_max = ((max_tokens as f32 * complexity * history_scale) as usize)
            .max(512)
            .min(8192.max(max_tokens * 2));

        let mut overflow_headlines: HashMap<PathBuf, String> = overflow.into_iter().collect();
        let mut seen = HashSet::new();
        let mut candidates = Vec::new();

        for (path, score) in full {
            if !seen.insert(path.clone()) {
                continue;
            }
            let boosted = self
                .entry_by_path(&path)
                .map(|entry| boost_score(entry, score))
                .unwrap_or(score);
            candidates.push((path, boosted));
        }
        for path in overflow_headlines.keys() {
            if !seen.insert(path.clone()) {
                continue;
            }
            let score = self
                .entry_by_path(path)
                .map(|entry| boost_score(entry, self.bm25_score(&terms, entry)))
                .unwrap_or(0.0);
            candidates.push((path.clone(), score));
        }

        candidates.sort_unstable_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));

        let mut reranked_full = Vec::new();
        let mut reranked_overflow = Vec::new();
        let mut used = 0usize;
        for (path, score) in candidates {
            let tokens = self
                .entry_by_path(&path)
                .map(|entry| entry.tokens)
                .unwrap_or(200);
            if used + tokens <= adjusted_max || reranked_full.is_empty() {
                used += tokens;
                reranked_full.push((path, score));
            } else {
                let headline = overflow_headlines
                    .remove(&path)
                    .unwrap_or_else(|| neuron_headline_for(&path));
                reranked_overflow.push((path, headline));
            }
        }

        (reranked_full, reranked_overflow)
    }

    /// B2: Expand query terms through per-neuron synonym clouds.
    ///
    /// For each activated neuron path, return any synonym-cloud terms that appear
    /// in the query — as augmented expansion terms for the next retrieval pass.
    /// Used during `get_contexts` vocabulary expansion phase.
    pub(crate) fn bm25_score_for_path(&self, terms: &[String], path: &Path) -> f32 {
        self.entry_by_path(path)
            .map(|entry| self.bm25_score(terms, entry))
            .unwrap_or(0.0)
    }

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
            if let Some(idxs) = self.retrieval.posting_list.get(term) {
                for &i in idxs {
                    let s = self.bm25_score(&terms, &self.retrieval.entries[i]);
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

        let Some(&new_idx) = self.retrieval.path_index.get(new_path) else {
            return;
        };

        // Snapshot new-entry data to avoid borrow conflicts below.
        let (new_module, new_ts, new_terms) = {
            let e = &self.retrieval.entries[new_idx];
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

        for i in 0..self.retrieval.entries.len() {
            if i == new_idx {
                continue;
            }
            let e = &self.retrieval.entries[i];
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
                self.retrieval.entries[i].staleness_multiplier =
                    (self.retrieval.entries[i].staleness_multiplier * 0.5).max(0.1);
                tracing::debug!(
                    old = ?self.retrieval.entries[i].neuron_path,
                    new = ?new_path,
                    overlap_ratio = ratio,
                    "Knowledge-update supersession: demoted older neuron"
                );
            }
        }
    }

    pub fn synonym_cloud_expansion(&self, query_terms: &[String]) -> Vec<String> {
        self.retrieval.synonym_cloud_expansion(query_terms)
    }

    /// F2: Record session token utilization for budget adaptation.
    ///
    /// Call at the end of each session (close_task) with the tokens used and the budget.
    /// Keeps the last 5 sessions' data. The next call to `adaptive_budget_scale()` uses
    /// this history to adjust max_tokens up or down.
    pub fn record_session_utilization(&mut self, tokens_used: usize, tokens_budget: usize) {
        self.feedback
            .record_session_utilization(tokens_used, tokens_budget);
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
        self.feedback.adaptive_budget_scale()
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

        for entry in &mut self.retrieval.entries {
            for syn in &mut entry.synapses {
                if syn.target == target_path {
                    // Cold-start init: seed from type multiplier so EMA starts at a
                    // sensible baseline rather than decaying from 0.
                    if syn.learned_weight.is_zero() {
                        syn.learned_weight = SynapseWeight::new(syn.edge_type.type_multiplier());
                    }
                    syn.learned_weight = SynapseWeight::new(
                        ALPHA * signal + (1.0 - ALPHA) * syn.learned_weight.get(),
                    );
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
        for e in &self.retrieval.entries {
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
        println!(
            "  Modules indexed:      {}",
            self.retrieval.module_index.len()
        );
        println!(
            "  Avg doc length:       {:.0} terms",
            self.retrieval.avg_doc_len
        );
    }
}
