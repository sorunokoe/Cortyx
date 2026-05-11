use super::*;

impl NeuronIndex {

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
}
