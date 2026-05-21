use super::*;

const COLD_START_CENTRALITY_MAX_WEIGHT: f32 = 0.2;
const COLD_START_CENTRALITY_WARM_ACTIVATIONS: u64 = 200;

pub(in crate::index::core) fn cold_start_centrality_blend(
    bm25_score: f32,
    centrality: f32,
    total_activations: u64,
) -> f32 {
    if total_activations >= COLD_START_CENTRALITY_WARM_ACTIVATIONS {
        return bm25_score;
    }
    let weight = COLD_START_CENTRALITY_MAX_WEIGHT
        * (1.0 - total_activations as f32 / COLD_START_CENTRALITY_WARM_ACTIVATIONS as f32);
    bm25_score * (1.0 + weight * centrality.max(0.0))
}

pub(in crate::index::core) fn query_touches_entry_module(
    terms: &[String],
    entry: &BM25Entry,
) -> bool {
    if terms.is_empty() {
        return false;
    }

    let mut module_tokens = HashSet::new();
    if let Some(module) = entry.module.as_deref() {
        module_tokens.extend(tokenize(module));
    }
    if let Some(stem) = entry.neuron_path.file_stem().and_then(|stem| stem.to_str()) {
        let cleaned = stem
            .trim_end_matches(".context")
            .replace("_rs", " ")
            .replace("_ts", " ")
            .replace("_py", " ")
            .replace("_go", " ");
        module_tokens.extend(tokenize(&cleaned));
    }

    !module_tokens.is_empty()
        && terms.iter().any(|term| {
            module_tokens.iter().any(|token| {
                token == term || token.contains(term.as_str()) || term.contains(token.as_str())
            })
        })
}

pub(in crate::index::core) fn apply_structural_centrality_prior(
    terms: &[String],
    entry: &BM25Entry,
    total_activations: u64,
    bm25_score: f32,
) -> f32 {
    if bm25_score <= 0.0
        || entry.structural_centrality <= 0.0
        || !query_touches_entry_module(terms, entry)
    {
        return bm25_score;
    }
    cold_start_centrality_blend(bm25_score, entry.structural_centrality, total_activations)
}

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
        self.retrieval.expand_query_terms(terms)
    }

    /// BM25 score for a single entry given query terms.
    ///
    /// Uses the precomputed `df_cache` for O(1) IDF lookup.
    /// Applies `entry.confidence_score` as a mild prior multiplier:
    /// committed + unmodified = 1.0 (neutral), modified = 0.9, untracked = 0.85.
    pub(in crate::index) fn bm25_score(&self, terms: &[String], entry: &BM25Entry) -> f32 {
        apply_structural_centrality_prior(
            terms,
            entry,
            self.total_activations(),
            self.retrieval.bm25_score(terms, entry),
        )
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
            let q_tf = 1.0f32;
            let d_tf = entry.term_freq.get(term).map(|v| v.get()).unwrap_or(0.0);
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
        self.retrieval.entry_by_path(path)
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
        self.retrieval.term_freq_overlap(path, tokens)
    }
}

#[cfg(test)]
mod tests {
    use super::{apply_structural_centrality_prior, cold_start_centrality_blend};
    use crate::index::core::BM25Entry;
    use std::path::PathBuf;

    #[test]
    fn cold_start_centrality_blend_decays() {
        assert!((cold_start_centrality_blend(10.0, 1.0, 0) - 12.0).abs() < 1e-6);
        assert!((cold_start_centrality_blend(10.0, 1.0, 100) - 11.0).abs() < 1e-6);
    }

    #[test]
    fn cold_start_centrality_zero_at_warm() {
        assert!((cold_start_centrality_blend(10.0, 1.0, 200) - 10.0).abs() < 1e-6);
        assert!((cold_start_centrality_blend(10.0, 1.0, 500) - 10.0).abs() < 1e-6);
    }

    #[test]
    fn structural_prior_only_boosts_query_touched_modules() {
        let entry = BM25Entry {
            neuron_path: PathBuf::from("auth_handler.context.md"),
            module: Some("auth".into()),
            structural_centrality: 1.0,
            ..Default::default()
        };

        assert!(
            (apply_structural_centrality_prior(&["auth".into()], &entry, 0, 10.0) - 12.0).abs()
                < 1e-6
        );
        assert!(
            (apply_structural_centrality_prior(&["render".into()], &entry, 0, 10.0) - 10.0).abs()
                < 1e-6
        );
    }
}
