//! Domain state for retrieval and query-time structures.

use super::super::pipeline::RetrievalStateView;
use super::super::*;
use std::path::Path;

#[cfg(feature = "embed")]
use crate::embedder::EmbeddingStore;

/// Owned retrieval state for `NeuronIndex`.
#[derive(Debug, Default)]
pub(crate) struct RetrievalState {
    pub(in crate::index) entries: Vec<BM25Entry>,
    pub(in crate::index) adjacency: HashMap<PathBuf, Vec<Synapse>>,
    pub(in crate::index) path_index: HashMap<PathBuf, usize>,
    pub(in crate::index) parent_index: HashMap<PathBuf, Vec<usize>>,
    pub(in crate::index) df_cache: HashMap<String, usize>,
    pub(in crate::index) posting_list: HashMap<String, Vec<usize>>,
    pub(in crate::index) avg_doc_len: f32,
    pub(in crate::index) avg_verbatim_doc_len: f32,
    pub(in crate::index) module_index: HashMap<String, Vec<usize>>,
    pub(in crate::index) vocab_bridge: HashMap<String, HashSet<String>>,
    pub(in crate::index) morpheme_map: HashMap<String, Vec<String>>,
    pub(in crate::index) session_index: HashMap<String, Vec<usize>>,
    pub(in crate::index) pmi_neighbors: HashMap<String, Vec<String>>,
    #[cfg(feature = "embed")]
    pub(in crate::index) embeddings: std::sync::Arc<EmbeddingStore>,
    pub(in crate::index) idf_n: usize,
}

impl RetrievalState {
    #[allow(dead_code)]
    pub(in crate::index::core) fn new() -> Self {
        Self::default()
    }

    #[allow(dead_code)]
    pub(in crate::index::core) fn view(&self) -> RetrievalStateView<'_> {
        RetrievalStateView {
            entries: &self.entries,
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
            session_index: &self.session_index,
            pmi_neighbors: &self.pmi_neighbors,
            #[cfg(feature = "embed")]
            embeddings: self.embeddings.as_ref(),
            idf_n: self.idf_n,
        }
    }

    pub(in crate::index::core) fn expand_query_terms(&self, terms: &[String]) -> Vec<String> {
        let mut expanded: HashSet<String> = terms.iter().cloned().collect();
        for term in terms {
            let term_lower = term.to_lowercase();

            for (fragment, vocab) in &self.vocab_bridge {
                if fragment.contains(term_lower.as_str()) || term_lower.contains(fragment.as_str())
                {
                    expanded.extend(vocab.iter().take(50).cloned());
                }
            }

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

            if let Some(pmi_nbrs) = self.pmi_neighbors.get(term_lower.as_str()) {
                expanded.extend(pmi_nbrs.iter().take(3).cloned());
            }

            for variant in morphological_variants(&term_lower) {
                if self.df_cache.contains_key(variant.as_str()) {
                    expanded.insert(variant);
                }
            }
        }
        expanded.into_iter().collect()
    }

    pub(in crate::index::core) fn bm25_score(&self, terms: &[String], entry: &BM25Entry) -> f32 {
        let n = self.idf_n.max(1) as f32;
        let avg = self.avg_doc_len.max(1.0);
        let dl = entry.term_count as f32;
        let len_norm = 1.0 - BM25_B + BM25_B * (dl / avg);

        let k1 = if matches!(entry.kind, NeuronKind::Verbatim) {
            1.5
        } else {
            BM25_K1
        };

        let raw: f32 = terms
            .iter()
            .map(|t| {
                let tf = entry.term_freq.get(t).map(|v| v.get()).unwrap_or(0.0);
                if tf == 0.0 {
                    return 0.0;
                }
                let df = self.df_cache.get(t).copied().unwrap_or(1) as f32;
                let idf = ((n - df + 0.5) / (df + 0.5) + 1.0).ln().max(0.0);
                const BM25_DELTA: f32 = 0.5;
                idf * (BM25_DELTA + (tf * (k1 + 1.0)) / (tf + k1 * len_norm))
            })
            .sum();

        let hit_multiplier = if entry.use_count < MIN_SAMPLE_SIZE {
            1.0
        } else {
            let hit_rate = entry.hit_count as f32 / entry.use_count as f32;
            (1.0 + hit_rate).min(1.5)
        };

        raw * entry.confidence_score
            * hit_multiplier
            * entry.staleness_multiplier
            * if entry.quality_score < 0.4 { 0.7 } else { 1.0 }
    }

    pub(in crate::index::core) fn entry_by_path(&self, path: &Path) -> Option<&BM25Entry> {
        self.path_index.get(path).map(|&i| &self.entries[i])
    }

    pub(in crate::index::core) fn term_freq_overlap(
        &self,
        path: &Path,
        tokens: &std::collections::HashSet<String>,
    ) -> usize {
        self.entry_by_path(path)
            .map(|entry| {
                tokens
                    .iter()
                    .filter(|token| entry.term_freq.contains_key(*token))
                    .count()
            })
            .unwrap_or(0)
    }

    pub(in crate::index::core) fn synonym_cloud_expansion(
        &self,
        query_terms: &[String],
    ) -> Vec<String> {
        let query_set: HashSet<&String> = query_terms.iter().collect();
        let mut expansion: HashSet<String> = HashSet::new();

        for entry in &self.entries {
            let neuron_has_query_term = entry.term_freq.keys().any(|term| query_set.contains(term));
            if neuron_has_query_term {
                for syn_term in &entry.synonym_cloud {
                    expansion.insert(syn_term.clone());
                }
            }
        }

        for term in query_terms {
            expansion.remove(term);
        }

        expansion.into_iter().collect()
    }

    pub(in crate::index::core) fn build_vocab_bridge(&mut self) {
        let mut bridge: HashMap<String, HashSet<String>> = HashMap::new();
        for entry in &self.entries {
            if matches!(entry.kind, NeuronKind::Aggregate) {
                continue;
            }
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
            if let Some(stem) = entry.neuron_path.file_stem().and_then(|s| s.to_str()) {
                let cleaned = stem
                    .trim_end_matches(".context")
                    .replace("_rs", "")
                    .replace("_ts", "")
                    .replace("_py", "")
                    .replace("_go", "")
                    .to_lowercase();
                for fragment in cleaned.split('_').filter(|fragment| fragment.len() >= 4) {
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
                        .filter(|term| term.len() >= 3)
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

    pub(in crate::index::core) fn merge_cooccurrence_into_vocab_bridge(
        &mut self,
        project_root: &Path,
    ) {
        let co_path = project_root.join(".cortyx").join("cooccurrence.json");
        if !co_path.exists() {
            return;
        }
        let Ok(json) = std::fs::read_to_string(&co_path) else {
            return;
        };
        let Ok(clusters): std::result::Result<HashMap<String, Vec<String>>, _> =
            serde_json::from_str(&json)
        else {
            return;
        };

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

    pub(in crate::index::core) fn load_pmi_neighbors(&mut self, project_root: &Path) {
        let co_path = project_root.join(".cortyx").join("cooccurrence.json");
        if !co_path.exists() {
            return;
        }
        let Ok(json) = std::fs::read_to_string(&co_path) else {
            return;
        };
        let Ok(clusters): std::result::Result<HashMap<String, Vec<String>>, _> =
            serde_json::from_str(&json)
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
                .filter(|neighbor| neighbor.len() >= 4)
                .take(5)
                .collect();
            if !valid.is_empty() {
                self.pmi_neighbors.insert(term, valid);
                loaded += 1;
            }
        }
        tracing::debug!(terms = loaded, "P1-A: PMI neighbors loaded (no global cap)");
    }

    pub(in crate::index::core) fn build_morpheme_map(&mut self) {
        let mut map: HashMap<String, Vec<String>> = HashMap::new();

        for entry in &self.entries {
            if matches!(entry.kind, NeuronKind::Aggregate) {
                continue;
            }
            for token in entry.term_freq.keys() {
                if token.len() < 4 {
                    continue;
                }
                let snake_parts: Vec<&str> = token.split('_').collect();
                let camel_parts = split_camel_case(token);

                let mut sub_tokens: HashSet<&str> = HashSet::new();
                for part in snake_parts.iter().chain(
                    camel_parts
                        .iter()
                        .map(|part| part.as_str())
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

        for values in map.values_mut() {
            values.sort_unstable();
            values.dedup();
        }

        self.morpheme_map = map;
    }

    pub(in crate::index::core) fn build_concept_clouds(&mut self) {
        const MAX_TERMS_PER_NEIGHBOUR: usize = 50;
        const MAX_CLOUD_SIZE: usize = 200;

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
                            .filter(|term| term.len() >= 3)
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
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_creates_empty_state() {
        let state = RetrievalState::new();
        assert!(state.entries.is_empty());
        assert!(state.adjacency.is_empty());
        assert!(state.path_index.is_empty());
        assert_eq!(state.idf_n, 0);
    }

    #[test]
    fn default_is_empty() {
        let state = RetrievalState::default();
        assert!(state.module_index.is_empty());
        assert!(state.vocab_bridge.is_empty());
        assert!(state.morpheme_map.is_empty());
        assert_eq!(state.avg_doc_len, 0.0);
        assert_eq!(state.avg_verbatim_doc_len, 0.0);
    }
}
