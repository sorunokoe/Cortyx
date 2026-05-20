use super::*;
use crate::index::core::pipeline::{ActivationPipeline, QueryContext, ScoredCandidate};
use crate::types::TermFrequency;
use std::collections::{HashMap, HashSet};

impl NeuronIndex {
    pub(super) fn phase1_candidates(&self, ctx: &QueryContext<'_>) -> Vec<ScoredCandidate> {
        let pipeline = ActivationPipeline::phase1();
        let mut candidates = Vec::new();
        pipeline.run(ctx, &mut candidates);

        if candidates.is_empty() && !ctx.concept_cloud_candidate_ids.is_empty() {
            candidates =
                self.score_candidate_ids(ctx, &ctx.concept_cloud_candidate_ids, &ctx.ranking_terms);
            pipeline.run(ctx, &mut candidates);
        }

        sort_scored_candidates(&mut candidates);
        candidates
    }

    pub(super) fn rerank_candidates(
        &self,
        ctx: &QueryContext<'_>,
        candidates: &mut Vec<ScoredCandidate>,
    ) {
        self.apply_named_person_move_rerank(candidates);
        self.aggregate_temporal_chain_scores(candidates);
        self.apply_lsh_fallback(ctx, candidates);
        self.apply_sparse_rerank(ctx, candidates);
        sort_scored_candidates(candidates);
    }

    pub(super) fn score_candidate_ids(
        &self,
        ctx: &QueryContext<'_>,
        candidate_ids: &HashSet<usize>,
        query_terms: &[String],
    ) -> Vec<ScoredCandidate> {
        let mut scored = candidate_ids
            .iter()
            .filter_map(|&idx| {
                let entry = &self.retrieval.entries[idx];
                if !ctx.kind_matches(entry) || !ctx.module_matches(idx) {
                    return None;
                }
                let mut score = ctx.score_entry_with_terms(query_terms, entry);
                if is_session_summary_path(&entry.neuron_path) {
                    if ctx.is_counting {
                        score *= 1.35;
                    } else if matches!(ctx.kind_lower.as_deref(), Some("conversation") | None) {
                        score *= 1.15;
                    }
                }
                if ctx.is_knowledge_update && matches!(entry.kind, NeuronKind::Verbatim) {
                    score *= 0.5;
                }
                (score > 0.0).then_some(ScoredCandidate::new(idx, score, entry.tokens))
            })
            .collect::<Vec<_>>();
        sort_scored_candidates(&mut scored);
        scored
    }

    fn apply_named_person_move_rerank(&self, candidates: &mut [ScoredCandidate]) {
        for candidate in candidates.iter_mut() {
            let entry = &self.retrieval.entries[candidate.entry_idx];
            if !matches!(entry.kind, NeuronKind::Verbatim) {
                continue;
            }
            if entry.has_move_residence_evidence {
                candidate.score *= 1.35;
            } else {
                candidate.score *= 0.55;
            }
        }
    }

    fn aggregate_temporal_chain_scores(&self, candidates: &mut [ScoredCandidate]) {
        let scored_path_map: HashMap<PathBuf, f32> = candidates
            .iter()
            .filter(|candidate| {
                matches!(
                    self.retrieval.entries[candidate.entry_idx].kind,
                    NeuronKind::Verbatim
                )
            })
            .map(|candidate| {
                (
                    self.retrieval.entries[candidate.entry_idx]
                        .neuron_path
                        .clone(),
                    candidate.score,
                )
            })
            .collect();
        if scored_path_map.is_empty() {
            return;
        }

        for candidate in candidates.iter_mut() {
            if !matches!(
                self.retrieval.entries[candidate.entry_idx].kind,
                NeuronKind::Verbatim
            ) {
                continue;
            }
            let anchor = self.retrieval.entries[candidate.entry_idx]
                .neuron_path
                .clone();
            let mut frontier = vec![anchor.clone()];
            let mut seen = HashSet::from([anchor]);
            let mut hop_discount = 0.5_f32;

            for _ in 0..3 {
                let mut next_frontier = Vec::new();
                for path in &frontier {
                    let Some(neighbors) = self.retrieval.adjacency.get(path) else {
                        continue;
                    };
                    for synapse in neighbors {
                        if synapse.edge_type != SynapseType::TemporalFollows
                            || seen.contains(&synapse.target)
                        {
                            continue;
                        }
                        seen.insert(synapse.target.clone());
                        if let Some(chain_score) = scored_path_map.get(&synapse.target) {
                            candidate.score += hop_discount * *chain_score;
                        }
                        next_frontier.push(synapse.target.clone());
                    }
                }
                if next_frontier.is_empty() {
                    break;
                }
                frontier = next_frontier;
                hop_discount *= 0.5;
            }
        }
    }

    fn apply_lsh_fallback(&self, ctx: &QueryContext<'_>, candidates: &mut Vec<ScoredCandidate>) {
        if candidates.len() >= 2 || ctx.active_scoring_terms.is_empty() {
            return;
        }

        let query_tf = ctx
            .active_scoring_terms
            .iter()
            .fold(HashMap::new(), |mut map, term| {
                *map.entry(term.clone()).or_insert(TermFrequency::ZERO) += 1.0;
                map
            });
        let query_fps = simhash_256(&query_tf);
        let already_scored = candidates
            .iter()
            .map(|candidate| candidate.entry_idx)
            .collect::<HashSet<_>>();

        for (idx, entry) in self.retrieval.entries.iter().enumerate() {
            if already_scored.contains(&idx)
                || ctx
                    .module_set
                    .as_ref()
                    .is_some_and(|module_set| !module_set.contains(&idx))
                || entry
                    .lsh_fingerprints
                    .iter()
                    .all(|fingerprint| *fingerprint == 0)
            {
                continue;
            }
            let matched = query_fps
                .iter()
                .zip(entry.lsh_fingerprints.iter())
                .any(|(&query_fp, &entry_fp)| hamming_distance(query_fp, entry_fp) <= 14);
            if matched {
                candidates.push(ScoredCandidate::new(idx, 0.5, entry.tokens));
            }
        }
    }

    fn apply_sparse_rerank(&self, ctx: &QueryContext<'_>, candidates: &mut Vec<ScoredCandidate>) {
        sort_scored_candidates(candidates);
        let mut top_score = candidates
            .first()
            .map(|candidate| candidate.score)
            .unwrap_or(0.0);

        if top_score < LOW_CONFIDENCE_THRESHOLD {
            const ITERATIVE_RRF_K: f32 = 60.0;
            let mut expansion_seed_terms = ctx.ranking_terms.clone();
            for candidate in candidates.iter().take(5) {
                expansion_seed_terms.extend(
                    self.retrieval.entries[candidate.entry_idx]
                        .concept_cloud
                        .iter()
                        .cloned(),
                );
            }
            expansion_seed_terms.sort();
            expansion_seed_terms.dedup();
            let expanded_terms = self.expand_query_terms(&expansion_seed_terms);
            if expanded_terms.len() > ctx.ranking_terms.len() {
                let expanded_candidate_set = expanded_terms
                    .iter()
                    .filter_map(|term| self.retrieval.posting_list.get(term))
                    .flat_map(|indices| indices.iter().copied())
                    .collect::<HashSet<_>>();
                let expanded_scored =
                    self.score_candidate_ids(ctx, &expanded_candidate_set, &expanded_terms);
                if !expanded_scored.is_empty() {
                    let original_top = top_score;
                    let mut merged_rrf: HashMap<usize, f32> = HashMap::new();
                    let mut merged_scores: HashMap<usize, f32> = HashMap::new();
                    for (rank, candidate) in candidates.iter().enumerate() {
                        *merged_rrf.entry(candidate.entry_idx).or_insert(0.0) +=
                            1.0 / (ITERATIVE_RRF_K + rank as f32);
                        merged_scores
                            .entry(candidate.entry_idx)
                            .and_modify(|existing| *existing = existing.max(candidate.score))
                            .or_insert(candidate.score);
                    }
                    for (rank, candidate) in expanded_scored.iter().enumerate() {
                        *merged_rrf.entry(candidate.entry_idx).or_insert(0.0) +=
                            1.0 / (ITERATIVE_RRF_K + rank as f32);
                        merged_scores
                            .entry(candidate.entry_idx)
                            .and_modify(|existing| *existing = existing.max(candidate.score))
                            .or_insert(candidate.score);
                    }
                    let mut merged_ranked = merged_scores
                        .into_iter()
                        .map(|(idx, score)| {
                            let rrf = merged_rrf.get(&idx).copied().unwrap_or(0.0);
                            (idx, score, rrf)
                        })
                        .collect::<Vec<_>>();
                    merged_ranked.sort_unstable_by(|a, b| {
                        b.2.total_cmp(&a.2)
                            .then_with(|| b.1.total_cmp(&a.1))
                            .then_with(|| a.0.cmp(&b.0))
                    });
                    let merged_top = merged_ranked
                        .first()
                        .map(|(_, score, _)| *score)
                        .unwrap_or(0.0);
                    if merged_top >= original_top {
                        *candidates = merged_ranked
                            .into_iter()
                            .map(|(idx, score, _)| {
                                ScoredCandidate::new(idx, score, self.retrieval.entries[idx].tokens)
                            })
                            .collect();
                        top_score = merged_top;
                    }
                }
            }
        }

        let run_tfidf =
            ctx.force_tfidf || (top_score < HIGH_CONFIDENCE_THRESHOLD && candidates.len() > 1);
        if run_tfidf && candidates.len() > 1 {
            let rerank_n = candidates.len().min(MAX_CORE_NEURONS * 3);
            for candidate in candidates.iter_mut().take(rerank_n) {
                let tfidf = Self::tfidf_cosine_sim_inner(
                    &ctx.terms,
                    &self.retrieval.entries[candidate.entry_idx],
                    &self.retrieval.df_cache,
                    self.retrieval.entries.len(),
                );
                candidate.score = 0.6 * candidate.score + 0.4 * tfidf;
            }
            sort_scored_candidates(&mut candidates[..rerank_n]);
        }

        #[cfg(feature = "embed")]
        {
            use crate::embedder::rrf_score;

            const K_ANN: usize = 20;

            if let Some(embeddings) = ctx.embeddings {
                if !embeddings.is_empty() {
                    let embed_result = (|| -> Option<Vec<f32>> {
                        static EMBEDDER: std::sync::OnceLock<
                            Option<crate::embedder::EmbeddingBackend>,
                        > = std::sync::OnceLock::new();
                        let backend =
                            EMBEDDER.get_or_init(|| crate::embedder::EmbeddingBackend::new().ok());
                        backend.as_ref()?.embed_query(ctx.task).ok()
                    })();

                    if let Some(query_vec) = embed_result {
                        let query_vec = {
                            let top3_embeddings: Vec<_> = candidates
                                .iter()
                                .take(3)
                                .filter_map(|candidate| {
                                    let path =
                                        &self.retrieval.entries[candidate.entry_idx].neuron_path;
                                    embeddings.get(path.as_path())
                                })
                                .collect();
                            if top3_embeddings.is_empty() {
                                query_vec
                            } else {
                                let mut blended = query_vec.clone();
                                let mean_weight = 0.25 / top3_embeddings.len() as f32;
                                for doc_vec in &top3_embeddings {
                                    for (blended_value, doc_value) in
                                        blended.iter_mut().zip(doc_vec.iter())
                                    {
                                        *blended_value += mean_weight * doc_value;
                                    }
                                }
                                let norm: f32 = blended
                                    .iter()
                                    .map(|value| value * value)
                                    .sum::<f32>()
                                    .sqrt();
                                if norm > 1e-8 {
                                    blended.iter_mut().for_each(|value| *value /= norm);
                                }
                                blended
                            }
                        };
                        let allow_paths = candidates
                            .iter()
                            .map(|candidate| {
                                self.retrieval.entries[candidate.entry_idx]
                                    .neuron_path
                                    .clone()
                            })
                            .collect::<HashSet<_>>();
                        let ann_hits = embeddings.search_filtered(&query_vec, K_ANN, &allow_paths);
                        if !ann_hits.is_empty() {
                            let ann_rank = ann_hits
                                .iter()
                                .enumerate()
                                .filter_map(|(rank, (_, path))| {
                                    self.retrieval
                                        .path_index
                                        .get(path)
                                        .copied()
                                        .map(|idx| (idx, rank))
                                })
                                .collect::<HashMap<_, _>>();
                            let missing_ann_rank = K_ANN;
                            let mut fused = candidates
                                .iter()
                                .enumerate()
                                .map(|(rank, candidate)| ScoredCandidate {
                                    entry_idx: candidate.entry_idx,
                                    score: rrf_score(
                                        rank,
                                        ann_rank
                                            .get(&candidate.entry_idx)
                                            .copied()
                                            .unwrap_or(missing_ann_rank),
                                    ),
                                    tokens: candidate.tokens,
                                })
                                .collect::<Vec<_>>();
                            sort_scored_candidates(&mut fused);
                            *candidates = fused;
                        }

                        let full_ann_hits = embeddings.search(&query_vec, K_ANN);
                        let bm25_paths: HashSet<_> = candidates
                            .iter()
                            .map(|candidate| {
                                &self.retrieval.entries[candidate.entry_idx].neuron_path
                            })
                            .collect();
                        let mut injected = false;
                        for (score, path) in &full_ann_hits {
                            if bm25_paths.contains(path) {
                                continue;
                            }
                            if let Some(&entry_idx) = self.retrieval.path_index.get(path) {
                                let entry = &self.retrieval.entries[entry_idx];
                                if !ctx.kind_matches(entry) {
                                    continue;
                                }
                                if let Some(mf) = ctx.module_filter {
                                    if entry.module.as_deref() != Some(mf) {
                                        continue;
                                    }
                                }
                                candidates.push(ScoredCandidate {
                                    entry_idx,
                                    score: *score * 0.6,
                                    tokens: entry.tokens,
                                });
                                injected = true;
                            }
                        }
                        if injected {
                            sort_scored_candidates(candidates);
                        }
                    }
                }
            }
        }

        #[cfg(feature = "rerank")]
        {
            let top_for_rerank = candidates
                .first()
                .map(|candidate| candidate.score)
                .unwrap_or(0.0);
            if top_for_rerank < LOW_CONFIDENCE_THRESHOLD {
                if let Some(reranker) =
                    crate::reranker::inner::global_reranker(&self.persistence.project_root)
                {
                    let max_bm25 = top_for_rerank.max(f32::EPSILON);
                    let rerank_n = candidates.len().min(10);
                    for candidate in candidates.iter_mut().take(rerank_n) {
                        let entry = &self.retrieval.entries[candidate.entry_idx];
                        let passage = std::fs::read_to_string(&entry.neuron_path)
                            .map(|text| text.chars().take(800).collect::<String>())
                            .unwrap_or_else(|_| {
                                entry
                                    .term_freq
                                    .keys()
                                    .cloned()
                                    .collect::<Vec<_>>()
                                    .join(" ")
                            });
                        let ce_score = reranker.score_pair(ctx.task, &passage);
                        let bm25_norm = candidate.score / max_bm25;
                        candidate.score = 0.80 * bm25_norm + 0.20 * ce_score;
                    }
                    sort_scored_candidates(&mut candidates[..rerank_n]);
                }
            }
        }
    }
}

fn sort_scored_candidates(candidates: &mut [ScoredCandidate]) {
    candidates.sort_unstable_by(|a, b| {
        b.score
            .total_cmp(&a.score)
            .then_with(|| a.entry_idx.cmp(&b.entry_idx))
    });
}
