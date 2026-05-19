pub mod stage;
pub mod stages;
pub mod types;

pub use stage::{ActivationPipeline, ActivationStage, ScoredCandidate};
pub use stages::{
    Bm25ScoringStage, CoactivationStage, CoreturnBoostStage, CountingAugmentStage,
    MorphemeBridgeStage, PmiExpansionStage, SessionClusterStage, SessionTfDecayStage,
    StalenessDecayStage, SynapseTraversalStage, UseCaseAugmentStage, VocabBridgeStage,
};
pub use types::{
    FeedbackSnapshot, FeedbackStateView, PersistenceStateView, QueryContext, RetrievalStateView,
    WatcherStateView,
};

use super::*;
use crate::types::QueryText;
use std::collections::HashSet;

impl NeuronIndex {
    pub(in crate::index) fn retrieval_state(&self) -> RetrievalStateView<'_> {
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
            embeddings: &self.embeddings,
            idf_n: self.idf_n,
        }
    }

    pub(in crate::index) fn feedback_state(&self) -> FeedbackStateView<'_> {
        FeedbackStateView {
            coactivation_counts: &self.coactivation_counts,
            co_return_counts: &self.co_return_counts,
            session_utilization: &self.session_utilization,
        }
    }

    pub(in crate::index) fn persistence_state(&self) -> PersistenceStateView<'_> {
        PersistenceStateView {
            project_root: &self.project_root,
            pending_append_count: self.pending_append_count,
            has_pending_updates: &self.has_pending_updates,
            delta_base: &self.delta_base,
            delta_dirty: &self.delta_dirty,
            structural_artifacts_dirty: &self.structural_artifacts_dirty,
            dirty_sidecars: &self.dirty_sidecars,
        }
    }

    pub(in crate::index) fn watcher_state(&self) -> WatcherStateView<'_> {
        WatcherStateView {
            dirty_set: &self.dirty_set,
        }
    }

    pub(in crate::index) fn build_query_context<'a>(
        &'a self,
        task: &'a str,
        max_tokens: usize,
        module: Option<&'a str>,
        kind: Option<&'a str>,
    ) -> Option<QueryContext<'a>> {
        let Ok(query) = QueryText::new(task) else {
            return None;
        };
        let terms = tokenize(query.as_str());

        let mut seed_candidate_ids: HashSet<usize> = HashSet::new();
        for term in &terms {
            if let Some(idxs) = self.posting_list.get(term) {
                seed_candidate_ids.extend(idxs.iter().copied());
            }
        }

        let module_set = module.map(|m| {
            self.module_index
                .get(m)
                .map(|v| v.iter().copied().collect::<HashSet<_>>())
                .unwrap_or_default()
        });

        let synonym_expansions = self.synonym_cloud_expansion(&terms);
        let morphological_expansions: Vec<String> = terms
            .iter()
            .flat_map(|term| morphological_variants(term))
            .filter(|variant| self.df_cache.contains_key(variant.as_str()))
            .collect();
        let terms_with_synonyms: Vec<String> =
            if !synonym_expansions.is_empty() || !morphological_expansions.is_empty() {
                let mut t = terms.clone();
                t.extend(synonym_expansions.iter().cloned());
                t.extend(morphological_expansions.iter().cloned());
                t.sort();
                t.dedup();
                t
            } else {
                terms.clone()
            };

        for term in synonym_expansions
            .iter()
            .chain(morphological_expansions.iter())
        {
            if let Some(idxs) = self.posting_list.get(term.as_str()) {
                seed_candidate_ids.extend(idxs.iter().copied());
            }
        }

        let synonym_expansions_empty =
            synonym_expansions.is_empty() && morphological_expansions.is_empty();
        let seed_scoring_terms = if synonym_expansions_empty {
            terms.clone()
        } else {
            terms_with_synonyms.clone()
        };

        let mut bridge_candidate_ids = HashSet::new();
        let mut bridge_scoring_terms = None;
        if seed_candidate_ids.is_empty() && !terms.is_empty() {
            let expanded = self.expand_query_terms(&terms_with_synonyms);
            if expanded.len() > terms_with_synonyms.len() {
                for term in &expanded {
                    if let Some(idxs) = self.posting_list.get(term) {
                        bridge_candidate_ids.extend(idxs.iter().copied());
                    }
                }
                if !bridge_candidate_ids.is_empty() {
                    tracing::debug!(
                        task,
                        original = terms.len(),
                        expanded = expanded.len(),
                        candidates = bridge_candidate_ids.len(),
                        "Vocabulary bridge: expanded query via module synonyms + morphemes + B2"
                    );
                    bridge_scoring_terms = Some(expanded);
                } else {
                    tracing::debug!(
                        task,
                        "Vocabulary gap: no posting-list candidates for query.                          Consider evolving relevant neurons to cover terms: {:?}",
                        &terms[..terms.len().min(5)]
                    );
                }
            } else {
                tracing::debug!(
                    task,
                    "Vocabulary gap: no posting-list candidates for query.                      Consider evolving relevant neurons to cover terms: {:?}",
                    &terms[..terms.len().min(5)]
                );
            }
        }

        let concept_cloud_candidate_ids = if seed_candidate_ids.is_empty()
            && bridge_candidate_ids.is_empty()
            && !terms.is_empty()
        {
            let term_set: HashSet<&str> = terms.iter().map(|s| s.as_str()).collect();
            let cloud_candidates: HashSet<usize> = self
                .entries
                .iter()
                .enumerate()
                .filter(|(_, e)| {
                    e.concept_cloud
                        .iter()
                        .any(|t| term_set.contains(t.as_str()))
                })
                .map(|(i, _)| i)
                .collect();
            if !cloud_candidates.is_empty() {
                tracing::debug!(
                    task,
                    candidates = cloud_candidates.len(),
                    "Concept cloud (R12-S1): found candidates via 1-hop graph vocabulary"
                );
            }
            cloud_candidates
        } else {
            HashSet::new()
        };

        let is_knowledge_update = detect_knowledge_update_query(task);
        let is_counting = detect_counting_query(task);
        let task_lower = task.to_ascii_lowercase();
        let explicit_current_state_query = has_explicit_current_state_marker(task);
        let named_person_move_query = count_proper_nouns(task) >= 1
            && (task_lower.contains(" move")
                || task_lower.contains(" moved")
                || task_lower.contains("relocation"));
        let expand_focus_terms = |base_terms: Vec<String>| {
            let mut expanded = base_terms.clone();
            for term in &base_terms {
                for variant in morphological_variants(term) {
                    if self.df_cache.contains_key(variant.as_str()) {
                        expanded.push(variant);
                    }
                }
            }
            expanded.sort();
            expanded.dedup();
            expanded
        };
        let raw_counting_focus_terms = if is_counting {
            extract_counting_focus_terms(&terms)
        } else {
            Vec::new()
        };
        let counting_focus_terms = if is_counting {
            expand_focus_terms(raw_counting_focus_terms.clone())
        } else {
            Vec::new()
        };
        let raw_knowledge_focus_terms = if !is_counting && is_knowledge_update {
            extract_knowledge_update_focus_terms(&terms)
        } else {
            Vec::new()
        };
        let knowledge_focus_terms = if !is_counting && is_knowledge_update {
            expand_focus_terms(raw_knowledge_focus_terms.clone())
        } else {
            Vec::new()
        };
        let bridge_scoring_terms_for_ranking = bridge_scoring_terms
            .clone()
            .unwrap_or_else(|| seed_scoring_terms.clone());
        let seed_ranking_terms = if !counting_focus_terms.is_empty() {
            counting_focus_terms.clone()
        } else if !knowledge_focus_terms.is_empty() {
            knowledge_focus_terms.clone()
        } else {
            seed_scoring_terms.clone()
        };
        let bridge_ranking_terms = if !counting_focus_terms.is_empty() {
            counting_focus_terms.clone()
        } else if !knowledge_focus_terms.is_empty() {
            knowledge_focus_terms.clone()
        } else {
            bridge_scoring_terms_for_ranking.clone()
        };
        let active_scoring_terms = if !bridge_candidate_ids.is_empty() {
            bridge_scoring_terms_for_ranking.clone()
        } else {
            seed_scoring_terms.clone()
        };
        let ranking_terms = if !bridge_candidate_ids.is_empty() {
            bridge_ranking_terms.clone()
        } else {
            seed_ranking_terms.clone()
        };
        let force_tfidf = is_knowledge_update;

        let kg_router_path: Option<PathBuf> =
            (!matches!(kind, Some(k) if k.eq_ignore_ascii_case("conversation")))
                .then_some(())
                .and_then(|_| detect_personal_fact_query(task))
                .and_then(|predicate| {
                    detect_personal_fact_entity(task).and_then(|entity| {
                        let kg_path = kg::kg_neuron_path(&self.project_root, &entity);
                        if !self.path_index.contains_key(&kg_path) {
                            return None;
                        }
                        let Ok(kg_entity) = kg::KgEntity::load(&kg_path) else {
                            return None;
                        };
                        let has_fact = kg_entity
                            .active_facts(None)
                            .iter()
                            .any(|f| f.predicate == predicate && !f.value.is_empty());
                        if has_fact {
                            tracing::debug!(
                                task,
                                predicate,
                                entity,
                                kind = kind.unwrap_or("all"),
                                "P2-B KG Router: routed personal-attribute query to exact KG neuron"
                            );
                            Some(kg_path)
                        } else {
                            None
                        }
                    })
                });

        let active_candidate_ids = if !bridge_candidate_ids.is_empty() {
            &bridge_candidate_ids
        } else {
            &seed_candidate_ids
        };
        let counting_augment: Vec<usize> = if is_counting {
            self.entries
                .iter()
                .enumerate()
                .filter(|(i, e)| {
                    matches!(e.kind, NeuronKind::Verbatim | NeuronKind::Aggregate)
                        && !active_candidate_ids.contains(i)
                })
                .map(|(i, _)| i)
                .collect()
        } else {
            vec![]
        };

        Some(QueryContext {
            task,
            task_lower,
            terms,
            seed_scoring_terms,
            active_scoring_terms,
            ranking_terms,
            seed_ranking_terms,
            bridge_ranking_terms,
            bridge_scoring_terms,
            module_filter: module,
            kind_filter: kind,
            kind_lower: kind.map(|k| k.to_lowercase()),
            max_tokens,
            session_id: None,
            is_counting,
            is_knowledge_update,
            force_tfidf,
            explicit_current_state_query,
            named_person_move_query,
            raw_counting_focus_terms,
            raw_knowledge_focus_terms,
            idf_n: self.idf_n,
            avg_doc_len: self.avg_doc_len,
            avg_verbatim_doc_len: self.avg_verbatim_doc_len,
            seed_candidate_ids,
            bridge_candidate_ids,
            concept_cloud_candidate_ids,
            module_set,
            counting_augment,
            kg_router_path,
            entries: &self.entries,
            posting_list: &self.posting_list,
            adjacency: &self.adjacency,
            path_index: &self.path_index,
            parent_index: &self.parent_index,
            module_index: &self.module_index,
            df_cache: &self.df_cache,
            vocab_bridge: &self.vocab_bridge,
            morpheme_map: &self.morpheme_map,
            pmi_neighbors: &self.pmi_neighbors,
            session_index: &self.session_index,
            project_root: self.project_root.as_path(),
            feedback: FeedbackSnapshot {
                coactivation_counts: &self.coactivation_counts,
                co_return_counts: &self.co_return_counts,
                session_utilization: &self.session_utilization,
            },
            #[cfg(feature = "embed")]
            embeddings: (!self.embeddings.is_empty()).then_some(&self.embeddings),
        })
    }
}
