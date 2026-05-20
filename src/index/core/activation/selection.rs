use super::*;
use crate::index::core::pipeline::{QueryContext, ScoredCandidate};
use std::collections::{HashMap, HashSet};

impl NeuronIndex {
    pub(super) fn select_paths(
        &self,
        ctx: &QueryContext<'_>,
        candidates: &[ScoredCandidate],
        module: Option<&str>,
    ) -> Vec<PathBuf> {
        let max_score = candidates
            .first()
            .map(|candidate| candidate.score)
            .unwrap_or(0.001)
            .max(0.001);
        let mut selected = Selected::new();

        if let Some(kg_path) = &ctx.kg_router_path {
            selected.insert(kg_path.clone());
        }
        self.inject_summary_if_needed(ctx, &mut selected);
        if let Some(answer_path) = self.synthetic_answer_path(ctx.task) {
            selected.insert(answer_path);
        }
        self.inject_count_aggregate_if_needed(ctx, &mut selected);

        for candidate in candidates {
            selected.insert(
                self.retrieval.entries[candidate.entry_idx]
                    .neuron_path
                    .clone(),
            );
        }

        let candidate_set = if !ctx.bridge_candidate_ids.is_empty() {
            &ctx.bridge_candidate_ids
        } else if !ctx.seed_candidate_ids.is_empty() {
            &ctx.seed_candidate_ids
        } else {
            &ctx.concept_cloud_candidate_ids
        };
        for &idx in candidate_set
            .iter()
            .filter(|&&idx| self.retrieval.entries[idx].kind == NeuronKind::Concept)
        {
            if let Some(module_name) = module {
                if self.retrieval.entries[idx].module.as_deref() != Some(module_name)
                    && self.retrieval.entries[idx].module.is_some()
                {
                    continue;
                }
            }
            let score = self.bm25_score(&ctx.ranking_terms, &self.retrieval.entries[idx]);
            if score > SYNAPSE_RELEVANCE_THRESHOLD * max_score {
                selected.insert(self.retrieval.entries[idx].neuron_path.clone());
            }
        }

        let local_results = self.trim_to_token_budget(selected.ordered, ctx.max_tokens);
        self.record_co_return_counts(&local_results);
        self.apply_global_fallback(local_results, &ctx.terms)
    }

    fn inject_summary_if_needed(&self, ctx: &QueryContext<'_>, selected: &mut Selected) {
        let should_inject_summary = !ctx.is_counting
            && !ctx.is_knowledge_update
            && matches!(ctx.kind_lower.as_deref(), Some("conversation") | None)
            && (ctx.task_lower.starts_with("what ")
                || ctx.task_lower.starts_with("where ")
                || ctx.task_lower.starts_with("who ")
                || ctx.task_lower.starts_with("which "))
            && (ctx.task_lower.contains(" my ")
                || ctx.task_lower.starts_with("what is my")
                || ctx.task_lower.starts_with("where did i")
                || ctx.task_lower.starts_with("who gave me"));
        if !should_inject_summary {
            return;
        }

        if let Some((_, summary_idx)) = self
            .retrieval
            .entries
            .iter()
            .enumerate()
            .filter(|(_, entry)| {
                matches!(entry.kind, NeuronKind::Verbatim)
                    && is_session_summary_path(&entry.neuron_path)
            })
            .filter_map(|(idx, entry)| {
                let bm25 = self.bm25_score(&ctx.ranking_terms, entry);
                if bm25 <= 0.0 {
                    return None;
                }
                let lexical_overlap = ctx
                    .ranking_terms
                    .iter()
                    .filter(|term| entry.term_freq.contains_key(term.as_str()))
                    .count() as f32;
                Some((bm25 * 1.5 + lexical_overlap, idx))
            })
            .max_by(|a, b| a.0.total_cmp(&b.0))
        {
            selected.insert(self.retrieval.entries[summary_idx].neuron_path.clone());
        }
    }

    fn inject_count_aggregate_if_needed(&self, ctx: &QueryContext<'_>, selected: &mut Selected) {
        if !ctx.is_counting {
            return;
        }
        let raw_focus_terms = if !ctx.raw_counting_focus_terms.is_empty() {
            &ctx.raw_counting_focus_terms
        } else if !ctx.raw_knowledge_focus_terms.is_empty() {
            &ctx.raw_knowledge_focus_terms
        } else {
            &ctx.terms
        };
        if !is_money_query(ctx.task) {
            return;
        }
        if let Some(aggregate_path) =
            best_matching_arithmetic_aggregate_path(&self.persistence.project_root, raw_focus_terms)
        {
            selected.insert(aggregate_path);
        }
    }

    fn record_co_return_counts(&self, local_results: &[PathBuf]) {
        let verbatim_ids = local_results
            .iter()
            .filter_map(|path| {
                let &idx = self.retrieval.path_index.get(path)?;
                matches!(self.retrieval.entries[idx].kind, NeuronKind::Verbatim).then_some(idx)
            })
            .collect::<Vec<_>>();
        if verbatim_ids.len() < 2 {
            return;
        }

        if let Ok(mut counts) = self.feedback.co_return_counts.lock() {
            const HEBBIAN_WIRED: u32 = u32::MAX;
            for i in 0..verbatim_ids.len() {
                for j in (i + 1)..verbatim_ids.len() {
                    let (a, b) = if verbatim_ids[i] <= verbatim_ids[j] {
                        (verbatim_ids[i], verbatim_ids[j])
                    } else {
                        (verbatim_ids[j], verbatim_ids[i])
                    };
                    let count = counts.entry((a, b)).or_insert(0);
                    if *count < HEBBIAN_WIRED {
                        *count += 1;
                    }
                }
            }
        }
    }

    fn apply_global_fallback(&self, local_results: Vec<PathBuf>, terms: &[String]) -> Vec<PathBuf> {
        if local_results.len() >= 3 || terms.is_empty() {
            return local_results;
        }

        let global_idx = global_index::GlobalIndex::load();
        let needed = 2usize.saturating_sub(local_results.len().saturating_sub(1));
        let global_paths = global_idx.query(terms, needed);
        if global_paths.is_empty() {
            return local_results;
        }

        let mut combined = local_results.clone();
        let local_set = local_results.iter().collect::<HashSet<_>>();
        for path in global_paths {
            if !local_set.contains(&path) {
                combined.push(path);
            }
        }
        combined
    }
}

struct Selected {
    set: HashSet<PathBuf>,
    ordered: Vec<PathBuf>,
}

impl Selected {
    fn new() -> Self {
        Self {
            set: HashSet::new(),
            ordered: Vec::new(),
        }
    }

    fn insert(&mut self, path: PathBuf) {
        if self.set.insert(path.clone()) {
            self.ordered.push(path);
        }
    }
}
