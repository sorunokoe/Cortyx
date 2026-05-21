//! Domain state for feedback, learning, and session adaptation.

use super::super::pipeline::{FeedbackSnapshot, FeedbackStateView};
use super::super::*;
use super::RetrievalState;
use std::path::Path;

/// Owned feedback state for `NeuronIndex`.
#[derive(Debug, Default)]
pub(crate) struct FeedbackState {
    pub(in crate::index) coactivation_counts: HashMap<PathBuf, HashMap<String, u32>>,
    pub(in crate::index) co_return_counts: std::sync::Mutex<HashMap<(usize, usize), u32>>,
    pub(in crate::index) session_utilization: Vec<[usize; 2]>,
}

impl FeedbackState {
    #[allow(dead_code)]
    pub(in crate::index::core) fn new() -> Self {
        Self::default()
    }

    #[allow(dead_code)]
    pub(in crate::index::core) fn view(&self) -> FeedbackStateView<'_> {
        FeedbackStateView {
            coactivation_counts: &self.coactivation_counts,
            co_return_counts: &self.co_return_counts,
            session_utilization: &self.session_utilization,
        }
    }

    #[allow(dead_code)]
    pub(in crate::index::core) fn snapshot(&self) -> FeedbackSnapshot<'_> {
        FeedbackSnapshot {
            coactivation_counts: &self.coactivation_counts,
            co_return_counts: &self.co_return_counts,
            session_utilization: &self.session_utilization,
        }
    }

    pub(in crate::index::core) fn record_coactivation(
        &mut self,
        neuron_path: &Path,
        query_terms: &[String],
        retrieval: &mut RetrievalState,
    ) {
        const SYNONYM_THRESHOLD: u32 = 30;

        let Some(&entry_idx) = retrieval.path_index.get(neuron_path) else {
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
            let cloud = &mut retrieval.entries[entry_idx].synonym_cloud;
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

        self.apply_pending_hebbian_synapses(retrieval);
    }

    pub(in crate::index::core) fn apply_pending_hebbian_synapses(
        &mut self,
        retrieval: &mut RetrievalState,
    ) {
        const MIN_HEBBIAN_COUNT: u32 = 3;
        const HEBBIAN_WIRED: u32 = u32::MAX;

        let candidates: Vec<(usize, usize, u32)> = {
            let Ok(counts) = self.co_return_counts.lock() else {
                return;
            };
            counts
                .iter()
                .filter(|(_, &count)| count >= MIN_HEBBIAN_COUNT && count != HEBBIAN_WIRED)
                .map(|(&(a, b), &count)| (a, b, count))
                .collect()
        };

        let pairs_to_wire: Vec<(usize, usize)> = candidates
            .into_iter()
            .filter(|&(a_idx, b_idx, count)| {
                let use_a = retrieval
                    .entries
                    .get(a_idx)
                    .map(|entry| entry.use_count.load(std::sync::atomic::Ordering::Relaxed))
                    .unwrap_or(0);
                let use_b = retrieval
                    .entries
                    .get(b_idx)
                    .map(|entry| entry.use_count.load(std::sync::atomic::Ordering::Relaxed))
                    .unwrap_or(0);
                let denominator = use_a.min(use_b).max(count);
                crate::index::core::query::wilson_lower_bound_z(count, denominator, 1.0) >= 0.10
            })
            .map(|(a, b, _)| (a, b))
            .collect();

        for (a_idx, b_idx) in pairs_to_wire {
            let (Some(a_entry), Some(b_entry)) =
                (retrieval.entries.get(a_idx), retrieval.entries.get(b_idx))
            else {
                continue;
            };
            let a = a_entry.neuron_path.clone();
            let b = b_entry.neuron_path.clone();

            if let Ok(mut counts) = self.co_return_counts.lock() {
                if let Some(count) = counts.get_mut(&(a_idx, b_idx)) {
                    *count = HEBBIAN_WIRED;
                }
            }

            let has_ab = retrieval.adjacency.get(&a).is_some_and(|synapses| {
                synapses.iter().any(|synapse| {
                    synapse.target == b && synapse.edge_type == SynapseType::SemanticRelated
                })
            });
            let has_ba = retrieval.adjacency.get(&b).is_some_and(|synapses| {
                synapses.iter().any(|synapse| {
                    synapse.target == a && synapse.edge_type == SynapseType::SemanticRelated
                })
            });
            if has_ab && has_ba {
                continue;
            }

            if !has_ab {
                let syn_ab = Synapse::new(
                    b.clone(),
                    SynapseType::SemanticRelated,
                    "hebbian:co-return".to_string(),
                );
                retrieval
                    .adjacency
                    .entry(a.clone())
                    .or_default()
                    .push(syn_ab);
            }
            if !has_ba {
                let syn_ba = Synapse::new(
                    a.clone(),
                    SynapseType::SemanticRelated,
                    "hebbian:co-return".to_string(),
                );
                retrieval
                    .adjacency
                    .entry(b.clone())
                    .or_default()
                    .push(syn_ba);
            }
            tracing::debug!(
                a = %a.display(),
                b = %b.display(),
                "C-2 Hebbian: SemanticRelated synapse created from co-return signal"
            );
        }
    }

    pub(in crate::index::core) fn record_session_utilization(
        &mut self,
        tokens_used: usize,
        tokens_budget: usize,
    ) {
        const MAX_HISTORY: usize = 5;
        self.session_utilization.push([tokens_used, tokens_budget]);
        if self.session_utilization.len() > MAX_HISTORY {
            self.session_utilization.remove(0);
        }
    }

    pub(in crate::index::core) fn adaptive_budget_scale(&self) -> f32 {
        let history = &self.session_utilization;
        if history.len() < 2 {
            return 1.0;
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
            0.8
        } else if overflowed >= 3 {
            1.2
        } else {
            1.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_is_empty() {
        let state = FeedbackState::default();
        assert!(state.coactivation_counts.is_empty());
        assert!(state.session_utilization.is_empty());
    }

    #[test]
    fn mutex_field_accepts_updates() {
        let state = FeedbackState::new();
        let mut guard = state.co_return_counts.lock().expect("mutex lock");
        guard.insert((1, 2), 3);
        drop(guard);
        let guard = state.co_return_counts.lock().expect("mutex relock");
        assert_eq!(guard.get(&(1, 2)), Some(&3));
    }
}
