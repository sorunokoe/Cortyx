//! Adaptive graph reasoner with bounded iterative deepening.

use crate::kg::KgEntity;

use super::{GraphReasoner, ReasonerNeuron, ReasonerSeed, ReasoningReport, TraversalOptions};

const MIN_COVERAGE: usize = 3;
const MAX_HOPS_CAP: u8 = 4;
const MAX_PASSES: u8 = 3;

/// Stats describing how many adaptive passes were used.
#[derive(Debug, Clone, Copy)]
pub struct IterationStats {
    pub passes: u8,
    pub final_options: TraversalOptions,
}

/// Wrapper around [`GraphReasoner`] that performs bounded iterative deepening.
pub struct AdaptiveReasoner {
    inner: GraphReasoner,
}

impl AdaptiveReasoner {
    pub fn new<I, J>(neurons: I, kg_entities: J) -> Self
    where
        I: IntoIterator<Item = ReasonerNeuron>,
        J: IntoIterator<Item = KgEntity>,
    {
        Self {
            inner: GraphReasoner::new(neurons, kg_entities),
        }
    }

    #[must_use]
    pub fn trace(
        &self,
        seeds: &[ReasonerSeed],
        options: TraversalOptions,
    ) -> (ReasoningReport, IterationStats) {
        let mut passes = 1;
        let mut current_options = options;
        let mut report = self.inner.trace(seeds, current_options);

        while passes < MAX_PASSES && should_retry(&report, current_options) {
            let mut next_options = current_options;
            if next_options.max_hops < MAX_HOPS_CAP {
                next_options.max_hops += 1;
            }
            if passes >= 2 && !next_options.include_reverse_edges {
                next_options.include_reverse_edges = true;
            }
            if next_options.max_hops == current_options.max_hops
                && next_options.include_reverse_edges == current_options.include_reverse_edges
            {
                break;
            }

            current_options = next_options;
            report = self.inner.trace(seeds, current_options);
            passes += 1;
        }

        (
            report,
            IterationStats {
                passes,
                final_options: current_options,
            },
        )
    }
}

fn should_retry(report: &ReasoningReport, options: TraversalOptions) -> bool {
    if !report.traversal_stats.converged {
        return true;
    }

    report.nodes.len() < MIN_COVERAGE
        && report.traversal_stats.max_depth_reached >= options.max_hops
        && (options.max_hops < MAX_HOPS_CAP || !options.include_reverse_edges)
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use crate::neuron::{NeuronKind, NeuronMeta, Synapse, SynapseType};

    use super::*;

    fn node(path: &str, targets: &[&str]) -> ReasonerNeuron {
        let path_buf = PathBuf::from(path);
        let mut meta = NeuronMeta::new_stub(&path_buf, NeuronKind::Core);
        meta.synapses = targets
            .iter()
            .map(|target| {
                Synapse::new(
                    PathBuf::from(target),
                    SynapseType::Imports,
                    "test edge".to_string(),
                )
            })
            .collect();
        ReasonerNeuron::new(path_buf, meta)
    }

    #[test]
    fn degenerate_single_node_graph_returns_in_one_pass() {
        let seed = ReasonerSeed::new("solo.md", 1.0);
        let reasoner = AdaptiveReasoner::new(vec![node("solo.md", &[])], std::iter::empty());

        let (_, stats) = reasoner.trace(&[seed], TraversalOptions::default());

        assert_eq!(stats.passes, 1);
    }

    #[test]
    fn non_converged_first_pass_is_capped_to_three_passes() {
        let nodes = vec![
            node("a.md", &["b.md", "c.md", "d.md"]),
            node("b.md", &["e.md"]),
            node("c.md", &["f.md"]),
            node("d.md", &["g.md"]),
            node("e.md", &[]),
            node("f.md", &[]),
            node("g.md", &[]),
        ];
        let reasoner = AdaptiveReasoner::new(nodes, std::iter::empty());
        let options = TraversalOptions {
            max_hops: 1,
            max_expansions: 1,
            min_propagated_score: 0.0,
            include_reverse_edges: false,
            include_inactive_facts: false,
        };

        let (_, stats) = reasoner.trace(&[ReasonerSeed::new("a.md", 1.0)], options);

        assert!(stats.passes <= 3);
    }

    #[test]
    fn iteration_stats_passes_stay_within_bounds() {
        let nodes = vec![
            node("a.md", &["b.md"]),
            node("b.md", &["c.md"]),
            node("c.md", &["d.md"]),
            node("d.md", &[]),
        ];
        let reasoner = AdaptiveReasoner::new(nodes, std::iter::empty());
        let options = TraversalOptions {
            max_hops: 1,
            max_expansions: 32,
            min_propagated_score: 0.0,
            include_reverse_edges: false,
            include_inactive_facts: false,
        };

        let (_, stats) = reasoner.trace(&[ReasonerSeed::new("a.md", 1.0)], options);

        assert!((1..=3).contains(&stats.passes));
    }
}
