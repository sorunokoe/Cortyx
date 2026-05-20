use super::super::{ActivationStage, QueryContext, ScoredCandidate};
use super::{apply_preceding_decay, sort_candidates, upsert_candidate};
use crate::index::core::config::{MAX_SYNAPSE_CANDIDATES, SYNAPSE_RELEVANCE_THRESHOLD};
use crate::neuron::SynapseType;
use std::collections::HashSet;

/// Traverses one hop of the synapse graph from high-confidence candidates.
pub struct SynapseTraversalStage;

impl ActivationStage for SynapseTraversalStage {
    fn name(&self) -> &'static str {
        "synapse_traversal"
    }

    fn apply(&self, ctx: &QueryContext<'_>, candidates: &mut Vec<ScoredCandidate>) {
        if candidates.is_empty() {
            return;
        }

        sort_candidates(candidates);
        let snapshot = candidates.clone();
        let max_score = snapshot
            .first()
            .map(|candidate| candidate.score)
            .unwrap_or(0.0);
        if max_score <= 0.0 {
            return;
        }

        let threshold = SYNAPSE_RELEVANCE_THRESHOLD * max_score;
        let mut inserted = 0usize;
        let mut visited: HashSet<usize> = candidates
            .iter()
            .map(|candidate| candidate.entry_idx)
            .collect();

        for candidate in snapshot {
            if candidate.score < threshold || inserted >= MAX_SYNAPSE_CANDIDATES {
                continue;
            }
            let path = &ctx.entry(candidate.entry_idx).neuron_path;
            let Some(neighbors) = ctx.adjacency.get(path) else {
                continue;
            };

            for synapse in neighbors {
                if inserted >= MAX_SYNAPSE_CANDIDATES {
                    break;
                }
                let Some(&neighbor_idx) = ctx.path_index.get(&synapse.target) else {
                    continue;
                };
                if visited.contains(&neighbor_idx) {
                    continue;
                }
                if synapse.edge_type == SynapseType::Contradicts {
                    continue;
                }
                let contradicts_existing =
                    ctx.adjacency
                        .get(&synapse.target)
                        .is_some_and(|other_edges| {
                            other_edges.iter().any(|edge| {
                                edge.edge_type == SynapseType::Contradicts
                                    && ctx
                                        .path_index
                                        .get(&edge.target)
                                        .is_some_and(|target_idx| visited.contains(target_idx))
                            })
                        });
                if contradicts_existing {
                    continue;
                }

                let lexical_score =
                    apply_preceding_decay(ctx, neighbor_idx, ctx.score_index(neighbor_idx));
                let traversal_score =
                    candidate.score * synapse.weight.get() * synapse.effective_weight();
                let weighted_lexical =
                    (lexical_score + 0.01) * synapse.weight.get() * synapse.effective_weight();
                let include = synapse.edge_type == SynapseType::ConceptExpands
                    || weighted_lexical >= threshold;
                if !include {
                    continue;
                }

                let neighbor = ctx.entry(neighbor_idx);
                upsert_candidate(
                    candidates,
                    neighbor_idx,
                    traversal_score.max(lexical_score),
                    neighbor.tokens,
                );
                visited.insert(neighbor_idx);
                inserted += 1;
            }
        }

        sort_candidates(candidates);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::index::core::pipeline::types::{test_entry, QueryContextFixture};
    use crate::neuron::{NeuronKind, Synapse, SynapseType};
    use std::path::PathBuf;

    fn synapse(target: &str, edge_type: SynapseType) -> Synapse {
        Synapse::new(PathBuf::from(target), edge_type, "test".into())
    }

    #[test]
    fn name_returns_expected_string() {
        assert_eq!(SynapseTraversalStage.name(), "synapse_traversal");
    }

    #[test]
    fn empty_candidates_is_passthrough() {
        let fixture = QueryContextFixture::new(vec![]);
        let ctx = fixture.ctx("anything");
        let mut candidates = Vec::new();

        SynapseTraversalStage.apply(&ctx, &mut candidates);

        assert!(candidates.is_empty());
    }

    #[test]
    fn injects_neighbor_for_high_scoring_anchor() {
        let anchor = test_entry("anchor.md", NeuronKind::Core, &[("auth", 2.0)]);
        let neighbor = test_entry("neighbor.md", NeuronKind::Core, &[("oauth", 8.0)]);
        let mut fixture = QueryContextFixture::new(vec![anchor, neighbor]);
        fixture.adjacency.insert(
            PathBuf::from("anchor.md"),
            vec![synapse("neighbor.md", SynapseType::SemanticRelated)],
        );
        let mut ctx = fixture.ctx("auth oauth");
        ctx.ranking_terms = vec!["oauth".into()];
        let mut candidates = vec![ScoredCandidate::new(0, 1.0, 32)];

        SynapseTraversalStage.apply(&ctx, &mut candidates);

        assert_eq!(candidates.len(), 2);
        assert!(candidates.iter().any(|candidate| candidate.entry_idx == 1));
    }

    #[test]
    fn concept_expands_edges_always_propagate() {
        let anchor = test_entry("anchor.md", NeuronKind::Concept, &[("auth", 2.0)]);
        let neighbor = test_entry("neighbor.md", NeuronKind::Core, &[("unmatched", 0.0)]);
        let mut fixture = QueryContextFixture::new(vec![anchor, neighbor]);
        fixture.adjacency.insert(
            PathBuf::from("anchor.md"),
            vec![synapse("neighbor.md", SynapseType::ConceptExpands)],
        );
        let ctx = fixture.ctx("auth");
        let mut candidates = vec![ScoredCandidate::new(0, 4.0, 32)];

        SynapseTraversalStage.apply(&ctx, &mut candidates);

        assert!(candidates.iter().any(|candidate| candidate.entry_idx == 1));
    }

    #[test]
    fn skips_neighbors_below_threshold() {
        let anchor = test_entry("anchor.md", NeuronKind::Core, &[("auth", 2.0)]);
        let neighbor = test_entry("neighbor.md", NeuronKind::Core, &[("tiny", 0.01)]);
        let mut fixture = QueryContextFixture::new(vec![anchor, neighbor]);
        fixture.adjacency.insert(
            PathBuf::from("anchor.md"),
            vec![synapse("neighbor.md", SynapseType::SemanticRelated)],
        );
        let mut ctx = fixture.ctx("auth");
        ctx.ranking_terms = vec!["absent".into()];
        let mut candidates = vec![ScoredCandidate::new(0, 4.0, 32)];

        SynapseTraversalStage.apply(&ctx, &mut candidates);

        assert_eq!(candidates.len(), 1);
    }

    #[test]
    fn skips_contradicting_neighbors() {
        let anchor = test_entry("anchor.md", NeuronKind::Core, &[("auth", 2.0)]);
        let selected = test_entry("selected.md", NeuronKind::Core, &[("oauth", 2.0)]);
        let neighbor = test_entry("neighbor.md", NeuronKind::Core, &[("oauth", 2.0)]);
        let mut fixture = QueryContextFixture::new(vec![anchor, selected, neighbor]);
        fixture.adjacency.insert(
            PathBuf::from("anchor.md"),
            vec![synapse("neighbor.md", SynapseType::SemanticRelated)],
        );
        fixture.adjacency.insert(
            PathBuf::from("neighbor.md"),
            vec![synapse("selected.md", SynapseType::Contradicts)],
        );
        let mut ctx = fixture.ctx("oauth");
        ctx.ranking_terms = vec!["oauth".into()];
        let mut candidates = vec![
            ScoredCandidate::new(0, 4.0, 32),
            ScoredCandidate::new(1, 3.0, 32),
        ];

        SynapseTraversalStage.apply(&ctx, &mut candidates);

        assert_eq!(candidates.len(), 2);
    }

    #[test]
    fn respects_max_synapse_candidate_cap() {
        let anchor = test_entry("anchor.md", NeuronKind::Core, &[("auth", 2.0)]);
        let neighbors = (0..6)
            .map(|idx| {
                test_entry(
                    &format!("neighbor_{idx}.md"),
                    NeuronKind::Core,
                    &[("oauth", 2.0)],
                )
            })
            .collect::<Vec<_>>();
        let mut all_entries = vec![anchor];
        all_entries.extend(neighbors);
        let mut fixture = QueryContextFixture::new(all_entries);
        let synapses = (1..=6)
            .map(|idx| {
                synapse(
                    &format!("neighbor_{}.md", idx - 1),
                    SynapseType::SemanticRelated,
                )
            })
            .collect::<Vec<_>>();
        fixture
            .adjacency
            .insert(PathBuf::from("anchor.md"), synapses);
        let mut ctx = fixture.ctx("oauth");
        ctx.ranking_terms = vec!["oauth".into()];
        let mut candidates = vec![ScoredCandidate::new(0, 4.0, 32)];

        SynapseTraversalStage.apply(&ctx, &mut candidates);

        assert!(candidates.len() <= 1 + MAX_SYNAPSE_CANDIDATES);
    }
}
