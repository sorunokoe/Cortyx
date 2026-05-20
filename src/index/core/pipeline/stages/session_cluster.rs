use super::super::{ActivationStage, QueryContext, ScoredCandidate};
use super::{apply_preceding_decay, sort_candidates, upsert_candidate};
use crate::neuron::NeuronKind;
use std::collections::HashSet;

/// Pulls in nearby session siblings when conversation chunks dominate the top results.
pub struct SessionClusterStage;

impl ActivationStage for SessionClusterStage {
    fn name(&self) -> &'static str {
        "session_cluster"
    }

    fn apply(&self, ctx: &QueryContext<'_>, candidates: &mut Vec<ScoredCandidate>) {
        if candidates.is_empty() {
            return;
        }

        sort_candidates(candidates);
        let snapshot = candidates.clone();
        let mut seen_sessions = HashSet::new();
        let mut existing = candidates
            .iter()
            .map(|candidate| candidate.entry_idx)
            .collect::<HashSet<_>>();

        for anchor in snapshot.into_iter().take(3) {
            let anchor_entry = ctx.entry(anchor.entry_idx);
            if !matches!(anchor_entry.kind, NeuronKind::Verbatim)
                || anchor_entry.session_id.is_empty()
                || !seen_sessions.insert(anchor_entry.session_id.clone())
            {
                continue;
            }
            let Some(sibling_indices) = ctx.session_index.get(&anchor_entry.session_id) else {
                continue;
            };
            let anchor_pos = sibling_indices
                .iter()
                .position(|&idx| idx == anchor.entry_idx)
                .unwrap_or(0);
            let mut ranked = sibling_indices
                .iter()
                .enumerate()
                .filter_map(|(position, &idx)| {
                    if existing.contains(&idx) {
                        return None;
                    }
                    let distance = anchor_pos.abs_diff(position);
                    let backward_penalty = usize::from(position < anchor_pos);
                    let lexical = apply_preceding_decay(ctx, idx, ctx.score_index(idx));
                    Some((distance, backward_penalty, lexical, idx))
                })
                .collect::<Vec<_>>();
            ranked.sort_unstable_by(|a, b| {
                a.0.cmp(&b.0)
                    .then_with(|| a.1.cmp(&b.1))
                    .then_with(|| b.2.total_cmp(&a.2))
                    .then_with(|| a.3.cmp(&b.3))
            });

            for (distance, _, lexical, idx) in ranked.into_iter().take(2) {
                let distance_u32 = u32::try_from(distance.max(1)).unwrap_or(u32::MAX);
                let proximity_score = anchor.score * (0.85 / distance_u32 as f32);
                let entry = ctx.entry(idx);
                upsert_candidate(candidates, idx, lexical.max(proximity_score), entry.tokens);
                existing.insert(idx);
            }
        }

        sort_candidates(candidates);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::index::core::pipeline::types::{test_entry, QueryContextFixture};
    use crate::neuron::NeuronKind;

    #[test]
    fn name_returns_expected_string() {
        assert_eq!(SessionClusterStage.name(), "session_cluster");
    }

    #[test]
    fn empty_candidates_is_passthrough() {
        let fixture = QueryContextFixture::new(vec![]);
        let ctx = fixture.ctx("anything");
        let mut candidates = Vec::new();

        SessionClusterStage.apply(&ctx, &mut candidates);

        assert!(candidates.is_empty());
    }

    #[test]
    fn injects_top_two_siblings_for_verbatim_anchor() {
        let mut anchor = test_entry("s1_0.md", NeuronKind::Verbatim, &[("topic", 2.0)]);
        anchor.session_id = "s1".into();
        let mut sibling_a = test_entry("s1_1.md", NeuronKind::Verbatim, &[("topic", 1.0)]);
        sibling_a.session_id = "s1".into();
        let mut sibling_b = test_entry("s1_2.md", NeuronKind::Verbatim, &[("topic", 1.0)]);
        sibling_b.session_id = "s1".into();
        let mut fixture = QueryContextFixture::new(vec![anchor, sibling_a, sibling_b]);
        fixture.session_index.insert("s1".into(), vec![0, 1, 2]);
        let mut ctx = fixture.ctx("topic");
        ctx.ranking_terms = vec!["topic".into()];
        let mut candidates = vec![ScoredCandidate::new(0, 3.0, 32)];

        SessionClusterStage.apply(&ctx, &mut candidates);

        assert_eq!(candidates.len(), 3);
        assert!(candidates.iter().any(|candidate| candidate.entry_idx == 1));
        assert!(candidates.iter().any(|candidate| candidate.entry_idx == 2));
    }

    #[test]
    fn ignores_non_verbatim_anchors() {
        let anchor = test_entry("core.md", NeuronKind::Core, &[("topic", 2.0)]);
        let mut sibling = test_entry("s1_1.md", NeuronKind::Verbatim, &[("topic", 1.0)]);
        sibling.session_id = "s1".into();
        let mut fixture = QueryContextFixture::new(vec![anchor, sibling]);
        fixture.session_index.insert("s1".into(), vec![1]);
        let mut ctx = fixture.ctx("topic");
        ctx.ranking_terms = vec!["topic".into()];
        let mut candidates = vec![ScoredCandidate::new(0, 3.0, 32)];

        SessionClusterStage.apply(&ctx, &mut candidates);

        assert_eq!(candidates.len(), 1);
    }

    #[test]
    fn only_top_three_anchors_are_considered() {
        let entries = (0..5)
            .map(|idx| {
                let mut entry = test_entry(
                    &format!("s{idx}_0.md"),
                    NeuronKind::Verbatim,
                    &[("topic", 1.0)],
                );
                entry.session_id = format!("s{idx}");
                entry
            })
            .collect::<Vec<_>>();
        let mut fixture = QueryContextFixture::new(entries);
        fixture.session_index.insert("s4".into(), vec![4]);
        let mut ctx = fixture.ctx("topic");
        ctx.ranking_terms = vec!["topic".into()];
        let mut candidates = (0..5)
            .map(|idx| ScoredCandidate::new(idx, 5.0 - idx as f32, 32))
            .collect::<Vec<_>>();

        SessionClusterStage.apply(&ctx, &mut candidates);

        assert_eq!(candidates.len(), 5);
    }

    #[test]
    fn preserves_existing_candidates_when_sibling_already_present() {
        let mut anchor = test_entry("s1_0.md", NeuronKind::Verbatim, &[("topic", 2.0)]);
        anchor.session_id = "s1".into();
        let mut sibling = test_entry("s1_1.md", NeuronKind::Verbatim, &[("topic", 1.0)]);
        sibling.session_id = "s1".into();
        let mut fixture = QueryContextFixture::new(vec![anchor, sibling]);
        fixture.session_index.insert("s1".into(), vec![0, 1]);
        let mut ctx = fixture.ctx("topic");
        ctx.ranking_terms = vec!["topic".into()];
        let mut candidates = vec![
            ScoredCandidate::new(0, 3.0, 32),
            ScoredCandidate::new(1, 2.0, 32),
        ];

        SessionClusterStage.apply(&ctx, &mut candidates);

        assert_eq!(candidates.len(), 2);
    }

    #[test]
    fn sibling_proximity_can_change_ordering() {
        let mut anchor = test_entry("s1_0.md", NeuronKind::Verbatim, &[("topic", 2.0)]);
        anchor.session_id = "s1".into();
        let mut sibling = test_entry("s1_1.md", NeuronKind::Verbatim, &[("topic", 1.0)]);
        sibling.session_id = "s1".into();
        let outsider = test_entry("other.md", NeuronKind::Verbatim, &[("topic", 1.0)]);
        let mut fixture = QueryContextFixture::new(vec![anchor, sibling, outsider]);
        fixture.session_index.insert("s1".into(), vec![0, 1]);
        let mut ctx = fixture.ctx("topic");
        ctx.ranking_terms = vec!["topic".into()];
        let mut candidates = vec![
            ScoredCandidate::new(0, 3.0, 32),
            ScoredCandidate::new(2, 1.0, 32),
        ];

        SessionClusterStage.apply(&ctx, &mut candidates);

        assert_eq!(candidates[1].entry_idx, 1);
    }
}
