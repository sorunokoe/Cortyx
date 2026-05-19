use super::super::{ActivationStage, QueryContext, ScoredCandidate};
use crate::index::core::is_session_summary_path;
use crate::neuron::NeuronKind;

pub struct VocabBridgeStage;

impl ActivationStage for VocabBridgeStage {
    fn name(&self) -> &'static str {
        "vocab_bridge"
    }

    fn apply(&self, ctx: &QueryContext<'_>, candidates: &mut Vec<ScoredCandidate>) {
        if !candidates.is_empty() || ctx.bridge_candidate_ids.is_empty() {
            return;
        }

        for &idx in &ctx.bridge_candidate_ids {
            let entry = ctx.entry(idx);
            if !ctx.kind_matches(entry) || !ctx.module_matches(idx) {
                continue;
            }
            let mut score = ctx.score_entry_with_terms(&ctx.bridge_ranking_terms, entry);
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
            if score > 0.0 {
                candidates.push(ScoredCandidate::new(idx, score, entry.tokens));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::index::core::pipeline::types::{test_entry, QueryContextFixture};
    use crate::neuron::NeuronKind;

    #[test]
    fn name_returns_expected_string() {
        assert_eq!(VocabBridgeStage.name(), "vocab_bridge");
    }

    #[test]
    fn empty_candidates_is_passthrough() {
        let fixture = QueryContextFixture::new(vec![]);
        let ctx = fixture.ctx("anything");
        let mut candidates = Vec::new();

        VocabBridgeStage.apply(&ctx, &mut candidates);

        assert!(candidates.is_empty());
    }

    #[test]
    fn empty_vocab_bridge_map_produces_no_extra_candidates() {
        let entry = test_entry("auth_guard.md", NeuronKind::Core, &[("auth_guard", 2.0)]);
        let fixture = QueryContextFixture::new(vec![entry]);
        let ctx = fixture.ctx("authentication");
        let mut candidates = Vec::new();

        assert!(ctx.vocab_bridge.is_empty());
        VocabBridgeStage.apply(&ctx, &mut candidates);

        assert!(candidates.is_empty());
    }

    #[test]
    fn only_runs_when_seed_stage_found_nothing() {
        let entry = test_entry("auth_guard.md", NeuronKind::Core, &[("auth_guard", 2.0)]);
        let fixture = QueryContextFixture::new(vec![entry]);
        let mut ctx = fixture.ctx("authentication");
        ctx.bridge_candidate_ids = [0usize].into_iter().collect();
        ctx.bridge_ranking_terms = vec!["auth_guard".into()];

        let mut candidates = vec![ScoredCandidate::new(99, 1.0, 1)];
        VocabBridgeStage.apply(&ctx, &mut candidates);
        assert_eq!(candidates.len(), 1);
        assert_eq!(candidates[0].entry_idx, 99);
    }

    #[test]
    fn scores_bridge_candidates_when_primary_path_is_empty() {
        let entry = test_entry("auth_guard.md", NeuronKind::Core, &[("auth_guard", 2.0)]);
        let fixture = QueryContextFixture::new(vec![entry]);
        let mut ctx = fixture.ctx("authentication");
        ctx.bridge_candidate_ids = [0usize].into_iter().collect();
        ctx.bridge_ranking_terms = vec!["auth_guard".into()];

        let mut candidates = Vec::new();
        VocabBridgeStage.apply(&ctx, &mut candidates);

        assert_eq!(candidates.len(), 1);
        assert_eq!(candidates[0].entry_idx, 0);
        assert!(candidates[0].score > 0.0);
    }
}
