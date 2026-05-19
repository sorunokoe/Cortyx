use super::super::{ActivationStage, QueryContext, ScoredCandidate};
use crate::index::core::is_session_summary_path;
use crate::neuron::NeuronKind;

pub struct Bm25ScoringStage;

impl ActivationStage for Bm25ScoringStage {
    fn name(&self) -> &'static str {
        "bm25_scoring"
    }

    fn apply(&self, ctx: &QueryContext<'_>, candidates: &mut Vec<ScoredCandidate>) {
        if !candidates.is_empty() {
            return;
        }

        for &idx in &ctx.seed_candidate_ids {
            let entry = ctx.entry(idx);
            if !ctx.kind_matches(entry) || !ctx.module_matches(idx) {
                continue;
            }
            let mut score = ctx.score_entry_with_terms(&ctx.seed_ranking_terms, entry);
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

    #[test]
    fn scores_only_matching_kinds_and_modules() {
        let mut core = test_entry("core.md", NeuronKind::Core, &[("auth", 2.0)]);
        core.module = Some("auth".into());
        let mut verbatim = test_entry("chat.md", NeuronKind::Verbatim, &[("auth", 1.0)]);
        verbatim.module = Some("chat".into());
        let fixture = QueryContextFixture::new(vec![core, verbatim]);
        let mut ctx = fixture.ctx("auth");
        ctx.seed_candidate_ids = [0usize, 1].into_iter().collect();
        ctx.seed_ranking_terms = vec!["auth".into()];
        ctx.kind_lower = Some("code".into());
        ctx.module_set = Some([0usize].into_iter().collect());

        let mut candidates = Vec::new();
        Bm25ScoringStage.apply(&ctx, &mut candidates);

        assert_eq!(candidates.len(), 1);
        assert_eq!(candidates[0].entry_idx, 0);
    }

    #[test]
    fn knowledge_update_demotes_verbatim_results() {
        let core = test_entry("core.md", NeuronKind::Core, &[("moved", 1.0)]);
        let verbatim = test_entry("chat.md", NeuronKind::Verbatim, &[("moved", 1.0)]);
        let fixture = QueryContextFixture::new(vec![core, verbatim]);
        let mut ctx = fixture.ctx("who moved");
        ctx.seed_candidate_ids = [0usize, 1].into_iter().collect();
        ctx.seed_ranking_terms = vec!["moved".into()];
        ctx.is_knowledge_update = true;

        let mut candidates = Vec::new();
        Bm25ScoringStage.apply(&ctx, &mut candidates);

        let mut scores = candidates
            .into_iter()
            .map(|candidate| (candidate.entry_idx, candidate.score))
            .collect::<std::collections::HashMap<_, _>>();
        assert!(scores.remove(&0).unwrap() > scores.remove(&1).unwrap());
    }
}
