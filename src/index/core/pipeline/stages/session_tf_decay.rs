use super::super::{ActivationStage, QueryContext, ScoredCandidate};
use super::sort_candidates;
use crate::index::core::config::SESSION_SAME_SCORE_DECAY;

/// Dampens candidates from the currently active session to reduce echo retrieval.
pub struct SessionTfDecayStage;

impl ActivationStage for SessionTfDecayStage {
    fn name(&self) -> &'static str {
        "session_tf_decay"
    }

    fn apply(&self, ctx: &QueryContext<'_>, candidates: &mut Vec<ScoredCandidate>) {
        let Some(session_id) = ctx.session_id else {
            return;
        };

        for candidate in candidates.iter_mut() {
            if ctx.entry(candidate.entry_idx).session_id == session_id {
                candidate.score *= SESSION_SAME_SCORE_DECAY;
            }
        }
        sort_candidates(candidates);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::index::core::config::SESSION_SAME_SCORE_DECAY;
    use crate::index::core::pipeline::types::{test_entry, QueryContextFixture};
    use crate::neuron::NeuronKind;

    #[test]
    fn name_returns_expected_string() {
        assert_eq!(SessionTfDecayStage.name(), "session_tf_decay");
    }

    #[test]
    fn empty_candidates_is_passthrough() {
        let fixture = QueryContextFixture::new(vec![]);
        let ctx = fixture.ctx("anything");
        let mut candidates = Vec::new();

        SessionTfDecayStage.apply(&ctx, &mut candidates);

        assert!(candidates.is_empty());
    }

    #[test]
    fn matching_session_candidates_are_decayed() {
        let mut entry = test_entry("a.md", NeuronKind::Verbatim, &[("foo", 1.0)]);
        entry.session_id = "s1".into();
        let fixture = QueryContextFixture::new(vec![entry]);
        let mut ctx = fixture.ctx("foo");
        ctx.session_id = Some("s1");
        let mut candidates = vec![ScoredCandidate::new(0, 2.0, 10)];

        SessionTfDecayStage.apply(&ctx, &mut candidates);

        assert!((candidates[0].score - (2.0 * SESSION_SAME_SCORE_DECAY)).abs() < f32::EPSILON);
    }

    #[test]
    fn nonmatching_session_candidates_are_unchanged() {
        let mut entry = test_entry("a.md", NeuronKind::Verbatim, &[("foo", 1.0)]);
        entry.session_id = "s1".into();
        let fixture = QueryContextFixture::new(vec![entry]);
        let mut ctx = fixture.ctx("foo");
        ctx.session_id = Some("s2");
        let mut candidates = vec![ScoredCandidate::new(0, 2.0, 10)];

        SessionTfDecayStage.apply(&ctx, &mut candidates);

        assert_eq!(candidates[0].score, 2.0);
    }

    #[test]
    fn matching_decay_can_change_ordering() {
        let mut first = test_entry("first.md", NeuronKind::Verbatim, &[("foo", 1.0)]);
        first.session_id = "s1".into();
        let mut second = test_entry("second.md", NeuronKind::Verbatim, &[("foo", 1.0)]);
        second.session_id = "s2".into();
        let fixture = QueryContextFixture::new(vec![first, second]);
        let mut ctx = fixture.ctx("foo");
        ctx.session_id = Some("s1");
        let mut candidates = vec![
            ScoredCandidate::new(0, 1.0, 10),
            ScoredCandidate::new(1, 0.9, 10),
        ];

        SessionTfDecayStage.apply(&ctx, &mut candidates);

        assert_eq!(
            candidates
                .iter()
                .map(|candidate| candidate.entry_idx)
                .collect::<Vec<_>>(),
            vec![1, 0]
        );
    }

    #[test]
    fn missing_context_session_leaves_scores_untouched() {
        let mut entry = test_entry("a.md", NeuronKind::Verbatim, &[("foo", 1.0)]);
        entry.session_id = "s1".into();
        let fixture = QueryContextFixture::new(vec![entry]);
        let ctx = fixture.ctx("foo");
        let mut candidates = vec![ScoredCandidate::new(0, 2.0, 10)];

        SessionTfDecayStage.apply(&ctx, &mut candidates);

        assert_eq!(candidates[0].score, 2.0);
    }

    #[test]
    fn integrates_with_preexisting_candidates() {
        let mut first = test_entry("first.md", NeuronKind::Verbatim, &[("foo", 1.0)]);
        first.session_id = "s1".into();
        let mut second = test_entry("second.md", NeuronKind::Verbatim, &[("foo", 1.0)]);
        second.session_id = "s1".into();
        let fixture = QueryContextFixture::new(vec![first, second]);
        let mut ctx = fixture.ctx("foo");
        ctx.session_id = Some("s1");
        let mut candidates = vec![
            ScoredCandidate::new(0, 2.0, 10),
            ScoredCandidate::new(1, 1.0, 10),
        ];

        SessionTfDecayStage.apply(&ctx, &mut candidates);

        assert!(candidates[0].score > candidates[1].score);
    }
}
