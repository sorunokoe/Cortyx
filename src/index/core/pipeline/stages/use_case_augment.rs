use super::super::{ActivationStage, QueryContext, ScoredCandidate};

pub struct UseCaseAugmentStage;

impl ActivationStage for UseCaseAugmentStage {
    fn name(&self) -> &'static str {
        "use_case_augment"
    }

    fn apply(&self, _ctx: &QueryContext<'_>, _candidates: &mut Vec<ScoredCandidate>) {}
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::index::core::pipeline::types::{test_entry, QueryContextFixture};
    use crate::neuron::NeuronKind;

    #[test]
    fn name_returns_expected_string() {
        assert_eq!(UseCaseAugmentStage.name(), "use_case_augment");
    }

    #[test]
    fn empty_candidates_is_passthrough() {
        let fixture = QueryContextFixture::new(vec![]);
        let ctx = fixture.ctx("anything");
        let mut candidates = Vec::new();

        UseCaseAugmentStage.apply(&ctx, &mut candidates);

        assert!(candidates.is_empty());
    }

    #[test]
    fn nonempty_candidates_unchanged_by_stub() {
        let entry = test_entry("a.md", NeuronKind::Core, &[("foo", 1.0)]);
        let fixture = QueryContextFixture::new(vec![entry]);
        let ctx = fixture.ctx("foo");
        let mut candidates = vec![ScoredCandidate::new(0, 1.0, 10)];

        UseCaseAugmentStage.apply(&ctx, &mut candidates);

        assert_eq!(candidates.len(), 1);
        assert_eq!(candidates[0].entry_idx, 0);
        assert_eq!(candidates[0].score, 1.0);
        assert_eq!(candidates[0].tokens, 10);
    }
}
