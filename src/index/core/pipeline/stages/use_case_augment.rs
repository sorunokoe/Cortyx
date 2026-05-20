use super::super::{ActivationStage, QueryContext, ScoredCandidate};
use super::{sort_candidates, upsert_candidate};
use crate::neuron::NeuronKind;

pub(super) const USE_CASE_SCORE_MULTIPLIER: f32 = 0.9;

/// Injects UseCase sub-neurons beneath already-activated parent cores.
pub struct UseCaseAugmentStage;

impl ActivationStage for UseCaseAugmentStage {
    fn name(&self) -> &'static str {
        "use_case_augment"
    }

    fn apply(&self, ctx: &QueryContext<'_>, candidates: &mut Vec<ScoredCandidate>) {
        let snapshot = candidates.clone();
        for candidate in snapshot {
            let parent_path = ctx.entry(candidate.entry_idx).neuron_path.clone();
            let Some(children) = ctx.parent_index.get(&parent_path) else {
                continue;
            };
            for &child_idx in children {
                let child = ctx.entry(child_idx);
                if !matches!(child.kind, NeuronKind::UseCase) {
                    continue;
                }
                upsert_candidate(
                    candidates,
                    child_idx,
                    candidate.score * USE_CASE_SCORE_MULTIPLIER,
                    child.tokens,
                );
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
    use std::path::PathBuf;

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
    fn injects_use_case_child_at_fraction_of_parent_score() {
        let core = test_entry("core.md", NeuronKind::Core, &[("oauth", 1.0)]);
        let mut child = test_entry("oauth.usecase.md", NeuronKind::UseCase, &[("oauth", 2.0)]);
        child.parent = Some(PathBuf::from("core.md"));
        let mut fixture = QueryContextFixture::new(vec![core, child]);
        fixture
            .parent_index
            .insert(PathBuf::from("core.md"), vec![1]);
        let ctx = fixture.ctx("oauth");
        let mut candidates = vec![ScoredCandidate::new(0, 2.0, 32)];

        UseCaseAugmentStage.apply(&ctx, &mut candidates);

        assert_eq!(candidates.len(), 2);
        let injected = candidates
            .iter()
            .find(|candidate| candidate.entry_idx == 1)
            .unwrap();
        assert!((injected.score - (2.0 * USE_CASE_SCORE_MULTIPLIER)).abs() < f32::EPSILON);
    }

    #[test]
    fn skips_non_use_case_children() {
        let core = test_entry("core.md", NeuronKind::Core, &[("oauth", 1.0)]);
        let child = test_entry("helper.md", NeuronKind::Core, &[("oauth", 2.0)]);
        let mut fixture = QueryContextFixture::new(vec![core, child]);
        fixture
            .parent_index
            .insert(PathBuf::from("core.md"), vec![1]);
        let ctx = fixture.ctx("oauth");
        let mut candidates = vec![ScoredCandidate::new(0, 2.0, 32)];

        UseCaseAugmentStage.apply(&ctx, &mut candidates);

        assert_eq!(candidates, vec![ScoredCandidate::new(0, 2.0, 32)]);
    }

    #[test]
    fn deduplicates_existing_use_case_candidates() {
        let core = test_entry("core.md", NeuronKind::Core, &[("oauth", 1.0)]);
        let child = test_entry("oauth.usecase.md", NeuronKind::UseCase, &[("oauth", 2.0)]);
        let mut fixture = QueryContextFixture::new(vec![core, child]);
        fixture
            .parent_index
            .insert(PathBuf::from("core.md"), vec![1]);
        let ctx = fixture.ctx("oauth");
        let mut candidates = vec![
            ScoredCandidate::new(0, 2.0, 32),
            ScoredCandidate::new(1, 0.5, 32),
        ];

        UseCaseAugmentStage.apply(&ctx, &mut candidates);

        assert_eq!(candidates.len(), 2);
        assert!(
            candidates
                .iter()
                .find(|candidate| candidate.entry_idx == 1)
                .unwrap()
                .score
                >= 1.8
        );
    }

    #[test]
    fn multiple_children_are_sorted_by_parent_score() {
        let core = test_entry("core.md", NeuronKind::Core, &[("oauth", 1.0)]);
        let child_a = test_entry("a.usecase.md", NeuronKind::UseCase, &[("oauth", 1.0)]);
        let child_b = test_entry("b.usecase.md", NeuronKind::UseCase, &[("oauth", 1.0)]);
        let mut fixture = QueryContextFixture::new(vec![core, child_a, child_b]);
        fixture
            .parent_index
            .insert(PathBuf::from("core.md"), vec![1, 2]);
        let ctx = fixture.ctx("oauth");
        let mut candidates = vec![ScoredCandidate::new(0, 3.0, 32)];

        UseCaseAugmentStage.apply(&ctx, &mut candidates);

        assert_eq!(candidates.len(), 3);
        assert_eq!(candidates[0].entry_idx, 0);
        assert!(candidates[1].score >= candidates[2].score);
    }

    #[test]
    fn integrates_with_multiple_parent_candidates() {
        let first = test_entry("first.md", NeuronKind::Core, &[("oauth", 1.0)]);
        let second = test_entry("second.md", NeuronKind::Core, &[("oauth", 1.0)]);
        let child = test_entry("oauth.usecase.md", NeuronKind::UseCase, &[("oauth", 2.0)]);
        let mut fixture = QueryContextFixture::new(vec![first, second, child]);
        fixture
            .parent_index
            .insert(PathBuf::from("first.md"), vec![2]);
        fixture
            .parent_index
            .insert(PathBuf::from("second.md"), vec![2]);
        let ctx = fixture.ctx("oauth");
        let mut candidates = vec![
            ScoredCandidate::new(0, 2.0, 32),
            ScoredCandidate::new(1, 1.0, 32),
        ];

        UseCaseAugmentStage.apply(&ctx, &mut candidates);

        assert_eq!(candidates.len(), 3);
        assert!(candidates.iter().any(|candidate| candidate.entry_idx == 2));
    }
}
