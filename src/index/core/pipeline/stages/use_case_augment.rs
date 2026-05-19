use super::super::{ActivationStage, QueryContext, ScoredCandidate};

pub struct UseCaseAugmentStage;

impl ActivationStage for UseCaseAugmentStage {
    fn name(&self) -> &'static str {
        "use_case_augment"
    }

    fn apply(&self, _ctx: &QueryContext<'_>, _candidates: &mut Vec<ScoredCandidate>) {}
}
