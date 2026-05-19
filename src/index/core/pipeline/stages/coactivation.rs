use super::super::{ActivationStage, QueryContext, ScoredCandidate};

pub struct CoactivationStage;

impl ActivationStage for CoactivationStage {
    fn name(&self) -> &'static str {
        "coactivation"
    }

    fn apply(&self, _ctx: &QueryContext<'_>, _candidates: &mut Vec<ScoredCandidate>) {}
}
