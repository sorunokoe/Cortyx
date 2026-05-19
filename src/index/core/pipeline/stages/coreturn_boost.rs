use super::super::{ActivationStage, QueryContext, ScoredCandidate};

pub struct CoreturnBoostStage;

impl ActivationStage for CoreturnBoostStage {
    fn name(&self) -> &'static str {
        "coreturn_boost"
    }

    fn apply(&self, _ctx: &QueryContext<'_>, _candidates: &mut Vec<ScoredCandidate>) {}
}
