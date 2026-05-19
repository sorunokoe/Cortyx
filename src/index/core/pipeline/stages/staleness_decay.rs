use super::super::{ActivationStage, QueryContext, ScoredCandidate};

pub struct StalenessDecayStage;

impl ActivationStage for StalenessDecayStage {
    fn name(&self) -> &'static str {
        "staleness_decay"
    }

    fn apply(&self, _ctx: &QueryContext<'_>, _candidates: &mut Vec<ScoredCandidate>) {}
}
