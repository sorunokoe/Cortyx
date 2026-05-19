use super::super::{ActivationStage, QueryContext, ScoredCandidate};

pub struct MorphemeBridgeStage;

impl ActivationStage for MorphemeBridgeStage {
    fn name(&self) -> &'static str {
        "morpheme_bridge"
    }

    fn apply(&self, _ctx: &QueryContext<'_>, _candidates: &mut Vec<ScoredCandidate>) {}
}
