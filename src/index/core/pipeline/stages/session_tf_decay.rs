use super::super::{ActivationStage, QueryContext, ScoredCandidate};

pub struct SessionTfDecayStage;

impl ActivationStage for SessionTfDecayStage {
    fn name(&self) -> &'static str {
        "session_tf_decay"
    }

    fn apply(&self, _ctx: &QueryContext<'_>, _candidates: &mut Vec<ScoredCandidate>) {}
}
