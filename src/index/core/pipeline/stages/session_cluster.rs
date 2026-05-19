use super::super::{ActivationStage, QueryContext, ScoredCandidate};

pub struct SessionClusterStage;

impl ActivationStage for SessionClusterStage {
    fn name(&self) -> &'static str {
        "session_cluster"
    }

    fn apply(&self, _ctx: &QueryContext<'_>, _candidates: &mut Vec<ScoredCandidate>) {}
}
