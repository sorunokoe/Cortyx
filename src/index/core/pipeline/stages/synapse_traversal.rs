use super::super::{ActivationStage, QueryContext, ScoredCandidate};

pub struct SynapseTraversalStage;

impl ActivationStage for SynapseTraversalStage {
    fn name(&self) -> &'static str {
        "synapse_traversal"
    }

    fn apply(&self, _ctx: &QueryContext<'_>, _candidates: &mut Vec<ScoredCandidate>) {}
}
