use super::super::{ActivationStage, QueryContext, ScoredCandidate};

pub struct PmiExpansionStage;

impl ActivationStage for PmiExpansionStage {
    fn name(&self) -> &'static str {
        "pmi_expansion"
    }

    fn apply(&self, _ctx: &QueryContext<'_>, _candidates: &mut Vec<ScoredCandidate>) {}
}
