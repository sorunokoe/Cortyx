use super::stages::{
    Bm25ScoringStage, CoactivationStage, CoreturnBoostStage, CountingAugmentStage,
    MorphemeBridgeStage, PmiExpansionStage, SessionClusterStage, SessionTfDecayStage,
    StalenessDecayStage, SynapseTraversalStage, TemporalProximityStage, UseCaseAugmentStage,
    VocabBridgeStage,
};
use super::types::QueryContext;
use std::sync::OnceLock;

/// A scored candidate entry during activation pipeline processing.
#[derive(Debug, Clone, PartialEq)]
pub struct ScoredCandidate {
    pub entry_idx: usize,
    pub score: f32,
    pub tokens: usize,
}

impl ScoredCandidate {
    pub fn new(entry_idx: usize, score: f32, tokens: usize) -> Self {
        Self {
            entry_idx,
            score,
            tokens,
        }
    }
}

/// A single, independently-testable activation stage.
pub trait ActivationStage: Send + Sync + 'static {
    #[allow(dead_code)]
    fn name(&self) -> &'static str;
    fn apply(&self, ctx: &QueryContext<'_>, candidates: &mut Vec<ScoredCandidate>);
}

/// Runs all stages in sequence.
pub struct ActivationPipeline {
    stages: Vec<Box<dyn ActivationStage>>,
}

impl ActivationPipeline {
    pub fn new(stages: Vec<Box<dyn ActivationStage>>) -> Self {
        Self { stages }
    }

    pub fn phase1() -> &'static Self {
        static PIPELINE: OnceLock<ActivationPipeline> = OnceLock::new();
        PIPELINE.get_or_init(|| {
            Self::new(vec![
                Box::new(Bm25ScoringStage),
                Box::new(VocabBridgeStage),
                Box::new(MorphemeBridgeStage),
                Box::new(PmiExpansionStage),
                Box::new(CountingAugmentStage),
                Box::new(StalenessDecayStage),
                Box::new(SessionTfDecayStage),
                Box::new(UseCaseAugmentStage),
                Box::new(SynapseTraversalStage),
                Box::new(CoactivationStage),
                Box::new(CoreturnBoostStage),
                Box::new(SessionClusterStage),
                Box::new(TemporalProximityStage),
            ])
        })
    }

    pub fn run(&self, ctx: &QueryContext<'_>, candidates: &mut Vec<ScoredCandidate>) {
        for stage in &self.stages {
            stage.apply(ctx, candidates);
        }
    }
}
