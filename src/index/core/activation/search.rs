use super::*;

impl NeuronIndex {
    /// Return the most relevant neuron paths for `task`, respecting `max_tokens`.
    pub fn get_contexts(
        &self,
        task: &str,
        max_tokens: usize,
        module: Option<&str>,
        kind: Option<&str>,
    ) -> Vec<PathBuf> {
        self.get_contexts_with_temporal_bias(task, max_tokens, module, kind, None)
    }

    pub(crate) fn get_contexts_with_temporal_bias(
        &self,
        task: &str,
        max_tokens: usize,
        module: Option<&str>,
        kind: Option<&str>,
        temporal_bias: Option<f32>,
    ) -> Vec<PathBuf> {
        let Ok(mut ctx) = self.build_query_context(task, max_tokens, module, kind) else {
            return Vec::new();
        };
        Self::set_ctx_temporal_bias(&mut ctx, temporal_bias);

        let mut candidates = self.phase1_candidates(&ctx);
        self.rerank_candidates(&ctx, &mut candidates);
        self.select_paths(&ctx, &candidates, module)
    }
}
