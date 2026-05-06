use super::super::*;
use rmcp::{handler::server::wrapper::Parameters, tool, tool_router};

#[tool_router(router = admin_tool_router, vis = "pub(super)")]
impl CortyxServer {
    /// Force a neuron to be marked stale.
    #[tool(
        name = "cortyx_invalidate",
        description = "Mark a neuron stale, forcing re-evaluation on the next cortyx_get_contexts call."
    )]
    pub(in crate::mcp) async fn invalidate(
        &self,
        Parameters(input): Parameters<InvalidateInput>,
    ) -> String {
        let rel = match validate_relative_path(&input.path) {
            Ok(p) => p,
            Err(e) => return format!("ERROR: Invalid path: {e}"),
        };
        let source = self.project_root.join(&rel);
        let mut idx = self.index.write().await;
        match idx.invalidate(&source) {
            Ok(()) => format!("Marked stale: {}", input.path),
            Err(e) => format!("ERROR: {e}"),
        }
    }
    /// Show neuron stats and cache-hit prediction.
    #[tool(
        name = "cortyx_status",
        description = "Show neuron count, synapse count, freshness, and cache-hit prediction."
    )]
    pub(in crate::mcp) async fn status(&self) -> String {
        let idx = self.index.read().await;
        let low_quality = idx.low_quality_count();
        let quality_note = if low_quality > 0 {
            format!("\nNeeds curation (quality<40%): {low_quality}")
        } else {
            String::new()
        };
        format!(
            "Cortyx Status\n\
             =============\n\
             Neurons (total):       {}\n\
             Synapses:              {}{}\n\
             \n\
             Prompt caching:        ✓ Static prefix byte-identical on every call\n\
             Activation latency:    ~BM25 in-memory (<10ms for <10k neurons)\n\
             Instructions: Call cortyx_get_contexts(task) at the start of each task.",
            idx.neuron_count(),
            idx.synapse_count(),
            quality_note
        )
    }
    /// Check for contradicting neuron pairs (S7 — NE6).
    ///
    /// Proactively scans all neurons (or a single neuron) for `Contradicts` synapse edges.
    /// Use before starting a task to surface known conflicts. Contradictions are also
    /// automatically surfaced by `cortyx_get_contexts` at query time.
    #[tool(
        name = "cortyx_check_consistency",
        description = "Check for contradictions in the neuron graph. Scans all Contradicts synapse edges and returns conflicting pairs with reasons. Scope to a single neuron with the optional path argument. Contradictions are also surfaced automatically during cortyx_get_contexts."
    )]
    pub(in crate::mcp) async fn check_consistency(
        &self,
        Parameters(input): Parameters<CheckConsistencyInput>,
    ) -> String {
        let path_filter: Option<PathBuf> = if let Some(ref p) = input.path {
            match validate_relative_path(p) {
                Ok(rel) => {
                    let src = self.project_root.join(&rel);
                    Some(core_neuron_path(&src, &self.project_root))
                },
                Err(e) => return format!("ERROR: Invalid path: {e}"),
            }
        } else {
            None
        };

        let idx = self.index.read().await;
        let pairs = idx.all_contradictions(path_filter.as_deref());

        // A4: semantic contradiction detection via PureReason (feature=verify).
        // Reads up to 30 neuron bodies (or just the filtered one), extracts logical
        // claims, and finds contradictions that have no explicit Contradicts synapse.
        let semantic_pairs: Vec<(String, String)> = {
            let bodies: Vec<String> = idx
                .neuron_bodies_for_consistency(path_filter.as_deref(), 30)
                .unwrap_or_default();
            let body_refs: Vec<&str> = bodies.iter().map(String::as_str).collect();
            verify_gate::find_semantic_contradictions(&body_refs)
        };

        let total = pairs.len() + semantic_pairs.len();
        if total == 0 {
            return "No contradictions detected.".to_string();
        }

        let mut out = format!("## Contradictions Found ({})\n\n", total);
        for (a, b, reason) in &pairs {
            let a_name = a.file_name().unwrap_or_default().to_string_lossy();
            let b_name = b.file_name().unwrap_or_default().to_string_lossy();
            out.push_str(&format!(
                "- **{}** ↔ **{}** *(synapse)*\n  Reason: {}\n  Action: use `cortyx_create_synapse` to update or \
                 `cortyx_invalidate` to retire the outdated neuron.\n\n",
                a_name, b_name, reason
            ));
        }
        for (claim_a, claim_b) in &semantic_pairs {
            out.push_str(&format!(
                "- *(semantic)* `{claim_a}` contradicts `{claim_b}`\n  Action: review neurons containing these claims.\n\n"
            ));
        }
        out
    }
}
