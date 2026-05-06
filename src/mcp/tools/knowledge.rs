use super::super::*;
use rmcp::{handler::server::wrapper::Parameters, tool, tool_router};

#[tool_router(router = knowledge_tool_router, vis = "pub(super)")]
impl CortyxServer {
    /// Add a synapse (cross-reference edge) between two neurons.
    #[tool(
        name = "cortyx_create_synapse",
        description = "Create a synapse between two neurons. The activation engine traverses 1-hop synapses to pull in related context for tasks spanning multiple files."
    )]
    pub(in crate::mcp) async fn create_synapse(
        &self,
        Parameters(input): Parameters<CreateSynapseInput>,
    ) -> String {
        // Validate both source and target are safe paths
        let source_rel = match validate_relative_path(&input.source) {
            Ok(p) => p,
            Err(e) => return format!("ERROR: Invalid source: {e}"),
        };
        let target_rel = match validate_relative_path(&input.target) {
            Ok(p) => p,
            Err(e) => return format!("ERROR: Invalid target: {e}"),
        };

        let ndir = neuron_dir(&self.project_root);
        let source_path = ndir.join(&source_rel);
        let target_path = ndir.join(&target_rel);

        for path in [&source_path, &target_path] {
            if !path.exists() {
                return format!(
                    "ERROR: Neuron not found: {}. Create it first with cortyx_evolve_context.",
                    path.display()
                );
            }
        }

        let mut content = match std::fs::read_to_string(&source_path) {
            Ok(c) => c,
            Err(e) => return format!("ERROR: Cannot read source neuron: {e}"),
        };

        if !content.contains("## CROSS-REFERENCES") {
            content.push_str("\n## CROSS-REFERENCES (synapses)\n");
        }
        // Use the relative path so neurons remain portable across machines.
        content.push_str(&format!(
            "\n- `{}` → {}",
            target_rel.display(),
            input.reason
        ));

        if let Err(e) = atomic_write(&source_path, content.as_bytes()) {
            return format!("ERROR: Failed to write synapse: {e}");
        }

        let meta_file = meta_path(&source_path);
        let mut meta = load_or_new_meta(&meta_file, &source_path, NeuronKind::Core);
        if let Some(source_hash) = hash_file(&meta.source_path) {
            meta.source_hash = source_hash;
        }
        meta.tokens = estimate_context_tokens(&content).get();
        meta.last_updated = now_iso8601();
        meta.status = NeuronStatus::Fresh;
        let edge_type = input.edge_type.unwrap_or(SynapseType::SemanticRelated);
        if !meta.synapses.iter().any(|s| s.target == target_path) {
            meta.synapses.push(Synapse::new(
                target_path.clone(),
                edge_type,
                input.reason.clone(),
            ));
        }
        if let Err(e) = save_meta(&meta_file, &meta) {
            tracing::warn!("Failed to save synapse meta: {e}");
        }
        let provenance_result = record_mutation_provenance(
            &source_path,
            &meta,
            &content,
            ProvenanceOperation::SectionUpdate,
            ProvenanceSource::Local,
            Some("cross-references".to_string()),
            Some(format!("added synapse to {}", target_rel.display())),
        );

        let mut idx = self.index.write().await;
        if let Err(e) = idx.upsert_neuron(&source_path, &content, &meta) {
            return format!("ERROR: Failed to update index: {e}");
        }
        finalize_mutation_message(
            format!(
                "Synapse created: {} → {} ({})",
                input.source, input.target, input.reason
            ),
            provenance_result,
        )
    }
    // ─── S4: Temporal Knowledge Graph (NE3) ──────────────────────────────────

    /// Add a fact to a KG entity neuron (creating the entity if needed).
    #[tool(
        name = "cortyx_kg_add",
        description = "Add a fact triple to a KG entity (creates entity if absent). \
                       KG neurons are git-tracked, BM25-indexed Markdown files. \
                       Example: entity='project_meta', predicate='language', value='Rust', valid_from='2024-01-01'."
    )]
    pub(in crate::mcp) async fn kg_add(&self, Parameters(input): Parameters<KgAddInput>) -> String {
        // A5: ECS verification gate — block factually risky claims from entering the KG.
        let fact_text = format!("{}: {} = {}", input.entity, input.predicate, input.value);
        let verdict = verify_gate::check(&fact_text);
        if verdict.risk_score > 0.70 {
            let summary = verdict
                .summary
                .as_deref()
                .unwrap_or("high hallucination risk");
            return format!(
                "REJECTED by ECS gate (risk={:.2}, ECS={}/100): {}. \
                 Review the fact before adding it to the KG.",
                verdict.risk_score,
                verdict.ecs_score(),
                summary
            );
        }

        let path = kg::kg_neuron_path(&self.project_root, &input.entity);
        let mut entity = match kg::KgEntity::load(&path) {
            Ok(e) => e,
            Err(e) => return format!("ERROR loading KG entity: {e}"),
        };
        entity.add_fact(&input.predicate, &input.value, input.valid_from.as_deref());
        if let Err(e) = entity.save() {
            return format!("ERROR saving KG entity: {e}");
        }
        let mut idx = self.index.write().await;
        if let Err(err) = index_kg_entity_path(&mut idx, &path) {
            return format!(
                "ERROR reloading KG entity {} after save: {err}",
                path.display()
            );
        }
        format!(
            "KG fact added: {entity} / {pred} = {val} (from: {from})",
            entity = input.entity,
            pred = input.predicate,
            val = input.value,
            from = input.valid_from.as_deref().unwrap_or(""),
        )
    }

    /// Query active facts for a KG entity as of an optional date.
    #[tool(
        name = "cortyx_kg_query",
        description = "Query active facts for a KG entity. Pass as_of (ISO-8601) to filter by date. \
                       Returns a Markdown table of active fact triples."
    )]
    pub(in crate::mcp) async fn kg_query(
        &self,
        Parameters(input): Parameters<KgQueryInput>,
    ) -> String {
        let path = kg::kg_neuron_path(&self.project_root, &input.entity);
        let entity = match kg::KgEntity::load(&path) {
            Ok(e) => e,
            Err(e) => return format!("ERROR loading KG entity: {e}"),
        };
        let facts = entity.active_facts(input.as_of.as_deref());
        if facts.is_empty() {
            return format!(
                "No active facts for entity '{}' (as_of: {:?})",
                input.entity, input.as_of
            );
        }
        let mut out = format!("## KG: {} (active facts)\n\n| predicate | value | valid_from | ended |\n|---|---|---|---|\n", input.entity);
        for f in facts {
            out.push_str(&format!(
                "| {} | {} | {} | {} |\n",
                f.predicate, f.value, f.valid_from, f.ended
            ));
        }
        out
    }

    /// Invalidate (end) an active KG fact by setting its `ended` date.
    #[tool(
        name = "cortyx_kg_invalidate",
        description = "Invalidate (end) the currently active fact for a predicate on a KG entity. \
                       Sets the `ended` date; does NOT delete the historical record."
    )]
    pub(in crate::mcp) async fn kg_invalidate(
        &self,
        Parameters(input): Parameters<KgInvalidateInput>,
    ) -> String {
        let path = kg::kg_neuron_path(&self.project_root, &input.entity);
        let mut entity = match kg::KgEntity::load(&path) {
            Ok(e) => e,
            Err(e) => return format!("ERROR loading KG entity: {e}"),
        };
        if let Err(e) = entity.invalidate_fact(&input.predicate, &input.ended) {
            return format!("ERROR: {e}");
        }
        if let Err(e) = entity.save() {
            return format!("ERROR saving KG entity: {e}");
        }
        let mut idx = self.index.write().await;
        if let Err(err) = index_kg_entity_path(&mut idx, &path) {
            return format!(
                "ERROR reloading KG entity {} after save: {err}",
                path.display()
            );
        }
        format!(
            "KG fact invalidated: {}/{} ended on {}",
            input.entity, input.predicate, input.ended
        )
    }

    /// Show the full temporal timeline for a predicate on a KG entity.
    #[tool(
        name = "cortyx_kg_timeline",
        description = "Show the full temporal history of a predicate on a KG entity — all past, \
                       present, and future values with their validity windows."
    )]
    pub(in crate::mcp) async fn kg_timeline(
        &self,
        Parameters(input): Parameters<KgTimelineInput>,
    ) -> String {
        let path = kg::kg_neuron_path(&self.project_root, &input.entity);
        let entity = match kg::KgEntity::load(&path) {
            Ok(e) => e,
            Err(e) => return format!("ERROR loading KG entity: {e}"),
        };
        let timeline = entity.timeline_for(&input.predicate);
        if timeline.is_empty() {
            return format!("No facts found for {}/{}", input.entity, input.predicate);
        }
        let mut out = format!(
            "## Timeline: {}/{}\n\n| # | value | valid_from | ended |\n|---|---|---|---|\n",
            input.entity, input.predicate
        );
        for (i, f) in timeline.iter().enumerate() {
            let ended = if f.ended.is_empty() {
                "active"
            } else {
                &f.ended
            };
            out.push_str(&format!(
                "| {} | {} | {} | {} |\n",
                i + 1,
                f.value,
                f.valid_from,
                ended
            ));
        }
        out
    }

    /// Return aggregate statistics for all KG entities in this project.
    #[tool(
        name = "cortyx_kg_stats",
        description = "Return aggregate statistics for all KG entities: entity count, total facts, \
                       active facts, ended/invalidated facts."
    )]
    pub(in crate::mcp) async fn kg_stats(&self, _params: Parameters<serde_json::Value>) -> String {
        let stats = kg::compute_stats(&self.project_root);
        format!(
            "KG stats: {} entities, {} total facts ({} active, {} ended)",
            stats.entity_count, stats.total_facts, stats.active_facts, stats.ended_facts
        )
    }
}
