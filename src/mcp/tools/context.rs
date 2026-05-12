use super::super::*;
use crate::types::QueryText;
use rmcp::{handler::server::wrapper::Parameters, tool, tool_router};

/// Read the `purpose` SECTION from a neuron file and return the first 2 sentences.
/// Returns an empty string if the file can't be read or the section is absent.
fn purpose_snippet(path: &std::path::Path) -> String {
    let content = match std::fs::read_to_string(path) {
        Ok(c) => c,
        Err(_) => return String::new(),
    };
    let sections = crate::neuron::parse_sections(&content);
    let body = match sections.get("purpose") {
        Some(b) => b.clone(),
        None => return String::new(),
    };
    // Take the first 2 non-empty, non-comment lines as the snippet
    let snippet: String = body
        .lines()
        .filter(|l| !l.trim().is_empty() && !l.trim_start().starts_with("<!--"))
        .take(2)
        .collect::<Vec<_>>()
        .join(" ");
    snippet
}

#[tool_router(router = context_tool_router, vis = "pub(super)")]
impl CortyxServer {
    #[tool(
        name = "cortyx",
        description = "Universal Cortyx entrypoint for the current local-first Cortyx surface. Routes a high-level intent to context retrieval, answer mode, wake-up priming, agent status, consistency checks, or a capability summary. Use intent='auto' or omit it to infer the best route from the supplied task/agent/person/path; call it with no task/agent/path inputs to get the current capability summary."
    )]
    pub(in crate::mcp) async fn cortyx(
        &self,
        Parameters(input): Parameters<CortyxInput>,
    ) -> String {
        let route = match derive_cortyx_route(&input) {
            Ok(route) => route,
            Err(err) => return format!("ERROR: {err}"),
        };

        match route.kind {
            CortyxRouteKind::Context | CortyxRouteKind::Answer => {
                let task = match QueryText::new(route.task.unwrap_or_default()) {
                    Ok(task) => task,
                    Err(err) => return format!("ERROR: {err}"),
                };
                let module = input.module.clone().or_else(|| {
                    route
                        .agent
                        .as_ref()
                        .map(|agent| format!("@agent/{}", agent.trim()))
                });
                self.get_contexts(Parameters(GetContextsInput {
                    task: task.into_string(),
                    max_tokens: input.max_tokens,
                    module,
                    person: input.person,
                    kind: input.kind,
                    min_confidence: input.min_confidence,
                    multi_hop: input.multi_hop,
                    previous_response: input.previous_response,
                    open_files: None,
                    error_context: None,
                    delta_mode: input.delta_mode,
                    context_handle: input.context_handle,
                    capsule_mode: input.capsule_mode,
                    answer_mode: Some(route.kind == CortyxRouteKind::Answer),
                    min_answer_confidence: input.min_answer_confidence,
                    provenance_mode: input.provenance_mode,
                    depth: None,
                }))
                .await
            },
            CortyxRouteKind::WakeUp => {
                self.wake_up(Parameters(WakeUpInput {
                    person: input.person,
                    agent: route.agent,
                    prefetch: None,
                }))
                .await
            },
            CortyxRouteKind::AgentStatus => {
                self.agent_status(Parameters(AgentStatusInput {
                    agent: route.agent.unwrap_or_default(),
                    include_timeline: input.include_timeline,
                }))
                .await
            },
            CortyxRouteKind::Consistency => {
                self.check_consistency(Parameters(CheckConsistencyInput { path: input.path }))
                    .await
            },
            CortyxRouteKind::Capabilities => self.render_cortyx_capability_summary().await,
        }
    }

    /// Activate the most relevant neurons for a task.
    ///
    /// Returns context files sorted lexicographically — place them AFTER the
    /// `cache_control: {type: "ephemeral"}` breakpoint in your prompt to keep
    /// the static prefix byte-identical across calls (enabling prompt cache hits
    /// on the static block).
    #[tool(
        name = "cortyx_get_contexts",
        description = "Get the most relevant local/project context neurons for a task. Returns 3-5 .context.md files, sorted deterministically. Inject after your cache_control breakpoint to keep the static prefix byte-identical for prompt caching. Pass your previous assistant response in `previous_response` to close the feedback loop automatically — no separate cortyx_close_task call needed. Set `delta_mode=true` and reuse `context_handle` to receive only added/changed context on iterative same-session work. Set `capsule_mode=true` to prepend stable module capsules and compress redundant same-module summaries into capsule + task delta. Set `answer_mode=true` to return an optional answer-layer derived from the selected contexts without changing the retrieval hot path. Set `min_answer_confidence` to require stronger answer support before answer-mode emits a result. Set `provenance_mode=true` to include lightweight source/explanation metadata."
    )]
    pub(in crate::mcp) async fn get_contexts(
        &self,
        Parameters(input): Parameters<GetContextsInput>,
    ) -> String {
        let mut input = input;
        if input.task.len() > MAX_TASK_BYTES {
            return format!("ERROR: task exceeds {MAX_TASK_BYTES} byte limit");
        }
        if let Some(prev_resp) = &input.previous_response {
            if prev_resp.len() > MAX_CONTENT_BYTES {
                return format!("ERROR: previous_response exceeds {MAX_CONTENT_BYTES} byte limit");
            }
        }

        // Guard total in-flight bytes across concurrent handlers.
        let estimated = input.task.len() + input.previous_response.as_deref().map_or(0, str::len);
        let prev = self
            .inflight_bytes
            .fetch_add(estimated, std::sync::atomic::Ordering::Relaxed);
        if prev + estimated > MAX_INFLIGHT_BYTES {
            self.inflight_bytes
                .fetch_sub(estimated, std::sync::atomic::Ordering::Relaxed);
            return format!(
                "ERROR: server busy — in-flight payload exceeds {MAX_INFLIGHT_BYTES} byte limit"
            );
        }
        // RAII decrement: use a guard so the counter is released even on early returns.
        struct InflightGuard<'a>(&'a std::sync::atomic::AtomicUsize, usize);
        impl Drop for InflightGuard<'_> {
            fn drop(&mut self) {
                self.0
                    .fetch_sub(self.1, std::sync::atomic::Ordering::Relaxed);
            }
        }
        let _guard = InflightGuard(&self.inflight_bytes, estimated);
        input.task = match QueryText::new(std::mem::take(&mut input.task)) {
            Ok(task) => task.into_string(),
            Err(err) => return format!("ERROR: {err}"),
        };

        // S6 — Implicit feedback: if the caller supplied their previous response,
        // apply soft-citation against last_activated before running the new query.
        // This eliminates the need for a separate cortyx_close_task call.
        if let Some(prev_resp) = &input.previous_response {
            let activated = self.last_activated.lock().await.clone();
            if !activated.is_empty() && !prev_resp.is_empty() {
                let response_lower = prev_resp.to_lowercase();
                let response_tokens: std::collections::HashSet<String> =
                    tokenize(prev_resp).into_iter().collect();
                let citation_decisions: Vec<(PathBuf, bool)> = {
                    let idx = self.index.read().await;
                    activated
                        .iter()
                        .map(|path| {
                            let stem = path
                                .file_stem()
                                .unwrap_or_default()
                                .to_string_lossy()
                                .to_lowercase();
                            let stem = stem.trim_end_matches(".context");
                            let explicit_cited = !stem.is_empty() && response_lower.contains(stem);
                            let soft_cited = if !explicit_cited && !response_tokens.is_empty() {
                                idx.term_freq_overlap(path, &response_tokens) >= 20
                            } else {
                                false
                            };
                            (path.clone(), explicit_cited || soft_cited)
                        })
                        .collect()
                };
                let mut idx = self.index.write().await;
                let mut implicit_hits = 0usize;
                for (path, cited) in &citation_decisions {
                    idx.record_hit(path, *cited);
                    if *cited {
                        implicit_hits += 1;
                    }
                }
                tracing::debug!(
                    hits = implicit_hits,
                    total = activated.len(),
                    "S6 implicit feedback applied from previous_response"
                );
            }
        }
        let max_tokens = input.max_tokens.unwrap_or(4096);
        let min_confidence = input.min_confidence.map(|value| value as f32);
        let multi_hop = input.multi_hop.unwrap_or(false);
        let capsule_mode = input.capsule_mode.unwrap_or(false);
        let answer_mode = input.answer_mode.unwrap_or(false);
        let min_answer_confidence = input.min_answer_confidence.map(|value| value as f32);
        let provenance_mode = input.provenance_mode.unwrap_or(false);
        let effective_module: Option<String> = input
            .person
            .as_ref()
            .map(|p| format!("@{}", p))
            .or_else(|| input.module.clone());

        // Clear the previous provisional buffer. Only explicit citation evidence should
        // train long-term ranking; carry-over paths are kept solely for in-session close_task.
        let old_provisional = std::mem::take(&mut *self.provisional_hits.lock().await);

        let augmented_task = {
            let idx = self.index.read().await;
            build_augmented_task(&idx, &input)
        };

        let (mut paths_with_scores, mut overflow) = {
            let idx = self.index.read().await;
            // S-I (R16): Multi-resolution emission — use scored variant for tiered output
            idx.get_contexts_with_scores_and_overflow(
                &augmented_task,
                max_tokens,
                effective_module.as_deref(),
                input.kind.as_deref(),
                min_confidence,
                multi_hop,
            )
        };

        let mut capsule_items = Vec::new();
        if capsule_mode {
            let idx = self.index.read().await;
            let path_modules = build_path_module_map(&paths_with_scores, &overflow, &idx);
            drop(idx);

            let candidate_modules = select_capsule_modules(
                &paths_with_scores,
                effective_module.as_deref(),
                &path_modules,
            );
            let available_capsules: Vec<(String, RenderedContextItem)> = candidate_modules
                .into_iter()
                .filter_map(|module| {
                    render_module_capsule(&self.project_root, &module).map(|item| (module, item))
                })
                .collect();

            if !available_capsules.is_empty() {
                let active_capsule_modules: HashSet<String> = available_capsules
                    .iter()
                    .map(|(module, _)| module.clone())
                    .collect();
                let keep_paths = select_capsule_anchor_paths(
                    &paths_with_scores,
                    &active_capsule_modules,
                    &path_modules,
                );
                paths_with_scores.retain(|(path, _)| match path_modules.get(path) {
                    Some(module) if active_capsule_modules.contains(module) => {
                        keep_paths.contains(path)
                    },
                    _ => true,
                });
                overflow.retain(|(path, _)| match path_modules.get(path) {
                    Some(module) => !active_capsule_modules.contains(module),
                    None => true,
                });
                capsule_items = available_capsules
                    .into_iter()
                    .map(|(_, item)| item)
                    .collect();
            }
        }

        // Flatten paths for backward-compatible downstream use
        let paths: Vec<PathBuf> = paths_with_scores.iter().map(|(p, _)| p.clone()).collect();
        if !old_provisional.is_empty() {
            tracing::debug!(
                cleared = old_provisional.len(),
                "Dropped provisional carry-over without applying implicit ranking feedback"
            );
        }

        // Increment use_count for all returned neurons — activates the feedback loop.
        // Also capture any Contradicts pairs for the warning block (S7).
        let contradictions = if !paths.is_empty() {
            let mut idx = self.index.write().await;
            idx.record_activation(&paths);
            // B2: Record co-activation of query terms with each activated neuron.
            // After ≥30 co-activations, terms are promoted to synonym clouds for
            // query expansion — improving recall for semantically related queries.
            // Use the effective augmented retrieval task, not only the raw user text.
            let terms = crate::index::tokenize(&augmented_task);
            for path in &paths {
                idx.record_coactivation(path, &terms);
            }
            // S7: Check for contradicting pairs among activated neurons.
            idx.find_contradictions(&paths)
        } else {
            Vec::new()
        };

        // Store for cortyx_close_task — replaces previous task's activation list.
        *self.last_activated.lock().await = paths.clone();
        // Set provisional carry-over for in-session close_task tracking only.
        *self.provisional_hits.lock().await = paths.clone();

        if paths.is_empty() && capsule_items.is_empty() {
            if input.min_confidence.is_some() {
                return "(no neurons matched — confidence below threshold)".to_string();
            }
            return "No relevant neurons found. Run `cortyx compile .` first, then call \
                cortyx_evolve_context to fill stubs."
                .to_string();
        }

        if answer_mode {
            let idx_read = self.index.read().await;
            return match answer_plane::render_answer_output_decision(
                &idx_read,
                &input.task,
                &paths_with_scores,
                provenance_mode,
                min_answer_confidence,
            ) {
                Ok(answer) => {
                    // A7: ECS filter — abstain if the generated answer is likely hallucinated.
                    let verdict = verify_gate::check(&answer);
                    if verdict.risk_score > 0.50 {
                        if provenance_mode {
                            return format!(
                                "(answer abstained — ECS={}/100, risk={:.2}: {})",
                                verdict.ecs_score(),
                                verdict.risk_score,
                                verdict.summary.as_deref().unwrap_or("high risk")
                            );
                        }
                        return String::new();
                    }
                    // Append ECS score to provenance output when available.
                    if provenance_mode {
                        format!("{answer}\n\n<!-- ECS: {}/100 -->", verdict.ecs_score())
                    } else {
                        answer
                    }
                },
                Err(answer_plane::AnswerAbstentionReason::LowFormConfidence)
                    if min_answer_confidence.is_some() =>
                {
                    "(no confident answer — answer confidence below threshold)".to_string()
                },
                Err(answer_plane::AnswerAbstentionReason::LowFormConfidence)
                | Err(answer_plane::AnswerAbstentionReason::Unsupported) => String::new(),
            };
        }

        // Filenames sorted lexicographically in the header — stable, byte-identical for the same
        // neuron set regardless of retrieval order. Used for cache-key validation by the client.
        // Bodies below are emitted in BM25-relevance order (most useful neuron first).
        let mut lex_names: Vec<String> = paths
            .iter()
            .map(|p| {
                p.file_name()
                    .unwrap_or_default()
                    .to_string_lossy()
                    .into_owned()
            })
            .collect();
        lex_names.sort();

        let mut out = format!(
            "<!-- CORTYX CONTEXT — injected after cache_control breakpoint -->\n\
             <!-- Task: {} -->\n\
             <!-- Neurons (lex): {} -->\n\n",
            sanitize_comment(&input.task),
            lex_names.join(", "),
        );
        if !capsule_items.is_empty() {
            let mut capsule_names: Vec<String> = capsule_items
                .iter()
                .map(|item| {
                    item.path
                        .file_name()
                        .unwrap_or_default()
                        .to_string_lossy()
                        .into_owned()
                })
                .collect();
            capsule_names.sort();
            out.push_str(&format!(
                "<!-- Module capsules: {} -->\n<!-- Capsule mode: stable module capsule + task delta -->\n\n",
                capsule_names.join(", "),
            ));
        }
        if provenance_mode {
            let idx_read = self.index.read().await;
            if let Some(block) =
                answer_plane::render_provenance_output(&idx_read, &paths_with_scores)
            {
                out.push_str(&block);
            }
        }

        // S-I (R16): Tiered emission — full body for Tier 2 (score ≥ 5.0),
        // summary for Tier 1 (1.5 ≤ score < 5.0), already handled as overflow for Tier 0.
        // TRIZ depth override: when depth is set, use fixed-level rendering instead.
        let render_terms = tokenize(&input.task);
        let idx_read = self.index.read().await;
        let mut rendered_chunks = capsule_items;
        if let Some(depth) = input.depth {
            rendered_chunks.extend(paths_with_scores.iter().map(|(path, score)| {
                render_context_item_at_depth(path, *score, depth, &render_terms, &idx_read)
            }));
        } else {
            rendered_chunks.extend(
                paths_with_scores.iter().map(|(path, score)| {
                    render_context_item(path, *score, &render_terms, &idx_read)
                }),
            );
        }
        drop(idx_read);
        let rendered_overflow: Vec<RenderedContextItem> = overflow
            .iter()
            .map(|(path, headline)| render_overflow_item(path, headline))
            .collect();

        let delta_mode = input.delta_mode.unwrap_or(false);
        let mut chunks_to_emit = rendered_chunks.clone();
        let mut overflow_to_emit = rendered_overflow.clone();
        if delta_mode {
            let context_handle = self
                .ensure_context_handle(input.context_handle.as_deref())
                .await;
            let previous_snapshot = self.load_context_snapshot(&context_handle).await;
            let chunk_delta = select_delta_items(
                &rendered_chunks,
                previous_snapshot.as_ref().map(|s| &s.chunks),
            );
            let overflow_delta = select_delta_items(
                &rendered_overflow,
                previous_snapshot.as_ref().map(|s| &s.overflow),
            );

            chunks_to_emit = chunk_delta.emitted;
            overflow_to_emit = overflow_delta.emitted;

            let emitted_total = chunks_to_emit.len() + overflow_to_emit.len();
            let unchanged_total = chunk_delta.unchanged + overflow_delta.unchanged;
            let removed_total = chunk_delta.removed + overflow_delta.removed;
            let mode_label = if previous_snapshot.is_some() {
                "delta"
            } else {
                "full"
            };

            out.push_str(&format!(
                "<!-- Context handle: {} -->\n<!-- Context mode: {mode_label}; emitted={emitted_total}; unchanged={unchanged_total}; removed={removed_total} -->\n",
                sanitize_comment(&context_handle),
            ));
            if emitted_total == 0 {
                out.push_str(
                    "<!-- Context delta: no new or changed chunks; reuse previously injected context. -->\n",
                );
            }
            out.push('\n');

            self.store_context_snapshot(context_handle, &rendered_chunks, &rendered_overflow)
                .await;
        }

        for chunk in &chunks_to_emit {
            out.push_str(&chunk.rendered);
        }

        // Compressed overflow neurons: emit one-line headlines for neurons that
        // were relevant but exceeded the token budget. Gives the LLM routing
        // signals at ~5% of the token cost of the full neuron.
        if !overflow_to_emit.is_empty() {
            out.push_str("<!-- === COMPRESSED CONTEXT (budget overflow) === -->\n");
            for item in &overflow_to_emit {
                out.push_str(&item.rendered);
            }
            out.push_str("<!-- === END COMPRESSED === -->\n");
        }

        // S7: Append contradiction warning block if any activated neurons conflict.
        if !contradictions.is_empty() {
            out.push_str(
                "\n## ⚠ Contradictions Detected\n\
                The following neuron pairs hold conflicting information. \
                Verify which is current before proceeding.\n\n",
            );
            for (a, b, reason) in &contradictions {
                let a_name = a.file_name().unwrap_or_default().to_string_lossy();
                let b_name = b.file_name().unwrap_or_default().to_string_lossy();
                out.push_str(&format!(
                    "- **{}** ↔ **{}**\n  Reason: {}\n\n",
                    a_name, b_name, reason
                ));
            }
        }

        out.push_str("<!-- === END CORTYX CONTEXT === -->\n");
        out
    }

    /// Rewrite a neuron with improved content (self-improvement during normal usage).
    #[tool(
        name = "cortyx_evolve_context",
        description = "Evolve (rewrite) a neuron with AI-curated content. Call after a task reveals better reasoning instructions, pitfalls, or cross-references. Atomically updates the .context.md file and refreshes the index. IMPORTANT: When writing neuron content for conversation/memory neurons, append a '## paraphrases' section containing 8-10 natural-language questions that this neuron directly answers. Example: '## paraphrases\\nWhat degree did she graduate with?\\nWhere did she go to school?\\nWhat did she study?' This pre-generates question vocabulary that BM25 uses at query time, dramatically improving recall without any model at query time."
    )]
    pub(in crate::mcp) async fn evolve_context(
        &self,
        Parameters(input): Parameters<EvolveContextInput>,
    ) -> String {
        let rel = match validate_relative_path(&input.path) {
            Ok(p) => p,
            Err(e) => return format!("ERROR: Invalid path: {e}"),
        };
        if input.content.is_empty() {
            return "ERROR: Content must not be empty".to_string();
        }
        if input.content.len() > MAX_CONTENT_BYTES {
            return format!("ERROR: content exceeds {MAX_CONTENT_BYTES} byte limit");
        }

        let source = self.project_root.join(&rel);
        let neuron_path = core_neuron_path(&source, &self.project_root);
        let existed_before = neuron_path.exists();

        if let Err(e) = std::fs::create_dir_all(neuron_path.parent().unwrap_or(Path::new("."))) {
            return format!("ERROR: Failed to create neuron dir: {e}");
        }

        // E2: Save full content shadow before overwriting — enables instant undo.
        if let Ok(prev_content) = std::fs::read_to_string(&neuron_path) {
            let meta_file_shadow = meta_path(&neuron_path);
            let mut meta_shadow = load_or_new_meta(&meta_file_shadow, &source, NeuronKind::Core);
            push_shadow(&mut meta_shadow.shadow_sections, "_full", prev_content);
            if let Err(e) = save_meta(&meta_file_shadow, &meta_shadow) {
                return format!("ERROR: Failed to save rollback shadow: {e}");
            }
        }

        if let Err(e) = atomic_write(&neuron_path, input.content.as_bytes()) {
            return format!("ERROR: Failed to write neuron: {e}");
        }

        let source_hash = hash_file(&source).unwrap_or_default();
        let now = now_iso8601();
        let meta_file = meta_path(&neuron_path);
        let mut meta = load_or_new_meta(&meta_file, &source, NeuronKind::Core);
        refresh_meta_after_content_write(&mut meta, &input.content);
        meta.source_hash = source_hash;
        meta.last_updated = now;

        if let Err(e) = save_meta(&meta_file, &meta) {
            return format!(
                "ERROR: Failed to save meta for {}: {e}",
                self.rel_display(&neuron_path)
            );
        }
        let provenance_result = record_mutation_provenance(
            &neuron_path,
            &meta,
            &input.content,
            if existed_before {
                ProvenanceOperation::Update
            } else {
                ProvenanceOperation::Create
            },
            ProvenanceSource::Local,
            None,
            Some(if existed_before {
                format!("rewrote neuron from {}", rel.display())
            } else {
                format!("created neuron from {}", rel.display())
            }),
        );

        let mut idx = self.index.write().await;
        if let Err(e) = idx.upsert_neuron(&neuron_path, &input.content, &meta) {
            return format!("ERROR: Failed to update index: {e}");
        }
        finalize_mutation_message(
            format!(
                "Neuron evolved: {} ({} tokens, {} synapses)",
                self.rel_display(&neuron_path),
                meta.tokens,
                meta.synapses.len()
            ),
            provenance_result,
        )
    }

    /// Update a single named section within a neuron — surgical and token-efficient.
    #[tool(
        name = "cortyx_evolve_section",
        description = "Update one named section (e.g. 'purpose', 'api', 'pitfalls') within a neuron. ~50 tokens instead of a full 1500-token rewrite. Use when only one section needs improving."
    )]
    pub(in crate::mcp) async fn evolve_section(
        &self,
        Parameters(input): Parameters<EvolveSectionInput>,
    ) -> String {
        let rel = match validate_relative_path(&input.path) {
            Ok(p) => p,
            Err(e) => return format!("ERROR: Invalid path: {e}"),
        };
        if input.section.is_empty() {
            return "ERROR: section must not be empty".to_string();
        }
        if input.content.is_empty() {
            return "ERROR: content must not be empty".to_string();
        }

        let source = self.project_root.join(&rel);
        let neuron_path = core_neuron_path(&source, &self.project_root);
        let section_key = input.section.to_lowercase();

        let existing = match std::fs::read_to_string(&neuron_path) {
            Ok(c) => c,
            Err(e) => {
                return format!("ERROR: Cannot read neuron (run `cortyx compile` first): {e}");
            },
        };

        // E2: Save previous section content to shadow before overwriting.
        {
            let meta_file_shadow = meta_path(&neuron_path);
            let mut meta_shadow = load_or_new_meta(&meta_file_shadow, &source, NeuronKind::Core);
            // Extract current section body from existing content and save as shadow
            let current_sections = parse_sections(&existing);
            if let Some(prev_body) = current_sections.get(&section_key) {
                push_shadow(
                    &mut meta_shadow.shadow_sections,
                    &section_key,
                    prev_body.clone(),
                );
            } else {
                // Save the full content as a fallback shadow
                push_shadow(&mut meta_shadow.shadow_sections, "_full", existing.clone());
            }
            if let Err(e) = save_meta(&meta_file_shadow, &meta_shadow) {
                return format!("ERROR: Failed to save rollback shadow: {e}");
            }
        }

        let new_content = replace_section(&existing, &input.section, &input.content);

        if let Err(e) = atomic_write(&neuron_path, new_content.as_bytes()) {
            return format!("ERROR: Failed to write neuron: {e}");
        }

        let now = now_iso8601();
        let meta_file = meta_path(&neuron_path);
        let mut meta = load_or_new_meta(&meta_file, &source, NeuronKind::Core);
        refresh_meta_after_content_write(&mut meta, &new_content);
        meta.last_updated = now;

        if let Err(e) = save_meta(&meta_file, &meta) {
            return format!("ERROR: Failed to save meta: {e}");
        }
        let provenance_result = record_mutation_provenance(
            &neuron_path,
            &meta,
            &new_content,
            ProvenanceOperation::SectionUpdate,
            ProvenanceSource::Local,
            Some(section_key.clone()),
            Some(format!(
                "updated {section_key} section for {}",
                rel.display()
            )),
        );

        let mut idx = self.index.write().await;
        if let Err(e) = idx.upsert_neuron(&neuron_path, &new_content, &meta) {
            return format!("ERROR: Failed to update index: {e}");
        }
        let sections = parse_sections(&new_content);
        finalize_mutation_message(
            format!(
                "Section '{}' updated in {} ({} tokens, {} sections)",
                input.section,
                self.rel_display(&neuron_path),
                meta.tokens,
                sections.len()
            ),
            provenance_result,
        )
    }

    /// Create a use-case neuron — a proven concrete chunk for a specific task pattern.
    #[tool(
        name = "cortyx_extract_from_raw",
        description = "Save a proven relevant chunk as a use-case neuron. Activated automatically for similar future tasks without re-reading the raw source."
    )]
    pub(in crate::mcp) async fn extract_from_raw(
        &self,
        Parameters(input): Parameters<ExtractFromRawInput>,
    ) -> String {
        let rel = match validate_relative_path(&input.path) {
            Ok(p) => p,
            Err(e) => return format!("ERROR: Invalid path: {e}"),
        };
        if input.chunk.len() > MAX_CONTENT_BYTES {
            return format!("ERROR: chunk exceeds {MAX_CONTENT_BYTES} byte limit");
        }

        let source = self.project_root.join(&rel);
        // Truncate kebab to avoid exceeding OS filename limits (max 255 chars total)
        let task_kebab = truncate_str(&to_kebab(&input.task_pattern), 64);
        let source_rel = rel.to_string_lossy().replace(['/', '\\'], "_");

        let neuron_filename = format!("{source_rel}.usecase.{task_kebab}.md");
        let neuron_path = neuron_dir(&self.project_root).join(&neuron_filename);
        let existed_before = neuron_path.exists();
        let now = now_iso8601();

        let content = format!(
            "<!-- Task pattern: {} -->\n\
             <!-- parent: {source_rel}.context.md -->\n\
             <!-- created: {now} | uses: 0 -->\n\n\
             **Exact relevant chunk (proven):**\n\n{}\n\n\
             **Why it was used:**\n{}\n",
            input.task_pattern, input.chunk, input.why
        );

        if let Err(e) = std::fs::create_dir_all(neuron_path.parent().unwrap_or(Path::new("."))) {
            return format!("ERROR: Failed to create dir: {e}");
        }
        if let Err(e) = atomic_write(&neuron_path, content.as_bytes()) {
            return format!("ERROR: Failed to write use-case neuron: {e}");
        }

        let parent_neuron = core_neuron_path(&source, &self.project_root);
        let mut meta = NeuronMeta::new_stub(&source, NeuronKind::UseCase);
        meta.task_pattern = Some(input.task_pattern.clone());
        meta.parent = Some(parent_neuron);
        meta.tokens = estimate_context_tokens(&content).get();
        meta.last_updated = now;
        meta.source_hash = hash_file(&source).unwrap_or_default();
        meta.status = NeuronStatus::Fresh;

        let meta_file = meta_path(&neuron_path);
        if let Err(e) = save_meta(&meta_file, &meta) {
            tracing::warn!("Failed to save meta for {}: {e}", neuron_path.display());
        }
        let provenance_result = record_mutation_provenance(
            &neuron_path,
            &meta,
            &content,
            if existed_before {
                ProvenanceOperation::Update
            } else {
                ProvenanceOperation::Create
            },
            ProvenanceSource::Import,
            None,
            Some(format!(
                "extracted raw chunk for pattern \"{}\"",
                input.task_pattern
            )),
        );

        let mut idx = self.index.write().await;
        if let Err(e) = idx.upsert_neuron(&neuron_path, &content, &meta) {
            return format!("ERROR: Failed to update index: {e}");
        }
        finalize_mutation_message(
            format!(
                "Use-case neuron created: {} for pattern \"{}\"",
                self.rel_display(&neuron_path),
                input.task_pattern
            ),
            provenance_result,
        )
    }
    // ── Hierarchy navigation tools (TRIZ R13-G2) ─────────────────────────────

    /// List all modules (directories and @person scopes) with their neuron count
    /// and average hit rate. Equivalent to MemPalace list_wings.
    #[tool(
        name = "cortyx_list_modules",
        description = "List all modules (code namespaces and @person scopes) with neuron count and avg hit rate. \
                       Equivalent to MemPalace list_wings. Returns JSON array."
    )]
    pub(in crate::mcp) async fn list_modules(&self) -> String {
        let idx = self.index.read().await;
        let modules = idx.list_modules();
        if modules.is_empty() {
            return "No modules found. Run cortyx_compile first.".to_string();
        }
        let rows: Vec<serde_json::Value> = modules
            .iter()
            .map(|m| {
                serde_json::json!({
                    "name": m.name,
                    "neuron_count": m.neuron_count,
                    "avg_hit_rate": format!("{:.2}", m.avg_hit_rate),
                    "person_scope": m.is_person_scope,
                })
            })
            .collect();
        serde_json::to_string_pretty(&rows)
            .unwrap_or_else(|_| "ERROR: serialization failed".to_string())
    }

    /// List neurons in a module (or all neurons if module is omitted).
    /// Returns neuron paths, kind, staleness, and hit rate.
    #[tool(
        name = "cortyx_list_neurons",
        description = "List neurons in a module (or all neurons if module is omitted). \
                       Returns path, kind, staleness, and hit_rate for each neuron."
    )]
    pub(in crate::mcp) async fn list_neurons(
        &self,
        Parameters(input): Parameters<ListNeuronsInput>,
    ) -> String {
        let idx = self.index.read().await;
        let neurons = idx.list_neurons(input.module.as_deref());
        if neurons.is_empty() {
            return format!(
                "No neurons found{}.",
                input
                    .module
                    .as_ref()
                    .map(|m| format!(" in module '{m}'"))
                    .unwrap_or_default()
            );
        }
        let summarize = input.summarize.unwrap_or(false);
        let rows: Vec<serde_json::Value> = neurons
            .iter()
            .map(|n| {
                let mut entry = serde_json::json!({
                    "path": self.rel_display(&n.path).as_ref().to_string(),
                    "kind": format!("{:?}", n.kind),
                    "staleness": format!("{:.1}", n.staleness_multiplier),
                    "hit_rate": format!("{:.2}", n.hit_rate),
                    "use_count": n.use_count,
                });
                if summarize {
                    entry["purpose"] = serde_json::Value::String(purpose_snippet(&n.path));
                }
                entry
            })
            .collect();
        serde_json::to_string_pretty(&rows)
            .unwrap_or_else(|_| "ERROR: serialization failed".to_string())
    }

    // ── DCI: composable search primitives ─────────────────────────────────────

    /// Read one named section of a neuron without loading the full body.
    ///
    /// Enables PageIndex-style triage: agents load capsules first, then drill into
    /// only the sections they actually need. Use `section="_full"` for the whole body.
    #[tool(
        name = "cortyx_read_section",
        description = "Read a single named section (e.g. 'purpose', 'api', 'pitfalls') from a \
                       neuron file without loading the full body. Use section='_full' to read \
                       the entire neuron. Path is the full neuron path as returned by \
                       cortyx_list_neurons. Enables token-efficient drill-down: load the \
                       'purpose' section first, then request 'api' only if needed."
    )]
    pub(in crate::mcp) async fn read_section(
        &self,
        Parameters(input): Parameters<ReadSectionInput>,
    ) -> String {
        let path = match resolve_neuron_store_path(&input.path, &self.project_root) {
            Ok(p) => p,
            Err(err) => return format!("ERROR: Invalid neuron path: {err}"),
        };
        let content = match std::fs::read_to_string(&path) {
            Ok(c) => c,
            Err(e) => return format!("ERROR: Cannot read neuron: {e}"),
        };
        if input.section == "_full" {
            return content;
        }
        let key = input.section.to_lowercase();
        let sections = parse_sections(&content);
        match sections.get(&key) {
            Some(body) => body.clone(),
            None => {
                let available: Vec<&str> = sections.keys().map(String::as_str).collect();
                format!(
                    "ERROR: Section '{}' not found. Available sections: [{}]",
                    input.section,
                    available.join(", ")
                )
            },
        }
    }

    /// Exact string search across all neuron bodies.
    ///
    /// Returns `(neuron_path, line_number, matched_line)` tuples — the same composable
    /// primitives DCI uses on raw corpora, applied to Cortyx's indexed neuron store.
    #[tool(
        name = "cortyx_search_literal",
        description = "Exact string search across all neuron bodies. Returns matched lines with \
                       neuron path and line number. Use for precise lookups that BM25 cannot \
                       express: exact function names, error messages, specific constants. \
                       Optionally scope to a module and/or enable case-insensitive matching."
    )]
    pub(in crate::mcp) async fn search_neurons_literal(
        &self,
        Parameters(input): Parameters<SearchNeuronsInput>,
    ) -> String {
        let ndir = neuron_dir(&self.project_root);
        let case_insensitive = input.case_insensitive.unwrap_or(false);
        let needle_owned;
        let needle: &str = if case_insensitive {
            needle_owned = input.term.to_lowercase();
            &needle_owned
        } else {
            &input.term
        };

        let mut matches: Vec<serde_json::Value> = Vec::new();
        let walker = walkdir::WalkDir::new(&ndir)
            .follow_links(false)
            .into_iter()
            .filter_map(|e| e.ok())
            .filter(|e| {
                e.file_type().is_file()
                    && e.path()
                        .extension()
                        .and_then(|s| s.to_str())
                        .map(|ext| ext == "md")
                        .unwrap_or(false)
            });

        for entry in walker {
            let path = entry.path();
            let rel = match path.strip_prefix(&ndir) {
                Ok(r) => r.to_string_lossy().to_string(),
                Err(_) => path.to_string_lossy().to_string(),
            };
            if let Some(ref scope) = input.scope {
                if !rel.contains(scope.as_str()) {
                    continue;
                }
            }
            let content = match std::fs::read_to_string(path) {
                Ok(c) => c,
                Err(_) => continue,
            };
            for (lineno, line) in content.lines().enumerate() {
                let haystack = if case_insensitive {
                    line.to_lowercase()
                } else {
                    line.to_string()
                };
                if haystack.contains(needle) {
                    matches.push(serde_json::json!({
                        "neuron": rel,
                        "line": lineno + 1,
                        "text": line,
                    }));
                    if matches.len() >= 200 {
                        break;
                    }
                }
            }
            if matches.len() >= 200 {
                break;
            }
        }

        if matches.is_empty() {
            return format!("No matches for {:?}", input.term);
        }
        serde_json::to_string_pretty(&matches)
            .unwrap_or_else(|_| "ERROR: serialization failed".to_string())
    }

    /// Regex search across all neuron bodies.
    #[tool(
        name = "cortyx_search_regex",
        description = "Regex search across all neuron bodies. Returns matched lines with neuron \
                       path and line number. Use for pattern-based searches that BM25 cannot \
                       express: identifiers with common prefixes, versioned symbols, structured \
                       values. Optionally scope to a module."
    )]
    pub(in crate::mcp) async fn search_neurons_regex(
        &self,
        Parameters(input): Parameters<SearchNeuronsRegexInput>,
    ) -> String {
        use regex::Regex;
        let re = match Regex::new(&input.pattern) {
            Ok(r) => r,
            Err(e) => return format!("ERROR: Invalid regex: {e}"),
        };
        let ndir = neuron_dir(&self.project_root);
        let mut matches: Vec<serde_json::Value> = Vec::new();

        let walker = walkdir::WalkDir::new(&ndir)
            .follow_links(false)
            .into_iter()
            .filter_map(|e| e.ok())
            .filter(|e| {
                e.file_type().is_file()
                    && e.path()
                        .extension()
                        .and_then(|s| s.to_str())
                        .map(|ext| ext == "md")
                        .unwrap_or(false)
            });

        for entry in walker {
            let path = entry.path();
            let rel = match path.strip_prefix(&ndir) {
                Ok(r) => r.to_string_lossy().to_string(),
                Err(_) => path.to_string_lossy().to_string(),
            };
            if let Some(ref scope) = input.scope {
                if !rel.contains(scope.as_str()) {
                    continue;
                }
            }
            let content = match std::fs::read_to_string(path) {
                Ok(c) => c,
                Err(_) => continue,
            };
            for (lineno, line) in content.lines().enumerate() {
                if re.is_match(line) {
                    matches.push(serde_json::json!({
                        "neuron": rel,
                        "line": lineno + 1,
                        "text": line,
                    }));
                    if matches.len() >= 200 {
                        break;
                    }
                }
            }
            if matches.len() >= 200 {
                break;
            }
        }

        if matches.is_empty() {
            return format!("No matches for pattern {:?}", input.pattern);
        }
        serde_json::to_string_pretty(&matches)
            .unwrap_or_else(|_| "ERROR: serialization failed".to_string())
    }

    /// Hierarchical tree navigation (PageIndex-style in-context index).
    ///
    /// Returns a compact JSON tree the agent uses to reason about where to drill in.
    /// No node → root module list. Module name → its neurons. Neuron path → its sections.
    #[tool(
        name = "cortyx_explore_tree",
        description = "Navigate the project's neuron hierarchy like a table of contents. \
                       No argument → list all modules with neuron counts. \
                       node='<module>' → list neurons in that module with Purpose snippets. \
                       node='<neuron_path>' → list named sections inside that neuron. \
                       Use this for PageIndex-style top-down navigation: read the ToC, \
                       pick a module, pick a neuron, then use cortyx_read_section to drill in."
    )]
    pub(in crate::mcp) async fn explore_tree(
        &self,
        Parameters(input): Parameters<ExploreTreeInput>,
    ) -> String {
        let idx = self.index.read().await;

        match &input.node {
            None => {
                // Root: list all modules with neuron counts
                let modules = idx.list_modules();
                if modules.is_empty() {
                    return "No modules found. Run `cortyx compile .` first.".to_string();
                }
                let rows: Vec<serde_json::Value> = modules
                    .iter()
                    .map(|m| {
                        serde_json::json!({
                            "module": m.name,
                            "neuron_count": m.neuron_count,
                            "hint": "call cortyx_explore_tree(node='<module>') to expand",
                        })
                    })
                    .collect();
                serde_json::to_string_pretty(&rows)
                    .unwrap_or_else(|_| "ERROR: serialization failed".to_string())
            },
            Some(node) => {
                // Try to resolve as a neuron path first
                let ndir = neuron_dir(&self.project_root);
                let as_path = ndir.join(node);
                if as_path.exists() && as_path.is_file() {
                    // Node is a neuron — list its sections
                    let content = match std::fs::read_to_string(&as_path) {
                        Ok(c) => c,
                        Err(e) => return format!("ERROR: Cannot read neuron: {e}"),
                    };
                    let sections = parse_sections(&content);
                    if sections.is_empty() {
                        return format!("No named sections found in {node}. Use cortyx_read_section(path='{node}', section='_full') to read the full body.");
                    }
                    let mut rows: Vec<serde_json::Value> = sections
                        .iter()
                        .map(|(name, body)| {
                            let snippet: String = body.lines().take(2).collect::<Vec<_>>().join(" ");
                            serde_json::json!({
                                "section": name,
                                "snippet": snippet,
                                "hint": format!("call cortyx_read_section(path='{node}', section='{name}') to read"),
                            })
                        })
                        .collect();
                    rows.sort_by(|a, b| {
                        a["section"]
                            .as_str()
                            .unwrap_or("")
                            .cmp(b["section"].as_str().unwrap_or(""))
                    });
                    serde_json::to_string_pretty(&rows)
                        .unwrap_or_else(|_| "ERROR: serialization failed".to_string())
                } else {
                    // Node is a module name — list its neurons with Purpose snippets
                    let neurons = idx.list_neurons(Some(node));
                    if neurons.is_empty() {
                        return format!("No neurons found in module '{node}'. Use cortyx_explore_tree() with no argument to see available modules.");
                    }
                    let rows: Vec<serde_json::Value> = neurons
                        .iter()
                        .map(|n| {
                            let rel = self.rel_display(&n.path).as_ref().to_string();
                            let snippet = purpose_snippet(&n.path);
                            serde_json::json!({
                                "neuron": rel,
                                "purpose": snippet,
                                "hint": format!("call cortyx_explore_tree(node='{rel}') to see sections"),
                            })
                        })
                        .collect();
                    serde_json::to_string_pretty(&rows)
                        .unwrap_or_else(|_| "ERROR: serialization failed".to_string())
                }
            },
        }
    }

    /// Raw-corpus search across source files (DCI fallback for unindexed content).
    ///
    /// Walks actual source files — not neuron bodies. Use when BM25 coverage is incomplete,
    /// for new files not yet compiled, or to discover what to index next.
    #[tool(
        name = "cortyx_search_raw",
        description = "Search raw source files directly (not neuron bodies). Returns \
                       (file_path, line_number, matched_line) tuples. Use for: files not yet \
                       compiled into the neuron index, cross-project searches, discovering \
                       what to index next. Set regex=true to treat pattern as a regex. \
                       Optionally scope by path or filetype (e.g. 'rs', 'ts')."
    )]
    pub(in crate::mcp) async fn search_raw(
        &self,
        Parameters(input): Parameters<SearchRawInput>,
    ) -> String {
        use regex::Regex;

        let search_root = if let Some(ref p) = input.path {
            let candidate = std::path::Path::new(p);
            if candidate.is_absolute() {
                candidate.to_path_buf()
            } else {
                self.project_root.join(p)
            }
        } else {
            self.project_root.clone()
        };

        // Skip hidden dirs and the neuron store itself
        let ndir = neuron_dir(&self.project_root);

        let use_regex = input.regex.unwrap_or(false);
        let re_opt: Option<Regex> = if use_regex {
            match Regex::new(&input.pattern) {
                Ok(r) => Some(r),
                Err(e) => return format!("ERROR: Invalid regex: {e}"),
            }
        } else {
            None
        };
        let literal = &input.pattern;

        let mut matches: Vec<serde_json::Value> = Vec::new();

        let walker = walkdir::WalkDir::new(&search_root)
            .follow_links(false)
            .into_iter()
            .filter_entry(|e| {
                // Skip hidden directories and the neuron store
                let name = e.file_name().to_string_lossy();
                if e.file_type().is_dir() {
                    if name.starts_with('.') {
                        return false;
                    }
                    if e.path() == ndir {
                        return false;
                    }
                }
                true
            })
            .filter_map(|e| e.ok())
            .filter(|e| {
                if !e.file_type().is_file() {
                    return false;
                }
                if let Some(ref ft) = input.filetype {
                    e.path()
                        .extension()
                        .and_then(|s| s.to_str())
                        .map(|ext| ext == ft.as_str())
                        .unwrap_or(false)
                } else {
                    true
                }
            });

        'outer: for entry in walker {
            let path = entry.path();
            let rel = match path.strip_prefix(&self.project_root) {
                Ok(r) => r.to_string_lossy().to_string(),
                Err(_) => path.to_string_lossy().to_string(),
            };
            let content = match std::fs::read_to_string(path) {
                Ok(c) => c,
                Err(_) => continue,
            };
            for (lineno, line) in content.lines().enumerate() {
                let hit = if let Some(ref re) = re_opt {
                    re.is_match(line)
                } else {
                    line.contains(literal.as_str())
                };
                if hit {
                    matches.push(serde_json::json!({
                        "file": rel,
                        "line": lineno + 1,
                        "text": line,
                    }));
                    if matches.len() >= 200 {
                        break 'outer;
                    }
                }
            }
        }

        if matches.is_empty() {
            return format!("No matches for {:?} in raw source files.", input.pattern);
        }
        serde_json::to_string_pretty(&matches)
            .unwrap_or_else(|_| "ERROR: serialization failed".to_string())
    }

    /// Return the first N lines of a neuron file for quick preview.
    #[tool(
        name = "cortyx_peek_neuron",
        description = "Return the first 20 lines of a neuron file for quick preview without full activation. \
                       Path is the full neuron path as returned by cortyx_list_neurons."
    )]
    pub(in crate::mcp) async fn peek_neuron(
        &self,
        Parameters(input): Parameters<PeekNeuronInput>,
    ) -> String {
        let path = match resolve_neuron_store_path(&input.path, &self.project_root) {
            Ok(path) => path,
            Err(err) => return format!("ERROR: Invalid neuron path: {err}"),
        };
        let preview = {
            let idx = self.index.read().await;
            idx.peek_neuron(&path, 20)
        };
        match preview {
            Some(p) => p,
            None => format!("ERROR: Neuron not found or unreadable: {}", input.path),
        }
    }

    // ── Person scope tools (TRIZ R13-G5) ─────────────────────────────────────

    /// Restore a single section of a neuron to its shadow copy (E2: section shadow, TRIZ R14).
    ///
    /// Before each evolve_context or evolve_section call, Cortyx automatically saves
    /// the previous content. Use this tool to step backward through recent evolutions.
    #[tool(
        name = "cortyx_rollback_section",
        description = "Restore a neuron section to its previous version (saved before recent evolve calls). \
                       Use section=\"_full\" to restore the entire neuron. \
                       Useful when an LLM evolution produces worse content than the original."
    )]
    pub(in crate::mcp) async fn rollback_section(
        &self,
        Parameters(input): Parameters<RollbackSectionInput>,
    ) -> String {
        use crate::neuron::NeuronMeta;

        let neuron_path = match resolve_neuron_store_path(&input.neuron_path, &self.project_root) {
            Ok(path) => path,
            Err(err) => return format!("ERROR: Invalid neuron path: {err}"),
        };
        let meta_file = meta_path(&neuron_path);

        let meta_data = match std::fs::read_to_string(&meta_file) {
            Ok(d) => d,
            Err(e) => return format!("ERROR: Cannot read sidecar: {e}"),
        };
        let mut meta: NeuronMeta = match serde_json::from_str(&meta_data) {
            Ok(m) => m,
            Err(e) => return format!("ERROR: Cannot parse sidecar: {e}"),
        };

        let shadow = match latest_shadow(&meta.shadow_sections, &input.section) {
            Some(s) => s.to_string(),
            None => {
                return format!(
                    "ERROR: No shadow for section '{}'. Shadows are saved before each evolve call.",
                    input.section
                )
            },
        };

        if input.section == "_full" {
            if let Err(e) = atomic_write(&neuron_path, shadow.as_bytes()) {
                return format!("ERROR: Failed to write neuron: {e}");
            }
            pop_shadow(&mut meta.shadow_sections, "_full");
            refresh_meta_after_content_write(&mut meta, &shadow);
            if let Err(e) = save_meta(&meta_file, &meta) {
                return format!("ERROR: Failed to save meta: {e}");
            }
            let provenance_result = record_mutation_provenance(
                &neuron_path,
                &meta,
                &shadow,
                ProvenanceOperation::Rollback,
                ProvenanceSource::Local,
                None,
                Some("restored full neuron from rollback shadow".to_string()),
            );
            let mut idx = self.index.write().await;
            if let Err(e) = idx.upsert_neuron(&neuron_path, &shadow, &meta) {
                return format!("ERROR: Failed to update index: {e}");
            }
            finalize_mutation_message(
                format!(
                    "✓ Restored full neuron {} from shadow.",
                    self.rel_display(&neuron_path)
                ),
                provenance_result,
            )
        } else {
            let existing = match std::fs::read_to_string(&neuron_path) {
                Ok(c) => c,
                Err(e) => return format!("ERROR: Cannot read neuron file: {e}"),
            };
            let restored = replace_section(&existing, &input.section, &shadow);
            if let Err(e) = atomic_write(&neuron_path, restored.as_bytes()) {
                return format!("ERROR: Failed to write neuron: {e}");
            }
            pop_shadow(&mut meta.shadow_sections, &input.section);
            refresh_meta_after_content_write(&mut meta, &restored);
            if let Err(e) = save_meta(&meta_file, &meta) {
                return format!("ERROR: Failed to save meta: {e}");
            }
            let section_key = input.section.to_lowercase();
            let provenance_result = record_mutation_provenance(
                &neuron_path,
                &meta,
                &restored,
                ProvenanceOperation::Rollback,
                ProvenanceSource::Local,
                Some(section_key.clone()),
                Some(format!("restored {section_key} from rollback shadow")),
            );
            let mut idx = self.index.write().await;
            if let Err(e) = idx.upsert_neuron(&neuron_path, &restored, &meta) {
                return format!("ERROR: Failed to update index: {e}");
            }
            finalize_mutation_message(
                format!(
                    "✓ Restored section '{}' in {} from shadow.",
                    input.section,
                    self.rel_display(&neuron_path)
                ),
                provenance_result,
            )
        }
    }
}
