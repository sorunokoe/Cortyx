use super::super::*;
use crate::agent_memory::{refine_entry, render_structured_diary_entry_from_entry};
use rmcp::{handler::server::wrapper::Parameters, tool, tool_router};

#[tool_router(router = memory_tool_router, vis = "pub(super)")]
impl CortyxServer {
    /// List all @person-scoped memory namespaces.
    #[tool(
        name = "cortyx_list_persons",
        description = "List all @person-scoped memory namespaces (created via mine_conversation with person=...). \
                       Returns person name, neuron count, and avg hit rate."
    )]
    pub(in crate::mcp) async fn list_persons(&self) -> String {
        let idx = self.index.read().await;
        let persons = idx.list_persons();
        if persons.is_empty() {
            return "No person-scoped memories found. Use mine_conversation with person=\"alice\" to create some.".to_string();
        }
        let rows: Vec<serde_json::Value> = persons
            .iter()
            .map(|p| {
                serde_json::json!({
                    "person": p.name.trim_start_matches('@'),
                    "module": p.name,
                    "neuron_count": p.neuron_count,
                    "avg_hit_rate": format!("{:.2}", p.avg_hit_rate),
                })
            })
            .collect();
        serde_json::to_string_pretty(&rows)
            .unwrap_or_else(|_| "ERROR: serialization failed".to_string())
    }

    // ── Conversation recall tool (TRIZ R13-G3) ────────────────────────────────

    /// Retrieve conversation memories (Verbatim neurons) for a query.
    /// Isolates episodic recall from code retrieval — equivalent to MemPalace's
    /// dedicated episodic store but with zero storage overhead (query-time predicate).
    #[tool(
        name = "cortyx_recall",
        description = "Retrieve conversation memories (Verbatim neurons) matching a query. \
                       Optionally scope to a person's memories with person=\"alice\". \
                       Use for 'what did I decide last month?' style queries."
    )]
    pub(in crate::mcp) async fn recall(
        &self,
        Parameters(input): Parameters<RecallInput>,
    ) -> String {
        let idx = self.index.read().await;
        let effective_module: Option<String> = input.person.as_ref().map(|p| format!("@{}", p));
        let paths = idx.get_contexts(
            &input.query,
            input.max_tokens.unwrap_or(4096),
            effective_module.as_deref(),
            Some("conversation"),
        );
        if paths.is_empty() {
            return "No conversation memories found for this query. \
                    Use cortyx_mine_conversation to index conversations first."
                .to_string();
        }
        let mut out = format!("<!-- cortyx:recall {} memories -->\n", paths.len());
        for path in &paths {
            if let Ok(content) = std::fs::read_to_string(path) {
                out.push_str(&content);
                out.push('\n');
            }
        }
        out
    }
    /// Mine a raw conversation turn into the live index as a Verbatim neuron.
    ///
    /// Accepts any format: Claude MD, ChatGPT JSON, LongMemEval JSON, or plain text.
    /// Consecutive calls automatically create TemporalFollows synapse chains.
    /// Use `module` to tag memories for namespace-filtered retrieval.
    #[tool(
        name = "cortyx_mine_conversation",
        description = "Mine a conversation turn (or whole file export) into Verbatim neurons for semantic recall. \
                       Returns the number of neurons created and the first neuron path."
    )]
    pub(in crate::mcp) async fn mine_conversation(
        &self,
        Parameters(input): Parameters<MineConversationInput>,
    ) -> String {
        if input.content.len() > MAX_CONTENT_BYTES {
            return format!("ERROR: content exceeds {MAX_CONTENT_BYTES} byte limit");
        }

        // ECS verification gate — blocks or quarantines hallucinated content before
        // it enters long-term memory. No-op when `--features verify` is absent.
        if !input.skip_verify.unwrap_or(false) {
            let verdict = verify_gate::check(&input.content);
            let block_threshold = input
                .min_ecs_threshold
                .unwrap_or(verify_gate::DEFAULT_BLOCK_THRESHOLD);
            if verdict.risk_score > block_threshold {
                let summary = verdict
                    .summary
                    .as_deref()
                    .unwrap_or("high hallucination risk");
                return format!(
                    "REJECTED by ECS gate (risk={:.2}, ECS={}/100): {}. \
                     Use skip_verify=true to override, or revise the content.",
                    verdict.risk_score,
                    verdict.ecs_score(),
                    summary
                );
            }
            // Medium-risk: quarantine annotation is stored in the neuron sidecar via
            // the miner metadata path (future: pass quarantine_tag into mine_text).
            // For now, surface the warning in the response so the agent is aware.
            if let Some(annotation) = verdict.quarantine_annotation() {
                tracing::debug!(
                    annotation = %annotation,
                    "mine_conversation: medium-risk content quarantined"
                );
            }
        }

        let mut idx = self.index.write().await;
        let effective_module: Option<String> = input
            .person
            .as_ref()
            .map(|p| format!("@{}", p))
            .or_else(|| input.module.clone());
        match miner::mine_text(
            &input.content,
            "mcp-inline",
            &self.project_root,
            &mut idx,
            effective_module.as_deref(),
            input.speaker.as_deref(),
            input.timestamp.as_deref(),
        ) {
            Ok(count) => format!(
                "Mined {count} Verbatim neuron(s). Total neurons: {}.",
                idx.neuron_count()
            ),
            Err(e) => format!("ERROR: {e}"),
        }
    }

    /// Implicit hit-rate feedback — scan the task response for cited neurons.
    ///
    /// Pass the assistant's full response text; Cortyx scans for neuron file stems
    /// and auto-increments hit_count for each match. No per-neuron tool calls needed.
    /// Call once at task end instead of cortyx_record_hit for each neuron.
    #[tool(
        name = "cortyx_close_task",
        description = "Pass the assistant response text. Cortyx auto-records hits for neurons whose filenames appear in the response. Zero friction — one call closes the feedback loop for the whole task."
    )]
    pub(in crate::mcp) async fn close_task(
        &self,
        Parameters(input): Parameters<CloseTaskInput>,
    ) -> String {
        if input.response_text.len() > MAX_CONTENT_BYTES {
            return format!("ERROR: response_text exceeds {MAX_CONTENT_BYTES} byte limit");
        }
        // R12-S2-B: Clear provisional hits — close_task provides actual citation evidence,
        // so the optimistic provisional buffer is no longer needed.
        self.feedback.provisional_hits.lock().await.clear();
        let activated = self.feedback.last_activated.lock().await.clone();
        if activated.is_empty() {
            return "No neurons from last cortyx_get_contexts call to evaluate.".to_string();
        }

        let response_lower = input.response_text.to_lowercase();
        // C1: Graded response-diff citation (TRIZ R14).
        // Tokenize response text and compute overlap with each activated neuron's vocabulary.
        // ≥15 terms → soft cite (record_hit once)
        // ≥30 terms → hard cite (record_hit twice — stronger feedback signal)
        // This is a tighter, graded version of the prior flat ≥20-term threshold.
        let response_tokens: std::collections::HashSet<String> =
            tokenize(&input.response_text).into_iter().collect();
        let mut hits = 0usize;

        // Phase 1 (immutable): decide citation for each activated neuron.
        // Returns (path, explicit_cited, soft_weight) where soft_weight: 0=miss, 1=soft, 2=hard
        let citation_decisions: Vec<(PathBuf, bool, u8)> = {
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
                    if explicit_cited {
                        return (path.clone(), true, 1u8);
                    }
                    // C1 graded: measure term overlap
                    let overlap = if !response_tokens.is_empty() {
                        idx.term_freq_overlap(path, &response_tokens)
                    } else {
                        0
                    };
                    let weight = if overlap >= 30 {
                        2 // hard cite
                    } else if overlap >= 15 {
                        1 // soft cite
                    } else {
                        0 // miss
                    };
                    (path.clone(), weight >= 1, weight)
                })
                .collect()
        };
        let hard_cited_paths: Vec<PathBuf> = citation_decisions
            .iter()
            .filter(|(_, _, weight)| *weight >= 2)
            .map(|(p, _, _)| p.clone())
            .collect();

        // Phase 2 (mutable): apply citation signals.
        let mut idx = self.index.write().await;
        for (path, cited, weight) in &citation_decisions {
            idx.record_hit(path, *cited);
            if *cited {
                hits += 1;
                // Hard cite: record a second hit to double the feedback signal
                if *weight >= 2 {
                    idx.record_hit(path, true);
                    tracing::debug!(path = %path.display(), "Hard citation (≥30 term overlap)");
                } else if *weight == 1 {
                    tracing::debug!(path = %path.display(), "Soft citation (≥15 term overlap)");
                }
            }
        }

        // S-VII (R16): Update last_co_activation_day for all cited neuron pairs (LTP).
        let cited_paths: Vec<PathBuf> = citation_decisions
            .iter()
            .filter(|(_, cited, _)| *cited)
            .map(|(p, _, _)| p.clone())
            .collect();
        if cited_paths.len() >= 2 {
            idx.touch_co_activation_day(&cited_paths);
        }

        // S-VIII (R16): Auto-mine UseCase stubs from code blocks in the response.
        // Code blocks ≥5 lines with ≥60% overlap with a cited neuron → write stub.
        let mined =
            auto_mine_code_blocks(&input.response_text, &cited_paths, &self.project_root, &idx);
        let mined_note = if mined > 0 {
            format!(" Auto-mined {mined} UseCase stub(s).")
        } else {
            String::new()
        };

        // F2: Record session token utilization for budget adaptation.
        // Count total tokens used in this session from the activated neurons.
        let tokens_used: usize = activated.iter().map(|p| idx.tokens_for(p)).sum();
        let tokens_budget = input.response_text.len() / 4; // rough estimate of budget from response size; actual budget not stored here
        if tokens_used > 0 {
            idx.record_session_utilization(tokens_used, tokens_budget.max(1));
            if let Err(e) = idx.save() {
                tracing::warn!("Failed to persist close_task feedback: {e}");
                return format!(
                    "Closed task: {hits}/{} neurons cited (auto-detected from response).{} Warning: feedback was applied in-memory but could not be saved: {e}",
                    activated.len(),
                    mined_note
                );
            }
        }

        // Release write lock before file I/O in surface enrichment.
        drop(idx);

        let surface_note = if !hard_cited_paths.is_empty() {
            let enriched = crate::answer_plane::surface_enricher::enrich_neuron_answer_surfaces(
                &input.response_text,
                &hard_cited_paths,
            );
            if enriched > 0 {
                format!(" Answer-surface enriched {enriched} neuron(s).")
            } else {
                String::new()
            }
        } else {
            String::new()
        };

        format!(
            "Closed task: {hits}/{} neurons cited (auto-detected from response).{mined_note}{surface_note}",
            activated.len()
        )
    }

    /// Record whether a neuron was actually cited — closes the self-improvement feedback loop.
    ///
    /// Call after each task for each neuron returned by cortyx_get_contexts.
    /// Cited neurons get a higher hit_rate → higher BM25 score → activated more readily in future.
    /// Irrelevant neurons are down-weighted over time without any manual curation.
    #[tool(
        name = "cortyx_record_hit",
        description = "Tell Cortyx whether a neuron was actually useful. was_cited=true boosts it; false down-weights it. Closes the self-improvement loop — neurons get smarter with every task."
    )]
    pub(in crate::mcp) async fn record_hit(
        &self,
        Parameters(input): Parameters<RecordHitInput>,
    ) -> String {
        let rel = match validate_relative_path(&input.path) {
            Ok(p) => p,
            Err(e) => return format!("ERROR: Invalid path: {e}"),
        };
        let source = self.project_root.join(&rel);
        let neuron_path = core_neuron_path(&source, &self.project_root);

        if !neuron_path.exists() {
            return format!(
                "ERROR: Neuron not found: {}",
                self.rel_display(&neuron_path)
            );
        }

        let mut idx = self.index.write().await;
        let hit_rate = idx.record_hit(&neuron_path, input.was_cited);
        let use_count = idx.use_count_for(&neuron_path);

        format!(
            "Recorded {} for {} — hit_rate now {:.0}% ({} uses)",
            if input.was_cited { "hit" } else { "miss" },
            self.rel_display(&neuron_path),
            hit_rate * 100.0,
            use_count
        )
    }

    /// Write an agent diary entry (S6 — NE5).
    ///
    /// Agent diaries are Verbatim neurons stored under the `@agent/{agent}` module namespace.
    /// They use the existing @prefix isolation mechanism — zero new storage, zero new retrieval
    /// logic. Each diary entry is BM25-indexed and searchable via cortyx_get_contexts.
    #[tool(
        name = "cortyx_diary_write",
        description = "Write an observation or decision to an agent's diary. Stored as a Verbatim neuron under @agent/{agent} — BM25-indexed, searchable via get_contexts. Optional title/status/goal/next_step/blocker/outcome/entities/depends_on fields turn it into structured agent-state memory without adding a new storage layer."
    )]
    pub(in crate::mcp) async fn diary_write(
        &self,
        Parameters(input): Parameters<DiaryWriteInput>,
    ) -> String {
        if input.agent.is_empty() {
            return "ERROR: agent name must not be empty".to_string();
        }
        let entities = input.entities.clone().unwrap_or_default();
        let depends_on = input.depends_on.clone().unwrap_or_default();
        let structured = has_structured_diary_fields(
            input.title.as_deref(),
            input.status.as_deref(),
            input.goal.as_deref(),
            input.next_step.as_deref(),
            input.blocker.as_deref(),
            input.outcome.as_deref(),
            &entities,
            &depends_on,
        );
        let body = if structured {
            render_structured_diary_entry(
                input.agent.trim(),
                &input.content,
                input.title.as_deref(),
                input.status.as_deref(),
                input.goal.as_deref(),
                input.next_step.as_deref(),
                input.blocker.as_deref(),
                input.outcome.as_deref(),
                &entities,
                &depends_on,
            )
        } else {
            input.content.trim().to_string()
        };
        if body.is_empty() {
            return "ERROR: content must not be empty unless structured diary fields are supplied"
                .to_string();
        }
        if body.len() > MAX_CONTENT_BYTES {
            return format!("ERROR: content exceeds {MAX_CONTENT_BYTES} byte limit");
        }
        let structured_entry = structured
            .then(|| parse_structured_diary_entry(&body))
            .flatten();
        let effective_timestamp = input.timestamp.clone().unwrap_or_else(now_iso8601);
        let module = format!("@agent/{}", input.agent.trim());
        let mut idx = self.index.write().await;
        match miner::mine_text(
            &body,
            "diary",
            &self.project_root,
            &mut idx,
            Some(&module),
            Some(input.agent.trim()),
            Some(effective_timestamp.as_str()),
        ) {
            Ok(count) => {
                if let Some(entry) = structured_entry.as_ref() {
                    if let Err(err) = sync_structured_diary_to_kg(
                        &self.project_root,
                        &mut idx,
                        input.agent.trim(),
                        entry,
                        &effective_timestamp,
                    ) {
                        return format!("ERROR syncing agent memory to temporal KG: {err}");
                    }
                    format!(
                        "Diary entry written for agent '{}' ({count} neuron(s) created, temporal KG synced).",
                        input.agent
                    )
                } else {
                    format!(
                        "Diary entry written for agent '{}' ({count} neuron(s) created).",
                        input.agent
                    )
                }
            },
            Err(e) => format!("ERROR: {e}"),
        }
    }

    /// Read recent diary entries for an agent (S6 — NE5).
    #[tool(
        name = "cortyx_diary_read",
        description = "Read recent diary entries for an agent. Returns last_n entries (default 10) from @agent/{agent} namespace, most recent first. Structured action memories are summarized with status/outcome/entity fields."
    )]
    pub(in crate::mcp) async fn diary_read(
        &self,
        Parameters(input): Parameters<DiaryReadInput>,
    ) -> String {
        if input.agent.is_empty() {
            return "ERROR: agent name must not be empty".to_string();
        }
        let last_n = input.last_n.unwrap_or(10);
        let module = format!("@agent/{}", input.agent.trim());
        let idx = self.index.read().await;
        let results = recent_module_paths(&idx, &module, last_n, Some(NeuronKind::Verbatim));
        if results.is_empty() {
            return format!("No diary entries found for agent '{}'.", input.agent);
        }
        let mut out = format!("## Agent Diary: {} (last {})\n\n", input.agent, last_n);
        for path in results {
            let timestamp_secs = idx
                .context_metadata_for(&path)
                .and_then(|metadata| metadata.timestamp_secs);
            match std::fs::read_to_string(&path) {
                Ok(content) => {
                    if let Some(entry) = parse_structured_diary_entry(&content) {
                        out.push_str(&render_structured_diary_history_entry(
                            &entry,
                            timestamp_secs,
                        ));
                    } else {
                        out.push_str(&format!("---\n{}\n", content));
                    }
                },
                Err(err) => {
                    out.push_str(&format!(
                        "- {} — read error: {}\n",
                        path.display(),
                        sanitize_comment(&err.to_string())
                    ));
                },
            }
        }
        out
    }

    /// Analyse a diary entry and populate refined_plan with a structured blocker decomposition.
    #[tool(
        name = "cortyx_diary_refine",
        description = "Analyse a recent diary entry for an agent and populate refined_plan with a heuristic decomposition suggestion. Returns the refined entry or a message if no refinement was needed. Pure heuristic — no LLM required."
    )]
    pub(in crate::mcp) async fn diary_refine(
        &self,
        Parameters(input): Parameters<DiaryRefineInput>,
    ) -> String {
        if input.agent.is_empty() {
            return "ERROR: agent name must not be empty".to_string();
        }

        let path = if let Some(entry_path) = input.entry_path.as_deref() {
            PathBuf::from(entry_path)
        } else {
            let module = format!("@agent/{}", input.agent.trim());
            let idx = self.index.read().await;
            let mut results = recent_module_paths(&idx, &module, 1, Some(NeuronKind::Verbatim));
            drop(idx);
            let Some(path) = results.pop() else {
                return format!("No diary entries found for agent '{}'.", input.agent);
            };
            path
        };

        let content = match std::fs::read_to_string(&path) {
            Ok(content) => content,
            Err(err) => {
                return format!(
                    "ERROR reading {}: {}",
                    path.display(),
                    sanitize_comment(&err.to_string())
                );
            },
        };
        let Some(mut entry) = parse_structured_diary_entry(&content) else {
            return format!("ERROR: {} is not a structured diary entry", path.display());
        };

        if !refine_entry(&mut entry) {
            return "No refinement needed".to_string();
        }

        let rendered = render_structured_diary_entry_from_entry(&entry);
        if let Err(err) = atomic_write(&path, rendered.as_bytes()) {
            return format!(
                "ERROR writing {}: {}",
                path.display(),
                sanitize_comment(&err.to_string())
            );
        }

        rendered
    }
    /// Session priming — load identity and critical-facts wake-up neurons (S5 — NE4).
    ///
    /// Returns both L0 (_identity) and L1 (_critical_facts) neurons (~170 tokens total)
    /// plus optional @person memories. Call at the start of a new session to prime the
    /// LLM with project identity — equivalent to MemPalace L0+L1 but lossless (plain
    /// Markdown, not AAAK-encoded).
    ///
    /// Zero tokens unless explicitly called — preserves Cortyx's token efficiency advantage.
    #[tool(
        name = "cortyx_wake_up",
        description = "Prime the LLM with project identity and critical facts. Returns _identity.context.md (~50 tokens) + _critical_facts.context.md (~120 tokens). Optionally include person memories and recent structured agent memories. Call once at session start — lossless Markdown (vs MemPalace AAAK-encoding)."
    )]
    pub(in crate::mcp) async fn wake_up(
        &self,
        Parameters(input): Parameters<WakeUpInput>,
    ) -> String {
        let ndir = neuron_dir(&self.project_root);
        let mut out = String::new();

        // Load identity neuron (L0)
        let identity_path = ndir.join("_identity.context.md");
        if identity_path.exists() {
            if let Ok(content) = std::fs::read_to_string(&identity_path) {
                out.push_str("<!-- CORTYX WAKE-UP: L0 identity -->\n");
                out.push_str(&content);
                out.push('\n');
            }
        } else {
            out.push_str(
                "<!-- _identity.context.md not found — run `cortyx compile .` first -->\n",
            );
        }

        // Load critical-facts neuron (L1)
        let critical_path = ndir.join("_critical_facts.context.md");
        if critical_path.exists() {
            if let Ok(content) = std::fs::read_to_string(&critical_path) {
                out.push_str("<!-- CORTYX WAKE-UP: L1 critical facts -->\n");
                out.push_str(&content);
                out.push('\n');
            }
        } else {
            out.push_str(
                "<!-- _critical_facts.context.md not found — run `cortyx compile .` first -->\n",
            );
        }

        if input.person.is_some() || input.agent.is_some() {
            let idx = self.index.read().await;

            // Optional @person memories (recent conversation neurons for the person)
            if let Some(ref person) = input.person {
                let module = format!("@{}", person.trim());
                let paths = idx.get_contexts("", 600, Some(&module), Some("conversation"));
                if !paths.is_empty() {
                    out.push_str(&format!("\n<!-- CORTYX WAKE-UP: @{person} memories -->\n"));
                    for path in paths.iter().take(3) {
                        if let Ok(content) = std::fs::read_to_string(path) {
                            out.push_str(&content);
                            out.push('\n');
                        }
                    }
                }
            }

            if let Some(ref agent) = input.agent {
                if let Some(block) = render_recent_agent_memory_block(&idx, agent, 3) {
                    out.push('\n');
                    out.push_str(&block);
                }
            }
        }

        // Optional git-prefetch: pre-include Level-0 (Purpose only) capsules for neurons
        // mapped to recently changed files, eliminating the orientation tax at session start.
        if input.prefetch.unwrap_or(false) {
            if let Some(prefetch_block) = self.build_git_prefetch_block().await {
                out.push_str("\n<!-- CORTYX WAKE-UP: git-prefetch capsules -->\n");
                out.push_str(&prefetch_block);
            }
        }

        if out.is_empty() {
            out.push_str("No wake-up neurons found. Run `cortyx compile .` to generate them.");
        }
        out
    }
}

impl CortyxServer {
    /// Run `git diff --name-only` and return Level-0 (Purpose section only) capsules
    /// for neurons that correspond to recently changed files. Returns None if git is
    /// unavailable or no relevant neurons are found.
    pub(super) async fn build_git_prefetch_block(&self) -> Option<String> {
        use tokio::process::Command;

        let output = Command::new(crate::git_util::git_binary())
            .args(["diff", "--name-only", "HEAD"])
            .current_dir(&self.project_root)
            .output()
            .await
            .ok()?;

        if !output.status.success() || output.stdout.is_empty() {
            return None;
        }

        let changed_files: Vec<String> = String::from_utf8_lossy(&output.stdout)
            .lines()
            .map(|l| l.to_string())
            .filter(|l| !l.is_empty())
            .collect();

        if changed_files.is_empty() {
            return None;
        }

        // Map changed source files to their neuron counterparts:
        // e.g. "src/auth/mod.rs" → ".cortyx/neurons/src_auth_mod_rs.context.md"
        // The index stores the source path in each neuron's metadata; we use the
        // index's list to find neurons whose `source` field matches a changed file.
        let idx = self.index.read().await;
        let all_neurons = idx.list_neurons(None);

        let ndir = crate::neuron::neuron_dir(&self.project_root);
        let mut capsules = String::new();
        let mut count = 0;

        for neuron in &all_neurons {
            let neuron_rel = match neuron.path.strip_prefix(&ndir) {
                Ok(r) => r.to_string_lossy().to_string(),
                Err(_) => continue,
            };
            // Check if this neuron's source path matches any changed file
            let is_relevant = changed_files.iter().any(|changed| {
                // Neuron path is typically derived by replacing path separators with '_'
                // and appending ".context.md" — match either direction
                neuron_rel.contains(changed.replace('/', "_").trim_end_matches(".rs"))
                    || changed.contains(neuron_rel.trim_end_matches(".context.md"))
            });

            if !is_relevant {
                continue;
            }

            // Read Purpose section only (Level-0 capsule, ~80 tokens)
            let content = match std::fs::read_to_string(&neuron.path) {
                Ok(c) => c,
                Err(_) => continue,
            };
            let sections = crate::neuron::parse_sections(&content);
            if let Some(purpose) = sections.get("purpose") {
                capsules.push_str(&format!("### {neuron_rel}\n{purpose}\n\n"));
                count += 1;
                if count >= 5 {
                    break;
                }
            }
        }

        if capsules.is_empty() {
            None
        } else {
            Some(format!(
                "<!-- {count} neurons prefetched from git diff -->\n{capsules}"
            ))
        }
    }
}
