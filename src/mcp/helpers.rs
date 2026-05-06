use super::*;

pub(in crate::mcp) async fn flush_provisional_hits_async(
    _index: &Arc<RwLock<NeuronIndex>>,
    provisional_hits: &Arc<Mutex<Vec<PathBuf>>>,
) -> Result<usize> {
    let pending = {
        let mut prov = provisional_hits.lock().await;
        if prov.is_empty() {
            return Ok(0);
        }
        std::mem::take(&mut *prov)
    };
    Ok(pending.len())
}

pub(in crate::mcp) fn flush_provisional_hits_blocking(
    _index: &Arc<RwLock<NeuronIndex>>,
    provisional_hits: &Arc<Mutex<Vec<PathBuf>>>,
) -> Result<usize> {
    let pending = {
        let mut prov = provisional_hits.blocking_lock();
        if prov.is_empty() {
            return Ok(0);
        }
        std::mem::take(&mut *prov)
    };
    Ok(pending.len())
}

/// Clear provisional carry-over on unexpected process exit (STDIO EOF).
///
/// Explicit citation feedback must come from `close_task`, `record_hit`, or previous-response
/// evidence. `Drop` remains a last-resort cleanup for abnormal shutdown so this buffer does not
/// leak across clones.
impl Drop for CortyxServer {
    fn drop(&mut self) {
        // Only flush when this is the last CortyxServer instance.
        // CortyxServer derives Clone; rmcp may hold short-lived clones per request.
        if Arc::strong_count(&self.provisional_hits) > 1 {
            return;
        }
        if tokio::runtime::Handle::try_current().is_ok() {
            tracing::debug!(
                "S2: skipping blocking provisional buffer clear from async Drop context"
            );
            return;
        }
        match flush_provisional_hits_blocking(&self.index, &self.provisional_hits) {
            Ok(0) => {},
            Ok(n) => tracing::info!("S2: Drop cleared {n} provisional paths on exit"),
            Err(e) => tracing::warn!("S2: failed to clear provisional buffer during Drop: {e}"),
        }
    }
}

impl CortyxServer {
    #[allow(dead_code)]
    pub fn for_benchmark(project_root: PathBuf, idx: NeuronIndex) -> Self {
        let index = Arc::new(RwLock::new(idx));
        let provisional_hits = Arc::new(Mutex::new(Vec::new()));
        let context_sessions = Arc::new(Mutex::new(HashMap::new()));
        let next_context_handle = Arc::new(AtomicU64::new(0));

        Self {
            project_root,
            index,
            last_activated: Arc::new(Mutex::new(Vec::new())),
            provisional_hits,
            context_sessions,
            next_context_handle,
            tool_router: CortyxServer::tool_router(),
        }
    }

    #[allow(dead_code)]
    pub async fn benchmark_get_contexts(&self, input: GetContextsInput) -> String {
        self.get_contexts(Parameters(input)).await
    }

    #[allow(dead_code)]
    pub async fn benchmark_cortyx(&self, input: CortyxInput) -> String {
        self.cortyx(Parameters(input)).await
    }

    #[allow(dead_code)]
    pub async fn benchmark_collaboration_status(&self, input: CollaborationStatusInput) -> String {
        self.collaboration_status(Parameters(input)).await
    }

    /// Return a project-relative display string for `path`.
    ///
    /// Strips the project root prefix so absolute internal filesystem paths
    /// (including usernames) are never exposed to MCP clients.
    pub(in crate::mcp) fn rel_display<'a>(&self, path: &'a Path) -> std::borrow::Cow<'a, str> {
        path.strip_prefix(&self.project_root)
            .unwrap_or(path)
            .to_string_lossy()
    }

    pub(in crate::mcp) async fn render_cortyx_capability_summary(&self) -> String {
        let (neuron_count, synapse_count, projection) = {
            let idx = self.index.read().await;
            (
                idx.neuron_count(),
                idx.synapse_count(),
                build_collaboration_projection(&idx, &self.project_root),
            )
        };
        let sync_statuses = load_collaboration_sync_statuses(&self.project_root);
        let pending_sync = sync_statuses
            .iter()
            .filter(|status| status.pending_incoming() || status.pending_outgoing())
            .count();
        let conflict_count: usize = sync_statuses
            .iter()
            .map(SyncTransportStatus::conflict_count)
            .sum();
        let collaboration_line =
            if projection.collaborators.is_empty() && projection.modules.is_empty() {
                " - collaboration: no tracked agent or shared-module state yet\n".to_string()
            } else {
                format!(
                    " - collaboration: {} collaborator(s), {} shared module(s)\n",
                    projection.collaborators.len(),
                    projection.modules.len()
                )
            };

        format!(
            "Cortyx capability summary\n\
             Default entrypoint: cortyx(task=\"...\") for auto routing, or cortyx(intent=\"context\", task=\"...\") when you need cache-safe raw context.\n\
             Call cortyx() with no args any time to see this summary again.\n\n\
             Available routes:\n\
             - context — retrieve the highest-signal local/project neurons for a task\n\
             - answer — reuse the retrieval path but return a concise answer layer\n\
             - wake_up — prime a person/agent session from diary + collaboration state\n\
             - agent_status — summarize one agent's focus, goal, blocker, and next step\n\
             - consistency — check a path's temporal KG surface for contradictions\n\n\
             Project snapshot:\n\
             - neurons: {neuron_count}, synapses: {synapse_count}\n\
             {collaboration_line}\
             - shared sync: {pending_sync} pending item(s), {conflict_count} conflict(s)\n\n\
             Examples:\n\
             - cortyx(task=\"trace the auth flow\")\n\
             - cortyx(intent=\"answer\", task=\"What is the reviewer's blocker?\", agent=\"reviewer\")\n\
             - cortyx(intent=\"wake_up\", agent=\"reviewer\")\n\
             - cortyx(intent=\"consistency\", path=\"src/auth.rs\")"
        )
    }

    pub(in crate::mcp) async fn ensure_context_handle(&self, requested: Option<&str>) -> String {
        requested
            .filter(|handle| !handle.trim().is_empty())
            .map(ToOwned::to_owned)
            .unwrap_or_else(|| {
                let next = self.next_context_handle.fetch_add(1, Ordering::Relaxed) + 1;
                format!("ctx-{next}")
            })
    }

    pub(in crate::mcp) async fn load_context_snapshot(
        &self,
        handle: &str,
    ) -> Option<ContextSnapshot> {
        self.context_sessions.lock().await.get(handle).cloned()
    }

    pub(in crate::mcp) async fn store_context_snapshot(
        &self,
        handle: String,
        chunks: &[RenderedContextItem],
        overflow: &[RenderedContextItem],
    ) {
        let order = self.next_context_handle.fetch_add(1, Ordering::Relaxed) + 1;
        let snapshot = ContextSnapshot {
            order,
            chunks: chunks
                .iter()
                .map(|item| (item.path.clone(), item.fingerprint.clone()))
                .collect(),
            overflow: overflow
                .iter()
                .map(|item| (item.path.clone(), item.fingerprint.clone()))
                .collect(),
        };

        let mut sessions = self.context_sessions.lock().await;
        if sessions.len() >= 128 {
            if let Some(oldest) = sessions
                .iter()
                .min_by_key(|(_, snapshot)| snapshot.order)
                .map(|(handle, _)| handle.clone())
            {
                sessions.remove(&oldest);
            }
        }
        sessions.insert(handle, snapshot);
    }
}

// ─── Helpers ─────────────────────────────────────────────────────────────────/// Convert a task pattern string to a URL-safe kebab-case identifier.
pub(super) fn to_kebab(s: &str) -> String {
    s.to_lowercase()
        .split(|c: char| !c.is_alphanumeric())
        .filter(|s: &&str| !s.is_empty())
        .collect::<Vec<_>>()
        .join("-")
}

/// Truncate a string to at most `max_chars` characters (byte boundary safe for ASCII).
pub(super) fn truncate_str(s: &str, max_chars: usize) -> String {
    s.chars().take(max_chars).collect()
}

pub(super) fn recent_module_paths(
    index: &NeuronIndex,
    module: &str,
    limit: usize,
    kind_filter: Option<NeuronKind>,
) -> Vec<PathBuf> {
    let mut items: Vec<(i64, PathBuf)> = index
        .list_neurons(Some(module))
        .into_iter()
        .filter(|summary| {
            kind_filter
                .as_ref()
                .map(|kind| summary.kind == *kind)
                .unwrap_or(true)
        })
        .map(|summary| {
            let timestamp = index
                .context_metadata_for(&summary.path)
                .and_then(|metadata| metadata.timestamp_secs)
                .unwrap_or(i64::MIN);
            (timestamp, summary.path)
        })
        .collect();
    items.sort_by(|a, b| b.0.cmp(&a.0).then_with(|| a.1.cmp(&b.1)));
    items
        .into_iter()
        .take(limit)
        .map(|(_, path)| path)
        .collect()
}

pub(super) const DEFAULT_COLLAB_SUMMARY_LIMIT: usize = 5;
pub(super) const DEFAULT_COLLAB_TIMELINE_LIMIT: usize = 8;

pub(super) fn build_collaboration_projection(
    index: &NeuronIndex,
    project_root: &Path,
) -> CollaborationStateProjection {
    let diaries = collect_collaboration_diaries(index);
    let sync_statuses = load_collaboration_sync_statuses(project_root);
    let kg_entities = load_collaboration_kg_entities(project_root);
    let reasoning = build_collaboration_reasoning(index, project_root, &diaries, &sync_statuses);
    project_collaboration_state(&diaries, &sync_statuses, &kg_entities, reasoning.as_ref())
}

pub(super) fn collect_collaboration_diaries(index: &NeuronIndex) -> Vec<CollaborationDiaryRecord> {
    let mut diaries = Vec::new();
    for summary in index.list_neurons(None) {
        if summary.kind != NeuronKind::Verbatim {
            continue;
        }
        let Some(module) = index.module_for(&summary.path) else {
            continue;
        };
        let Some(collaborator) = module.strip_prefix("@agent/") else {
            continue;
        };
        let collaborator = collaborator.trim();
        if collaborator.is_empty() {
            continue;
        }
        let Ok(content) = std::fs::read_to_string(&summary.path) else {
            continue;
        };
        let Some(entry) = parse_structured_diary_entry(&content) else {
            continue;
        };
        let when = index
            .context_metadata_for(&summary.path)
            .and_then(|metadata| metadata.timestamp_secs)
            .map(format_timestamp_secs);
        diaries.push(CollaborationDiaryRecord {
            collaborator: collaborator.to_string(),
            when,
            path: Some(summary.path.clone()),
            entry,
        });
    }
    diaries.sort_by(|left, right| {
        right
            .when
            .cmp(&left.when)
            .then_with(|| left.collaborator.cmp(&right.collaborator))
            .then_with(|| left.path.cmp(&right.path))
    });
    diaries
}

pub(super) fn load_collaboration_sync_statuses(project_root: &Path) -> Vec<SyncTransportStatus> {
    let sync_root = sync_transport_dir(project_root);
    if !sync_root.exists() {
        return Vec::new();
    }
    SyncTransportRepository::for_project(project_root)
        .list_statuses()
        .unwrap_or_default()
}

pub(super) fn load_collaboration_kg_entities(project_root: &Path) -> Vec<kg::KgEntity> {
    kg::list_kg_paths(project_root)
        .into_iter()
        .filter_map(|path| kg::KgEntity::load(&path).ok())
        .collect()
}

pub(super) fn build_collaboration_reasoning(
    index: &NeuronIndex,
    project_root: &Path,
    diaries: &[CollaborationDiaryRecord],
    sync_statuses: &[SyncTransportStatus],
) -> Option<ReasoningReport> {
    let mut seeds = HashMap::new();
    for diary in diaries {
        if let Some(path) = &diary.path {
            merge_collaboration_seed(&mut seeds, path.clone(), 0.6);
        }
        for entity in diary
            .entry
            .entities
            .iter()
            .chain(diary.entry.depends_on.iter())
        {
            let kg_path = kg::kg_neuron_path(project_root, entity);
            if kg_path.exists() {
                merge_collaboration_seed(&mut seeds, kg_path, 0.8);
            }
        }
    }
    for status in sync_statuses {
        if let Some(source_path) = status.source_path() {
            let source = if source_path.is_absolute() {
                source_path.to_path_buf()
            } else {
                project_root.join(source_path)
            };
            if source.starts_with(project_root) {
                let neuron_path = core_neuron_path(&source, project_root);
                if neuron_path.exists() {
                    merge_collaboration_seed(&mut seeds, neuron_path, 1.0);
                }
            }
        }
        if let Some(module) = status.module() {
            let kg_path = kg::kg_neuron_path(project_root, module);
            if kg_path.exists() {
                merge_collaboration_seed(&mut seeds, kg_path, 0.7);
            }
        }
    }
    if seeds.is_empty() {
        return None;
    }

    let report = index.reason_over_paths(
        &seeds.into_iter().collect::<Vec<_>>(),
        TraversalOptions {
            max_hops: 1,
            max_expansions: 32,
            ..TraversalOptions::default()
        },
    );
    (!report.nodes.is_empty() || !report.facts.is_empty() || !report.conflicts.is_empty())
        .then_some(report)
}

pub(super) fn merge_collaboration_seed(
    seeds: &mut HashMap<PathBuf, f32>,
    path: PathBuf,
    score: f32,
) {
    seeds
        .entry(path)
        .and_modify(|existing| *existing = existing.max(score))
        .or_insert(score);
}

/// Load existing metadata or create a fresh stub.
pub(super) fn load_or_new_meta(meta_file: &Path, source: &Path, kind: NeuronKind) -> NeuronMeta {
    if let Ok(data) = std::fs::read_to_string(meta_file) {
        if let Ok(meta) = serde_json::from_str::<NeuronMeta>(&data) {
            return meta;
        }
    }
    NeuronMeta::new_stub(source, kind)
}

/// Serialize and write metadata to disk atomically.
pub(super) fn save_meta(meta_file: &Path, meta: &NeuronMeta) -> Result<()> {
    Ok(atomic_write_json(meta_file, meta)?)
}

pub(super) fn refresh_meta_after_content_write(meta: &mut NeuronMeta, content: &str) {
    if let Some(source_hash) = hash_file(&meta.source_path) {
        meta.source_hash = source_hash;
    }
    meta.tokens = estimate_context_tokens(content).get();
    meta.last_updated = now_iso8601();
    meta.status = NeuronStatus::Fresh;
    meta.synapses = parse_synapses_from_content(content);
}

pub(super) fn record_mutation_provenance(
    neuron_path: &Path,
    meta: &NeuronMeta,
    content: &str,
    operation: ProvenanceOperation,
    source: ProvenanceSource,
    section: Option<String>,
    summary: Option<String>,
) -> Result<()> {
    Ok(record_content_provenance_edit(
        neuron_path,
        meta,
        content,
        ProvenanceEdit {
            operation,
            source,
            section,
            summary,
            ..Default::default()
        },
    )
    .map(|_| ())?)
}

pub(super) fn finalize_mutation_message(message: String, provenance_result: Result<()>) -> String {
    match provenance_result {
        Ok(()) => message,
        Err(err) => format!("{message}\nWARNING: Failed to record provenance: {err}"),
    }
}

pub(super) fn resolve_neuron_store_path(raw_path: &str, project_root: &Path) -> Result<PathBuf> {
    let neuron_root = neuron_dir(project_root)
        .canonicalize()
        .map_err(|err| anyhow::anyhow!("cannot access neuron directory: {err}"))?;
    let candidate = Path::new(raw_path);
    let resolved = if candidate.is_absolute() {
        candidate.to_path_buf()
    } else {
        neuron_root.join(candidate)
    };
    let canonical = resolved
        .canonicalize()
        .map_err(|err| anyhow::anyhow!("cannot access neuron path: {err}"))?;
    if !canonical.starts_with(&neuron_root) {
        anyhow::bail!(
            "path {} is outside neuron directory {}",
            canonical.display(),
            neuron_root.display()
        );
    }
    if canonical.is_dir() {
        anyhow::bail!(
            "path {} is a directory, not a neuron file",
            canonical.display()
        );
    }
    Ok(canonical)
}

pub(super) fn build_augmented_task(index: &NeuronIndex, input: &GetContextsInput) -> String {
    let mut extra = String::new();

    if let Some(ref open_files) = input.open_files {
        if !open_files.is_empty() {
            let soft = index.soft_terms_for_editor_context(open_files, 8);
            if !soft.is_empty() {
                extra.push(' ');
                extra.push_str(&soft.join(" "));
                tracing::debug!(
                    files = open_files.len(),
                    soft_terms = soft.len(),
                    "S-V: editor context injected"
                );
            }
        }
    }

    if let Some(ref err_ctx) = input.error_context {
        if !err_ctx.is_empty() {
            let err_terms = tokenize(err_ctx);
            if !err_terms.is_empty() {
                extra.push(' ');
                extra.push_str(&err_terms.join(" "));
                tracing::debug!(err_terms = err_terms.len(), "S-V: error_context injected");
            }
        }
    }

    if extra.is_empty() {
        input.task.clone()
    } else {
        format!("{}{}", input.task, extra)
    }
}

pub(super) fn index_kg_entity_path(index: &mut NeuronIndex, path: &Path) -> Result<()> {
    let content = std::fs::read_to_string(path)
        .map_err(|err| anyhow::anyhow!("reload KG entity {}: {err}", path.display()))?;
    let mut meta = NeuronMeta::new_stub(path, NeuronKind::Concept);
    meta.module = Some("@kg".to_string());
    meta.tokens = estimate_context_tokens(&content).get();
    index.index_neuron(path, &content, &meta);
    Ok(())
}

pub(super) fn sync_structured_diary_to_kg(
    project_root: &Path,
    index: &mut NeuronIndex,
    agent: &str,
    entry: &StructuredDiaryEntry,
    effective_from: &str,
) -> Result<()> {
    let path = kg::kg_neuron_path(project_root, &agent_entity_name(agent));
    let mut entity = kg::KgEntity::load(&path)?;
    entity.replace_active_fact(
        AGENT_FOCUS_PREDICATE,
        entry.title.as_deref().unwrap_or(""),
        effective_from,
    );
    entity.replace_active_fact(
        AGENT_STATUS_PREDICATE,
        entry.status.as_deref().unwrap_or(""),
        effective_from,
    );
    entity.replace_active_fact(
        AGENT_GOAL_PREDICATE,
        entry.goal.as_deref().unwrap_or(""),
        effective_from,
    );
    entity.replace_active_fact(
        AGENT_NEXT_STEP_PREDICATE,
        entry.next_step.as_deref().unwrap_or(""),
        effective_from,
    );
    entity.replace_active_fact(
        AGENT_BLOCKER_PREDICATE,
        entry.blocker.as_deref().unwrap_or(""),
        effective_from,
    );
    entity.replace_active_fact(
        AGENT_OUTCOME_PREDICATE,
        entry.outcome.as_deref().unwrap_or(""),
        effective_from,
    );
    entity.replace_active_fact(
        AGENT_ACTION_PREDICATE,
        entry.action.as_deref().unwrap_or(""),
        effective_from,
    );
    entity.sync_active_values(
        AGENT_RELATED_ENTITY_PREDICATE,
        &entry.entities,
        effective_from,
    );
    entity.sync_active_values(
        AGENT_DEPENDS_ON_PREDICATE,
        &entry.depends_on,
        effective_from,
    );
    entity.save()?;
    index_kg_entity_path(index, &path)
}

/// Strip HTML comment delimiters and control characters from user-supplied strings
/// before embedding them in comment blocks, preventing comment breakout and prompt injection.
pub(super) fn sanitize_comment(s: &str) -> String {
    let clean: String = s
        .chars()
        .map(|c| {
            if c.is_ascii_control() && c != '\t' {
                ' '
            } else {
                c
            }
        })
        .collect();
    let clean = clean.replace("-->", "—>").replace("<!--", "<—");
    // Truncate to 500 chars to prevent unbounded comment sections
    clean.chars().take(500).collect()
}

pub(super) fn render_recent_agent_memory_block(
    index: &NeuronIndex,
    agent: &str,
    limit: usize,
) -> Option<String> {
    let module = format!("@agent/{}", agent.trim());
    let paths = recent_module_paths(index, &module, limit, Some(NeuronKind::Verbatim));
    if paths.is_empty() {
        return None;
    }

    let mut out = format!(
        "<!-- CORTYX WAKE-UP: @agent/{} memories -->\n",
        agent.trim()
    );
    for path in paths {
        let timestamp_secs = index
            .context_metadata_for(&path)
            .and_then(|metadata| metadata.timestamp_secs);
        match std::fs::read_to_string(&path) {
            Ok(content) => {
                out.push_str(&render_agent_memory_summary(&content, timestamp_secs));
                out.push('\n');
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
    Some(out)
}

pub(super) fn render_agent_memory_summary(content: &str, timestamp_secs: Option<i64>) -> String {
    let timestamp = timestamp_secs
        .map(format_timestamp_secs)
        .unwrap_or_else(|| "unknown-time".to_string());
    if let Some(entry) = parse_structured_diary_entry(content) {
        format!(
            "- {timestamp} — {}",
            summarize_structured_diary_entry(&entry)
        )
    } else {
        format!(
            "- {timestamp} — {}",
            truncate_str(&summarize_plain_diary_content(content), 180)
        )
    }
}

pub(super) fn render_structured_diary_history_entry(
    entry: &crate::agent_memory::StructuredDiaryEntry,
    timestamp_secs: Option<i64>,
) -> String {
    let timestamp = timestamp_secs
        .map(format_timestamp_secs)
        .unwrap_or_else(|| "unknown-time".to_string());
    let mut out = format!(
        "- {timestamp} — {}",
        summarize_structured_diary_entry(entry)
    );
    if let Some(action) = &entry.action {
        out.push_str(&format!(
            "\n  action: {}",
            truncate_str(&summarize_plain_diary_content(action), 200)
        ));
    }
    if let Some(goal) = &entry.goal {
        out.push_str(&format!("\n  goal: {}", truncate_str(goal, 200)));
    }
    if let Some(next_step) = &entry.next_step {
        out.push_str(&format!("\n  next step: {}", truncate_str(next_step, 200)));
    }
    if let Some(blocker) = &entry.blocker {
        out.push_str(&format!("\n  blocker: {}", truncate_str(blocker, 200)));
    }
    if let Some(outcome) = &entry.outcome {
        out.push_str(&format!(
            "\n  outcome: {}",
            truncate_str(&summarize_plain_diary_content(outcome), 200)
        ));
    }
    if !entry.depends_on.is_empty() {
        out.push_str(&format!("\n  depends on: {}", entry.depends_on.join(", ")));
    }
    out.push('\n');
    out
}

pub(super) fn render_agent_status_report(
    index: &NeuronIndex,
    project_root: &Path,
    agent: &str,
    include_timeline: bool,
) -> Option<String> {
    let projection = build_collaboration_projection(index, project_root);
    let summary = projection
        .collaborators
        .iter()
        .find(|summary| matches_collaboration_filter(&summary.collaborator, agent))?;
    Some(render_collaborator_status_report(
        summary,
        &projection,
        include_timeline,
    ))
}

pub(super) fn render_collaboration_status_report(
    projection: &CollaborationStateProjection,
    agent: Option<&str>,
    module: Option<&str>,
    include_timeline: bool,
) -> Option<String> {
    if let Some(agent) = agent.filter(|agent| !agent.trim().is_empty()) {
        let summary = projection
            .collaborators
            .iter()
            .find(|summary| matches_collaboration_filter(&summary.collaborator, agent))?;
        return Some(render_collaborator_status_report(
            summary,
            projection,
            include_timeline,
        ));
    }
    if let Some(module) = module.filter(|module| !module.trim().is_empty()) {
        let state = projection
            .modules
            .iter()
            .find(|state| matches_collaboration_filter(&state.module, module))?;
        return Some(render_module_status_report(
            state,
            projection,
            include_timeline,
        ));
    }
    if projection.collaborators.is_empty()
        && projection.modules.is_empty()
        && projection.timeline.is_empty()
    {
        return None;
    }
    Some(render_project_collaboration_status(
        projection,
        include_timeline,
    ))
}

pub(super) fn render_project_collaboration_status(
    projection: &CollaborationStateProjection,
    include_timeline: bool,
) -> String {
    let latest_activity = projection
        .timeline
        .iter()
        .find_map(|event| event.at.clone())
        .or_else(|| {
            projection
                .collaborators
                .iter()
                .filter_map(|summary| summary.last_updated.clone())
                .max()
        })
        .or_else(|| {
            projection
                .modules
                .iter()
                .filter_map(|state| state.last_updated.clone())
                .max()
        });
    let pending_sync_count = projection
        .timeline
        .iter()
        .filter(|event| matches!(event.kind, CollaborationTimelineKind::Sync))
        .count();
    let conflict_count = projection
        .timeline
        .iter()
        .filter(|event| matches!(event.kind, CollaborationTimelineKind::Conflict))
        .count();
    let trust_scores = projection
        .collaborators
        .iter()
        .filter_map(|summary| summary.trust_score)
        .chain(
            projection
                .modules
                .iter()
                .filter_map(|state| state.trust_score),
        )
        .collect::<Vec<_>>();
    let average_trust_score = (!trust_scores.is_empty())
        .then_some(trust_scores.iter().sum::<f32>() / trust_scores.len() as f32);
    let handoff_risk_count = projection
        .collaborators
        .iter()
        .map(|summary| summary.evidence.untrusted_handoff_count)
        .sum::<usize>();
    let integrity_issue_count = projection
        .collaborators
        .iter()
        .map(|summary| summary.evidence.integrity_issue_count)
        .sum::<usize>();
    let top_attention = match (projection.collaborators.first(), projection.modules.first()) {
        (Some(collaborator), Some(module))
            if module.attention_score > collaborator.attention_score =>
        {
            module.attention
        },
        (Some(collaborator), _) => collaborator.attention,
        (None, Some(module)) => module.attention,
        (None, None) => CollaborationAttention::Nominal,
    };

    let mut out = "## Collaboration Status\n\n".to_string();
    out.push_str(&format!(
        "- collaborators: {}\n",
        projection.collaborators.len()
    ));
    out.push_str(&format!("- modules: {}\n", projection.modules.len()));
    if let Some(latest_activity) = latest_activity {
        out.push_str(&format!("- latest activity: {latest_activity}\n"));
    }
    if top_attention != CollaborationAttention::Nominal {
        out.push_str(&format!(
            "- attention: {}\n",
            format_collaboration_attention(top_attention)
        ));
    }
    out.push_str(&format!("- pending sync items: {pending_sync_count}\n"));
    out.push_str(&format!("- sync conflicts: {conflict_count}\n"));
    if let Some(trust_score) = average_trust_score {
        out.push_str(&format!(
            "- average trust score: {}\n",
            format_trust_score(trust_score)
        ));
    }
    if handoff_risk_count > 0 {
        out.push_str(&format!("- handoff risks: {handoff_risk_count}\n"));
    }
    if integrity_issue_count > 0 {
        out.push_str(&format!("- integrity issues: {integrity_issue_count}\n"));
    }

    if !projection.collaborators.is_empty() {
        out.push_str("\n## Top collaborators\n");
        for summary in projection
            .collaborators
            .iter()
            .take(DEFAULT_COLLAB_SUMMARY_LIMIT)
        {
            out.push_str(&format!("- {}\n", render_collaborator_brief(summary)));
        }
    }
    if !projection.modules.is_empty() {
        out.push_str("\n## Shared modules\n");
        for state in projection.modules.iter().take(DEFAULT_COLLAB_SUMMARY_LIMIT) {
            out.push_str(&format!("- {}\n", render_module_brief(state)));
        }
    }
    if include_timeline {
        append_collaboration_timeline(&mut out, &projection.timeline, None, None);
    }
    out
}

pub(super) fn render_collaborator_status_report(
    summary: &CollaboratorSummary,
    projection: &CollaborationStateProjection,
    include_timeline: bool,
) -> String {
    let mut out = format!("## Agent Status: {}\n\n", summary.collaborator);
    if let Some(updated) = &summary.last_updated {
        out.push_str(&format!("- updated: {updated}\n"));
    }
    out.push_str(&format!(
        "- attention: {}\n",
        format_collaboration_attention(summary.attention)
    ));
    if let Some(focus) = &summary.focus {
        out.push_str(&format!("- focus: {focus}\n"));
    }
    if let Some(status) = &summary.status {
        out.push_str(&format!("- status: {status}\n"));
    }
    if let Some(goal) = &summary.goal {
        out.push_str(&format!("- goal: {goal}\n"));
    }
    if let Some(next_step) = &summary.next_step {
        out.push_str(&format!("- next step: {next_step}\n"));
    }
    if let Some(blocker) = &summary.blocker {
        out.push_str(&format!("- blocker: {blocker}\n"));
    }
    if let Some(action) = &summary.action {
        out.push_str(&format!(
            "- action: {}\n",
            truncate_str(&summarize_plain_diary_content(action), 220)
        ));
    }
    if let Some(outcome) = &summary.outcome {
        out.push_str(&format!(
            "- outcome: {}\n",
            truncate_str(&summarize_plain_diary_content(outcome), 220)
        ));
    }
    if !summary.related_entities.is_empty() {
        out.push_str(&format!(
            "- related entities: {}\n",
            summary.related_entities.join(", ")
        ));
    }
    if !summary.depends_on.is_empty() {
        out.push_str(&format!(
            "- depends on: {}\n",
            summary.depends_on.join(", ")
        ));
    }
    if !summary.touched_modules.is_empty() {
        out.push_str(&format!(
            "- touched modules: {}\n",
            summary.touched_modules.join(", ")
        ));
    }
    out.push_str(&format!(
        "- pending sync: {}\n",
        if summary.pending_sync { "yes" } else { "no" }
    ));
    if let Some(trust_score) = summary.trust_score {
        out.push_str(&format!(
            "- trust score: {}\n",
            format_trust_score(trust_score)
        ));
    }
    let evidence = format_collaboration_evidence(&summary.evidence);
    if !evidence.is_empty() {
        out.push_str(&format!("- evidence: {evidence}\n"));
    }
    out.push_str(&format!(
        "- sources: @agent/{} diary, @kg/{}\n",
        summary.collaborator, summary.entity
    ));

    append_supporting_facts(&mut out, &summary.supporting_facts);

    let related_modules: Vec<&ModuleCollaborationState> = projection
        .modules
        .iter()
        .filter(|state| {
            summary
                .touched_modules
                .iter()
                .any(|module| matches_collaboration_filter(&state.module, module))
        })
        .take(3)
        .collect();
    if !related_modules.is_empty() {
        out.push_str("\n## Shared modules\n");
        for state in related_modules {
            out.push_str(&format!("- {}\n", render_module_brief(state)));
        }
    }

    if include_timeline {
        append_collaboration_timeline(
            &mut out,
            &projection.timeline,
            Some(&summary.collaborator),
            None,
        );
    }
    out
}

pub(super) fn render_module_status_report(
    state: &ModuleCollaborationState,
    projection: &CollaborationStateProjection,
    include_timeline: bool,
) -> String {
    let mut out = format!("## Collaboration Module: {}\n\n", state.module);
    if let Some(updated) = &state.last_updated {
        out.push_str(&format!("- updated: {updated}\n"));
    }
    out.push_str(&format!(
        "- attention: {}\n",
        format_collaboration_attention(state.attention)
    ));
    if !state.collaborators.is_empty() {
        out.push_str(&format!(
            "- collaborators: {}\n",
            state.collaborators.join(", ")
        ));
    }
    if !state.focuses.is_empty() {
        out.push_str(&format!("- focuses: {}\n", state.focuses.join(", ")));
    }
    if !state.blockers.is_empty() {
        out.push_str(&format!("- blockers: {}\n", state.blockers.join(", ")));
    }
    if !state.related_entities.is_empty() {
        out.push_str(&format!(
            "- related entities: {}\n",
            state.related_entities.join(", ")
        ));
    }
    out.push_str(&format!(
        "- pending sync: {}\n",
        if state.pending_sync { "yes" } else { "no" }
    ));
    if let Some(trust_score) = state.trust_score {
        out.push_str(&format!(
            "- trust score: {}\n",
            format_trust_score(trust_score)
        ));
    }
    if let Some(edit_id) = &state.latest_sync_edit_id {
        out.push_str(&format!("- latest sync edit: {edit_id}\n"));
    }
    let evidence = format_collaboration_evidence(&state.evidence);
    if !evidence.is_empty() {
        out.push_str(&format!("- evidence: {evidence}\n"));
    }

    append_supporting_facts(&mut out, &state.supporting_facts);

    let matching_collaborators: Vec<&CollaboratorSummary> = projection
        .collaborators
        .iter()
        .filter(|summary| {
            summary
                .touched_modules
                .iter()
                .any(|module| matches_collaboration_filter(module, &state.module))
        })
        .take(3)
        .collect();
    if !matching_collaborators.is_empty() {
        out.push_str("\n## Collaborators\n");
        for summary in matching_collaborators {
            out.push_str(&format!("- {}\n", render_collaborator_brief(summary)));
        }
    }

    if include_timeline {
        append_collaboration_timeline(&mut out, &projection.timeline, None, Some(&state.module));
    }
    out
}

pub(super) fn summarize_plain_diary_content(content: &str) -> String {
    content
        .lines()
        .map(str::trim)
        .find(|line| !line.is_empty() && !line.starts_with("<!--") && !line.starts_with('#'))
        .unwrap_or("(empty diary entry)")
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

pub(super) fn format_timestamp_secs(timestamp_secs: i64) -> String {
    if timestamp_secs < 0 {
        return timestamp_secs.to_string();
    }
    let (y, mo, d, h, mi, s) = unix_secs_to_datetime(timestamp_secs as u64);
    format!("{y:04}-{mo:02}-{d:02}T{h:02}:{mi:02}:{s:02}Z")
}

#[allow(dead_code)]
pub(super) fn latest_active_kg_value(entity: &kg::KgEntity, predicate: &str) -> Option<String> {
    entity
        .active_values_for_predicate(predicate, None)
        .last()
        .map(|fact| fact.value.clone())
}

#[allow(dead_code)]
pub(super) fn active_kg_values(entity: &kg::KgEntity, predicate: &str) -> Vec<String> {
    entity
        .active_values_for_predicate(predicate, None)
        .into_iter()
        .map(|fact| fact.value.clone())
        .collect()
}

pub(super) fn render_collaborator_brief(summary: &CollaboratorSummary) -> String {
    let mut details = vec![format_collaboration_attention(summary.attention).to_string()];
    if let Some(focus) = &summary.focus {
        details.push(format!("focus: {}", truncate_str(focus, 80)));
    }
    if !summary.touched_modules.is_empty() {
        details.push(format!("modules: {}", summary.touched_modules.join(", ")));
    }
    if summary.pending_sync {
        details.push("pending sync".to_string());
    }
    if summary.evidence.conflict_count > 0 {
        details.push(format!("conflicts: {}", summary.evidence.conflict_count));
    }
    if summary.evidence.integrity_issue_count > 0 {
        details.push(format!(
            "integrity issues: {}",
            summary.evidence.integrity_issue_count
        ));
    }
    if let Some(trust_score) = summary.trust_score {
        details.push(format!("trust: {}", format_trust_score(trust_score)));
    }
    format!("{} — {}", summary.collaborator, details.join("; "))
}

pub(super) fn render_module_brief(state: &ModuleCollaborationState) -> String {
    let mut details = vec![format_collaboration_attention(state.attention).to_string()];
    if !state.collaborators.is_empty() {
        details.push(format!("collaborators: {}", state.collaborators.join(", ")));
    }
    if state.pending_sync {
        details.push("pending sync".to_string());
    }
    if state.evidence.conflict_count > 0 {
        details.push(format!("conflicts: {}", state.evidence.conflict_count));
    }
    if state.evidence.integrity_issue_count > 0 {
        details.push(format!(
            "integrity issues: {}",
            state.evidence.integrity_issue_count
        ));
    }
    if let Some(trust_score) = state.trust_score {
        details.push(format!("trust: {}", format_trust_score(trust_score)));
    }
    format!("{} — {}", state.module, details.join("; "))
}

pub(super) fn append_supporting_facts(out: &mut String, facts: &[String]) {
    if facts.is_empty() {
        return;
    }
    out.push_str("\n## Supporting facts\n");
    for fact in facts.iter().take(6) {
        out.push_str(&format!("- {fact}\n"));
    }
}

pub(super) fn append_collaboration_timeline(
    out: &mut String,
    timeline: &[CollaborationTimelineEvent],
    collaborator: Option<&str>,
    module: Option<&str>,
) {
    let mut filtered = timeline
        .iter()
        .filter(|event| {
            collaborator
                .map(|target| {
                    event
                        .collaborator
                        .as_deref()
                        .map(|value| matches_collaboration_filter(value, target))
                        .unwrap_or(false)
                })
                .unwrap_or(true)
                && module
                    .map(|target| {
                        event
                            .module
                            .as_deref()
                            .map(|value| matches_collaboration_filter(value, target))
                            .unwrap_or(false)
                    })
                    .unwrap_or(true)
        })
        .take(DEFAULT_COLLAB_TIMELINE_LIMIT)
        .peekable();
    if filtered.peek().is_none() {
        return;
    }
    out.push_str("\n## Collaboration timeline\n");
    for event in filtered {
        out.push_str(&format!(
            "- {}\n",
            render_collaboration_timeline_event(event)
        ));
    }
}

pub(super) fn render_collaboration_timeline_event(event: &CollaborationTimelineEvent) -> String {
    let at = event.at.as_deref().unwrap_or("unknown-time");
    let mut scope = Vec::new();
    if let Some(collaborator) = &event.collaborator {
        scope.push(collaborator.as_str());
    }
    if let Some(module) = &event.module {
        scope.push(module.as_str());
    }
    let prefix = if scope.is_empty() {
        String::new()
    } else {
        format!("{} — ", scope.join(" / "))
    };
    let mut rendered = format!("{at} — {prefix}{}", event.summary);
    if event.attention != CollaborationAttention::Nominal {
        rendered.push_str(&format!(
            " ({})",
            format_collaboration_attention(event.attention)
        ));
    }
    rendered
}

pub(super) fn format_collaboration_attention(attention: CollaborationAttention) -> &'static str {
    match attention {
        CollaborationAttention::Nominal => "nominal",
        CollaborationAttention::NeedsFollowUp => "needs_follow_up",
        CollaborationAttention::Blocked => "blocked",
        CollaborationAttention::SyncConflict => "sync_conflict",
    }
}

pub(super) fn format_collaboration_evidence(evidence: &CollaborationEvidenceSummary) -> String {
    let mut parts = Vec::new();
    if evidence.diary_entries > 0 {
        parts.push(format!("diary={}", evidence.diary_entries));
    }
    if evidence.kg_fact_count > 0 {
        parts.push(format!("kg={}", evidence.kg_fact_count));
    }
    if evidence.sync_neuron_count > 0 {
        parts.push(format!("sync={}", evidence.sync_neuron_count));
    }
    if evidence.verified_sync_count > 0 {
        parts.push(format!("verified_sync={}", evidence.verified_sync_count));
    }
    if evidence.pending_sync_count > 0 {
        parts.push(format!("pending_sync={}", evidence.pending_sync_count));
    }
    if evidence.conflict_count > 0 {
        parts.push(format!("conflicts={}", evidence.conflict_count));
    }
    if evidence.integrity_issue_count > 0 {
        parts.push(format!(
            "integrity_issues={}",
            evidence.integrity_issue_count
        ));
    }
    if evidence.untrusted_handoff_count > 0 {
        parts.push(format!(
            "handoff_risks={}",
            evidence.untrusted_handoff_count
        ));
    }
    if evidence.reasoning_node_count > 0 {
        parts.push(format!("reasoning_nodes={}", evidence.reasoning_node_count));
    }
    if evidence.reasoning_fact_count > 0 {
        parts.push(format!("reasoning_facts={}", evidence.reasoning_fact_count));
    }
    parts.join(", ")
}

pub(super) fn format_trust_score(score: f32) -> String {
    format!("{score:.1}")
}

pub(super) fn matches_collaboration_filter(value: &str, filter: &str) -> bool {
    kg::slugify(value) == kg::slugify(filter)
}

pub(super) fn fingerprint_rendered_context(rendered: &str) -> String {
    blake3::hash(rendered.as_bytes()).to_hex()[..16].to_string()
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum EmissionTier {
    Full,
    Focused,
    Summary,
}

pub(super) fn render_context_item(
    path: &Path,
    score: f32,
    task_terms: &[String],
    index: &NeuronIndex,
) -> RenderedContextItem {
    let rendered = match std::fs::read_to_string(path) {
        Ok(content) => {
            let content = strip_render_only_sections(&content);
            match select_emission_tier(score, &content) {
                EmissionTier::Full => format!(
                    "<!-- === NEURON: {} === -->\n{}\n\n",
                    path.display(),
                    content
                ),
                EmissionTier::Focused => {
                    let focused = build_focused_context(&content, task_terms);
                    format!(
                        "<!-- === NEURON (focused, score={:.1}): {} === -->\n{}\n\n",
                        score,
                        path.file_name().unwrap_or_default().to_string_lossy(),
                        focused
                    )
                },
                EmissionTier::Summary => {
                    let summary = index
                        .summary_for(path)
                        .filter(|s| !s.is_empty())
                        .map(|s| s.to_string())
                        .unwrap_or_else(|| {
                            fallback_excerpt(&content, 3)
                                .lines()
                                .take(3)
                                .collect::<Vec<_>>()
                                .join("\n")
                        });
                    format!(
                        "<!-- === NEURON (summary, score={:.1}): {} === -->\n{}\n\n",
                        score,
                        path.file_name().unwrap_or_default().to_string_lossy(),
                        sanitize_comment(&summary),
                    )
                },
            }
        },
        Err(err) => {
            if score >= 5.0 {
                format!("<!-- NEURON {} — read error: {err} -->\n\n", path.display())
            } else {
                tracing::warn!(
                    "Failed to read {} while building summary context: {}",
                    path.display(),
                    err
                );
                format!(
                    "<!-- === NEURON (summary, score={:.1}): {} === -->\n{}\n\n",
                    score,
                    path.file_name().unwrap_or_default().to_string_lossy(),
                    sanitize_comment(&format!("(read error: {err})")),
                )
            }
        },
    };

    RenderedContextItem {
        path: path.to_path_buf(),
        fingerprint: fingerprint_rendered_context(&rendered),
        rendered,
    }
}

pub(super) fn select_emission_tier(score: f32, content: &str) -> EmissionTier {
    let tokens = estimate_context_tokens(content).get();
    if score < 5.0 {
        EmissionTier::Summary
    } else if score >= 9.0 || tokens <= 160 {
        EmissionTier::Full
    } else {
        EmissionTier::Focused
    }
}

pub(super) fn build_focused_context(content: &str, task_terms: &[String]) -> String {
    if let Some(sectioned) = render_focused_sections(content, task_terms) {
        return sectioned;
    }
    render_focused_excerpt(content, task_terms)
}

pub(super) fn render_focused_sections(content: &str, task_terms: &[String]) -> Option<String> {
    let sections = parse_markdown_sections(content);
    if sections.len() < 2 {
        return None;
    }

    let focus_terms = significant_task_terms(task_terms);
    let debug_task = focus_terms.iter().any(|term| {
        matches!(
            term.as_str(),
            "fix" | "bug" | "error" | "errors" | "failing" | "failure" | "debug" | "issue"
        )
    });
    let guidance_task = focus_terms.iter().any(|term| {
        matches!(
            term.as_str(),
            "implement" | "implementation" | "how" | "why" | "use" | "usage" | "example"
        )
    });

    let mut selected = std::collections::BTreeSet::new();
    if let Some(idx) = sections
        .iter()
        .position(|(name, _)| name.eq_ignore_ascii_case("purpose"))
    {
        selected.insert(idx);
    }

    let mut scored: Vec<(i32, usize)> = sections
        .iter()
        .enumerate()
        .map(|(idx, (name, body))| {
            let lower_name = name.to_ascii_lowercase();
            let section_terms: std::collections::HashSet<String> =
                tokenize(body).into_iter().collect();
            let overlap = section_terms
                .iter()
                .filter(|term| focus_terms.contains(*term))
                .count() as i32;
            let mut score = overlap * 10;
            match lower_name.as_str() {
                "purpose" => score += 15,
                "api" => score += 12,
                "pitfalls" if debug_task => score += 14,
                "patterns" | "examples" if guidance_task => score += 12,
                "auto_evolved" => score += 6,
                "notes" => score += 2,
                _ => {},
            }
            (score, idx)
        })
        .collect();
    scored.sort_unstable_by(|a, b| b.0.cmp(&a.0).then_with(|| a.1.cmp(&b.1)));
    for (_, idx) in scored {
        selected.insert(idx);
        if selected.len() >= 3 {
            break;
        }
    }

    if selected.is_empty() {
        return None;
    }

    let title = content
        .lines()
        .find(|line| line.trim_start().starts_with("# "))
        .map(str::trim)
        .filter(|line| !line.is_empty());
    let mut out = String::new();
    if let Some(title) = title {
        out.push_str(title);
        out.push_str("\n\n");
    }
    for idx in selected {
        let (name, body) = &sections[idx];
        let body = trim_body_lines(body, 8);
        if body.is_empty() {
            continue;
        }
        out.push_str("## ");
        out.push_str(name);
        out.push('\n');
        out.push_str(&body);
        out.push_str("\n\n");
    }

    let trimmed = out.trim().to_string();
    (!trimmed.is_empty()).then_some(trimmed)
}

pub(super) fn render_focused_excerpt(content: &str, task_terms: &[String]) -> String {
    let focus_terms = significant_task_terms(task_terms);
    let lines: Vec<&str> = content.lines().collect();
    let mut scored: Vec<(usize, usize)> = lines
        .iter()
        .enumerate()
        .filter_map(|(idx, line)| {
            let trimmed = line.trim();
            if trimmed.is_empty() {
                return None;
            }
            let overlap = tokenize(trimmed)
                .into_iter()
                .filter(|term| focus_terms.contains(term))
                .count();
            if overlap == 0 {
                return None;
            }
            let speaker_bonus =
                usize::from(trimmed.starts_with("User:") || trimmed.starts_with("Assistant:"));
            Some((overlap + speaker_bonus, idx))
        })
        .collect();

    if scored.is_empty() {
        return fallback_excerpt(content, 6);
    }

    scored.sort_unstable_by(|a, b| b.0.cmp(&a.0).then_with(|| a.1.cmp(&b.1)));
    let mut chosen = std::collections::BTreeSet::new();
    for (_, idx) in scored.into_iter().take(3) {
        let start = idx.saturating_sub(1);
        let end = (idx + 1).min(lines.len().saturating_sub(1));
        for line_idx in start..=end {
            if !lines[line_idx].trim().is_empty() {
                chosen.insert(line_idx);
            }
        }
    }

    let excerpt_lines: Vec<&str> = chosen
        .into_iter()
        .filter_map(|idx| lines.get(idx).copied())
        .filter(|line| !line.trim().is_empty())
        .take(10)
        .collect();
    if excerpt_lines.is_empty() {
        fallback_excerpt(content, 6)
    } else {
        excerpt_lines.join("\n")
    }
}

pub(super) fn parse_markdown_sections(content: &str) -> Vec<(String, String)> {
    let mut sections = Vec::new();
    let mut current_name: Option<String> = None;
    let mut body_lines: Vec<&str> = Vec::new();

    for line in content.lines() {
        if let Some(name) = line.trim_start().strip_prefix("## ") {
            if let Some(prev) = current_name.take() {
                sections.push((prev, body_lines.join("\n").trim().to_string()));
                body_lines.clear();
            }
            current_name = Some(name.trim().to_string());
        } else if current_name.is_some() {
            body_lines.push(line);
        }
    }

    if let Some(name) = current_name {
        sections.push((name, body_lines.join("\n").trim().to_string()));
    }

    sections.retain(|(_, body)| !body.is_empty());
    sections
}

pub(super) fn trim_body_lines(body: &str, max_nonempty_lines: usize) -> String {
    body.lines()
        .filter(|line| !line.trim().is_empty())
        .take(max_nonempty_lines)
        .collect::<Vec<_>>()
        .join("\n")
}

pub(super) fn fallback_excerpt(content: &str, max_nonempty_lines: usize) -> String {
    content
        .lines()
        .filter(|line| !line.trim().is_empty())
        .take(max_nonempty_lines)
        .collect::<Vec<_>>()
        .join("\n")
}

pub(super) fn significant_task_terms(task_terms: &[String]) -> std::collections::HashSet<String> {
    const STOPWORDS: &[&str] = &[
        "a", "an", "and", "are", "as", "at", "be", "by", "did", "do", "for", "from", "have", "how",
        "i", "in", "into", "is", "it", "me", "my", "of", "on", "or", "our", "that", "the", "their",
        "them", "they", "this", "to", "was", "what", "when", "where", "which", "who", "why",
        "with", "you", "your",
    ];
    task_terms
        .iter()
        .map(|term| term.to_ascii_lowercase())
        .filter(|term| term.len() >= 3 && !STOPWORDS.contains(&term.as_str()))
        .collect()
}

pub(super) fn strip_render_only_sections(content: &str) -> String {
    let without_query = strip_named_render_section(content, "query_surface");
    strip_named_render_section(&without_query, "answer_surface")
}

pub(super) fn strip_named_render_section(content: &str, section_name: &str) -> String {
    let header = format!("## {section_name}");
    let marker = format!("<!-- SECTION: {section_name} -->");
    let end_marker = "<!-- /SECTION -->";
    let Some(header_start) = content.find(&header) else {
        return content.to_string();
    };
    let Some(section_start_rel) = content[header_start..].find(&marker) else {
        return content.to_string();
    };
    let section_start = header_start + section_start_rel;
    let Some(section_end_rel) = content[section_start..].find(end_marker) else {
        return content.to_string();
    };
    let section_end = section_start + section_end_rel + end_marker.len();

    let mut stripped = String::with_capacity(content.len());
    stripped.push_str(content[..header_start].trim_end());
    if !stripped.ends_with('\n') {
        stripped.push('\n');
    }
    let tail = content[section_end..].trim_start_matches('\n');
    if !tail.is_empty() {
        stripped.push('\n');
        stripped.push_str(tail);
    }
    stripped
}

pub(super) fn render_overflow_item(path: &Path, headline: &str) -> RenderedContextItem {
    let rendered = format!(
        "<!-- NEURON (compressed): {} — {} -->\n",
        path.file_name().unwrap_or_default().to_string_lossy(),
        sanitize_comment(headline),
    );
    RenderedContextItem {
        path: path.to_path_buf(),
        fingerprint: fingerprint_rendered_context(&rendered),
        rendered,
    }
}

pub(super) fn render_module_capsule(
    project_root: &Path,
    module: &str,
) -> Option<RenderedContextItem> {
    let path = module_capsule_path(project_root, module);
    let content = match std::fs::read_to_string(&path) {
        Ok(content) => content,
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => return None,
        Err(err) => {
            tracing::warn!(
                "Failed to read module capsule {} for {}: {}",
                path.display(),
                module,
                err
            );
            return Some(RenderedContextItem {
                path,
                fingerprint: fingerprint_rendered_context(&format!(
                    "<!-- MODULE CAPSULE {} — read error: {} -->\n\n",
                    sanitize_comment(module),
                    sanitize_comment(&err.to_string())
                )),
                rendered: format!(
                    "<!-- MODULE CAPSULE {} — read error: {} -->\n\n",
                    sanitize_comment(module),
                    sanitize_comment(&err.to_string())
                ),
            });
        },
    };
    let rendered = format!(
        "<!-- === MODULE CAPSULE: {} === -->\n{}\n\n",
        sanitize_comment(module),
        content
    );
    Some(RenderedContextItem {
        path,
        fingerprint: fingerprint_rendered_context(&rendered),
        rendered,
    })
}

pub(super) fn build_path_module_map(
    paths_with_scores: &[(PathBuf, f32)],
    overflow: &[(PathBuf, String)],
    index: &NeuronIndex,
) -> HashMap<PathBuf, String> {
    let mut path_modules = HashMap::new();
    for (path, _) in paths_with_scores {
        if let Some(module) = index.module_for(path) {
            path_modules.insert(path.clone(), module.to_string());
        }
    }
    for (path, _) in overflow {
        if let Some(module) = index.module_for(path) {
            path_modules.insert(path.clone(), module.to_string());
        }
    }
    path_modules
}

pub(super) fn select_capsule_modules(
    paths_with_scores: &[(PathBuf, f32)],
    explicit_module: Option<&str>,
    path_modules: &HashMap<PathBuf, String>,
) -> Vec<String> {
    if let Some(module) = explicit_module {
        return if is_capsule_module(module) {
            vec![module.to_string()]
        } else {
            Vec::new()
        };
    }

    let module_tagged_total = paths_with_scores
        .iter()
        .filter(|(path, _)| {
            path_modules
                .get(path)
                .is_some_and(|module| is_capsule_module(module))
        })
        .count();
    if module_tagged_total < 2 {
        return Vec::new();
    }

    let mut counts: HashMap<&str, usize> = HashMap::new();
    for (path, _) in paths_with_scores {
        let Some(module) = path_modules.get(path) else {
            continue;
        };
        if !is_capsule_module(module) {
            continue;
        }
        *counts.entry(module.as_str()).or_insert(0) += 1;
    }

    let mut ranked: Vec<(&str, usize)> = counts.into_iter().collect();
    ranked.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(b.0)));
    ranked
        .into_iter()
        .find(|(_, count)| *count >= 2 && *count * 2 >= module_tagged_total)
        .map(|(module, _)| vec![module.to_string()])
        .unwrap_or_default()
}

pub(super) fn select_capsule_anchor_paths(
    paths_with_scores: &[(PathBuf, f32)],
    capsule_modules: &HashSet<String>,
    path_modules: &HashMap<PathBuf, String>,
) -> HashSet<PathBuf> {
    const CAPSULE_DYNAMIC_NEURONS_PER_MODULE: usize = 2;
    const CAPSULE_FULL_BODY_SCORE_THRESHOLD: f32 = 5.0;

    let mut grouped: HashMap<&str, Vec<(&PathBuf, f32)>> = HashMap::new();
    for (path, score) in paths_with_scores {
        let Some(module) = path_modules.get(path) else {
            continue;
        };
        if capsule_modules.contains(module) {
            grouped
                .entry(module.as_str())
                .or_default()
                .push((path, *score));
        }
    }

    let mut keep = HashSet::new();
    for items in grouped.values_mut() {
        items.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(b.0)));
        let mut kept = 0usize;
        for (path, score) in items.iter() {
            if *score >= CAPSULE_FULL_BODY_SCORE_THRESHOLD
                && kept < CAPSULE_DYNAMIC_NEURONS_PER_MODULE
            {
                keep.insert((*path).clone());
                kept += 1;
            }
        }
        if kept == 0 {
            if let Some((path, _)) = items.first() {
                keep.insert((*path).clone());
            }
        }
    }

    keep
}

pub(super) fn select_delta_items(
    items: &[RenderedContextItem],
    previous: Option<&HashMap<PathBuf, String>>,
) -> DeltaSelection {
    let current_paths: std::collections::HashSet<&PathBuf> =
        items.iter().map(|item| &item.path).collect();

    let mut emitted = Vec::new();
    let mut unchanged = 0usize;
    for item in items {
        if previous
            .and_then(|snapshot| snapshot.get(&item.path))
            .is_some_and(|fingerprint| fingerprint == &item.fingerprint)
        {
            unchanged += 1;
        } else {
            emitted.push(item.clone());
        }
    }

    let removed = previous
        .map(|snapshot| {
            snapshot
                .keys()
                .filter(|path| !current_paths.contains(*path))
                .count()
        })
        .unwrap_or(0);

    DeltaSelection {
        emitted,
        unchanged,
        removed,
    }
}

/// S-VIII (R16): Auto-mine UseCase stubs from code blocks in an LLM response.
///
/// Scans `response_text` for fenced code blocks (``` ... ```) with ≥5 lines.
/// For each block, finds the cited neuron with the highest term overlap.
/// If overlap ≥ 60% of the neuron's own terms, writes a UseCase stub to
/// `.cortyx/neurons/{neuron}.usecase.auto-{hash}.md` with `status: Stub`.
///
/// Returns the count of stubs written.
pub(super) fn auto_mine_code_blocks(
    response_text: &str,
    cited_paths: &[PathBuf],
    project_root: &Path,
    index: &NeuronIndex,
) -> usize {
    if cited_paths.is_empty() {
        return 0;
    }

    // Extract fenced code blocks: ```[lang]\n<body>\n```
    let mut blocks: Vec<String> = Vec::new();
    let mut in_block = false;
    let mut current_block = Vec::new();
    for line in response_text.lines() {
        let trimmed = line.trim();
        if !in_block && trimmed.starts_with("```") {
            in_block = true;
            current_block.clear();
        } else if in_block && trimmed.starts_with("```") {
            if current_block.len() >= 5 {
                blocks.push(current_block.join("\n"));
            }
            in_block = false;
            current_block.clear();
        } else if in_block {
            current_block.push(line.to_string());
        }
    }

    if blocks.is_empty() {
        return 0;
    }

    let ndir = neuron_dir(project_root);
    let mut written = 0usize;

    for block in &blocks {
        let block_terms: std::collections::HashSet<String> = tokenize(block).into_iter().collect();
        if block_terms.is_empty() {
            continue;
        }

        // Find the cited neuron with highest term overlap
        let best = cited_paths
            .iter()
            .filter_map(|path| {
                let overlap = index.term_freq_overlap(path, &block_terms);
                let total_neuron_terms = index.term_count_for(path);
                if total_neuron_terms == 0 {
                    return None;
                }
                let ratio = overlap as f32 / total_neuron_terms as f32;
                Some((ratio, path))
            })
            .max_by(|a, b| a.0.total_cmp(&b.0));

        let Some((ratio, best_path)) = best else {
            continue;
        };
        if ratio < 0.6 {
            continue;
        }

        // Derive the output filename from the parent neuron stem + a short hash
        let stem = best_path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("unknown")
            .trim_end_matches(".context")
            .to_string();

        let hash_bytes = blake3::hash(block.as_bytes());
        let short_hash = &hash_bytes.to_hex()[..8];
        let usecase_filename = format!("{stem}.usecase.auto-{short_hash}.md");
        let usecase_path = ndir.join(&usecase_filename);

        if usecase_path.exists() {
            continue; // already mined
        }

        let content = format!(
            "# {stem} — auto-mined UseCase\n\
             status: Stub\n\
             source: auto-mined from close_task\n\n\
             ## task\n\
             (edit: describe the task pattern this code solves)\n\n\
             ## example\n\
             ```\n{block}\n```\n"
        );
        if let Err(e) = std::fs::write(&usecase_path, &content) {
            tracing::warn!(
                "S-VIII: failed to write UseCase stub {:?}: {e}",
                usecase_path
            );
        } else {
            tracing::debug!(
                "S-VIII: wrote UseCase stub {:?} (ratio={ratio:.2})",
                usecase_path
            );
            written += 1;
        }
    }

    written
}
