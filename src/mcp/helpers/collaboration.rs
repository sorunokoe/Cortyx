//! Collaboration state building, projection, and rendering.

use super::super::*;
use std::collections::{HashMap, HashSet};

pub(super) const DEFAULT_COLLAB_SUMMARY_LIMIT: usize = 5;
pub(super) const DEFAULT_COLLAB_TIMELINE_LIMIT: usize = 8;

pub fn build_collaboration_projection(
    index: &NeuronIndex,
    project_root: &Path,
) -> CollaborationStateProjection {
    let diaries = collect_collaboration_diaries(index);
    let sync_statuses = load_collaboration_sync_statuses(project_root);
    let kg_entities = load_collaboration_kg_entities(project_root);
    let reasoning = build_collaboration_reasoning(index, project_root, &diaries, &sync_statuses);
    project_collaboration_state(&diaries, &sync_statuses, &kg_entities, reasoning.as_ref())
}

pub fn collect_collaboration_diaries(index: &NeuronIndex) -> Vec<CollaborationDiaryRecord> {
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

pub fn load_collaboration_sync_statuses(project_root: &Path) -> Vec<SyncTransportStatus> {
    let sync_root = sync_transport_dir(project_root);
    if !sync_root.exists() {
        return Vec::new();
    }
    SyncTransportRepository::for_project(project_root)
        .list_statuses()
        .unwrap_or_default()
}

pub fn load_collaboration_kg_entities(project_root: &Path) -> Vec<kg::KgEntity> {
    kg::list_kg_paths(project_root)
        .into_iter()
        .filter_map(|path| kg::KgEntity::load(&path).ok())
        .collect()
}

pub fn build_collaboration_reasoning(
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

pub fn merge_collaboration_seed(seeds: &mut HashMap<PathBuf, f32>, path: PathBuf, score: f32) {
    seeds
        .entry(path)
        .and_modify(|existing| *existing = existing.max(score))
        .or_insert(score);
}

pub fn render_collaboration_status_report(
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

pub fn render_project_collaboration_status(
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

pub fn render_collaborator_status_report(
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

pub fn render_module_status_report(
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

pub fn summarize_plain_diary_content(content: &str) -> String {
    content
        .lines()
        .map(str::trim)
        .find(|line| !line.is_empty() && !line.starts_with("<!--") && !line.starts_with('#'))
        .unwrap_or("(empty diary entry)")
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

pub fn format_timestamp_secs(timestamp_secs: i64) -> String {
    if timestamp_secs < 0 {
        return timestamp_secs.to_string();
    }
    let (y, mo, d, h, mi, s) = unix_secs_to_datetime(timestamp_secs as u64);
    format!("{y:04}-{mo:02}-{d:02}T{h:02}:{mi:02}:{s:02}Z")
}

pub fn render_collaborator_brief(summary: &CollaboratorSummary) -> String {
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

pub fn render_module_brief(state: &ModuleCollaborationState) -> String {
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

pub fn append_supporting_facts(out: &mut String, facts: &[String]) {
    if facts.is_empty() {
        return;
    }
    out.push_str("\n## Supporting facts\n");
    for fact in facts.iter().take(6) {
        out.push_str(&format!("- {fact}\n"));
    }
}

pub fn append_collaboration_timeline(
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

pub fn render_collaboration_timeline_event(event: &CollaborationTimelineEvent) -> String {
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

pub fn format_collaboration_attention(attention: CollaborationAttention) -> &'static str {
    match attention {
        CollaborationAttention::Nominal => "nominal",
        CollaborationAttention::NeedsFollowUp => "needs_follow_up",
        CollaborationAttention::Blocked => "blocked",
        CollaborationAttention::SyncConflict => "sync_conflict",
    }
}

pub fn format_collaboration_evidence(evidence: &CollaborationEvidenceSummary) -> String {
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

pub fn format_trust_score(score: f32) -> String {
    format!("{score:.1}")
}

pub fn matches_collaboration_filter(value: &str, filter: &str) -> bool {
    kg::slugify(value) == kg::slugify(filter)
}

fn truncate_str(s: &str, max_chars: usize) -> String {
    s.chars().take(max_chars).collect()
}
