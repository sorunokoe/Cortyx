use super::*;

pub(super) fn render_answer_output(
    index: &NeuronIndex,
    task: &str,
    paths_with_scores: &[(PathBuf, f32)],
    include_provenance: bool,
    min_answer_confidence: Option<f32>,
) -> Option<String> {
    render_answer_output_decision(
        index,
        task,
        paths_with_scores,
        include_provenance,
        min_answer_confidence,
    )
    .ok()
}

pub(super) fn render_answer_output_decision(
    index: &NeuronIndex,
    task: &str,
    paths_with_scores: &[(PathBuf, f32)],
    include_provenance: bool,
    min_answer_confidence: Option<f32>,
) -> Result<String, AnswerAbstentionReason> {
    let precomputed_answer_path = index.derived_answer_path_for_task(task);
    let deferred_answer_path = precomputed_answer_path
        .as_ref()
        .filter(|path| should_defer_precomputed_answer(task, path))
        .cloned();
    if deferred_answer_path.is_none() {
        if let Some(answer_path) = precomputed_answer_path.as_ref() {
            return render_precomputed_answer(
                index,
                task,
                paths_with_scores,
                include_provenance,
                answer_path,
            )
            .ok_or(AnswerAbstentionReason::Unsupported);
        }
    }

    let (evidence, reasoning) = collect_evidence_with_reasoning(index, task, paths_with_scores);
    if evidence.is_empty() {
        return deferred_answer_path
            .as_ref()
            .and_then(|answer_path| {
                render_precomputed_answer(
                    index,
                    task,
                    paths_with_scores,
                    include_provenance,
                    answer_path,
                )
            })
            .ok_or(AnswerAbstentionReason::Unsupported);
    }

    let base_candidate = select_answer_internal(task, &evidence, None, true);
    let base_answer = validate_selected_answer(task, base_candidate.clone(), None);
    let answer_candidate = if min_answer_confidence.is_some() {
        select_answer_internal(task, &evidence, min_answer_confidence, true)
    } else {
        base_candidate.clone()
    };
    let answer = validate_selected_answer(task, answer_candidate.clone(), min_answer_confidence);
    let Some(answer) = answer else {
        if let Some(answer_path) = deferred_answer_path.as_ref() {
            if let Some(rendered) = render_precomputed_answer(
                index,
                task,
                paths_with_scores,
                include_provenance,
                answer_path,
            ) {
                return Ok(rendered);
            }
        }
        let read_error = evidence.iter().find_map(|item| {
            item.snippet
                .contains("read error")
                .then_some(item.snippet.clone())
        });
        let Some(answer) = read_error else {
            return Err(
                if answer_candidate.is_some() || base_answer.is_some() || base_candidate.is_some() {
                    AnswerAbstentionReason::LowFormConfidence
                } else {
                    AnswerAbstentionReason::Unsupported
                },
            );
        };
        if !include_provenance {
            return Ok(format!("{answer}\n"));
        }
        return Ok(render_answer_with_provenance(
            &answer,
            &evidence,
            Some(&reasoning),
        ));
    };
    if !include_provenance {
        return Ok(format!("{answer}\n"));
    }
    Ok(render_answer_with_provenance(
        &answer,
        &evidence,
        Some(&reasoning),
    ))
}

fn render_precomputed_answer(
    index: &NeuronIndex,
    task: &str,
    paths_with_scores: &[(PathBuf, f32)],
    include_provenance: bool,
    answer_path: &Path,
) -> Option<String> {
    let content = std::fs::read_to_string(answer_path).ok()?;
    let answer = extract_derived_answer(&content)?;
    if !include_provenance {
        if std::env::var_os("CORTYX_EMPTY_ABSTENTION").is_some()
            && derived_answer_is_explicit_abstention(&answer)
        {
            return Some("\n".to_string());
        }
        return Some(format!("{answer}\n"));
    }

    let mut evidence = collect_evidence(index, task, paths_with_scores);
    if !evidence.iter().any(|item| item.path == answer_path) {
        let metadata = index.context_metadata_for(answer_path);
        let snippet = metadata
            .as_ref()
            .map(|m| sanitize_inline(&m.summary))
            .filter(|summary| !summary.is_empty())
            .or_else(|| extract_derived_answer(&content))
            .unwrap_or_else(|| fallback_snippet(answer_path));
        let score = paths_with_scores
            .iter()
            .find_map(|(path, score)| (path == answer_path).then_some(*score))
            .unwrap_or(0.0);
        evidence.insert(
            0,
            EvidenceItem {
                path: answer_path.to_path_buf(),
                score,
                metadata,
                snippet,
            },
        );
    }
    let (evidence, reasoning) = augment_evidence_with_reasoning(index, task, evidence);
    Some(render_answer_with_provenance(
        &answer,
        &evidence,
        Some(&reasoning),
    ))
}

pub(super) fn should_defer_precomputed_answer(task: &str, answer_path: &Path) -> bool {
    let lower = task.to_ascii_lowercase();
    if lower.contains("move from")
        || lower.contains("moved from")
        || lower.contains("home country")
        || lower.contains("origin country")
    {
        return false;
    }
    if parse_temporal_elapsed_query(task).is_some() {
        return true;
    }
    let file_name = answer_path
        .file_name()
        .unwrap_or_default()
        .to_string_lossy()
        .to_ascii_lowercase();
    if is_temporal_sequence_query(task) && !file_name.contains("temporal") {
        return true;
    }
    file_name.contains("answer-surface") && is_temporal_reasoning_query(task)
}

fn render_answer_with_provenance(
    answer: &str,
    evidence: &[EvidenceItem],
    reasoning: Option<&ReasoningEnhancement>,
) -> String {
    let mut out = String::new();
    out.push_str(answer);
    out.push_str("\n\n");
    out.push_str("<!-- CORTYX PROVENANCE -->\n");
    for item in evidence.iter().take(3) {
        out.push_str("- ");
        out.push_str(&format_provenance_line(item));
        out.push('\n');
    }
    out.push_str("<!-- END PROVENANCE -->\n");
    append_reasoning_block(&mut out, reasoning);
    out
}

pub(super) fn render_provenance_output(
    index: &NeuronIndex,
    paths_with_scores: &[(PathBuf, f32)],
) -> Option<String> {
    if paths_with_scores.is_empty() {
        return None;
    }
    let evidence = paths_with_scores
        .iter()
        .take(5)
        .map(|(path, score)| {
            let metadata = index.context_metadata_for(path);
            let snippet = metadata
                .as_ref()
                .map(|m| sanitize_inline(&m.summary))
                .filter(|summary| !summary.is_empty())
                .unwrap_or_else(|| fallback_snippet(path));
            EvidenceItem {
                path: path.clone(),
                score: *score,
                metadata,
                snippet,
            }
        })
        .collect::<Vec<_>>();
    let reasoning = build_reasoning_enhancement(index, None, &evidence);
    let mut out = String::from("<!-- CORTYX PROVENANCE -->\n");
    for item in &evidence {
        out.push_str("- ");
        out.push_str(&format_provenance_line(item));
        out.push('\n');
    }
    out.push_str("<!-- END PROVENANCE -->\n");
    append_reasoning_block(&mut out, Some(&reasoning));
    out.push('\n');
    Some(out)
}

fn collect_evidence(
    index: &NeuronIndex,
    task: &str,
    paths_with_scores: &[(PathBuf, f32)],
) -> Vec<EvidenceItem> {
    let task_terms = salient_query_terms(task);
    let temporal_query = is_temporal_reasoning_query(task);
    let mut evidence = Vec::new();
    for (path, score) in paths_with_scores {
        if temporal_query
            && path
                .file_name()
                .unwrap_or_default()
                .to_string_lossy()
                .to_ascii_lowercase()
                .starts_with("_answer_")
        {
            continue;
        }
        let metadata = index.context_metadata_for(path);
        let snippet = match read_context_text(path, "answer evidence collection") {
            Some(content) => best_evidence_snippet(&content, metadata.as_ref(), &task_terms)
                .unwrap_or_else(|| fallback_snippet(path)),
            None => explicit_read_error_snippet(path),
        };
        evidence.push(EvidenceItem {
            path: path.clone(),
            score: *score,
            metadata,
            snippet,
        });
        if evidence.len() >= 5 {
            break;
        }
    }
    evidence
}

fn collect_evidence_with_reasoning(
    index: &NeuronIndex,
    task: &str,
    paths_with_scores: &[(PathBuf, f32)],
) -> (Vec<EvidenceItem>, ReasoningEnhancement) {
    augment_evidence_with_reasoning(
        index,
        task,
        collect_evidence(index, task, paths_with_scores),
    )
}

fn augment_evidence_with_reasoning(
    index: &NeuronIndex,
    task: &str,
    mut evidence: Vec<EvidenceItem>,
) -> (Vec<EvidenceItem>, ReasoningEnhancement) {
    let reasoning = build_reasoning_enhancement(index, Some(task), &evidence);
    if !reasoning.supplemental_evidence.is_empty() {
        evidence.extend(reasoning.supplemental_evidence.iter().cloned());
    }
    (evidence, reasoning)
}

fn build_reasoning_enhancement(
    index: &NeuronIndex,
    task: Option<&str>,
    evidence: &[EvidenceItem],
) -> ReasoningEnhancement {
    if evidence.is_empty() {
        return ReasoningEnhancement::default();
    }

    let temporal_query = task.map(is_temporal_reasoning_query).unwrap_or(false);
    let defaults = TraversalOptions::default();
    let report = index.reason_over_paths(
        &evidence
            .iter()
            .map(|item| (item.path.clone(), item.score))
            .collect::<Vec<_>>(),
        TraversalOptions {
            max_hops: if temporal_query {
                10
            } else {
                defaults.max_hops
            },
            max_expansions: if temporal_query { 160 } else { 32 },
            min_propagated_score: if temporal_query { 0.015 } else { 0.18 },
            ..defaults
        },
    );
    let supplemental_evidence = task
        .map(|task| {
            let mut supplemental = supplemental_temporal_chunk_evidence(task, index, evidence);
            let mut temporal_evidence = evidence.to_vec();
            temporal_evidence.extend(supplemental.iter().cloned());
            let mut seen_paths = supplemental
                .iter()
                .map(|item| item.path.clone())
                .collect::<HashSet<_>>();
            for item in
                supplemental_temporal_current_anchor_evidence(task, index, &temporal_evidence)
            {
                if seen_paths.insert(item.path.clone()) {
                    temporal_evidence.push(item.clone());
                    supplemental.push(item);
                }
            }
            for item in supplemental_node_evidence_from_reasoning(task, index, evidence, &report) {
                if seen_paths.insert(item.path.clone()) {
                    supplemental.push(item);
                }
            }
            let limit = if is_temporal_reasoning_query(task) {
                12
            } else {
                2
            };
            for item in supplemental_kg_evidence_from_reasoning(task, index, evidence, &report) {
                if seen_paths.insert(item.path.clone()) {
                    supplemental.push(item);
                }
                if supplemental.len() >= limit {
                    break;
                }
            }
            supplemental
        })
        .unwrap_or_default();
    if report.nodes.is_empty() && report.facts.is_empty() && report.conflicts.is_empty() {
        return ReasoningEnhancement {
            supplemental_evidence,
            summary_lines: Vec::new(),
            chain_lines: Vec::new(),
        };
    }
    let seed_paths: HashSet<PathBuf> = evidence.iter().map(|item| item.path.clone()).collect();
    let mut summary_report = report.clone();
    summary_report
        .nodes
        .retain(|node| !seed_paths.contains(&node.path));

    ReasoningEnhancement {
        supplemental_evidence,
        summary_lines: summary_report.summary_lines(2, 2),
        chain_lines: summary_report.chain_lines(3),
    }
}

fn supplemental_kg_evidence_from_reasoning(
    task: &str,
    index: &NeuronIndex,
    evidence: &[EvidenceItem],
    report: &ReasoningReport,
) -> Vec<EvidenceItem> {
    let task_terms = salient_query_terms(task);
    if task_terms.is_empty() {
        return Vec::new();
    }

    let max_seed_score = evidence
        .iter()
        .map(|item| item.score)
        .fold(0.0_f32, f32::max);
    if max_seed_score <= 0.0 {
        return Vec::new();
    }

    let existing_paths: HashSet<PathBuf> = evidence.iter().map(|item| item.path.clone()).collect();
    let mut ranked = report
        .facts
        .iter()
        .filter_map(|fact| {
            let overlap = reasoned_fact_task_overlap(&task_terms, fact);
            (overlap > 0).then_some((overlap, fact.score, fact.supporting_paths.len(), fact))
        })
        .collect::<Vec<_>>();
    ranked.sort_by(|a, b| {
        b.0.cmp(&a.0)
            .then_with(|| b.1.total_cmp(&a.1))
            .then_with(|| b.2.cmp(&a.2))
            .then_with(|| a.3.entity_path.cmp(&b.3.entity_path))
    });

    let mut supplemental = Vec::new();
    let mut seen_paths = HashSet::new();
    for (_, fact_score, _, fact) in ranked {
        if existing_paths.contains(&fact.entity_path)
            || !seen_paths.insert(fact.entity_path.clone())
        {
            continue;
        }
        supplemental.push(EvidenceItem {
            path: fact.entity_path.clone(),
            score: (max_seed_score * fact_score.clamp(0.0, 1.0)).max(0.1),
            metadata: index.context_metadata_for(&fact.entity_path),
            snippet: format!("kg: {}.{} = {}", fact.entity, fact.predicate, fact.value),
        });
        if supplemental.len() >= 2 {
            break;
        }
    }

    supplemental
}

fn supplemental_temporal_chunk_evidence(
    task: &str,
    index: &NeuronIndex,
    evidence: &[EvidenceItem],
) -> Vec<EvidenceItem> {
    if !is_temporal_reasoning_query(task) {
        return Vec::new();
    }

    let max_seed_score = evidence
        .iter()
        .map(|item| item.score)
        .fold(0.0_f32, f32::max);
    if max_seed_score <= 0.0 {
        return Vec::new();
    }

    let existing_paths: HashSet<PathBuf> = evidence.iter().map(|item| item.path.clone()).collect();
    let mut discovered = Vec::new();
    let mut seen_paths = HashSet::new();
    for item in evidence {
        let Some(seed) = temporal_chunk_seed(&item.path) else {
            continue;
        };
        let Some(parent) = item.path.parent() else {
            continue;
        };
        let Ok(entries) = std::fs::read_dir(parent) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if existing_paths.contains(&path) || !seen_paths.insert(path.clone()) {
                continue;
            }
            let Some((candidate_family, candidate_index)) = temporal_chunk_member(&path) else {
                continue;
            };
            if candidate_family != seed.family {
                continue;
            }
            let distance = seed
                .index
                .map_or(0, |seed_index| seed_index.abs_diff(candidate_index));
            discovered.push((distance, candidate_index, path));
        }
    }

    discovered.sort_by(|left, right| left.cmp(right));
    let mut supplemental = Vec::new();
    for (distance, _, path) in discovered {
        let metadata = index.context_metadata_for(&path);
        let snippet = metadata
            .as_ref()
            .map(|value| sanitize_inline(&value.summary))
            .filter(|summary| !summary.is_empty())
            .unwrap_or_else(|| fallback_snippet(&path));
        supplemental.push(EvidenceItem {
            path,
            score: (max_seed_score * (0.45 / (distance as f32 + 1.0))).max(0.1),
            metadata,
            snippet,
        });
        if supplemental.len() >= 16 {
            break;
        }
    }
    supplemental
}

fn supplemental_temporal_current_anchor_evidence(
    task: &str,
    index: &NeuronIndex,
    evidence: &[EvidenceItem],
) -> Vec<EvidenceItem> {
    if parse_temporal_elapsed_query(task).is_none() {
        return Vec::new();
    }

    let candidates = collect_temporal_candidates(evidence, "temporal current-anchor seed");
    if best_calendar_grounded_current_anchor_rank(&candidates).is_some() {
        return Vec::new();
    }

    let max_seed_score = evidence
        .iter()
        .map(|item| item.score)
        .fold(0.0_f32, f32::max);
    if max_seed_score <= 0.0 {
        return Vec::new();
    }

    let existing_paths: HashSet<PathBuf> = evidence.iter().map(|item| item.path.clone()).collect();
    let module_scope = evidence.iter().find_map(|item| {
        item.metadata
            .as_ref()
            .and_then(|metadata| metadata.module.clone())
    });

    let mut best: Option<(i32, EvidenceItem)> = None;
    for path in index.recent_verbatim_paths_with_current_markers(module_scope.as_deref(), 2048) {
        if existing_paths.contains(&path) {
            continue;
        }

        let metadata = index.context_metadata_for(&path);
        let snippet = metadata
            .as_ref()
            .map(|value| sanitize_inline(&value.summary))
            .filter(|summary| !summary.is_empty())
            .unwrap_or_else(|| fallback_snippet(&path));
        let item = EvidenceItem {
            path,
            score: (max_seed_score * 0.3).max(0.1),
            metadata,
            snippet,
        };
        let anchor_candidates = collect_temporal_candidates(
            std::slice::from_ref(&item),
            "temporal current-anchor supplement",
        );
        let Some(rank) = best_calendar_grounded_current_anchor_rank(&anchor_candidates) else {
            continue;
        };
        let should_replace = best
            .as_ref()
            .map(|(best_rank, best_item)| {
                rank > *best_rank || (rank == *best_rank && item.path < best_item.path)
            })
            .unwrap_or(true);
        if should_replace {
            best = Some((rank, item));
        }
    }

    best.map(|(_, item)| vec![item]).unwrap_or_default()
}

#[derive(Debug, Clone)]
struct TemporalChunkSeed {
    family: String,
    index: Option<i32>,
}

fn temporal_chunk_seed(path: &Path) -> Option<TemporalChunkSeed> {
    let file_name = path.file_name()?.to_string_lossy();
    if file_name.ends_with("_summary.md") {
        return Some(TemporalChunkSeed {
            family: file_name.trim_end_matches("_summary.md").to_string(),
            index: None,
        });
    }
    let (family, index) = temporal_chunk_member(path)?;
    Some(TemporalChunkSeed {
        family,
        index: Some(index),
    })
}

fn temporal_chunk_member(path: &Path) -> Option<(String, i32)> {
    let file_name = path.file_name()?.to_string_lossy();
    if !file_name.ends_with(".md") || !file_name.contains("_chunk.verbatim.md") {
        return None;
    }
    let marker = "_chunk";
    let marker_index = file_name.find(marker)?;
    let prefix = &file_name[..marker_index];
    let chunk_digits = prefix.rsplit('_').next()?;
    if chunk_digits.is_empty() || !chunk_digits.chars().all(|c| c.is_ascii_digit()) {
        return None;
    }
    let family = prefix[..prefix.len().saturating_sub(chunk_digits.len())]
        .trim_end_matches('_')
        .to_string();
    Some((family, chunk_digits.parse().ok()?))
}

fn supplemental_node_evidence_from_reasoning(
    task: &str,
    index: &NeuronIndex,
    evidence: &[EvidenceItem],
    report: &ReasoningReport,
) -> Vec<EvidenceItem> {
    if !is_temporal_reasoning_query(task) {
        return Vec::new();
    }

    let mut task_terms = temporal_focus_terms(task);
    if task_terms.is_empty() {
        task_terms = salient_query_terms(task);
    }
    if task_terms.is_empty() {
        return Vec::new();
    }

    let max_seed_score = evidence
        .iter()
        .map(|item| item.score)
        .fold(0.0_f32, f32::max);
    if max_seed_score <= 0.0 {
        return Vec::new();
    }

    let existing_paths: HashSet<PathBuf> = evidence.iter().map(|item| item.path.clone()).collect();
    let mut ranked = report
        .nodes
        .iter()
        .filter_map(|node| {
            if node.is_seed || node.is_kg_entity || existing_paths.contains(&node.path) {
                return None;
            }

            let metadata = index.context_metadata_for(&node.path);
            let summary = node
                .summary
                .clone()
                .filter(|value| !value.trim().is_empty())
                .or_else(|| metadata.as_ref().map(|value| value.summary.clone()))
                .unwrap_or_default();
            let overlap = task_overlap_count(&summary, &task_terms);
            let edge_bonus = match node.strongest_step.as_ref().map(|step| &step.edge_type) {
                Some(SynapseType::TemporalFollows) => 8.0,
                Some(SynapseType::Derived) => 3.5,
                Some(SynapseType::SemanticRelated) => 1.0,
                _ => 0.0,
            };
            if overlap == 0 && edge_bonus < 8.0 {
                return None;
            }

            let score = overlap as f32 * 10.0
                + node.score * 12.0
                + edge_bonus
                + if matches!(
                    metadata.as_ref().map(|value| &value.kind),
                    Some(&NeuronKind::Verbatim)
                ) {
                    1.5
                } else {
                    0.0
                }
                - node.depth as f32 * 0.5;
            Some((score, node.path.clone(), metadata, summary, node.score))
        })
        .collect::<Vec<_>>();
    ranked.sort_by(|a, b| b.0.total_cmp(&a.0).then_with(|| a.1.cmp(&b.1)));

    let mut supplemental = Vec::new();
    let mut seen_paths = HashSet::new();
    for (_, path, metadata, summary, node_score) in ranked {
        if !seen_paths.insert(path.clone()) {
            continue;
        }
        supplemental.push(EvidenceItem {
            path: path.clone(),
            score: (max_seed_score * node_score.clamp(0.0, 1.0)).max(0.1),
            metadata,
            snippet: if summary.trim().is_empty() {
                fallback_snippet(&path)
            } else {
                sanitize_inline(&summary)
            },
        });
        if supplemental.len() >= 8 {
            break;
        }
    }
    supplemental
}

fn reasoned_fact_task_overlap(task_terms: &[String], fact: &ReasonedFact) -> usize {
    let mut context_terms = kg_predicate_query_terms(&fact.predicate);
    context_terms.extend(kg_entity_query_terms(&fact.entity));
    context_terms.extend(
        fact.value
            .split(|c: char| !c.is_ascii_alphanumeric())
            .filter(|token| token.len() >= 3)
            .map(|token| token.to_ascii_lowercase()),
    );
    task_overlap_count(&context_terms.join(" "), task_terms)
}

fn append_reasoning_block(out: &mut String, reasoning: Option<&ReasoningEnhancement>) {
    let Some(reasoning) = reasoning else {
        return;
    };
    if reasoning.summary_lines.is_empty() && reasoning.chain_lines.is_empty() {
        return;
    }

    out.push_str("<!-- CORTYX GRAPH REASONING -->\n");
    for line in &reasoning.chain_lines {
        out.push_str("- chain: ");
        out.push_str(line);
        out.push('\n');
    }
    for line in &reasoning.summary_lines {
        out.push_str("- ");
        out.push_str(line);
        out.push('\n');
    }
    out.push_str("<!-- END GRAPH REASONING -->\n");
}

fn best_evidence_snippet(
    content: &str,
    metadata: Option<&ContextMetadata>,
    task_terms: &[String],
) -> Option<String> {
    if let Some(answer) = extract_derived_answer(content) {
        return Some(answer);
    }
    let mut best: Option<(f32, String)> = None;
    for line in answer_candidate_lines(content) {
        let clean = sanitize_answer_text(&line);
        if clean.is_empty() {
            continue;
        }
        let score = candidate_weight(&clean, task_terms, 0.0, false);
        if best
            .as_ref()
            .map(|(best_score, _)| score > *best_score)
            .unwrap_or(true)
        {
            best = Some((score, clean));
        }
    }
    best.map(|(_, line)| line).or_else(|| {
        metadata
            .map(|m| sanitize_answer_text(&m.summary))
            .filter(|summary| !summary.is_empty())
    })
}

pub(super) fn answer_candidate_lines(content: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut in_generated_section = false;
    for raw_line in content.lines().map(str::trim) {
        if should_skip_generated_answer_line(raw_line, &mut in_generated_section) {
            continue;
        }
        out.push(raw_line.to_string());
        for fragment in split_candidate_fragments(raw_line) {
            if fragment != raw_line {
                out.push(fragment);
            }
        }
    }
    out
}

pub(super) fn read_context_text(path: &Path, stage: &str) -> Option<String> {
    match std::fs::read_to_string(path) {
        Ok(content) => Some(content),
        Err(err) => {
            tracing::warn!(
                "Failed to read context file {} during {}: {}",
                path.display(),
                stage,
                err
            );
            None
        },
    }
}

fn explicit_read_error_snippet(path: &Path) -> String {
    format!("(read error: {})", fallback_snippet(path))
}
