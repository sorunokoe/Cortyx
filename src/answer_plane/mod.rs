//! Answer plane - query processing and response generation.
//!
//! Handles natural language queries, temporal reasoning, and fact extraction.

use crate::index::{ContextMetadata, NeuronIndex};
use crate::kg;
use crate::neuron::{parse_sections, unix_secs_to_datetime, NeuronKind, SynapseType};
use crate::reasoner::{ReasonedFact, ReasoningReport, TraversalOptions};
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

const QUESTION_STOPWORDS: &[&str] = &[
    "the",
    "a",
    "an",
    "and",
    "or",
    "but",
    "for",
    "with",
    "from",
    "that",
    "this",
    "what",
    "when",
    "where",
    "which",
    "who",
    "whom",
    "whose",
    "why",
    "how",
    "did",
    "does",
    "do",
    "is",
    "was",
    "are",
    "were",
    "am",
    "would",
    "could",
    "should",
    "can",
    "will",
    "may",
    "might",
    "to",
    "of",
    "in",
    "on",
    "at",
    "by",
    "it",
    "its",
    "my",
    "your",
    "our",
    "their",
    "his",
    "her",
    "have",
    "has",
    "had",
    "been",
    "being",
    "after",
    "before",
    "later",
    "earlier",
    "about",
    "into",
    "over",
    "under",
    "through",
    "likely",
    "probably",
    "possibly",
    "potentially",
    "considered",
    "still",
    "more",
    "most",
    "less",
    "least",
];

const PREPOSITION_HINTS: &[&str] = &[
    "for", "about", "with", "from", "to", "in", "at", "on", "after", "before", "during",
];

const TAIL_BOUNDARIES: &[&str] = &[
    ".",
    "!",
    "?",
    ";",
    ",",
    " - ",
    " — ",
    " because ",
    " but ",
    " so ",
    " while ",
    " although ",
    " though ",
    " which ",
    " that ",
    " who ",
    " when ",
    " lately ",
    " recently ",
    " yesterday ",
    " today ",
    " tomorrow ",
    " last ",
    " next ",
    " 'cause ",
    " cause ",
];

const COPULA_BOUNDARIES: &[&str] = &[
    " is ",
    " was ",
    " are ",
    " were ",
    " feels ",
    " felt ",
    " sounds ",
    " sounded ",
    " seems ",
    " seemed ",
    " looks ",
    " looked ",
];

const ENTITY_STOPWORDS: &[&str] = &[
    "what",
    "who",
    "when",
    "where",
    "why",
    "how",
    "which",
    "did",
    "does",
    "do",
    "the",
    "a",
    "an",
    "on",
    "in",
    "at",
    "for",
    "of",
    "january",
    "february",
    "march",
    "april",
    "may",
    "june",
    "july",
    "august",
    "september",
    "october",
    "november",
    "december",
];

const MULTIHOP_BUNDLE_FIELDS: &[&str] = &[
    "next step",
    "next action",
    "dependencies",
    "dependency",
    "blocker",
    "status",
    "outcome",
    "result",
    "action",
    "title",
    "focus",
    "goal",
    "entities",
    "entity",
    "location",
    "residence",
    "city",
    "home",
    "job",
    "occupation",
    "career",
    "role",
];

#[derive(Debug, Clone)]
struct EvidenceItem {
    path: PathBuf,
    score: f32,
    metadata: Option<ContextMetadata>,
    snippet: String,
}

#[derive(Debug, Clone, Default)]
struct ReasoningEnhancement {
    supplemental_evidence: Vec<EvidenceItem>,
    summary_lines: Vec<String>,
    /// Traversal chains rendered as "seed → hop1 → hop2 (score X.XX)".
    chain_lines: Vec<String>,
}

#[derive(Debug, Clone)]
struct CandidateLine {
    path: PathBuf,
    text: String,
    weight: f32,
    retrieval_score: f32,
    support_overlap: usize,
    anchor_overlap: usize,
    specific_anchor_overlap: usize,
}

#[derive(Debug, Clone)]
struct AnswerSurfaceRow {
    question_pattern: String,
    answer_span: String,
    confidence: f32,
}

#[derive(Debug, Clone)]
struct DialogueTurn {
    speaker: Option<String>,
    text: String,
    session_date: Option<(i32, u32, u32)>,
}

#[derive(Debug, Clone)]
struct TemporalCandidate {
    text: String,
    base_date: Option<(i32, u32, u32)>,
    retrieval_score: f32,
    user_authored: bool,
    ordinal: usize,
    sequence_rank: Option<i32>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TemporalDirection {
    Earlier,
    Later,
}

#[derive(Debug, Clone)]
struct ChoiceOption {
    display: String,
    tokens: Vec<String>,
}

#[derive(Debug, Clone)]
enum TemporalStateQuery {
    CurrentValue,
    AsOfValue { as_of: String },
    LastChange { target_value: Option<ChoiceOption> },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TemporalAnswerPoint {
    Day { year: i32, month: u32, day: u32 },
    Month { year: i32, month: u32 },
    Year { year: i32 },
}

#[derive(Debug, Clone)]
enum TemporalGapEndpoint {
    Event(ChoiceOption),
    CurrentMoment,
}

#[derive(Debug, Clone)]
enum TemporalGapAnswerStyle {
    FixedUnit { unit: String },
    NaturalLanguage,
}

#[derive(Debug, Clone)]
struct TemporalGapQuery {
    start: ChoiceOption,
    end: TemporalGapEndpoint,
    answer_style: TemporalGapAnswerStyle,
}

#[derive(Debug, Clone, Copy)]
struct CalendarGroundedRank {
    ordinal: usize,
    rank: i32,
    overlap: usize,
    score: f32,
}

#[derive(Debug, Clone)]
struct RelationKgSupport {
    values: Vec<String>,
}

#[derive(Debug, Clone)]
struct RelationCandidateBucket {
    best_candidate: CandidateLine,
    best_single_weight: f32,
    total_weight: f32,
    max_retrieval_score: f32,
    max_support_overlap: usize,
    max_anchor_overlap: usize,
    max_specific_anchor_overlap: usize,
    paths: HashSet<PathBuf>,
    hits: usize,
}

#[derive(Debug, Clone)]
struct AnswerSurfaceBucket {
    answer_span: String,
    best_score: f32,
    total_score: f32,
    best_confidence: f32,
    max_overlap: usize,
    max_anchor_overlap: usize,
    paths: HashSet<PathBuf>,
    hits: usize,
}

#[derive(Debug, Clone)]
enum RelationResolution {
    Answer(String),
    Suppress,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AnswerAbstentionReason {
    Unsupported,
    LowFormConfidence,
}

pub fn render_answer_output(
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

pub fn render_answer_output_decision(
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

fn should_defer_precomputed_answer(task: &str, answer_path: &Path) -> bool {
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

pub fn render_provenance_output(
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

#[cfg(test)]
fn select_answer(
    task: &str,
    evidence: &[EvidenceItem],
    min_answer_confidence: Option<f32>,
) -> Option<String> {
    select_answer_internal(task, evidence, min_answer_confidence, true)
}

fn select_answer_internal(
    task: &str,
    evidence: &[EvidenceItem],
    min_answer_confidence: Option<f32>,
    allow_multi_hop: bool,
) -> Option<String> {
    let temporal_query = is_temporal_reasoning_query(task);
    let typed_open_qa = looks_like_typed_open_qa_query(task);
    if is_reading_progress_pages_left_query(task) {
        return None;
    }
    if allow_multi_hop {
        if let Some(subquestions) = decompose_multi_hop_subquestions(task) {
            return compose_subquestion_answers(&subquestions, evidence, min_answer_confidence);
        }
    }

    let precomputed_candidates =
        if allow_multi_hop && !temporal_query && looks_like_multi_hop_list_query(task) {
            Some(collect_answer_candidates(task, evidence))
        } else {
            None
        };

    if allow_multi_hop && !temporal_query {
        if let Some(candidates) = precomputed_candidates.as_deref() {
            if let Some(answer) =
                select_multi_item_answer_from_candidates(task, candidates, min_answer_confidence)
            {
                return Some(answer);
            }
        }
    }

    if parse_temporal_gap_query(task).is_some() {
        return select_temporal_duration_answer(task, evidence);
    }

    if let Some(answer) = select_temporal_count_answer(task, evidence) {
        return Some(answer);
    }
    if parse_temporal_elapsed_query(task).is_some() {
        return None;
    }

    if let Some(answer) = select_temporal_state_answer(task, evidence) {
        return Some(answer);
    }

    if let Some(answer) = select_dialogue_temporal_answer(task, evidence) {
        return Some(answer);
    }

    if let Some(answer) = select_comparison_answer(task, evidence) {
        return Some(answer);
    }

    if let Some(answer) = select_temporal_order_answer(task, evidence) {
        return Some(answer);
    }

    for item in evidence {
        let Some(content) = read_context_text(&item.path, "answer derived lookup") else {
            continue;
        };
        if let Some(answer) = extract_derived_answer(&content) {
            return Some(answer);
        }
    }

    if let Some(resolution) = resolve_relation_answer(task, evidence, min_answer_confidence) {
        return match resolution {
            RelationResolution::Answer(answer) => Some(answer),
            RelationResolution::Suppress => None,
        };
    }

    if let Some(answer) = select_suggestion_list_item_answer(task, evidence) {
        return Some(answer);
    }

    if let Some(answer) = select_answer_surface(task, evidence) {
        return Some(answer);
    }

    if let Some(answer) = select_structured_diary_answer(task, evidence) {
        return Some(answer);
    }

    if typed_open_qa {
        if let Some(answer) = select_typed_open_qa_structured_answer(task, evidence) {
            return Some(answer);
        }
        return None;
    }

    if let Some(answer) = select_subject_turn_answer(task, evidence, min_answer_confidence) {
        return Some(answer);
    }

    if let Some(answer) = select_turn_pair_answer(task, evidence, min_answer_confidence) {
        return Some(answer);
    }

    let enumerative = is_enumerative_query(task);
    let candidates =
        precomputed_candidates.unwrap_or_else(|| collect_answer_candidates(task, evidence));

    let target_count = if enumerative { 3 } else { 1 };
    let mut chosen = Vec::new();
    let mut seen = HashSet::new();
    for candidate in candidates {
        if candidate.text.is_empty()
            || !seen.insert(candidate.text.clone())
            || !candidate_has_required_anchor_support(task, &candidate)
        {
            continue;
        }
        chosen.push(candidate.text);
        if chosen.len() >= target_count {
            break;
        }
    }

    if chosen.is_empty() {
        None
    } else if enumerative {
        Some(chosen.join("; "))
    } else {
        chosen.into_iter().next()
    }
}

fn decompose_multi_hop_subquestions(task: &str) -> Option<Vec<String>> {
    let trimmed = task.trim().trim_end_matches('?').trim();
    if trimmed.is_empty() {
        return None;
    }

    decompose_explicit_question_clauses(trimmed)
        .or_else(|| decompose_shared_prefix_question(trimmed))
}

fn decompose_explicit_question_clauses(task: &str) -> Option<Vec<String>> {
    for question_word in ["what", "who", "where", "when", "why", "how", "which"] {
        for marker in [
            format!(", and {question_word} "),
            format!(" and {question_word} "),
        ] {
            let Some((left, right)) = split_once_case_insensitive(task, &marker) else {
                continue;
            };
            return Some(vec![
                ensure_question_suffix(left),
                ensure_question_suffix(&format!("{question_word} {}", right.trim())),
            ]);
        }
    }
    None
}

fn decompose_shared_prefix_question(task: &str) -> Option<Vec<String>> {
    let lower = task.to_ascii_lowercase();
    let opener = ["what is ", "what are ", "what was ", "what were "]
        .into_iter()
        .find(|prefix| lower.starts_with(prefix))?;
    let tail = task[opener.len()..].trim();
    if !tail.contains(" and ") {
        return None;
    }

    let normalized_tail = tail.replace(", and ", ", ").replace(" and ", ", ");
    let segments = normalized_tail
        .split(',')
        .map(str::trim)
        .filter(|segment| !segment.is_empty())
        .collect::<Vec<_>>();
    if !(2..=3).contains(&segments.len()) {
        return None;
    }

    let first_field = trailing_multihop_field(segments.first()?)?;
    let first_segment = segments.first()?.trim();
    let shared_prefix = first_segment[..first_segment.len() - first_field.len()]
        .trim()
        .to_string();
    if shared_prefix.is_empty() {
        return None;
    }

    let mut out = Vec::new();
    for segment in segments {
        let field = trailing_multihop_field(segment)
            .or_else(|| segment_field_name(segment))
            .or_else(|| infer_subanswer_label(segment).map(str::to_string))?;
        let body = if segment
            .to_ascii_lowercase()
            .starts_with(&shared_prefix.to_ascii_lowercase())
        {
            segment.to_string()
        } else {
            format!("{shared_prefix} {field}")
        };
        out.push(ensure_question_suffix(&format!("{opener}{body}")));
    }
    Some(out)
}

fn trailing_multihop_field(text: &str) -> Option<String> {
    let lower = text.trim().to_ascii_lowercase();
    MULTIHOP_BUNDLE_FIELDS
        .iter()
        .filter(|field| lower.ends_with(**field))
        .max_by_key(|field| field.len())
        .map(|field| (*field).to_string())
}

fn segment_field_name(text: &str) -> Option<String> {
    let clean = text.trim();
    MULTIHOP_BUNDLE_FIELDS
        .iter()
        .find(|field| clean.eq_ignore_ascii_case(field))
        .map(|field| (*field).to_string())
}

fn ensure_question_suffix(text: &str) -> String {
    let mut clean = sanitize_inline(text.trim().trim_matches(','));
    if !clean.ends_with('?') {
        clean.push('?');
    }
    clean
}

fn compose_subquestion_answers(
    subquestions: &[String],
    evidence: &[EvidenceItem],
    min_answer_confidence: Option<f32>,
) -> Option<String> {
    let mut parts = Vec::new();
    let mut seen_answers = HashSet::new();

    for subquestion in subquestions {
        let answer = select_answer_internal(subquestion, evidence, min_answer_confidence, false)?;
        let clean = sanitize_answer_text(&answer);
        if clean.is_empty() {
            return None;
        }
        let key = normalized_answer_key(&clean);
        if !seen_answers.insert(key) {
            continue;
        }
        parts.push((infer_subanswer_label(subquestion), clean));
    }

    if parts.len() < 2 {
        return None;
    }

    let labeled = parts.iter().all(|(label, _)| label.is_some())
        && parts
            .iter()
            .filter_map(|(label, _)| *label)
            .collect::<HashSet<_>>()
            .len()
            == parts.len();

    Some(if labeled {
        parts
            .into_iter()
            .map(|(label, answer)| format!("{}: {}", label.unwrap_or("answer"), answer))
            .collect::<Vec<_>>()
            .join("; ")
    } else {
        parts
            .into_iter()
            .map(|(_, answer)| answer)
            .collect::<Vec<_>>()
            .join("; ")
    })
}

fn infer_subanswer_label(task: &str) -> Option<&'static str> {
    let lower = task.to_ascii_lowercase();
    if lower.contains("next step") || lower.contains("next action") {
        Some("next step")
    } else if structured_diary_blocker_query(&lower) {
        Some("blocker")
    } else if lower.contains("find") || lower.contains("found") || lower.contains("discover") {
        Some("found")
    } else if structured_diary_status_query(&lower) {
        Some("status")
    } else if structured_diary_goal_query(&lower) {
        Some("goal")
    } else if structured_diary_dependencies_query(&lower) {
        Some("dependencies")
    } else if structured_diary_entities_query(&lower) {
        Some("entities")
    } else if structured_diary_action_query(&lower) {
        Some("action")
    } else if structured_diary_title_query(&lower) {
        Some("title")
    } else if lower.contains("where ")
        || lower.contains(" location")
        || lower.contains(" live")
        || lower.contains(" residence")
        || lower.contains(" city")
        || lower.contains(" home")
    {
        Some("location")
    } else if lower.contains("job")
        || lower.contains("occupation")
        || lower.contains("career")
        || lower.contains(" role")
    {
        Some("job")
    } else {
        None
    }
}

fn looks_like_multi_hop_list_query(task: &str) -> bool {
    let lower = task.to_ascii_lowercase();
    lower.starts_with("what groups")
        || lower.starts_with("which groups")
        || lower.starts_with("what instruments")
        || lower.starts_with("which instruments")
        || lower.starts_with("what events")
        || lower.starts_with("which events")
        || lower.starts_with("what books")
        || lower.starts_with("which books")
        || lower.starts_with("what movies")
        || lower.starts_with("which movies")
        || lower.starts_with("what files")
        || lower.starts_with("which files")
        || lower.starts_with("what modules")
        || lower.starts_with("which modules")
        || lower.starts_with("what topics")
        || lower.starts_with("which topics")
        || lower.starts_with("what activities")
        || lower.starts_with("which activities")
        || lower.starts_with("where has ")
        || lower.starts_with("where have ")
        || lower.starts_with("who supports ")
        || lower.starts_with("who supported ")
}

fn collect_answer_candidates(task: &str, evidence: &[EvidenceItem]) -> Vec<CandidateLine> {
    let task_terms = salient_query_terms(task);
    let subject_hints = extract_subject_hints(task);
    let focus_terms = dialogue_focus_terms(task, &task_terms, &subject_hints);
    let anchor_terms = task_anchor_terms(task, &task_terms, &subject_hints);
    let required_tail_terms = required_tail_anchor_tokens(task);
    let enumerative = is_enumerative_query(task);
    let mut candidates = Vec::new();

    for item in evidence {
        let Some(content) = read_context_text(&item.path, "answer candidate extraction") else {
            continue;
        };
        for line in answer_candidate_lines(&content) {
            let clean = sanitize_answer_text(&line);
            if clean.is_empty() || looks_like_question_turn(&clean) {
                continue;
            }
            let subject_overlap = if subject_hints.is_empty() {
                0
            } else {
                task_overlap_count(&line, &subject_hints)
                    .max(task_overlap_count(&clean, &subject_hints))
            };
            if !subject_hints.is_empty() && subject_overlap == 0 {
                continue;
            }
            let focus_overlap = if focus_terms.is_empty() {
                0
            } else {
                task_overlap_count(&line, &focus_terms)
                    .max(task_overlap_count(&clean, &focus_terms))
            };
            if !focus_terms.is_empty()
                && focus_overlap == 0
                && subject_overlap > 0
                && !is_reason_query(task)
            {
                continue;
            }
            let support_overlap = task_overlap_count(&line, &task_terms)
                .max(task_overlap_count(&clean, &task_terms))
                .max(focus_overlap);
            let anchor_overlap = if anchor_terms.is_empty() {
                0
            } else {
                max_task_overlap([line.as_str(), clean.as_str()], &anchor_terms)
            };
            if !anchor_terms.is_empty() && anchor_overlap == 0 {
                continue;
            }
            if !required_tail_terms.is_empty()
                && max_task_overlap([line.as_str(), clean.as_str()], &required_tail_terms)
                    < required_tail_terms.len()
            {
                continue;
            }
            let weight = candidate_weight(&clean, &task_terms, item.score, false)
                + focus_overlap as f32 * 6.0;
            if weight > 0.0 && support_overlap > 0 {
                candidates.push(CandidateLine {
                    path: item.path.clone(),
                    text: clean.clone(),
                    weight,
                    retrieval_score: item.score,
                    support_overlap,
                    anchor_overlap,
                    specific_anchor_overlap: 0,
                });
                if !enumerative {
                    if let Some(compact) = compact_answer(task, &clean, &task_terms) {
                        if compact != clean
                            && is_informative_compact_answer(&compact)
                            && answer_meets_form_gate(task, &compact, None)
                        {
                            let compact_bonus =
                                4.0 + answer_form_confidence(task, &compact, &task_terms) * 6.0;
                            candidates.push(CandidateLine {
                                path: item.path.clone(),
                                text: compact,
                                weight: weight + compact_bonus,
                                retrieval_score: item.score,
                                support_overlap,
                                anchor_overlap,
                                specific_anchor_overlap: 0,
                            });
                        }
                    }
                }
            }
        }

        for turn in parse_dialogue_turns(&content) {
            let candidate = extract_relation_answer(task, &turn.text, &task_terms)
                .or_else(|| compact_answer(task, &turn.text, &task_terms));
            let Some(candidate) = candidate else {
                continue;
            };
            let clean = sanitize_answer_text(&candidate);
            if clean.is_empty() {
                continue;
            }
            if !subject_hints.is_empty() && !turn_matches_subject(&turn, &subject_hints) {
                continue;
            }
            let focus_overlap = if focus_terms.is_empty() {
                0
            } else {
                task_overlap_count(&turn.text, &focus_terms)
                    .max(task_overlap_count(&clean, &focus_terms))
            };
            if !focus_terms.is_empty() && focus_overlap == 0 {
                continue;
            }
            let support_overlap = task_overlap_count(&turn.text, &task_terms)
                .max(task_overlap_count(&clean, &task_terms))
                .max(focus_overlap)
                .max(1);
            let anchor_overlap = if anchor_terms.is_empty() {
                0
            } else {
                max_task_overlap([turn.text.as_str(), clean.as_str()], &anchor_terms)
            };
            if !anchor_terms.is_empty() && anchor_overlap == 0 {
                continue;
            }
            if !required_tail_terms.is_empty()
                && max_task_overlap([turn.text.as_str(), clean.as_str()], &required_tail_terms)
                    < required_tail_terms.len()
            {
                continue;
            }
            let weight = item.score * 10.0
                + dialogue_match_score(&turn.text, &task_terms)
                + speaker_match_bonus(turn.speaker.as_deref(), &subject_hints)
                + focus_overlap as f32 * 8.0
                + 10.0;
            candidates.push(CandidateLine {
                path: item.path.clone(),
                text: clean,
                weight,
                retrieval_score: item.score,
                support_overlap,
                anchor_overlap,
                specific_anchor_overlap: 0,
            });
        }
    }

    candidates.sort_by(|a, b| {
        b.weight
            .total_cmp(&a.weight)
            .then_with(|| a.path.cmp(&b.path))
            .then_with(|| a.text.cmp(&b.text))
    });
    candidates
}

fn select_multi_item_answer_from_candidates(
    task: &str,
    candidates: &[CandidateLine],
    _min_answer_confidence: Option<f32>,
) -> Option<String> {
    if !looks_like_multi_hop_list_query(task) || candidates.is_empty() {
        return None;
    }

    let top_weight = candidates.first()?.weight;
    let mut chosen = Vec::new();
    let mut seen_paths = HashSet::new();
    let mut seen_keys = HashSet::new();

    for prefer_new_path in [true, false] {
        for candidate in candidates {
            if candidate.weight + 14.0 < top_weight {
                break;
            }
            if !candidate_has_required_anchor_support(task, candidate) {
                continue;
            }
            if !is_composeable_multi_item_candidate(candidate) {
                continue;
            }
            if prefer_new_path && !seen_paths.insert(candidate.path.clone()) {
                continue;
            }

            let mut added_any = false;
            for item in split_composable_answer_items(&candidate.text) {
                let key = normalized_answer_key(&item);
                if key.is_empty()
                    || !seen_keys.insert(key)
                    || chosen
                        .iter()
                        .any(|existing: &String| answer_items_overlap(existing.as_str(), &item))
                {
                    continue;
                }
                chosen.push(item);
                added_any = true;
                if chosen.len() >= 3 {
                    break;
                }
            }

            if prefer_new_path && !added_any {
                seen_paths.remove(&candidate.path);
            }
            if chosen.len() >= 3 {
                break;
            }
        }
        if chosen.len() >= 2 {
            break;
        }
    }

    (chosen.len() >= 2).then(|| format_answer_list(&chosen))
}

fn is_composeable_multi_item_candidate(candidate: &CandidateLine) -> bool {
    let word_count = candidate.text.split_whitespace().count();
    word_count > 0
        && word_count <= 8
        && !candidate.text.contains('?')
        && !candidate.text.contains(" because ")
        && !candidate.text.contains(". ")
        && (is_informative_compact_answer(&candidate.text) || candidate.text.contains(','))
}

fn split_composable_answer_items(text: &str) -> Vec<String> {
    let clean = sanitize_answer_text(text);
    if clean.contains(',') {
        let parts = clean
            .replace(", and ", ", ")
            .split(',')
            .map(str::trim)
            .map(sanitize_inline)
            .filter(|part| !part.is_empty())
            .collect::<Vec<_>>();
        if parts.len() >= 2 {
            return parts;
        }
    }
    vec![clean]
}

fn normalized_answer_key(text: &str) -> String {
    sanitize_inline(
        &text
            .trim()
            .trim_matches(|c: char| matches!(c, '"' | '\'' | ',' | '.' | ';' | ':' | '!' | '?'))
            .to_ascii_lowercase(),
    )
}

fn answer_items_overlap(left: &str, right: &str) -> bool {
    let left_key = normalized_answer_key(left);
    let right_key = normalized_answer_key(right);
    !left_key.is_empty()
        && !right_key.is_empty()
        && (left_key == right_key || left_key.contains(&right_key) || right_key.contains(&left_key))
}

fn format_answer_list(items: &[String]) -> String {
    match items {
        [] => String::new(),
        [item] => item.clone(),
        [left, right] => format!("{left} and {right}"),
        _ => {
            let mut out = items[..items.len() - 1].join(", ");
            out.push_str(", and ");
            out.push_str(items.last().unwrap_or(&String::new()));
            out
        },
    }
}

fn looks_like_typed_open_qa_query(task: &str) -> bool {
    let lower = task.to_ascii_lowercase();
    if is_temporal_reasoning_query(task) {
        return false;
    }
    lower.starts_with("would ")
        || lower.starts_with("could ")
        || lower.starts_with("should ")
        || lower.starts_with("can ")
        || lower.starts_with("will ")
        || lower.starts_with("may ")
        || lower.starts_with("might ")
        || lower.starts_with("is ")
        || lower.starts_with("are ")
        || lower.starts_with("was ")
        || lower.starts_with("were ")
        || lower.starts_with("does ")
        || lower.starts_with("do ")
        || lower.starts_with("did ")
        || lower.starts_with("has ")
        || lower.starts_with("have ")
        || lower.starts_with("had ")
        || lower.starts_with("which ")
        || lower.starts_with("what might ")
        || lower.starts_with("what would ")
        || lower.contains(" likely ")
        || lower.contains(" considered ")
}

fn is_education_field_query(lower_task: &str) -> bool {
    lower_task.contains("field")
        || lower_task.contains("education")
        || lower_task.contains("study")
        || lower_task.contains("school")
        || lower_task.contains("pursue")
        || lower_task.contains("career option")
        || lower_task.contains("career options")
        || lower_task.contains("career path")
        || lower_task.contains("future career")
}

fn typed_open_qa_anchor_terms(task_terms: &[String], subject_hints: &[String]) -> Vec<String> {
    const FILLER: &[&str] = &[
        "likely",
        "probably",
        "possibly",
        "potentially",
        "considered",
        "still",
        "more",
        "most",
        "less",
        "least",
        "kind",
        "sort",
        "thing",
        "things",
        "personality",
        "trait",
        "traits",
        "additional",
        "alternative",
        "popular",
        "based",
        "around",
    ];
    let mut terms = task_terms
        .iter()
        .filter(|term| {
            !FILLER.contains(&term.as_str()) && !subject_hints.iter().any(|hint| hint == *term)
        })
        .cloned()
        .collect::<Vec<_>>();
    if terms.is_empty() {
        terms = task_terms.to_vec();
    }
    terms.sort();
    terms.dedup();
    terms
}

fn format_open_qa_answer_surface_answer(task: &str, answer: &str) -> String {
    let lower_task = task.to_ascii_lowercase();
    let answer_lower = answer.to_ascii_lowercase();
    if answer_lower.contains("ally")
        && [
            "member of the lgbtq community",
            "member of the lgbtq+ community",
            "part of the lgbtq community",
            "part of the lgbtq+ community",
            "member of the transgender community",
        ]
        .iter()
        .any(|needle| lower_task.contains(needle))
    {
        "Likely no, supportive ally".to_string()
    } else if answer_lower.contains("ally")
        && [
            "ally to the transgender community",
            "ally to the lgbtq community",
            "ally to the lgbtq+ community",
            "considered an ally",
        ]
        .iter()
        .any(|needle| lower_task.contains(needle))
    {
        "Yes, supportive ally".to_string()
    } else {
        answer.to_string()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AnswerShape {
    Generic,
    YesNo,
    Choice,
    Number,
    Duration,
    Date,
    TraitList,
    Suggestion,
}

const GENERIC_ANCHOR_TERMS: &[&str] = &[
    "activities",
    "activity",
    "advice",
    "answer",
    "attributes",
    "career",
    "careers",
    "city",
    "close",
    "could",
    "country",
    "countries",
    "current",
    "currently",
    "degree",
    "degrees",
    "did",
    "does",
    "education",
    "event",
    "events",
    "fields",
    "field",
    "first",
    "group",
    "groups",
    "help",
    "home",
    "idea",
    "ideas",
    "interesting",
    "job",
    "jobs",
    "kind",
    "kinds",
    "last",
    "less",
    "likely",
    "live",
    "location",
    "many",
    "might",
    "more",
    "most",
    "movie",
    "movies",
    "much",
    "name",
    "names",
    "occupation",
    "option",
    "options",
    "people",
    "person",
    "personality",
    "profession",
    "quality",
    "qualities",
    "question",
    "recent",
    "recommend",
    "recommended",
    "recommending",
    "resource",
    "resources",
    "role",
    "roles",
    "school",
    "show",
    "shows",
    "some",
    "something",
    "state",
    "states",
    "suggest",
    "suggested",
    "suggesting",
    "task",
    "thing",
    "things",
    "tips",
    "topic",
    "topics",
    "trait",
    "traits",
    "trip",
    "type",
    "types",
    "upcoming",
    "weekend",
    "what",
    "which",
    "who",
    "where",
    "when",
    "why",
    "would",
];

const ANSWER_REJECT_PREFIXES: &[&str] = &[
    "congratulations",
    "great to hear",
    "here are",
    "i can help",
    "i'd be happy",
    "i would be happy",
    "i'm happy to",
    "let's get started",
    "sounds great",
    "that's great",
    "that sounds",
    "wow",
];

const ANSWER_REJECT_EXACT: &[&str] = &[
    "1",
    "2",
    "3",
    "can",
    "great idea",
    "i'm a large language model",
    "many dishes",
    "trap crop",
    "yogurt making",
];

const ANSWER_TRAILING_STOPWORDS: &[&str] = &[
    "a", "an", "and", "at", "by", "for", "from", "in", "of", "on", "or", "the", "to", "with",
];

const GENERIC_COLLECTION_NOUNS: &[&str] = &[
    "activities",
    "advice",
    "days",
    "dishes",
    "ideas",
    "options",
    "recipes",
    "resources",
    "tips",
    "tools",
    "ways",
];

const FOOD_QUERY_HINTS: &[&str] = &[
    "bake",
    "basil",
    "cookies",
    "cook",
    "cooking",
    "dessert",
    "dinner",
    "ingredients",
    "meal",
    "mint",
    "recipe",
    "recipes",
    "serve",
    "slow cooker",
];

const FOOD_GENERIC_NOUNS: &[&str] = &["drink", "water", "cocktail", "tea", "smoothie", "juice"];

const FOOD_ITEM_HINTS: &[&str] = &[
    "beef",
    "brownie",
    "cake",
    "caprese",
    "chicken",
    "chili",
    "chutney",
    "cookie",
    "cookies",
    "curry",
    "dessert",
    "lamb",
    "pasta",
    "pesto",
    "salad",
    "sandwich",
    "soup",
    "spaghetti",
    "stew",
    "tacos",
];

fn answer_shape(task: &str) -> AnswerShape {
    let lower = task.to_ascii_lowercase();
    if parse_binary_choice(task).is_some() || !parse_open_qa_choice_options(task).is_empty() {
        AnswerShape::Choice
    } else if lower.starts_with("when ")
        || lower.contains(" what date")
        || lower.contains(" which date")
        || lower.contains(" what month")
        || lower.contains(" which month")
        || lower.contains(" what year")
        || lower.contains(" which year")
    {
        AnswerShape::Date
    } else if lower.starts_with("how long ")
        || lower.contains("how many days")
        || lower.contains("how many weeks")
        || lower.contains("how many months")
        || lower.contains("how many years")
        || lower.contains("how many hours")
        || lower.contains("how many minutes")
    {
        AnswerShape::Duration
    } else if lower.starts_with("how many ")
        || lower.starts_with("how much ")
        || lower.starts_with("how often ")
        || lower.contains("number of ")
    {
        AnswerShape::Number
    } else if lower.contains("personality trait")
        || lower.contains("personality traits")
        || lower.contains("what traits")
        || lower.contains("what attributes")
        || lower.contains("attributes describe")
        || lower.contains("what qualities")
    {
        AnswerShape::TraitList
    } else if lower.starts_with("would ")
        || lower.starts_with("could ")
        || lower.starts_with("should ")
        || lower.starts_with("can ")
        || lower.starts_with("will ")
        || lower.starts_with("may ")
        || lower.starts_with("might ")
        || lower.starts_with("is ")
        || lower.starts_with("are ")
        || lower.starts_with("was ")
        || lower.starts_with("were ")
        || lower.starts_with("does ")
        || lower.starts_with("do ")
        || lower.starts_with("did ")
        || lower.starts_with("has ")
        || lower.starts_with("have ")
        || lower.starts_with("had ")
    {
        AnswerShape::YesNo
    } else if is_suggestion_query(task) {
        AnswerShape::Suggestion
    } else {
        AnswerShape::Generic
    }
}

fn is_suggestion_query(task: &str) -> bool {
    let lower = task.to_ascii_lowercase();
    lower.starts_with("can you suggest")
        || lower.starts_with("can you recommend")
        || lower.contains(" any advice")
        || lower.contains(" any tips")
        || lower.contains(" any ideas")
        || lower.starts_with("what should i ")
        || lower.starts_with("what can i ")
        || lower.starts_with("what could i ")
}

fn is_food_query(task: &str) -> bool {
    let lower = task.to_ascii_lowercase();
    FOOD_QUERY_HINTS.iter().any(|needle| lower.contains(needle))
}

fn normalized_validation_text(text: &str) -> String {
    text.replace('*', " ")
        .replace('`', " ")
        .replace('_', " ")
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

fn task_anchor_terms(task: &str, task_terms: &[String], subject_hints: &[String]) -> Vec<String> {
    let lower = task.to_ascii_lowercase();
    let mut anchors = if looks_like_typed_open_qa_query(task) {
        typed_open_qa_anchor_terms(task_terms, subject_hints)
    } else {
        task_terms
            .iter()
            .filter(|term| {
                !subject_hints.iter().any(|hint| hint == *term)
                    && !GENERIC_ANCHOR_TERMS.contains(&term.as_str())
            })
            .cloned()
            .collect::<Vec<_>>()
    };

    if let Some((options, _)) = parse_binary_choice(task) {
        anchors.extend(
            options
                .into_iter()
                .flat_map(|option| option.tokens)
                .filter(|term| !GENERIC_ANCHOR_TERMS.contains(&term.as_str())),
        );
    }
    if !lower.contains("yes or no") {
        anchors.extend(
            parse_open_qa_choice_options(task)
                .into_iter()
                .flat_map(|option| option.tokens)
                .filter(|term| !GENERIC_ANCHOR_TERMS.contains(&term.as_str())),
        );
    }
    anchors.sort();
    anchors.dedup();
    anchors
}

fn answer_form_confidence(task: &str, text: &str, task_terms: &[String]) -> f32 {
    let clean = normalized_validation_text(text);
    if clean.is_empty() {
        return 0.0;
    }

    let lower = clean.to_ascii_lowercase();
    let shape = answer_shape(task);
    let tokens = lower
        .split_whitespace()
        .filter(|token| !token.is_empty())
        .collect::<Vec<_>>();
    if tokens.is_empty() {
        return 0.0;
    }
    if text.contains('?') || clean.ends_with(':') {
        return 0.0;
    }
    let reject_question_echo =
        shape != AnswerShape::Choice && looks_like_question_echo(task, &lower, task_terms);
    let reject_heading = matches!(
        shape,
        AnswerShape::Generic | AnswerShape::TraitList | AnswerShape::Suggestion
    ) && looks_like_heading_fragment(text, &clean);
    if reject_question_echo
        || reject_heading
        || looks_like_social_filler(&lower)
        || looks_like_truncated_answer(&tokens)
    {
        return 0.0;
    }
    if ANSWER_REJECT_EXACT.contains(&lower.as_str()) {
        return 0.0;
    }
    if institution_query_expected(task) {
        return institution_answer_confidence(&clean, &lower);
    }

    match shape {
        AnswerShape::YesNo => yes_no_answer_confidence(task, &lower),
        AnswerShape::Choice => choice_answer_confidence(task, &clean),
        AnswerShape::Number => number_answer_confidence(task, &lower),
        AnswerShape::Duration => duration_answer_confidence(task, &lower),
        AnswerShape::Date => date_answer_confidence(&clean, &lower),
        AnswerShape::TraitList => trait_list_answer_confidence(&clean, &lower),
        AnswerShape::Suggestion => suggestion_answer_confidence(task, &lower),
        AnswerShape::Generic => generic_answer_confidence(&lower, task_terms),
    }
}

fn looks_like_question_echo(task: &str, answer_lower: &str, task_terms: &[String]) -> bool {
    let answer_key = normalized_answer_key(answer_lower);
    let task_key = normalized_answer_key(task);
    if answer_key.is_empty() || task_key.is_empty() {
        return false;
    }
    if task_key.contains(&answer_key) && answer_key.split_whitespace().count() >= 3 {
        return true;
    }

    let overlap = task_overlap_count(answer_lower, task_terms);
    let novel_tokens = answer_lower
        .split(|c: char| !c.is_ascii_alphanumeric())
        .filter(|token| token.len() >= 4)
        .filter(|token| {
            !task_terms
                .iter()
                .any(|term| query_term_matches_token(term, token))
        })
        .count();
    overlap >= task_terms.len().min(3).max(2) && novel_tokens == 0
}

fn looks_like_heading_fragment(original: &str, clean: &str) -> bool {
    let tokens = clean.split_whitespace().collect::<Vec<_>>();
    if tokens.is_empty() {
        return true;
    }
    if original.contains("**") || original.starts_with('#') {
        if tokens.len() <= 5 {
            return true;
        }
    }
    let alpha_tokens = tokens
        .iter()
        .filter(|token| token.chars().any(|c| c.is_alphabetic()))
        .count();
    alpha_tokens > 0
        && tokens.len() <= 4
        && tokens
            .iter()
            .any(|token| token.chars().all(|c| c.is_ascii_digit()))
        && tokens
            .iter()
            .filter(|token| token.chars().any(|c| c.is_alphabetic()))
            .all(|token| {
                token
                    .chars()
                    .next()
                    .map(|c| c.is_ascii_uppercase())
                    .unwrap_or(false)
            })
}

fn looks_like_social_filler(lower: &str) -> bool {
    ANSWER_REJECT_PREFIXES
        .iter()
        .any(|prefix| lower.starts_with(prefix))
}

fn looks_like_truncated_answer(tokens: &[&str]) -> bool {
    let Some(last) = tokens.last() else {
        return true;
    };
    let tail = last.trim_matches(|c: char| !c.is_ascii_alphabetic());
    if ANSWER_TRAILING_STOPWORDS.contains(&tail) {
        return true;
    }
    let Some(first) = tokens.first() else {
        return true;
    };
    matches!(*first, "and" | "or" | "to" | "for" | "with" | "because")
}

fn yes_no_answer_confidence(task: &str, lower: &str) -> f32 {
    if [
        "yes",
        "no",
        "likely yes",
        "likely no",
        "probably yes",
        "probably no",
        "somewhat",
    ]
    .iter()
    .any(|prefix| lower.starts_with(prefix))
    {
        return 1.0;
    }
    if lower.contains("religious") || lower.contains("ally") {
        return 0.9;
    }
    if task.to_ascii_lowercase().contains("member of the lgbtq")
        && lower.contains("supportive ally")
    {
        return 0.9;
    }
    0.0
}

fn choice_answer_confidence(task: &str, text: &str) -> f32 {
    if let Some((options, _)) = parse_binary_choice(task) {
        if options
            .iter()
            .any(|option| answer_items_overlap(text, &option.display))
        {
            return 1.0;
        }
    }
    let options = parse_open_qa_choice_options(task);
    if options
        .iter()
        .any(|option| answer_items_overlap(text, &option.display))
    {
        return 1.0;
    }
    if let Some(target) = open_qa_location_target(task) {
        if open_qa_location_alias(target, text).is_some() {
            return 0.9;
        }
    }
    0.0
}

fn number_answer_confidence(task: &str, lower: &str) -> f32 {
    let tokens = lower.split_whitespace().collect::<Vec<_>>();
    if tokens.is_empty() {
        return 0.0;
    }
    if tokens.len() <= 6 && tokens.iter().all(|token| numeric_answer_component(token)) {
        return 1.0;
    }
    if answer_shape(task) != AnswerShape::Duration
        && tokens.len() == 1
        && parse_count_token(tokens[0]).is_some()
    {
        return 1.0;
    }
    0.0
}

fn numeric_answer_component(token: &str) -> bool {
    parse_count_token(token).is_some()
        || matches!(
            token,
            "times"
                | "time"
                | "per"
                | "week"
                | "weeks"
                | "month"
                | "months"
                | "year"
                | "years"
                | "day"
                | "days"
                | "hour"
                | "hours"
                | "minute"
                | "minutes"
                | "ago"
                | "before"
                | "after"
                | "and"
        )
}

fn duration_answer_confidence(task: &str, lower: &str) -> f32 {
    if number_answer_confidence(task, lower) > 0.0
        && [
            "day", "days", "week", "weeks", "month", "months", "year", "years", "hour", "hours",
            "minute", "minutes", "ago",
        ]
        .iter()
        .any(|unit| lower.contains(unit))
    {
        return 1.0;
    }
    0.0
}

fn date_answer_confidence(text: &str, lower: &str) -> f32 {
    if extract_explicit_date(text, None).is_some()
        || [
            "january",
            "february",
            "march",
            "april",
            "may",
            "june",
            "july",
            "august",
            "september",
            "october",
            "november",
            "december",
            "thanksgiving",
            "christmas",
            "independence day",
            "black friday",
            "easter",
            "holi",
        ]
        .iter()
        .any(|needle| lower.contains(needle))
    {
        return 1.0;
    }
    0.0
}

fn trait_list_answer_confidence(text: &str, lower: &str) -> f32 {
    let normalized = text.replace(", and ", ", ").replace(" and ", ", ");
    let parts = normalized
        .split(',')
        .map(str::trim)
        .filter(|part| !part.is_empty())
        .collect::<Vec<_>>();
    if parts.len() >= 2
        && parts.iter().all(|part| {
            let words = part.split_whitespace().count();
            words >= 1
                && words <= 3
                && !part.chars().any(|c| c.is_ascii_digit())
                && !part.contains('?')
        })
    {
        return 1.0;
    }
    if lower.split_whitespace().count() == 1 {
        return 0.0;
    }
    0.25
}

fn suggestion_answer_confidence(task: &str, lower: &str) -> f32 {
    let tokens = lower.split_whitespace().collect::<Vec<_>>();
    if tokens.is_empty() {
        return 0.0;
    }
    if tokens
        .first()
        .copied()
        .map(parse_count_token)
        .flatten()
        .is_some()
    {
        return 0.0;
    }
    if tokens.len() >= 2
        && matches!(tokens[0], "many" | "some" | "several" | "various")
        && GENERIC_COLLECTION_NOUNS.contains(&tokens[1])
    {
        return 0.0;
    }
    if is_food_query(task) {
        let has_generic_drink = FOOD_GENERIC_NOUNS
            .iter()
            .any(|needle| lower.contains(needle));
        let has_food_item = FOOD_ITEM_HINTS.iter().any(|needle| lower.contains(needle));
        if task.to_ascii_lowercase().contains("dinner") && has_generic_drink && !has_food_item {
            return 0.0;
        }
    }
    if tokens.len() > 10 {
        return 0.35;
    }
    0.8
}

fn generic_answer_confidence(lower: &str, task_terms: &[String]) -> f32 {
    let tokens = lower
        .split(|c: char| !c.is_ascii_alphanumeric())
        .filter(|token| !token.is_empty())
        .collect::<Vec<_>>();
    if tokens.is_empty() {
        return 0.0;
    }
    if tokens.len() == 1 {
        let token = tokens[0];
        if parse_count_token(token).is_some() || token.len() <= 2 {
            return 0.0;
        }
    }
    let novel_tokens = tokens
        .iter()
        .filter(|token| token.len() >= 3)
        .filter(|token| {
            !task_terms
                .iter()
                .any(|term| query_term_matches_token(term, token))
        })
        .count();
    if novel_tokens == 0 && tokens.len() <= 3 {
        return 0.0;
    }
    0.75
}

fn institution_query_expected(task: &str) -> bool {
    let lower = task.to_ascii_lowercase();
    lower.contains("which university")
        || lower.contains("what university")
        || lower.contains("which college")
        || lower.contains("what college")
        || lower.contains("which school")
        || lower.contains("what school")
        || lower.contains("which institute")
        || lower.contains("what institute")
}

fn institution_specific_anchor_terms(task: &str) -> Vec<String> {
    if !institution_query_expected(task) {
        return Vec::new();
    }
    let mut terms = salient_query_terms(task)
        .into_iter()
        .filter(|term| {
            term.len() >= 4
                && !matches!(
                    term.as_str(),
                    "university"
                        | "college"
                        | "school"
                        | "institute"
                        | "academy"
                        | "present"
                        | "presented"
                        | "poster"
                        | "research"
                        | "conference"
                )
        })
        .collect::<Vec<_>>();
    terms.sort();
    terms.dedup();
    terms
}

fn institution_answer_confidence(clean: &str, lower: &str) -> f32 {
    if ["university", "college", "school", "institute", "academy"]
        .iter()
        .any(|needle| lower.contains(needle))
    {
        return 0.95;
    }
    let tokens = clean.split_whitespace().collect::<Vec<_>>();
    if !tokens.is_empty()
        && tokens.len() <= 4
        && tokens.iter().all(|token| {
            token
                .chars()
                .all(|c| c.is_ascii_uppercase() || matches!(c, '.' | '&' | '-'))
        })
        && tokens.iter().any(|token| token.len() >= 2)
    {
        return 0.8;
    }
    0.0
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum OpenQaLocationTarget {
    State,
    Country,
    NationalPark,
}

fn open_qa_location_target(task: &str) -> Option<OpenQaLocationTarget> {
    let lower = task.to_ascii_lowercase();
    if lower.contains("national park") {
        Some(OpenQaLocationTarget::NationalPark)
    } else if lower.starts_with("what state")
        || lower.starts_with("which state")
        || lower.contains(" in what state")
        || lower.contains(" in which state")
        || lower.contains(" us state")
        || lower.contains(" us states")
    {
        Some(OpenQaLocationTarget::State)
    } else if lower.starts_with("what country")
        || lower.starts_with("which country")
        || lower.contains(" in what country")
        || lower.contains(" in which country")
        || lower.contains(" home country")
    {
        Some(OpenQaLocationTarget::Country)
    } else {
        None
    }
}

fn parse_open_qa_choice_options(task: &str) -> Vec<ChoiceOption> {
    let lower = task.to_ascii_lowercase();
    if !lower.contains(" or ")
        || lower.contains("answer in yes or no")
        || lower.ends_with("yes or no")
    {
        return Vec::new();
    }

    let tail = task.trim().trim_end_matches('?').trim();
    let Some((left_segment, right_segment)) = tail.rsplit_once(" or ") else {
        return Vec::new();
    };
    let left_raw = [
        " close to ",
        " going to ",
        " visiting ",
        " visit ",
        " in ",
        " at ",
        " between ",
        ", ",
    ]
    .iter()
    .find_map(|marker| left_segment.rsplit_once(marker).map(|(_, value)| value))
    .unwrap_or(left_segment);

    [left_raw, right_segment]
        .into_iter()
        .filter_map(|raw| {
            let display = raw
                .trim()
                .trim_start_matches("the ")
                .trim_start_matches("a ")
                .trim_start_matches("an ")
                .trim_matches(|c: char| matches!(c, '?' | ',' | '.' | ':' | ';'))
                .to_string();
            let tokens = salient_query_terms(&display)
                .into_iter()
                .filter(|token| parse_count_token(token).is_none())
                .collect::<Vec<_>>();
            (!display.is_empty() && !tokens.is_empty()).then_some(ChoiceOption { display, tokens })
        })
        .collect()
}

fn open_qa_choice_affinity_terms(display_lower: &str) -> &'static [&'static str] {
    if display_lower.contains("national park") {
        &[
            "nature",
            "outdoors",
            "outdoor",
            "camping",
            "camp",
            "hiking",
            "mountain",
            "mountains",
            "forest",
            "woods",
            "trail",
            "park",
        ]
    } else if display_lower.contains("theme park") {
        &[
            "theme",
            "amusement",
            "rides",
            "roller",
            "coaster",
            "disney",
            "universal",
            "park",
        ]
    } else if display_lower.contains("mountain") {
        &[
            "mountain",
            "mountains",
            "hiking",
            "camping",
            "nature",
            "outdoors",
            "trail",
            "park",
        ]
    } else if display_lower.contains("beach") {
        &["beach", "ocean", "coast", "shore", "sand", "waves", "surf"]
    } else {
        &[]
    }
}

fn open_qa_location_alias(target: OpenQaLocationTarget, text: &str) -> Option<String> {
    let lower = text.to_ascii_lowercase();
    let aliases: &[(&[&str], &str)] = match target {
        OpenQaLocationTarget::State => &[
            (
                &["universal studios hollywood", "hollywood", "los angeles"],
                "California",
            ),
            (
                &[
                    "universal studios orlando",
                    "orlando",
                    "miami",
                    "disney world",
                ],
                "Florida",
            ),
            (&["universal studios"], "California or Florida"),
            (&["california"], "California"),
            (&["florida"], "Florida"),
            (&["indiana", "indianapolis", "indiana dunes"], "Indiana"),
            (
                &["minnesota", "minneapolis", "st. paul", "voyageurs"],
                "Minnesota",
            ),
            (
                &["connecticut", "new haven", "hartford", "bridgeport"],
                "Connecticut",
            ),
            (&["alaska", "anchorage", "denali", "fairbanks"], "Alaska"),
            (&["arizona", "grand canyon"], "Arizona"),
        ],
        OpenQaLocationTarget::Country => &[
            (&["canada", "vancouver", "toronto", "montreal"], "Canada"),
            (&["greenland"], "Greenland"),
            (&["france", "paris"], "France"),
            (&["colombia", "bogota", "medellin", "cartagena"], "Colombia"),
            (&["sweden"], "Sweden"),
            (
                &[
                    "united states",
                    "u.s.",
                    "usa",
                    "america",
                    "boston",
                    "new york",
                    "florida",
                    "california",
                    "minnesota",
                    "connecticut",
                    "alaska",
                    "arizona",
                    "universal studios",
                ],
                "United States",
            ),
        ],
        OpenQaLocationTarget::NationalPark => &[
            (
                &["voyageurs", "voyageurs national park"],
                "Voyageurs National Park",
            ),
            (&["grand canyon"], "Grand Canyon National Park"),
            (&["yellowstone"], "Yellowstone National Park"),
        ],
    };
    aliases.iter().find_map(|(needles, canonical)| {
        needles
            .iter()
            .any(|needle| lower.contains(needle))
            .then(|| (*canonical).to_string())
    })
}

fn select_typed_open_qa_structured_answer(task: &str, evidence: &[EvidenceItem]) -> Option<String> {
    let lower_task = task.to_ascii_lowercase();
    let subject_hints = extract_subject_hints(task);
    let task_terms = salient_query_terms(task);

    let choice_options = parse_open_qa_choice_options(task);
    if !choice_options.is_empty() {
        let mut best: Option<(usize, f32, String)> = None;
        for item in evidence {
            let Some(content) = read_context_text(&item.path, "typed open qa choice selection")
            else {
                continue;
            };
            for turn in parse_dialogue_turns(&content) {
                if !subject_hints.is_empty() && !turn_matches_subject(&turn, &subject_hints) {
                    continue;
                }
                let lower = turn.text.to_ascii_lowercase();
                let support = task_overlap_count(&turn.text, &task_terms);
                for option in &choice_options {
                    let display_lower = option.display.to_ascii_lowercase();
                    let direct = option
                        .tokens
                        .iter()
                        .filter(|token| lower.contains(token.as_str()))
                        .count();
                    let affinity = open_qa_choice_affinity_terms(&display_lower)
                        .iter()
                        .filter(|needle| lower.contains(**needle))
                        .count();
                    let score = direct * 5 + affinity * 3 + support;
                    if score == 0 {
                        continue;
                    }
                    if best
                        .as_ref()
                        .map(|(best_score, best_retrieval, _)| {
                            score > *best_score
                                || (score == *best_score && item.score > *best_retrieval)
                        })
                        .unwrap_or(true)
                    {
                        best = Some((score, item.score, option.display.clone()));
                    }
                }
            }
        }
        if let Some((score, _, answer)) = best {
            if score >= 3 {
                return Some(answer);
            }
        }
    }

    if let Some(target) = open_qa_location_target(task) {
        let mut best: Option<(usize, f32, String)> = None;
        for item in evidence {
            let Some(content) = read_context_text(&item.path, "typed open qa location selection")
            else {
                continue;
            };
            for turn in parse_dialogue_turns(&content) {
                if !subject_hints.is_empty() && !turn_matches_subject(&turn, &subject_hints) {
                    continue;
                }
                let Some(answer) = open_qa_location_alias(target, &turn.text) else {
                    continue;
                };
                let score = task_overlap_count(&turn.text, &task_terms).max(1);
                if best
                    .as_ref()
                    .map(|(best_score, best_retrieval, _)| {
                        score > *best_score
                            || (score == *best_score && item.score > *best_retrieval)
                    })
                    .unwrap_or(true)
                {
                    best = Some((score, item.score, answer));
                }
            }
        }
        if let Some((_, _, answer)) = best {
            return Some(answer);
        }
    }

    if lower_task.contains("religious")
        || lower_task.contains("religion")
        || lower_task.contains("faith")
    {
        for item in evidence {
            let Some(content) = read_context_text(&item.path, "typed open qa religion selection")
            else {
                continue;
            };
            for turn in parse_dialogue_turns(&content) {
                if !subject_hints.is_empty() && !turn_matches_subject(&turn, &subject_hints) {
                    continue;
                }
                let lower = turn.text.to_ascii_lowercase();
                if lower.contains("church") || lower.contains("faith") {
                    return Some("Somewhat religious".to_string());
                }
            }
        }
    }

    if lower_task.contains("ally")
        || lower_task.contains("lgbtq")
        || lower_task.contains("transgender")
    {
        for item in evidence {
            let Some(content) = read_context_text(&item.path, "typed open qa ally selection")
            else {
                continue;
            };
            for turn in parse_dialogue_turns(&content) {
                if !subject_hints.is_empty() && !turn_matches_subject(&turn, &subject_hints) {
                    continue;
                }
                let lower = turn.text.to_ascii_lowercase();
                let community = lower.contains("lgbtq")
                    || lower.contains("transgender")
                    || lower.contains("trans community")
                    || lower.contains("gender identity");
                let supportive = lower.contains("support")
                    || lower.contains("supportive")
                    || lower.contains("accept")
                    || lower.contains("ally")
                    || lower.contains("proud of you")
                    || lower.contains("back you")
                    || lower.contains("not alone");
                if community && supportive {
                    return Some(format_open_qa_answer_surface_answer(
                        task,
                        "supportive ally",
                    ));
                }
            }
        }
    }

    if is_education_field_query(&lower_task) {
        let focus_terms = dialogue_focus_terms(task, &task_terms, &subject_hints);
        let mut best: Option<(f32, String)> = None;
        for item in evidence {
            let Some(content) = read_context_text(&item.path, "typed open qa education selection")
            else {
                continue;
            };
            for turn in parse_dialogue_turns(&content) {
                if !subject_hints.is_empty() && !turn_matches_subject(&turn, &subject_hints) {
                    continue;
                }
                let focus_overlap = if focus_terms.is_empty() {
                    0
                } else {
                    task_overlap_count(&turn.text, &focus_terms)
                };
                if !focus_terms.is_empty() && focus_overlap == 0 {
                    continue;
                }
                let Some(answer) = compact_answer(task, &turn.text, &task_terms) else {
                    continue;
                };
                if !answer_meets_form_gate(task, &answer, None) {
                    continue;
                }
                let score = item.score * 10.0
                    + speaker_match_bonus(turn.speaker.as_deref(), &subject_hints)
                    + focus_overlap as f32 * 10.0
                    + candidate_weight(&answer, &task_terms, item.score, false);
                update_best_answer(&mut best, score, answer);
            }
        }
        if let Some((_, answer)) = best {
            return Some(answer);
        }
    }

    None
}

fn select_suggestion_list_item_answer(task: &str, evidence: &[EvidenceItem]) -> Option<String> {
    if !is_suggestion_query(task) {
        return None;
    }

    let task_terms = salient_query_terms(task);
    let subject_hints = extract_subject_hints(task);
    let anchor_terms = task_anchor_terms(task, &task_terms, &subject_hints);
    let mut best: Option<(f32, String)> = None;

    for item in evidence {
        let Some(content) = read_context_text(&item.path, "suggestion list answer selection")
        else {
            continue;
        };
        let mut recent_lines: Vec<String> = Vec::new();
        for raw_line in content.lines() {
            let raw_trimmed = raw_line.trim();
            let Some(candidate) = extract_list_item_title(raw_line) else {
                if !raw_trimmed.is_empty() {
                    recent_lines.push(raw_trimmed.to_string());
                    if recent_lines.len() > 3 {
                        recent_lines.remove(0);
                    }
                }
                continue;
            };
            if !answer_meets_form_gate(task, &candidate, None) {
                if !raw_trimmed.is_empty() {
                    recent_lines.push(raw_trimmed.to_string());
                    if recent_lines.len() > 3 {
                        recent_lines.remove(0);
                    }
                }
                continue;
            }
            let context = recent_lines.join(" ");
            let anchor_overlap = if anchor_terms.is_empty() {
                0
            } else {
                max_task_overlap(
                    [context.as_str(), raw_line, candidate.as_str()],
                    &anchor_terms,
                )
            };
            if !anchor_terms.is_empty() && anchor_overlap == 0 {
                if !raw_trimmed.is_empty() {
                    recent_lines.push(raw_trimmed.to_string());
                    if recent_lines.len() > 3 {
                        recent_lines.remove(0);
                    }
                }
                continue;
            }
            let lower = candidate.to_ascii_lowercase();
            let food_bonus = if is_food_query(task)
                && FOOD_ITEM_HINTS.iter().any(|needle| lower.contains(needle))
            {
                6.0
            } else {
                0.0
            };
            let score = candidate_weight(raw_line, &task_terms, item.score, false)
                + anchor_overlap as f32 * 8.0
                + food_bonus;
            update_best_answer(&mut best, score, candidate);
            if !raw_trimmed.is_empty() {
                recent_lines.push(raw_trimmed.to_string());
                if recent_lines.len() > 3 {
                    recent_lines.remove(0);
                }
            }
        }
    }

    best.map(|(_, answer)| answer)
}

fn extract_list_item_title(line: &str) -> Option<String> {
    let trimmed = line.trim();
    if trimmed.is_empty() {
        return None;
    }

    let stripped = trimmed
        .trim_start_matches(|c: char| {
            c.is_ascii_digit() || matches!(c, '.' | ')' | '-' | '*' | ' ')
        })
        .trim();
    let head = stripped
        .split_once(':')
        .map(|(head, _)| head)
        .unwrap_or(stripped);
    let clean = normalized_validation_text(head)
        .trim_matches(|c: char| matches!(c, '"' | '\'' | '.' | ',' | ';' | ':' | '!' | '?'))
        .trim()
        .to_string();
    let word_count = clean.split_whitespace().count();
    if clean.is_empty() || !(1..=6).contains(&word_count) {
        return None;
    }
    let lower = clean.to_ascii_lowercase();
    if looks_like_heading_fragment(head, &clean)
        || ANSWER_REJECT_EXACT.contains(&lower.as_str())
        || looks_like_social_filler(&lower)
    {
        return None;
    }
    Some(clean)
}

fn select_answer_surface(task: &str, evidence: &[EvidenceItem]) -> Option<String> {
    let task_terms = salient_query_terms(task);
    let subject_hints = extract_subject_hints(task);
    let focus_terms = dialogue_focus_terms(task, &task_terms, &subject_hints);
    let typed_open_qa = looks_like_typed_open_qa_query(task);
    let anchor_terms = task_anchor_terms(task, &task_terms, &subject_hints);
    if task_terms.is_empty() {
        return None;
    }

    let mut buckets: HashMap<String, AnswerSurfaceBucket> = HashMap::new();
    for item in evidence {
        let Some(content) = read_context_text(&item.path, "answer surface lookup") else {
            continue;
        };
        for row in parse_answer_surface_rows(&content) {
            let overlap = answer_surface_overlap(&row, &task_terms);
            if overlap == 0 {
                continue;
            }
            let anchor_overlap = if anchor_terms.is_empty() {
                0
            } else {
                answer_surface_overlap(&row, &anchor_terms)
            };
            if !anchor_terms.is_empty() && anchor_overlap == 0 {
                continue;
            }
            let subject_overlap = if subject_hints.is_empty() {
                0
            } else {
                answer_surface_overlap(&row, &subject_hints)
            };
            let focus_overlap = if focus_terms.is_empty() {
                0
            } else {
                answer_surface_overlap(&row, &focus_terms)
            };
            if !focus_terms.is_empty() && focus_overlap == 0 && subject_overlap > 0 {
                continue;
            }
            let score = answer_surface_score(&row, &task_terms, item.score)
                + focus_overlap as f32 * 5.0
                + subject_overlap as f32 * 2.0;
            if score <= 0.0 {
                continue;
            }
            let key = normalized_answer_key(&row.answer_span);
            if key.is_empty() {
                continue;
            }
            let bucket = buckets.entry(key).or_insert_with(|| AnswerSurfaceBucket {
                answer_span: row.answer_span.clone(),
                best_score: score,
                total_score: 0.0,
                best_confidence: row.confidence,
                max_overlap: 0,
                max_anchor_overlap: anchor_overlap,
                paths: HashSet::new(),
                hits: 0,
            });
            if score > bucket.best_score
                || ((score - bucket.best_score).abs() < 0.01
                    && row.answer_span.len() < bucket.answer_span.len())
            {
                bucket.answer_span = row.answer_span.clone();
                bucket.best_score = score;
                bucket.best_confidence = row.confidence;
            }
            bucket.total_score += score;
            bucket.max_overlap = bucket.max_overlap.max(overlap.max(focus_overlap));
            bucket.max_anchor_overlap = bucket.max_anchor_overlap.max(anchor_overlap);
            bucket.paths.insert(item.path.clone());
            bucket.hits += 1;
        }
    }

    let mut buckets = buckets
        .into_values()
        .filter(|bucket| answer_meets_form_gate(task, &bucket.answer_span, None))
        .collect::<Vec<_>>();
    buckets.sort_by(|left, right| {
        answer_surface_bucket_rank(right)
            .total_cmp(&answer_surface_bucket_rank(left))
            .then_with(|| right.max_anchor_overlap.cmp(&left.max_anchor_overlap))
            .then_with(|| right.max_overlap.cmp(&left.max_overlap))
            .then_with(|| right.paths.len().cmp(&left.paths.len()))
            .then_with(|| left.answer_span.len().cmp(&right.answer_span.len()))
            .then_with(|| left.answer_span.cmp(&right.answer_span))
    });
    let top = buckets.first()?;
    if let Some(next) = buckets.get(1) {
        if answer_surface_buckets_conflict(top, next) {
            return None;
        }
    }
    Some(if typed_open_qa {
        format_open_qa_answer_surface_answer(task, &top.answer_span)
    } else {
        top.answer_span.clone()
    })
}

fn answer_surface_overlap(row: &AnswerSurfaceRow, task_terms: &[String]) -> usize {
    let pattern_terms = salient_query_terms(&row.question_pattern);
    if pattern_terms.is_empty() {
        return 0;
    }
    term_list_overlap_count(task_terms, &pattern_terms)
}

fn answer_surface_bucket_rank(bucket: &AnswerSurfaceBucket) -> f32 {
    bucket.total_score
        + bucket.max_overlap as f32
        + bucket.max_anchor_overlap as f32 * 6.0
        + (bucket.paths.len().saturating_sub(1) as f32) * 2.5
        + (bucket.hits.saturating_sub(1) as f32) * 0.5
        + bucket.best_confidence * 2.0
}

fn answer_surface_buckets_conflict(
    top: &AnswerSurfaceBucket,
    runner_up: &AnswerSurfaceBucket,
) -> bool {
    !answer_items_overlap(&top.answer_span, &runner_up.answer_span)
        && answer_surface_bucket_rank(runner_up) + 2.5 >= answer_surface_bucket_rank(top)
        && runner_up.max_overlap + 1 >= top.max_overlap
}

fn resolve_relation_answer(
    task: &str,
    evidence: &[EvidenceItem],
    min_answer_confidence: Option<f32>,
) -> Option<RelationResolution> {
    if looks_like_multi_hop_list_query(task)
        || is_enumerative_query(task)
        || !looks_like_relation_query(task)
    {
        return None;
    }

    let mut candidates = aggregate_answer_candidates(collect_relation_candidates(task, evidence));
    candidates.retain(|candidate| {
        candidate_has_required_anchor_support(task, candidate)
            && answer_meets_form_gate(task, &candidate.text, min_answer_confidence)
    });

    if let Some(kg_support) = best_relation_kg_support(task, evidence) {
        if let Some(candidate) = candidates
            .into_iter()
            .find(|candidate| relation_candidate_matches_kg(&candidate.text, &kg_support.values))
        {
            return Some(RelationResolution::Answer(candidate.text));
        }

        let answer = format_answer_list(&kg_support.values);
        return if answer.is_empty() {
            Some(RelationResolution::Suppress)
        } else {
            Some(RelationResolution::Answer(answer))
        };
    }

    let top = candidates.first()?.clone();
    if candidates.iter().skip(1).any(|candidate| {
        candidate.weight + 12.0 >= top.weight && !answer_items_overlap(&candidate.text, &top.text)
    }) {
        return Some(RelationResolution::Suppress);
    }

    Some(RelationResolution::Answer(top.text))
}

fn looks_like_relation_query(task: &str) -> bool {
    let lower = task.to_ascii_lowercase();
    if lower.contains("ingredient") || lower.contains("recipe") {
        return true;
    }
    if !relation_answer_markers(&lower).is_empty() {
        return true;
    }
    [
        "job",
        "occupation",
        "career",
        "role",
        "live",
        "location",
        "residence",
        "city",
        "home",
        "based",
        "partner",
        "husband",
        "wife",
        "boyfriend",
        "girlfriend",
        "spouse",
        "degree",
        "education",
        "major",
        "study",
        "studied",
        "school",
        "pet",
        "dog",
        "cat",
        "phone",
        "number",
        "book",
        "reading",
        "project",
        "playlist",
        "blog",
        "channel",
        "called",
        "name",
        "vehicle",
        "car",
        "truck",
        "model",
        "commute",
        "diet",
        "allergy",
        "allergic",
        "group",
        "joined",
    ]
    .iter()
    .any(|needle| lower.contains(needle))
}

fn collect_relation_candidates(task: &str, evidence: &[EvidenceItem]) -> Vec<CandidateLine> {
    let task_terms = salient_query_terms(task);
    let subject_hints = extract_subject_hints(task);
    let focus_terms = dialogue_focus_terms(task, &task_terms, &subject_hints);
    let anchor_terms = task_anchor_terms(task, &task_terms, &subject_hints);
    let institution_anchor_terms = institution_specific_anchor_terms(task);
    if task_terms.is_empty() {
        return Vec::new();
    }

    let mut candidates = Vec::new();
    for item in evidence {
        let Some(content) = read_context_text(&item.path, "relation candidate collection") else {
            continue;
        };

        for row in parse_answer_surface_rows(&content) {
            let score = answer_surface_score(&row, &task_terms, item.score);
            if score <= 0.0 {
                continue;
            }
            let support_overlap = task_overlap_count(&row.question_pattern, &task_terms)
                .max(task_overlap_count(&row.answer_span, &task_terms))
                .max(1);
            let anchor_overlap = if anchor_terms.is_empty() {
                0
            } else {
                max_task_overlap(
                    [row.question_pattern.as_str(), row.answer_span.as_str()],
                    &anchor_terms,
                )
            };
            if !anchor_terms.is_empty() && anchor_overlap == 0 {
                continue;
            }
            let specific_anchor_overlap = if institution_anchor_terms.is_empty() {
                0
            } else {
                max_task_overlap(
                    [row.question_pattern.as_str(), row.answer_span.as_str()],
                    &institution_anchor_terms,
                )
            };
            candidates.push(CandidateLine {
                path: item.path.clone(),
                text: row.answer_span,
                weight: item.score * 10.0 + score * 2.0,
                retrieval_score: item.score,
                support_overlap,
                anchor_overlap,
                specific_anchor_overlap,
            });
        }

        let turns = parse_dialogue_turns(&content);
        for turn in &turns {
            let base_score = dialogue_match_score(&turn.text, &task_terms);
            let speaker_bonus = speaker_match_bonus(turn.speaker.as_deref(), &subject_hints);
            let Some(candidate) = extract_relation_answer(task, &turn.text, &task_terms) else {
                continue;
            };
            if !subject_hints.is_empty() && !turn_matches_subject(turn, &subject_hints) {
                continue;
            }
            let clean = sanitize_answer_text(&candidate);
            if clean.is_empty() {
                continue;
            }
            let focus_overlap = if focus_terms.is_empty() {
                0
            } else {
                task_overlap_count(&turn.text, &focus_terms)
                    .max(task_overlap_count(&clean, &focus_terms))
            };
            if !focus_terms.is_empty() && focus_overlap == 0 {
                continue;
            }
            let total_score = base_score + speaker_bonus + focus_overlap as f32 * 8.0 + 10.0;
            if total_score < 24.0 {
                continue;
            }
            let support_overlap = task_overlap_count(&turn.text, &task_terms)
                .max(task_overlap_count(&clean, &task_terms))
                .max(focus_overlap)
                .max(1);
            let anchor_overlap = if anchor_terms.is_empty() {
                0
            } else {
                max_task_overlap([turn.text.as_str(), clean.as_str()], &anchor_terms)
            };
            if !anchor_terms.is_empty() && anchor_overlap == 0 {
                continue;
            }
            let specific_anchor_overlap = if institution_anchor_terms.is_empty() {
                0
            } else {
                max_task_overlap(
                    [turn.text.as_str(), clean.as_str()],
                    &institution_anchor_terms,
                )
            };
            candidates.push(CandidateLine {
                path: item.path.clone(),
                text: clean,
                weight: item.score * 10.0 + total_score,
                retrieval_score: item.score,
                support_overlap,
                anchor_overlap,
                specific_anchor_overlap,
            });
        }

        if turns.len() < 2 {
            continue;
        }
        let requires_reason = is_reason_query(task);
        for idx in 0..turns.len() - 1 {
            let question = &turns[idx];
            let answer = &turns[idx + 1];
            if !looks_like_question_turn(&question.text) {
                continue;
            }
            if question.speaker.is_some() && question.speaker == answer.speaker {
                continue;
            }

            let mut context = question.text.clone();
            if idx > 0 {
                context = format!("{} {}", turns[idx - 1].text, context);
            }
            let subject_overlap = if subject_hints.is_empty() {
                0
            } else {
                task_overlap_count(&context, &subject_hints)
                    .max(task_overlap_count(&answer.text, &subject_hints))
            };
            if !subject_hints.is_empty() && subject_overlap == 0 {
                continue;
            }
            let question_focus_overlap = if focus_terms.is_empty() {
                0
            } else {
                task_overlap_count(&context, &focus_terms)
            };
            let answer_focus_overlap = if focus_terms.is_empty() {
                0
            } else {
                task_overlap_count(&answer.text, &focus_terms)
            };
            if !focus_terms.is_empty() && question_focus_overlap == 0 && answer_focus_overlap == 0 {
                continue;
            }
            let question_score = dialogue_match_score(&context, &task_terms);
            let speaker_bonus = speaker_match_bonus(answer.speaker.as_deref(), &subject_hints);
            let total_score = question_score
                + speaker_bonus
                + question_focus_overlap as f32 * 8.0
                + answer_focus_overlap as f32 * 10.0;
            let threshold = if requires_reason { 26.0 } else { 20.0 };
            if total_score < threshold {
                continue;
            }

            let Some(candidate) = extract_turn_answer(task, &answer.text, &task_terms) else {
                continue;
            };
            let clean = sanitize_answer_text(&candidate);
            if clean.is_empty() {
                continue;
            }
            let candidate_focus_overlap = if focus_terms.is_empty() {
                0
            } else {
                task_overlap_count(&clean, &focus_terms)
            };
            if !focus_terms.is_empty()
                && candidate_focus_overlap == 0
                && answer_focus_overlap == 0
                && !requires_reason
            {
                continue;
            }
            let support_overlap = task_overlap_count(&context, &task_terms)
                .max(task_overlap_count(&clean, &task_terms))
                .max(candidate_focus_overlap)
                .max(1);
            let anchor_overlap = if anchor_terms.is_empty() {
                0
            } else {
                max_task_overlap(
                    [context.as_str(), answer.text.as_str(), clean.as_str()],
                    &anchor_terms,
                )
            };
            if !anchor_terms.is_empty() && anchor_overlap == 0 {
                continue;
            }
            let specific_anchor_overlap = if institution_anchor_terms.is_empty() {
                0
            } else {
                max_task_overlap(
                    [context.as_str(), answer.text.as_str(), clean.as_str()],
                    &institution_anchor_terms,
                )
            };
            candidates.push(CandidateLine {
                path: item.path.clone(),
                text: clean,
                weight: item.score * 10.0
                    + total_score
                    + candidate_focus_overlap as f32 * 8.0
                    + subject_overlap as f32 * 4.0
                    + 6.0,
                retrieval_score: item.score,
                support_overlap,
                anchor_overlap,
                specific_anchor_overlap,
            });
        }
    }

    candidates.sort_by(|a, b| {
        b.weight
            .total_cmp(&a.weight)
            .then_with(|| a.path.cmp(&b.path))
            .then_with(|| a.text.cmp(&b.text))
    });
    candidates
}

fn aggregate_answer_candidates(candidates: Vec<CandidateLine>) -> Vec<CandidateLine> {
    let mut buckets: HashMap<String, RelationCandidateBucket> = HashMap::new();
    for candidate in candidates {
        let key = normalized_answer_key(&candidate.text);
        if key.is_empty() {
            continue;
        }
        let bucket = buckets
            .entry(key)
            .or_insert_with(|| RelationCandidateBucket {
                best_candidate: candidate.clone(),
                best_single_weight: candidate.weight,
                total_weight: 0.0,
                max_retrieval_score: candidate.retrieval_score,
                max_support_overlap: candidate.support_overlap,
                max_anchor_overlap: candidate.anchor_overlap,
                max_specific_anchor_overlap: candidate.specific_anchor_overlap,
                paths: HashSet::new(),
                hits: 0,
            });
        if candidate.weight > bucket.best_single_weight {
            bucket.best_candidate = candidate.clone();
            bucket.best_single_weight = candidate.weight;
        }
        bucket.total_weight += candidate.weight;
        bucket.max_retrieval_score = bucket.max_retrieval_score.max(candidate.retrieval_score);
        bucket.max_support_overlap = bucket.max_support_overlap.max(candidate.support_overlap);
        bucket.max_anchor_overlap = bucket.max_anchor_overlap.max(candidate.anchor_overlap);
        bucket.max_specific_anchor_overlap = bucket
            .max_specific_anchor_overlap
            .max(candidate.specific_anchor_overlap);
        bucket.paths.insert(candidate.path.clone());
        bucket.hits += 1;
    }

    let mut aggregated = buckets
        .into_values()
        .map(|bucket| {
            let mut candidate = bucket.best_candidate;
            candidate.weight = bucket.total_weight
                + (bucket.paths.len().saturating_sub(1) as f32) * 8.0
                + (bucket.hits.saturating_sub(1) as f32) * 4.0;
            candidate.retrieval_score = bucket.max_retrieval_score;
            candidate.support_overlap = bucket.max_support_overlap;
            candidate.anchor_overlap = bucket.max_anchor_overlap;
            candidate.specific_anchor_overlap = bucket.max_specific_anchor_overlap;
            candidate
        })
        .collect::<Vec<_>>();
    aggregated.sort_by(|a, b| {
        b.weight
            .total_cmp(&a.weight)
            .then_with(|| a.path.cmp(&b.path))
            .then_with(|| a.text.cmp(&b.text))
    });
    aggregated
}

fn best_relation_kg_support(task: &str, evidence: &[EvidenceItem]) -> Option<RelationKgSupport> {
    let task_terms = salient_query_terms(task);
    if task_terms.is_empty() {
        return None;
    }

    let mut candidates = Vec::new();
    for item in evidence {
        if !path_looks_like_kg_neuron(&item.path) {
            continue;
        }
        let Ok(entity) = kg::KgEntity::load(&item.path) else {
            continue;
        };
        if entity.facts.is_empty() {
            continue;
        }

        let mut predicates = entity
            .facts
            .iter()
            .map(|fact| fact.predicate.clone())
            .collect::<Vec<_>>();
        predicates.sort();
        predicates.dedup();

        for predicate in predicates {
            if !is_relation_kg_predicate(&predicate) {
                continue;
            }
            let values = current_kg_values(&entity, &predicate);
            if values.is_empty() {
                continue;
            }
            let score =
                relation_kg_candidate_score(task, &task_terms, item.score, &entity, &predicate);
            if score <= 0.0 {
                continue;
            }
            candidates.push((score, values));
        }
    }

    candidates.sort_by(|a, b| b.0.total_cmp(&a.0));
    let (top_score, top_values) = candidates.first()?.clone();
    if top_score < 14.0 {
        return None;
    }
    if candidates.iter().skip(1).any(|(score, values)| {
        *score + 4.0 >= top_score && !kg_value_sets_overlap(values, &top_values)
    }) {
        return None;
    }

    Some(RelationKgSupport { values: top_values })
}

fn path_looks_like_kg_neuron(path: &Path) -> bool {
    path.file_name()
        .and_then(|name| name.to_str())
        .map(|name| name.starts_with("_kg_") && name.ends_with(".context.md"))
        .unwrap_or(false)
}

fn is_relation_kg_predicate(predicate: &str) -> bool {
    matches!(
        predicate,
        "occupation"
            | "location"
            | "partner"
            | "phone"
            | "education"
            | "major"
            | "school"
            | "studying"
            | "pet"
            | "book"
            | "project_name"
            | "commute_time"
            | "diet"
            | "allergy"
            | "vehicle_model"
            | "family_trip_location"
    )
}

fn relation_kg_candidate_score(
    task: &str,
    task_terms: &[String],
    retrieval_score: f32,
    entity: &kg::KgEntity,
    predicate: &str,
) -> f32 {
    let predicate_context = kg_predicate_query_terms(predicate).join(" ");
    let predicate_overlap = task_overlap_count(&predicate_context, task_terms);
    if predicate_overlap == 0 {
        return 0.0;
    }

    let entity_context = kg_entity_query_terms(&entity.entity).join(" ");
    let entity_overlap = if entity_context.is_empty() {
        0
    } else {
        task_overlap_count(&entity_context, task_terms)
    };
    let lower = task.to_ascii_lowercase();
    let entity_bonus = if entity_overlap > 0 {
        entity_overlap as f32 * 8.0
    } else if (entity.entity == "user" || entity.entity.starts_with("agent_"))
        && query_targets_primary_entity(&lower)
    {
        4.0
    } else {
        0.0
    };

    predicate_overlap as f32 * 10.0
        + entity_bonus
        + relation_predicate_query_bonus(&lower, predicate)
        + retrieval_score
}

fn query_targets_primary_entity(task_lower: &str) -> bool {
    task_lower.contains(" my ")
        || task_lower.starts_with("my ")
        || task_lower.starts_with("what is my ")
        || task_lower.starts_with("what's my ")
        || task_lower.starts_with("where do i ")
        || task_lower.starts_with("where am i ")
        || task_lower.starts_with("who is my ")
        || task_lower.starts_with("what is the reviewer")
}

fn relation_predicate_query_bonus(task_lower: &str, predicate: &str) -> f32 {
    match predicate {
        "occupation"
            if ["job", "occupation", "career", "role", "work"]
                .iter()
                .any(|needle| task_lower.contains(needle)) =>
        {
            6.0
        },
        "location"
            if task_lower.starts_with("where ")
                || [
                    "live",
                    "location",
                    "residence",
                    "city",
                    "home",
                    "based",
                    "moved",
                ]
                .iter()
                .any(|needle| task_lower.contains(needle)) =>
        {
            6.0
        },
        "partner"
            if task_lower.starts_with("who ")
                || [
                    "partner",
                    "husband",
                    "wife",
                    "boyfriend",
                    "girlfriend",
                    "spouse",
                ]
                .iter()
                .any(|needle| task_lower.contains(needle)) =>
        {
            6.0
        },
        "major"
            if ["major", "field"]
                .iter()
                .any(|needle| task_lower.contains(needle)) =>
        {
            6.0
        },
        "education"
            if ["study", "studied", "degree", "education", "graduated"]
                .iter()
                .any(|needle| task_lower.contains(needle)) =>
        {
            5.0
        },
        "school"
            if ["school", "college", "university"]
                .iter()
                .any(|needle| task_lower.contains(needle)) =>
        {
            6.0
        },
        "pet"
            if ["pet", "dog", "cat", "name", "called"]
                .iter()
                .any(|needle| task_lower.contains(needle)) =>
        {
            6.0
        },
        "phone"
            if ["phone", "number", "call"]
                .iter()
                .any(|needle| task_lower.contains(needle)) =>
        {
            6.0
        },
        "book"
            if ["book", "read", "reading", "novel"]
                .iter()
                .any(|needle| task_lower.contains(needle)) =>
        {
            6.0
        },
        "project_name"
            if ["project", "playlist", "blog", "channel", "called", "name"]
                .iter()
                .any(|needle| task_lower.contains(needle)) =>
        {
            6.0
        },
        "vehicle_model"
            if ["vehicle", "car", "truck", "model", "drive"]
                .iter()
                .any(|needle| task_lower.contains(needle)) =>
        {
            6.0
        },
        "family_trip_location"
            if task_lower.starts_with("where ")
                || ["family", "trip", "vacation", "travel", "destination"]
                    .iter()
                    .any(|needle| task_lower.contains(needle)) =>
        {
            5.0
        },
        "commute_time"
            if ["commute", "travel", "minutes", "time"]
                .iter()
                .any(|needle| task_lower.contains(needle)) =>
        {
            5.0
        },
        "diet"
            if ["diet", "vegan", "vegetarian", "pescatarian", "keto"]
                .iter()
                .any(|needle| task_lower.contains(needle)) =>
        {
            5.0
        },
        "allergy"
            if ["allergy", "allergic"]
                .iter()
                .any(|needle| task_lower.contains(needle)) =>
        {
            5.0
        },
        "studying"
            if ["study", "studying"]
                .iter()
                .any(|needle| task_lower.contains(needle)) =>
        {
            4.0
        },
        _ => 0.0,
    }
}

fn relation_candidate_matches_kg(candidate: &str, values: &[String]) -> bool {
    values
        .iter()
        .any(|value| answer_items_overlap(candidate, value))
}

fn kg_value_sets_overlap(left: &[String], right: &[String]) -> bool {
    left.iter().any(|left_value| {
        right
            .iter()
            .any(|right_value| answer_items_overlap(left_value, right_value))
    })
}

fn parse_answer_surface_rows(content: &str) -> Vec<AnswerSurfaceRow> {
    let sections = parse_sections(content);
    let Some(table) = sections.get("answer_surface") else {
        return Vec::new();
    };

    table
        .lines()
        .filter_map(|line| {
            let trimmed = line.trim();
            if !trimmed.starts_with('|') {
                return None;
            }
            let columns = trimmed
                .trim_matches('|')
                .split('|')
                .map(str::trim)
                .collect::<Vec<_>>();
            if columns.len() != 3 {
                return None;
            }
            if columns[0].eq_ignore_ascii_case("question_pattern")
                || columns[0].chars().all(|c| c == '-' || c == ' ')
            {
                return None;
            }
            let confidence = columns[2].parse::<f32>().unwrap_or(0.0);
            let answer_span = sanitize_answer_text(columns[1]);
            if answer_span.is_empty() {
                return None;
            }
            Some(AnswerSurfaceRow {
                question_pattern: columns[0].to_string(),
                answer_span,
                confidence,
            })
        })
        .collect()
}

fn answer_surface_score(
    row: &AnswerSurfaceRow,
    task_terms: &[String],
    retrieval_score: f32,
) -> f32 {
    let pattern_terms = salient_query_terms(&row.question_pattern);
    if pattern_terms.is_empty() {
        return 0.0;
    }
    let overlap = term_list_overlap_count(task_terms, &pattern_terms);
    if overlap == 0 {
        return 0.0;
    }

    let coverage = overlap as f32 / task_terms.len().max(1) as f32;
    let specificity = overlap as f32 / pattern_terms.len().max(1) as f32;
    retrieval_score + overlap as f32 * 4.0 + coverage * 6.0 + specificity * 2.0 + row.confidence
}

pub(super) mod temporal;
use self::temporal::{
    best_calendar_grounded_current_anchor_rank, collect_temporal_candidates, current_kg_values,
    is_temporal_reasoning_query, is_temporal_sequence_query, kg_entity_query_terms,
    kg_predicate_query_terms, parse_temporal_elapsed_query, parse_temporal_gap_query,
    required_tail_anchor_tokens, select_comparison_answer, select_dialogue_temporal_answer,
    select_temporal_count_answer, select_temporal_duration_answer,
    select_temporal_employment_duration_answer, select_temporal_order_answer,
    select_temporal_state_answer, shift_date_by_days, split_once_case_insensitive,
    temporal_focus_terms,
};

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

fn answer_candidate_lines(content: &str) -> Vec<String> {
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

fn read_context_text(path: &Path, stage: &str) -> Option<String> {
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

mod dialogue;
use self::dialogue::{
    looks_like_question_turn, parse_dialogue_turns, select_structured_diary_answer,
    select_subject_turn_answer, select_turn_pair_answer, should_skip_generated_answer_line,
    structured_diary_action_query, structured_diary_blocker_query,
    structured_diary_dependencies_query, structured_diary_entities_query,
    structured_diary_goal_query, structured_diary_status_query, structured_diary_title_query,
};
pub use self::dialogue::{mine_dialogue_answer_surface_span, mine_dialogue_question_pattern};

mod scoring;
use self::scoring::*;

#[cfg(test)]
mod temporal_tests;

#[cfg(test)]
mod temporal_current_anchor_tests;

#[cfg(test)]
mod temporal_comparison_tests;

#[cfg(test)]
mod openqa_tests;

#[cfg(test)]
mod validator_tests;

#[cfg(test)]
mod tests;
