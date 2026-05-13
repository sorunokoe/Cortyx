//! Answer plane - query processing and response generation.
//!
//! Handles natural language queries, temporal reasoning, and fact extraction.

use crate::index::{ContextMetadata, NeuronIndex};
use crate::kg;
use crate::neuron::{parse_sections, unix_secs_to_datetime, NeuronKind, SynapseType};
use crate::reasoner::{ReasonedFact, ReasoningReport, TraversalOptions};
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AnswerAbstentionReason {
    Unsupported,
    LowFormConfidence,
}

pub(crate) mod types;
use self::types::*;

mod output;
use self::output::{answer_candidate_lines, read_context_text};

pub(crate) mod surface_enricher;

mod multihop;
use self::multihop::*;

mod validation;
use self::validation::*;

mod openqa;
use self::openqa::*;

mod suggestions;
use self::suggestions::*;

mod surface;
use self::surface::*;

mod relation;
use self::relation::*;

pub mod temporal;
#[cfg(test)]
use self::temporal::select_temporal_employment_duration_answer;
pub(crate) use self::temporal::*;

mod dialogue;
use self::dialogue::*;
pub use self::dialogue::{mine_dialogue_answer_surface_span, mine_dialogue_question_pattern};

mod scoring;
pub(crate) use self::scoring::*;

pub fn render_answer_output(
    index: &NeuronIndex,
    task: &str,
    paths_with_scores: &[(PathBuf, f32)],
    include_provenance: bool,
    min_answer_confidence: Option<f32>,
) -> Option<String> {
    output::render_answer_output(
        index,
        task,
        paths_with_scores,
        include_provenance,
        min_answer_confidence,
    )
}

pub fn render_answer_output_decision(
    index: &NeuronIndex,
    task: &str,
    paths_with_scores: &[(PathBuf, f32)],
    include_provenance: bool,
    min_answer_confidence: Option<f32>,
) -> Result<String, AnswerAbstentionReason> {
    output::render_answer_output_decision(
        index,
        task,
        paths_with_scores,
        include_provenance,
        min_answer_confidence,
    )
}

pub fn render_provenance_output(
    index: &NeuronIndex,
    paths_with_scores: &[(PathBuf, f32)],
) -> Option<String> {
    output::render_provenance_output(index, paths_with_scores)
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

    if let Some(answer) = select_activity_completion_duration_answer(task, evidence) {
        return Some(answer);
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

    if let Some(answer) = select_simple_count_span_answer(task, evidence) {
        return Some(answer);
    }

    if let Some(answer) = select_previous_state_answer(task, evidence) {
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
