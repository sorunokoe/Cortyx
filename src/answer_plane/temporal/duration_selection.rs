//! Duration pair selection and scoring: candidate rank computation for interval queries.
//!
//! This module handles duration pair selection from evidence items,
//! specificity scoring, and temporal reference resolution.

use super::*;

pub(crate) fn best_temporal_duration_pair(
    start: &ChoiceOption,
    end: &ChoiceOption,
    evidence: &[EvidenceItem],
) -> Option<(i32, i32)> {
    let candidates = collect_temporal_candidates(evidence, "temporal duration answer selection");
    best_temporal_duration_pair_from_candidates(start, end, &candidates)
}

pub(crate) fn best_temporal_duration_pair_from_candidates(
    start: &ChoiceOption,
    end: &ChoiceOption,
    candidates: &[TemporalCandidate],
) -> Option<(i32, i32)> {
    let start_rank = best_calendar_grounded_temporal_candidate_rank(start, candidates)?;
    let end_rank = best_calendar_grounded_temporal_candidate_rank(end, candidates)?;
    if start_rank.rank != end_rank.rank {
        return Some((start_rank.rank, end_rank.rank));
    }

    let specific_start = best_calendar_grounded_temporal_candidate_rank_with_competing_option(
        start,
        Some(end),
        candidates,
    )?;
    let specific_end = best_calendar_grounded_temporal_candidate_rank_with_competing_option(
        end,
        Some(start),
        candidates,
    )?;
    if specific_start.rank != specific_end.rank || specific_start.ordinal != specific_end.ordinal {
        return Some((specific_start.rank, specific_end.rank));
    }

    if start_rank.ordinal == end_rank.ordinal && start_rank.rank == end_rank.rank {
        return None;
    }
    Some((start_rank.rank, end_rank.rank))
}

pub(crate) fn best_temporal_duration_pair_strict(
    start: &ChoiceOption,
    end: &ChoiceOption,
    evidence: &[EvidenceItem],
) -> Option<(i32, i32)> {
    best_temporal_duration_pair(start, end, evidence)
}

pub(crate) fn temporal_specificity_score(
    candidate: &TemporalCandidate,
    target: &ChoiceOption,
    competing: Option<&ChoiceOption>,
) -> f32 {
    let score = temporal_candidate_score(candidate, target);
    if score <= 0.0 {
        return 0.0;
    }
    let competing_score = competing
        .map(|option| {
            temporal_event_match_score(&candidate.text, option, candidate.retrieval_score)
        })
        .unwrap_or(0.0);
    score - competing_score * 0.55
        + if candidate_temporal_base_rank(candidate).is_some() {
            1.0
        } else {
            0.0
        }
}

pub(crate) fn extract_relative_unit_amount(text: &str, unit: &str) -> Option<i32> {
    let lower = text.to_ascii_lowercase();
    match unit {
        "day" if lower.contains("yesterday") => return Some(1),
        "week" if lower.contains("last week") => return Some(1),
        "month" if lower.contains("last month") => return Some(1),
        "year" if lower.contains("last year") => return Some(1),
        _ => {},
    }

    for marker in [format!("{unit} ago"), format!("{unit}s ago")] {
        if !lower.contains(&marker) {
            continue;
        }
        let Some(prefix) = lower.split(&marker).next() else {
            continue;
        };
        if let Some(amount) = extract_trailing_count(prefix) {
            return Some(amount);
        }
    }
    None
}

pub(crate) fn resolve_temporal_reference_rank(
    target: &TemporalCandidate,
    candidates: &[TemporalCandidate],
) -> Option<i32> {
    let reference_terms = temporal_reference_terms(&target.text);
    if reference_terms.is_empty() {
        return None;
    }

    let mut best: Option<(f32, i32)> = None;
    for candidate in candidates {
        if candidate.ordinal == target.ordinal {
            continue;
        }
        let Some(rank) = candidate_temporal_base_rank(candidate) else {
            continue;
        };
        let overlap =
            term_list_overlap_count(&reference_terms, &salient_query_terms(&candidate.text));
        if overlap == 0 {
            continue;
        }
        let distance_penalty = target.ordinal.abs_diff(candidate.ordinal) as f32 * 0.25;
        let score = overlap as f32 * 8.0
            + candidate.retrieval_score * 2.0
            + if candidate.user_authored { 4.0 } else { 0.0 }
            - distance_penalty;
        if best
            .as_ref()
            .map(|(best_score, _)| score > *best_score)
            .unwrap_or(true)
        {
            best = Some((score, rank));
        }
    }
    best.map(|(_, rank)| rank)
}

pub(crate) fn temporal_reference_terms(text: &str) -> Vec<String> {
    let mut terms = salient_query_terms(text);
    terms.retain(|term| {
        !matches!(
            term.as_str(),
            "day"
                | "days"
                | "week"
                | "weeks"
                | "month"
                | "months"
                | "year"
                | "years"
                | "ago"
                | "advance"
                | "before"
                | "after"
                | "later"
                | "book"
                | "booked"
                | "booking"
                | "exactly"
                | "about"
                | "around"
        )
    });
    terms.sort();
    terms.dedup();
    terms
}
