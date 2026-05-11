//! Candidate ranking: best match selection for temporal answers.

use super::*;

pub(crate) fn best_temporal_candidate_rank(
    option: &ChoiceOption,
    candidates: &[TemporalCandidate],
) -> Option<i32> {
    best_temporal_candidate_rank_for_comparison(option, candidates, false)
}

pub(crate) fn best_temporal_candidate_rank_for_comparison(
    option: &ChoiceOption,
    candidates: &[TemporalCandidate],
    acquisition_focus: bool,
) -> Option<i32> {
    let mut best: Option<(usize, f32, i32)> = None;
    for candidate in candidates {
        let rank = if acquisition_focus {
            candidate_temporal_rank_for_acquisition_option(candidate, option)
                .or_else(|| candidate_temporal_rank(candidate))
        } else {
            candidate_temporal_rank(candidate)
        };
        let Some(rank) = rank else {
            continue;
        };
        let overlap = temporal_event_overlap_count(&candidate.text, option);
        if overlap == 0 {
            continue;
        }
        let score = temporal_candidate_score(candidate, option);
        if score < 10.0 {
            continue;
        }
        if best
            .as_ref()
            .map(|(best_overlap, best_score, _)| {
                overlap > *best_overlap || (overlap == *best_overlap && score > *best_score)
            })
            .unwrap_or(true)
        {
            best = Some((overlap, score, rank));
        }
    }
    best.map(|(_, _, rank)| rank)
}

fn candidate_temporal_rank_for_acquisition_option(
    candidate: &TemporalCandidate,
    option: &ChoiceOption,
) -> Option<i32> {
    temporal_option_focus_tail(&candidate.text, option)
        .and_then(|tail| extract_acquisition_completion_rank(tail, candidate.base_date))
        .or_else(|| extract_acquisition_completion_rank(&candidate.text, candidate.base_date))
}

fn extract_acquisition_completion_rank(
    text: &str,
    base_date: Option<(i32, u32, u32)>,
) -> Option<i32> {
    let lower = text.to_ascii_lowercase();
    let has_completion_marker = ["arrived", "delivered", "received", "showed up"]
        .iter()
        .any(|marker| lower.contains(marker));
    let has_lead_time_marker = ["pre-ordered", "preordered", "ordered", "expected arrival"]
        .iter()
        .any(|marker| lower.contains(marker));
    if !has_completion_marker || !has_lead_time_marker {
        return None;
    }

    temporal_rank_clauses(text)
        .into_iter()
        .rev()
        .find_map(|clause| {
            ["arrived", "delivered", "received", "showed up"]
                .iter()
                .any(|marker| clause.to_ascii_lowercase().contains(marker))
                .then_some(clause)
        })
        .and_then(|clause| extract_explicit_date(&clause, base_date))
        .map(|(year, month, day)| ymd_to_days(year, month, day))
}

fn temporal_rank_clauses(text: &str) -> Vec<String> {
    text.replace(", and ", ". ")
        .replace(" and ", ". ")
        .replace(", but ", ". ")
        .replace(" but ", ". ")
        .split('.')
        .map(str::trim)
        .filter(|clause| !clause.is_empty())
        .map(ToString::to_string)
        .collect()
}

pub(crate) fn best_calendar_grounded_temporal_candidate_rank(
    option: &ChoiceOption,
    candidates: &[TemporalCandidate],
) -> Option<CalendarGroundedRank> {
    best_calendar_grounded_temporal_candidate_rank_with_competing_option(option, None, candidates)
}

pub(crate) fn best_calendar_grounded_temporal_candidate_rank_with_competing_option(
    option: &ChoiceOption,
    competing: Option<&ChoiceOption>,
    candidates: &[TemporalCandidate],
) -> Option<CalendarGroundedRank> {
    let mut best: Option<CalendarGroundedRank> = None;
    let prioritize_specificity = competing.is_some();
    for candidate in candidates {
        let overlap = temporal_event_overlap_count(&candidate.text, option);
        if overlap == 0 {
            continue;
        }
        let Some((rank, grounding_bonus)) =
            candidate_calendar_grounded_rank_for_option(candidate, option, candidates)
        else {
            continue;
        };
        let score = if let Some(competing) = competing {
            temporal_specificity_score(candidate, option, Some(competing)) + grounding_bonus
        } else {
            temporal_candidate_score(candidate, option) + grounding_bonus
        };
        if score < 8.0 {
            continue;
        }
        let grounded = CalendarGroundedRank {
            ordinal: candidate.ordinal,
            rank,
            overlap,
            score,
        };
        if best
            .as_ref()
            .map(|current| {
                if prioritize_specificity {
                    score > current.score || (score == current.score && overlap > current.overlap)
                } else {
                    overlap > current.overlap
                        || (overlap == current.overlap && score > current.score)
                }
            })
            .unwrap_or(true)
        {
            best = Some(grounded);
        }
    }
    best
}

fn candidate_calendar_grounded_rank_for_option(
    candidate: &TemporalCandidate,
    option: &ChoiceOption,
    candidates: &[TemporalCandidate],
) -> Option<(i32, f32)> {
    if let Some(tail) = temporal_option_focus_tail(&candidate.text, option) {
        if let Some(rank) = extract_calendar_grounded_rank(tail, candidate.base_date) {
            return Some((rank, 3.0));
        }
        if let Some(rank) = resolve_calendar_grounded_reference_rank(candidate, tail, candidates) {
            return Some((rank, 2.0));
        }
    }
    candidate_calendar_grounded_base_rank(candidate).map(|rank| (rank, 0.5))
}

fn candidate_calendar_grounded_base_rank(candidate: &TemporalCandidate) -> Option<i32> {
    extract_calendar_grounded_rank(&candidate.text, candidate.base_date)
}

fn extract_calendar_grounded_rank(text: &str, base_date: Option<(i32, u32, u32)>) -> Option<i32> {
    extract_self_anchored_temporal_rank(text, base_date).or_else(|| {
        extract_explicit_date(text, base_date)
            .map(|(year, month, day)| ymd_to_days(year, month, day))
    })
}

fn extract_self_anchored_temporal_rank(
    text: &str,
    base_date: Option<(i32, u32, u32)>,
) -> Option<i32> {
    let lower = text.to_ascii_lowercase();
    for (unit, scale) in [("day", 1), ("week", 7), ("month", 30), ("year", 365)] {
        for (marker, direction) in [
            (format!("{unit} in advance"), TemporalDirection::Earlier),
            (format!("{unit}s in advance"), TemporalDirection::Earlier),
            (format!("{unit} before"), TemporalDirection::Earlier),
            (format!("{unit}s before"), TemporalDirection::Earlier),
            (format!("{unit} after"), TemporalDirection::Later),
            (format!("{unit}s after"), TemporalDirection::Later),
            (format!("{unit} later"), TemporalDirection::Later),
            (format!("{unit}s later"), TemporalDirection::Later),
        ] {
            if !lower.contains(&marker) {
                continue;
            }
            let (prefix, suffix) = split_once_case_insensitive(text, &marker)?;
            let amount = extract_trailing_count(&prefix.to_ascii_lowercase())? * scale;
            let (year, month, day) = extract_explicit_date(suffix, base_date)?;
            let anchor_rank = ymd_to_days(year, month, day);
            return Some(match direction {
                TemporalDirection::Earlier => anchor_rank - amount,
                TemporalDirection::Later => anchor_rank + amount,
            });
        }
    }
    None
}

fn resolve_calendar_grounded_reference_rank(
    target: &TemporalCandidate,
    text: &str,
    candidates: &[TemporalCandidate],
) -> Option<i32> {
    let (direction, offset_days) = extract_relative_reference_offset_days(text)?;
    let reference_rank = resolve_calendar_reference_anchor_rank(target, text, candidates)?;
    Some(match direction {
        TemporalDirection::Earlier => reference_rank - offset_days,
        TemporalDirection::Later => reference_rank + offset_days,
    })
}

fn resolve_calendar_reference_anchor_rank(
    target: &TemporalCandidate,
    text: &str,
    candidates: &[TemporalCandidate],
) -> Option<i32> {
    let reference_terms = temporal_reference_terms(text);
    if reference_terms.is_empty() {
        return None;
    }

    let mut best: Option<(f32, i32)> = None;
    for candidate in candidates {
        if candidate.ordinal == target.ordinal {
            continue;
        }
        let Some(rank) = candidate_calendar_grounded_base_rank(candidate) else {
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

pub(crate) fn best_temporal_candidate_strict_rank(
    option: &ChoiceOption,
    candidates: &[TemporalCandidate],
) -> Option<i32> {
    let mut best: Option<(usize, f32, i32)> = None;
    for candidate in candidates {
        let Some(rank) = candidate_temporal_rank(candidate) else {
            continue;
        };
        let overlap = temporal_event_overlap_count(&candidate.text, option);
        if overlap == 0 {
            continue;
        }
        let score = temporal_candidate_score(candidate, option);
        if score < 10.0 {
            continue;
        }
        if best
            .as_ref()
            .map(|(best_overlap, best_score, _)| {
                overlap > *best_overlap || (overlap == *best_overlap && score > *best_score)
            })
            .unwrap_or(true)
        {
            best = Some((overlap, score, rank));
        }
    }
    best.map(|(_, _, rank)| rank)
}

pub(crate) fn best_calendar_grounded_current_anchor_rank(
    candidates: &[TemporalCandidate],
) -> Option<i32> {
    candidates
        .iter()
        .filter_map(candidate_calendar_grounded_current_rank)
        .max()
        .or_else(|| {
            candidates
                .iter()
                .filter_map(candidate_calendar_grounded_base_rank)
                .max()
        })
}

fn candidate_calendar_grounded_current_rank(candidate: &TemporalCandidate) -> Option<i32> {
    let base_date = candidate.base_date?;
    temporal_current_rank_from_text(&candidate.text, Some(base_date), candidate.user_authored)
}

pub(crate) fn best_temporal_candidate_sequence_rank(
    option: &ChoiceOption,
    candidates: &[TemporalCandidate],
) -> Option<i32> {
    let mut best: Option<(usize, f32, i32)> = None;
    for candidate in candidates {
        let Some(rank) = candidate.sequence_rank else {
            continue;
        };
        let overlap = temporal_event_overlap_count(&candidate.text, option);
        if overlap == 0 {
            continue;
        }
        let score = temporal_candidate_score(candidate, option)
            + if candidate_temporal_rank(candidate).is_none() {
                1.5
            } else {
                0.0
            };
        if score < 8.0 {
            continue;
        }
        if best
            .as_ref()
            .map(|(best_overlap, best_score, _)| {
                overlap > *best_overlap || (overlap == *best_overlap && score > *best_score)
            })
            .unwrap_or(true)
        {
            best = Some((overlap, score, rank));
        }
    }
    best.map(|(_, _, rank)| rank)
}

pub(crate) fn best_temporal_candidate_loose_sequence_rank(
    option: &ChoiceOption,
    candidates: &[TemporalCandidate],
) -> Option<i32> {
    let mut best: Option<(usize, f32, i32)> = None;
    for candidate in candidates {
        let rank = candidate
            .sequence_rank
            .or_else(|| candidate_temporal_order_rank_for_option(candidate, option, candidates))?;
        let overlap = temporal_event_overlap_count(&candidate.text, option);
        if overlap == 0 {
            continue;
        }
        let score = temporal_candidate_score(candidate, option)
            + if candidate.sequence_rank.is_some() {
                1.5
            } else {
                0.0
            };
        if score < 4.5 && overlap < 2 {
            continue;
        }
        if best
            .as_ref()
            .map(|(best_overlap, best_score, _)| {
                overlap > *best_overlap || (overlap == *best_overlap && score > *best_score)
            })
            .unwrap_or(true)
        {
            best = Some((overlap, score, rank));
        }
    }
    best.map(|(_, _, rank)| rank)
}

pub(crate) fn temporal_candidate_score(
    candidate: &TemporalCandidate,
    option: &ChoiceOption,
) -> f32 {
    temporal_event_match_score(&candidate.text, option, candidate.retrieval_score)
        + if candidate.user_authored { 4.0 } else { 0.0 }
}

fn temporal_event_overlap_count(line: &str, option: &ChoiceOption) -> usize {
    let lower = line.to_ascii_lowercase();
    option
        .tokens
        .iter()
        .filter(|token| line_matches_event_token(&lower, token))
        .count()
}

pub(crate) fn temporal_target_score(candidate: &TemporalCandidate, target_terms: &[String]) -> f32 {
    let overlap = task_overlap_count(&candidate.text, target_terms);
    if overlap == 0 {
        return 0.0;
    }
    candidate_weight(
        &candidate.text,
        target_terms,
        candidate.retrieval_score,
        false,
    ) + overlap as f32 * 6.0
        + if candidate.user_authored { 4.0 } else { 0.0 }
}

pub(crate) fn extract_relative_reference_offset_days(
    text: &str,
) -> Option<(TemporalDirection, i32)> {
    let lower = text.to_ascii_lowercase();
    for (unit, scale) in [("day", 1), ("week", 7), ("month", 30), ("year", 365)] {
        for marker in [
            (format!("{unit} in advance"), TemporalDirection::Earlier),
            (format!("{unit}s in advance"), TemporalDirection::Earlier),
            (format!("{unit} before"), TemporalDirection::Earlier),
            (format!("{unit}s before"), TemporalDirection::Earlier),
            (format!("{unit} after"), TemporalDirection::Later),
            (format!("{unit}s after"), TemporalDirection::Later),
            (format!("{unit} later"), TemporalDirection::Later),
            (format!("{unit}s later"), TemporalDirection::Later),
        ] {
            if !lower.contains(&marker.0) {
                continue;
            }
            let Some(prefix) = lower.split(&marker.0).next() else {
                continue;
            };
            if let Some(amount) = extract_trailing_count(prefix) {
                return Some((marker.1, amount * scale));
            }
        }
    }
    None
}
