//! Comparison and temporal order selection.

use super::*;
use std::collections::HashMap;

pub(crate) fn select_comparison_answer(task: &str, evidence: &[EvidenceItem]) -> Option<String> {
    let (options, direction) = parse_binary_choice(task)?;
    if options.len() != 2 {
        return None;
    }

    let candidates = collect_temporal_candidates(evidence, "comparison answer selection");
    let acquisition_focus = comparison_prefers_acquisition_rank(task);
    let explicit_hits = options
        .iter()
        .map(|option| {
            best_temporal_candidate_rank_for_comparison(option, &candidates, acquisition_focus)
        })
        .collect::<Vec<_>>();
    let best_hits = if explicit_hits.iter().all(Option::is_some) {
        explicit_hits
    } else {
        let sequence_hits = options
            .iter()
            .map(|option| best_temporal_candidate_sequence_rank(option, &candidates))
            .collect::<Vec<_>>();
        if sequence_hits.iter().all(Option::is_some) {
            sequence_hits
        } else {
            explicit_hits
        }
    };

    match (best_hits[0], best_hits[1]) {
        (Some(left), Some(right)) => {
            let pick_left = match direction {
                TemporalDirection::Earlier => left <= right,
                TemporalDirection::Later => left >= right,
            };
            Some(if pick_left {
                options[0].display.clone()
            } else {
                options[1].display.clone()
            })
        },
        (Some(_), None) | (None, Some(_)) => None,
        (None, None) => None,
    }
}

fn comparison_prefers_acquisition_rank(task: &str) -> bool {
    let lower = task.to_ascii_lowercase();
    lower.contains(" got ")
        || lower.starts_with("got ")
        || lower.contains(" get ")
        || lower.contains(" received ")
        || lower.contains(" receive ")
        || lower.contains(" arrived ")
        || lower.contains(" arrival ")
}

pub(crate) fn select_temporal_order_answer(
    task: &str,
    evidence: &[EvidenceItem],
) -> Option<String> {
    select_temporal_sequence_answer(task, evidence)
        .or_else(|| select_temporal_anchor_order_answer(task, evidence))
        .or_else(|| select_temporal_window_answer(task, evidence))
}

fn select_temporal_sequence_answer(task: &str, evidence: &[EvidenceItem]) -> Option<String> {
    if !is_temporal_sequence_query(task) {
        return None;
    }

    let lower = task.to_ascii_lowercase();
    let direction = if lower.contains("latest to earliest")
        || lower.contains("most recent to earliest")
        || lower.contains("starting from the latest")
    {
        TemporalDirection::Later
    } else {
        TemporalDirection::Earlier
    };
    let focus_terms = temporal_sequence_focus_terms(task);
    if focus_terms.is_empty() {
        return None;
    }

    let candidates = collect_temporal_candidates(evidence, "temporal sequence answer selection");
    if let Some(options) = parse_temporal_sequence_options(task) {
        let mut ordered = options
            .iter()
            .filter_map(|option| {
                let rank = best_temporal_candidate_rank(option, &candidates)
                    .or_else(|| best_temporal_candidate_sequence_rank(option, &candidates))
                    .or_else(|| best_temporal_candidate_loose_sequence_rank(option, &candidates))?;
                Some((rank, strip_leading_temporal_actor(&option.display)))
            })
            .collect::<Vec<_>>();
        if ordered.len() >= 2 {
            ordered.sort_by(|left, right| match direction {
                TemporalDirection::Earlier => {
                    left.0.cmp(&right.0).then_with(|| left.1.cmp(&right.1))
                },
                TemporalDirection::Later => right.0.cmp(&left.0).then_with(|| left.1.cmp(&right.1)),
            });
            let items = ordered
                .into_iter()
                .map(|(_, item)| item)
                .take(parse_requested_sequence_count(task).max(2))
                .collect::<Vec<_>>();
            if let Some(answer) = render_temporal_sequence_answer(&items) {
                return Some(answer);
            }
        }
    }
    let require_completed_events = lower.contains("past ")
        || lower.contains(" i took ")
        || lower.contains(" i attended ")
        || lower.contains(" i participated ")
        || lower.contains("from earliest")
        || lower.contains("from latest")
        || lower.contains("order of");
    let requested = parse_requested_sequence_count(task).max(2);
    let explicit_candidate_count = candidates
        .iter()
        .filter(|candidate| {
            (!require_completed_events || looks_like_completed_temporal_event(&candidate.text))
                && candidate_temporal_rank(candidate).is_some()
                && temporal_target_score(candidate, &focus_terms) >= 8.0
        })
        .count();
    let use_sequence_rank = explicit_candidate_count < 2;
    let mut buckets: HashMap<String, (i32, f32, String)> = HashMap::new();

    for candidate in &candidates {
        let rank = if use_sequence_rank {
            candidate.sequence_rank
        } else {
            candidate_temporal_rank(candidate)
        };
        let Some(rank) = rank else {
            continue;
        };
        if require_completed_events && !looks_like_completed_temporal_event(&candidate.text) {
            continue;
        }

        let score = temporal_target_score(candidate, &focus_terms)
            + if candidate_temporal_rank(candidate).is_some() {
                4.0
            } else {
                0.0
            };
        if score < 8.0 {
            continue;
        }

        let mut summary = compact_temporal_event_summary(&candidate.text);
        if summary.split_whitespace().count() < 2 {
            summary =
                compact_temporal_event_summary(&summarize_turn_text(&candidate.text, &focus_terms));
        }
        let key = normalized_answer_key(&summary);
        if key.is_empty() {
            continue;
        }
        let replace = buckets
            .get(&key)
            .map(|(best_rank, best_score, _)| {
                (match direction {
                    TemporalDirection::Earlier => rank < *best_rank,
                    TemporalDirection::Later => rank > *best_rank,
                }) || (rank == *best_rank && score > *best_score)
            })
            .unwrap_or(true);
        if replace {
            buckets.insert(key, (rank, score, summary));
        }
    }

    let mut ordered = buckets.into_values().collect::<Vec<_>>();
    if ordered.len() < 2 {
        return None;
    }
    ordered.sort_by(|left, right| match direction {
        TemporalDirection::Earlier => left
            .0
            .cmp(&right.0)
            .then_with(|| right.1.total_cmp(&left.1))
            .then_with(|| left.2.cmp(&right.2)),
        TemporalDirection::Later => right
            .0
            .cmp(&left.0)
            .then_with(|| right.1.total_cmp(&left.1))
            .then_with(|| left.2.cmp(&right.2)),
    });

    let items = ordered
        .into_iter()
        .map(|(_, _, summary)| summary)
        .take(requested)
        .collect::<Vec<_>>();
    render_temporal_sequence_answer(&items)
}

fn select_temporal_anchor_order_answer(task: &str, evidence: &[EvidenceItem]) -> Option<String> {
    let lower = task.to_ascii_lowercase();
    if parse_binary_choice(task).is_some()
        || (!lower.contains(" first ")
            && !lower.starts_with("first ")
            && !lower.contains(" last ")
            && !lower.contains(" latest ")
            && !lower.contains(" earliest ")
            && !lower.contains(" most recent "))
    {
        return None;
    }

    let answer_direction = parse_temporal_order_direction(task)?;
    let (target_text, anchor_text, relation) = if let Some((left, right)) =
        split_once_case_insensitive(task.trim().trim_end_matches('?'), " after ")
    {
        (left, right, TemporalDirection::Later)
    } else if let Some((left, right)) =
        split_once_case_insensitive(task.trim().trim_end_matches('?'), " before ")
    {
        (left, right, TemporalDirection::Earlier)
    } else {
        return None;
    };

    let anchor = build_temporal_event_option(anchor_text)?;
    let mut target_terms = temporal_focus_terms(target_text);
    target_terms.retain(|term| !anchor.tokens.iter().any(|anchor_term| anchor_term == term));
    if target_terms.is_empty() {
        return None;
    }

    let candidates = collect_temporal_candidates(evidence, "temporal order selection");
    let anchor_explicit_rank = best_temporal_candidate_rank(&anchor, &candidates);
    let anchor_rank = anchor_explicit_rank
        .or_else(|| best_temporal_candidate_sequence_rank(&anchor, &candidates))?;
    let mut best: Option<(i32, f32, TemporalCandidate)> = None;

    for candidate in &candidates {
        let rank = if anchor_explicit_rank.is_some() {
            candidate_temporal_order_rank(candidate, &candidates)
        } else {
            candidate.sequence_rank
        };
        let Some(rank) = rank else {
            continue;
        };
        let matches_relation = match relation {
            TemporalDirection::Earlier => rank < anchor_rank,
            TemporalDirection::Later => rank > anchor_rank,
        };
        if !matches_relation {
            continue;
        }

        let score = temporal_target_score(&candidate, &target_terms);
        if score < 8.0 {
            continue;
        }

        let replace = match &best {
            Some((best_rank, best_score, _)) => match answer_direction {
                TemporalDirection::Earlier => {
                    rank < *best_rank || (rank == *best_rank && score > *best_score)
                },
                TemporalDirection::Later => {
                    rank > *best_rank || (rank == *best_rank && score > *best_score)
                },
            },
            None => true,
        };
        if replace {
            best = Some((rank, score, candidate.clone()));
        }
    }

    best.map(|(_, _, candidate)| render_temporal_candidate_answer(task, &candidate, &target_terms))
}

fn select_temporal_window_answer(task: &str, evidence: &[EvidenceItem]) -> Option<String> {
    let lower = task.to_ascii_lowercase();
    if !(lower.starts_with("what ") || lower.starts_with("which "))
        || lower.starts_with("when ")
        || parse_binary_choice(task).is_some()
    {
        return None;
    }

    let query_rank = extract_temporal_rank(task, None)?;
    let target_terms = temporal_focus_terms(task);
    if target_terms.is_empty() {
        return None;
    }

    let candidates = collect_temporal_candidates(evidence, "temporal window selection");
    let mut best: Option<(f32, TemporalCandidate)> = None;
    for candidate in &candidates {
        let Some(rank) = candidate_temporal_event_rank(candidate, &candidates) else {
            continue;
        };
        let target_score = temporal_target_score(candidate, &target_terms);
        if target_score < 7.0 {
            continue;
        }

        let distance_penalty = (rank - query_rank).abs() as f32 / 7.0;
        let score = target_score * 2.0 - distance_penalty;
        if best
            .as_ref()
            .map(|(best_score, _)| score > *best_score)
            .unwrap_or(true)
        {
            best = Some((score, candidate.clone()));
        }
    }

    best.map(|(_, candidate)| render_temporal_candidate_answer(task, &candidate, &target_terms))
}
