//! All `select_*` answer functions for temporal queries.

use super::*;
use crate::kg;
use std::collections::{HashMap, HashSet};

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

pub(crate) fn select_temporal_duration_answer(
    task: &str,
    evidence: &[EvidenceItem],
) -> Option<String> {
    let query = parse_temporal_gap_query(task)?;
    let legacy_day_query = task.to_ascii_lowercase().starts_with("how many days")
        && parse_temporal_duration_events(task).is_some();
    let candidates = collect_temporal_candidates(evidence, "temporal duration answer selection");
    let (start_rank, end_rank) = match &query.end {
        TemporalGapEndpoint::Event(end) => {
            let pair = if legacy_day_query {
                best_temporal_duration_pair_strict(&query.start, end, evidence)
            } else {
                best_temporal_duration_pair_from_candidates(&query.start, end, &candidates)
            };
            pair?
        },
        TemporalGapEndpoint::CurrentMoment => (
            best_calendar_grounded_temporal_candidate_rank(&query.start, &candidates)?.rank,
            best_calendar_grounded_current_anchor_rank(&candidates)?,
        ),
    };
    render_temporal_gap_answer((end_rank - start_rank).abs(), &query.answer_style)
}

pub(crate) fn select_temporal_count_answer(
    task: &str,
    evidence: &[EvidenceItem],
) -> Option<String> {
    select_temporal_elapsed_answer(task, evidence)
        .or_else(|| select_temporal_event_count_answer(task, evidence))
        .or_else(|| select_temporal_employment_duration_answer(task, evidence))
}

fn select_temporal_elapsed_answer(task: &str, evidence: &[EvidenceItem]) -> Option<String> {
    let (unit, option) = parse_temporal_elapsed_query(task)?;
    let candidates = collect_temporal_candidates(evidence, "temporal elapsed answer selection");
    let current_anchor_rank = best_calendar_grounded_current_anchor_rank(&candidates);
    let mut best: Option<(f32, String)> = None;

    for candidate in &candidates {
        let score = temporal_candidate_score(candidate, &option);
        if score < 8.0 {
            continue;
        }

        if let Some(amount) = resolve_from_temporal_option_focus(&candidate.text, &option, |text| {
            extract_relative_unit_amount(text, &unit)
        }) {
            update_best_answer(
                &mut best,
                score + 3.0,
                render_relative_elapsed(&unit, amount),
            );
            continue;
        }

        let uses_relative_reference =
            resolve_from_temporal_option_focus(&candidate.text, &option, |text| {
                extract_relative_reference_offset_days(text)
            })
            .is_some();
        let rank = candidate_temporal_event_rank_for_option(candidate, &option, &candidates);
        if let Some(rank) = rank {
            let fallback_anchor_rank = candidate
                .base_date
                .map(|(year, month, day)| ymd_to_days(year, month, day));
            if let Some(elapsed_days) =
                elapsed_days_since_anchor(current_anchor_rank.or(fallback_anchor_rank), rank)
            {
                if let Some(amount) = convert_days_to_elapsed_unit(elapsed_days, &unit) {
                    update_best_answer(
                        &mut best,
                        score + if uses_relative_reference { 8.0 } else { 4.0 },
                        render_relative_elapsed(&unit, amount),
                    );
                }
            }
        }
    }

    best.map(|(_, answer)| answer)
}

fn select_temporal_event_count_answer(task: &str, evidence: &[EvidenceItem]) -> Option<String> {
    let lower = task.to_ascii_lowercase();
    if !lower.starts_with("how many ")
        || lower.contains("how many days")
        || lower.contains("how many weeks")
        || lower.contains("how many months")
        || lower.contains("how many years")
    {
        return None;
    }

    let (prefix, anchor_raw, relation) = if let Some((left, right)) =
        split_once_case_insensitive(task.trim().trim_end_matches('?'), " before ")
    {
        (left, right, TemporalDirection::Earlier)
    } else if let Some((left, right)) =
        split_once_case_insensitive(task.trim().trim_end_matches('?'), " after ")
    {
        (left, right, TemporalDirection::Later)
    } else {
        return None;
    };

    let anchor = build_temporal_event_option(anchor_raw)?;
    let target_terms = temporal_focus_terms(prefix);
    if target_terms.is_empty() {
        return None;
    }

    let candidates = collect_temporal_candidates(evidence, "temporal count answer selection");
    let anchor_rank = best_temporal_candidate_strict_rank(&anchor, &candidates)?;
    let mut seen = HashSet::new();
    let mut count = 0usize;

    for candidate in &candidates {
        let Some(rank) = candidate_temporal_rank(candidate) else {
            continue;
        };
        if !looks_like_completed_temporal_event(&candidate.text) {
            continue;
        }
        let matches_relation = match relation {
            TemporalDirection::Earlier => rank < anchor_rank,
            TemporalDirection::Later => rank > anchor_rank,
        };
        if !matches_relation {
            continue;
        }

        let score = temporal_target_score(candidate, &target_terms);
        if score < 6.0 {
            continue;
        }
        if candidate
            .text
            .to_ascii_lowercase()
            .contains("similar to the one i attended")
        {
            continue;
        }

        let mut summary = compact_temporal_event_summary(&candidate.text);
        if summary.split_whitespace().count() < 2 {
            summary = compact_temporal_event_summary(&summarize_turn_text(
                &candidate.text,
                &target_terms,
            ));
        }
        if summary.is_empty() {
            continue;
        }
        if !seen.insert(normalized_answer_key(&summary)) {
            continue;
        }
        count += 1;
    }

    (count > 0).then(|| count.to_string())
}

pub(crate) fn select_temporal_employment_duration_answer(
    task: &str,
    evidence: &[EvidenceItem],
) -> Option<String> {
    let lower = task.to_ascii_lowercase();
    if !lower.starts_with("how long have i been working before i started my current job at ") {
        return None;
    }

    let (_, organization) =
        split_once_case_insensitive(task.trim().trim_end_matches('?'), "current job at ")?;
    let organization_terms = salient_query_terms(organization);
    if organization_terms.is_empty() {
        return None;
    }

    let candidates =
        collect_temporal_candidates(evidence, "temporal employment duration selection");
    let mut total_months = None;
    let mut current_months = None;

    for candidate in &candidates {
        if !candidate.user_authored {
            continue;
        }
        let lower_text = candidate.text.to_ascii_lowercase();
        if lower_text.contains("working professionally")
            || lower_text.contains("been in this field")
            || lower_text.contains("years of experience")
        {
            merge_duration_max(
                &mut total_months,
                extract_duration_months_near_phrases(
                    &candidate.text,
                    &[
                        "working professionally",
                        "been in this field",
                        "years of experience",
                    ],
                ),
            );
        }
        if organization_terms
            .iter()
            .all(|term| lower_text.contains(term.as_str()))
            && (lower_text.contains("working at")
                || lower_text.contains("work at")
                || lower_text.contains("been working at"))
        {
            merge_duration_max(
                &mut current_months,
                extract_duration_months_near_phrases(
                    &candidate.text,
                    &["working at", "work at", "been working at"],
                ),
            );
        }
    }

    for candidate in &candidates {
        let lower_text = candidate.text.to_ascii_lowercase();
        if lower_text.contains("working professionally")
            || lower_text.contains("been in this field")
            || lower_text.contains("years of experience")
        {
            merge_duration_max(
                &mut total_months,
                extract_duration_months_near_phrases(
                    &candidate.text,
                    &[
                        "working professionally",
                        "been in this field",
                        "years of experience",
                    ],
                ),
            );
        }
        if organization_terms
            .iter()
            .all(|term| lower_text.contains(term.as_str()))
            && (lower_text.contains("working at")
                || lower_text.contains("work at")
                || lower_text.contains("been working at"))
        {
            merge_duration_max(
                &mut current_months,
                extract_duration_months_near_phrases(
                    &candidate.text,
                    &["working at", "work at", "been working at"],
                ),
            );
        }
    }

    let total_months = total_months?;
    let current_months = current_months?;
    (total_months > current_months).then(|| format_duration_months(total_months - current_months))
}

pub(crate) fn select_temporal_state_answer(
    task: &str,
    evidence: &[EvidenceItem],
) -> Option<String> {
    let query = parse_temporal_state_query(task)?;
    let task_terms = salient_query_terms(task);
    if task_terms.is_empty() {
        return None;
    }

    let mut best: Option<(f32, String)> = None;
    for item in evidence {
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
            let score =
                temporal_state_candidate_score(&task_terms, item.score, &entity, &predicate);
            if score <= 0.0 {
                continue;
            }

            let Some(answer) = render_temporal_state_kg_answer(&query, &entity, &predicate) else {
                continue;
            };
            update_best_answer(&mut best, score, answer);
        }
    }

    best.map(|(_, answer)| answer)
}

pub(crate) fn select_dialogue_temporal_answer(
    task: &str,
    evidence: &[EvidenceItem],
) -> Option<String> {
    let lower = task.to_ascii_lowercase();
    let asks_when = lower.starts_with("when ")
        || lower.contains(" what date")
        || lower.contains(" which date")
        || lower.contains(" at what time");
    if !asks_when {
        return None;
    }

    let task_terms = salient_query_terms(task);
    if task_terms.is_empty() {
        return None;
    }
    let subject_hints = extract_subject_hints(task);
    let focus_terms = dialogue_focus_terms(task, &task_terms, &subject_hints);
    if focus_terms.is_empty() {
        return None;
    }
    let required_terms = required_tail_anchor_tokens(task);

    let mut candidates = Vec::new();
    for item in evidence {
        let Some(content) = read_context_text(&item.path, "dialogue temporal answer selection")
        else {
            continue;
        };
        for turn in parse_dialogue_turns(&content) {
            if !turn_matches_subject(&turn, &subject_hints) {
                continue;
            }
            let lower_turn = turn.text.to_ascii_lowercase();
            if !required_terms.is_empty()
                && !required_terms
                    .iter()
                    .all(|token| line_matches_event_token(&lower_turn, token))
            {
                continue;
            }
            let event_overlap = task_overlap_count(&turn.text, &focus_terms);
            if event_overlap == 0 {
                continue;
            }
            let Some(point) = extract_turn_temporal_answer(&turn) else {
                continue;
            };
            let subject_overlap = if subject_hints.is_empty() {
                0
            } else {
                task_overlap_count(&turn.text, &subject_hints)
            };
            let score = item.score * 10.0
                + dialogue_match_score(&turn.text, &task_terms)
                + event_overlap as f32 * 12.0
                + speaker_match_bonus(turn.speaker.as_deref(), &subject_hints)
                + subject_overlap as f32 * 4.0
                + temporal_point_specificity_bonus(point);
            candidates.push((score, render_temporal_answer(point)));
        }
    }

    candidates.sort_by(|left, right| {
        right
            .0
            .total_cmp(&left.0)
            .then_with(|| left.1.len().cmp(&right.1.len()))
            .then_with(|| left.1.cmp(&right.1))
    });
    let (top_score, top_answer) = candidates.first()?.clone();
    if candidates.iter().skip(1).any(|(score, answer)| {
        *score + 8.0 >= top_score && !answer_items_overlap(answer, &top_answer)
    }) {
        return None;
    }
    Some(top_answer)
}

fn extract_turn_temporal_answer(turn: &DialogueTurn) -> Option<TemporalAnswerPoint> {
    if let Some((year, month, day)) = extract_explicit_date(&turn.text, turn.session_date) {
        return Some(TemporalAnswerPoint::Day { year, month, day });
    }

    let base_date = turn.session_date?;
    let lower = turn.text.to_ascii_lowercase();

    if lower.contains("yesterday") {
        let (year, month, day) = shift_date_by_days(base_date, -1);
        return Some(TemporalAnswerPoint::Day { year, month, day });
    }
    if lower.contains("today") {
        return Some(TemporalAnswerPoint::Day {
            year: base_date.0,
            month: base_date.1,
            day: base_date.2,
        });
    }
    if lower.contains("tomorrow") {
        let (year, month, day) = shift_date_by_days(base_date, 1);
        return Some(TemporalAnswerPoint::Day { year, month, day });
    }
    if lower.contains("a couple of days ago") {
        let (year, month, day) = shift_date_by_days(base_date, -2);
        return Some(TemporalAnswerPoint::Day { year, month, day });
    }
    if lower.contains("a few days ago") {
        let (year, month, day) = shift_date_by_days(base_date, -3);
        return Some(TemporalAnswerPoint::Day { year, month, day });
    }
    if (lower.contains(" day ago") || lower.contains(" days ago"))
        && !lower.contains("week")
        && !lower.contains("month")
    {
        if let Some(days) = extract_relative_days(&turn.text) {
            let (year, month, day) = shift_date_by_days(base_date, -days);
            return Some(TemporalAnswerPoint::Day { year, month, day });
        }
    }
    if lower.contains("next month") {
        let (year, month) = shift_month(base_date.0, base_date.1, 1);
        return Some(TemporalAnswerPoint::Month { year, month });
    }
    if lower.contains("last month") {
        let (year, month) = shift_month(base_date.0, base_date.1, -1);
        return Some(TemporalAnswerPoint::Month { year, month });
    }
    if lower.contains("this month") {
        return Some(TemporalAnswerPoint::Month {
            year: base_date.0,
            month: base_date.1,
        });
    }
    if lower.contains("next year") {
        return Some(TemporalAnswerPoint::Year {
            year: base_date.0 + 1,
        });
    }
    if lower.contains("last year") {
        return Some(TemporalAnswerPoint::Year {
            year: base_date.0 - 1,
        });
    }
    if lower.contains("this year") {
        return Some(TemporalAnswerPoint::Year { year: base_date.0 });
    }
    None
}

fn render_temporal_answer(point: TemporalAnswerPoint) -> String {
    match point {
        TemporalAnswerPoint::Day { year, month, day } => {
            format!("{day} {} {year}", month_name(month))
        },
        TemporalAnswerPoint::Month { year, month } => format!("{} {year}", month_name(month)),
        TemporalAnswerPoint::Year { year } => year.to_string(),
    }
}

fn temporal_point_specificity_bonus(point: TemporalAnswerPoint) -> f32 {
    match point {
        TemporalAnswerPoint::Day { .. } => 12.0,
        TemporalAnswerPoint::Month { .. } => 8.0,
        TemporalAnswerPoint::Year { .. } => 6.0,
    }
}
