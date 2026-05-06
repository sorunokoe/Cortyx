use super::{
    answer_candidate_lines,
    // additional shared utilities:
    answer_items_overlap,
    candidate_weight,
    collapse_inline_whitespace,
    compact_answer,
    contains_standalone_token,
    dialogue_focus_terms,
    dialogue_match_score,
    extract_explicit_date,
    extract_relative_days,
    extract_session_base_date,
    extract_subject_hints,
    extract_temporal_rank,
    extract_trailing_count,
    normalized_answer_key,
    parse_binary_choice,
    parse_count_token,
    parse_dialogue_turns,
    read_context_text,
    salient_query_terms,
    sanitize_answer_text,
    sanitize_inline,
    speaker_match_bonus,
    split_candidate_fragments,
    strip_temporal_discourse_prefix,
    summarize_turn_text,
    task_overlap_count,
    term_list_overlap_count,
    trim_answer_tail,
    turn_matches_subject,
    update_best_answer,
    ymd_to_days,
    CalendarGroundedRank,
    ChoiceOption,
    DialogueTurn,
    EvidenceItem,
    TemporalAnswerPoint,
    TemporalCandidate,
    TemporalDirection,
    TemporalGapAnswerStyle,
    TemporalGapEndpoint,
    TemporalGapQuery,
    TemporalStateQuery,
    GENERIC_ANCHOR_TERMS,
    QUESTION_STOPWORDS,
};
use crate::kg;
use std::collections::{HashMap, HashSet};
use std::path::Path;

pub(super) fn select_comparison_answer(task: &str, evidence: &[EvidenceItem]) -> Option<String> {
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

pub(super) fn select_temporal_order_answer(
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

pub(super) fn select_temporal_duration_answer(
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

pub(super) fn select_temporal_count_answer(
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

pub(super) fn select_temporal_employment_duration_answer(
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

pub(super) fn select_temporal_state_answer(
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

pub(super) fn select_dialogue_temporal_answer(
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

pub(super) fn shift_date_by_days(base: (i32, u32, u32), delta_days: i32) -> (i32, u32, u32) {
    let rank = ymd_to_days(base.0, base.1, base.2) + delta_days;
    days_to_ymd(rank)
}

fn shift_month(year: i32, month: u32, delta_months: i32) -> (i32, u32) {
    let base = year * 12 + month as i32 - 1 + delta_months;
    let shifted_year = base.div_euclid(12);
    let shifted_month = base.rem_euclid(12) + 1;
    (shifted_year, shifted_month as u32)
}

fn days_to_ymd(mut days: i32) -> (i32, u32, u32) {
    let mut year = 1970;
    loop {
        let year_days = if is_leap_year(year) { 366 } else { 365 };
        if days < year_days {
            break;
        }
        days -= year_days;
        year += 1;
    }

    let month_lengths = [
        31,
        if is_leap_year(year) { 29 } else { 28 },
        31,
        30,
        31,
        30,
        31,
        31,
        30,
        31,
        30,
        31,
    ];
    let mut month = 1u32;
    for month_length in month_lengths {
        if days < month_length {
            break;
        }
        days -= month_length;
        month += 1;
    }
    (year, month, days as u32 + 1)
}

fn is_leap_year(year: i32) -> bool {
    (year % 4 == 0 && year % 100 != 0) || year % 400 == 0
}

fn month_name(month: u32) -> &'static str {
    match month {
        1 => "January",
        2 => "February",
        3 => "March",
        4 => "April",
        5 => "May",
        6 => "June",
        7 => "July",
        8 => "August",
        9 => "September",
        10 => "October",
        11 => "November",
        12 => "December",
        _ => "Unknown",
    }
}

fn parse_temporal_state_query(task: &str) -> Option<TemporalStateQuery> {
    if let Some(as_of) = extract_temporal_as_of_point(task) {
        return Some(TemporalStateQuery::AsOfValue { as_of });
    }

    let lower = task.to_ascii_lowercase();
    let asks_when = lower.starts_with("when ")
        || lower.contains(" what date")
        || lower.contains(" which date")
        || lower.contains(" at what time");
    let asks_change_time = asks_when
        && (lower.contains(" last change")
            || lower.contains(" last changed")
            || lower.contains(" latest change")
            || lower.contains(" most recent change")
            || lower.contains(" change ")
            || lower.contains(" changed ")
            || lower.contains(" become ")
            || lower.contains(" became "));
    if asks_change_time {
        return Some(TemporalStateQuery::LastChange {
            target_value: parse_temporal_change_target(task),
        });
    }

    let asks_current_state = lower.contains(" current ")
        || lower.contains(" currently ")
        || lower.contains(" right now")
        || lower.contains(" now ")
        || lower.ends_with(" now?")
        || lower.contains(" latest ")
        || lower.starts_with("what is my latest")
        || lower.contains(" still ")
        || lower.contains(" present ");
    asks_current_state.then_some(TemporalStateQuery::CurrentValue)
}

fn parse_temporal_change_target(task: &str) -> Option<ChoiceOption> {
    let trimmed = task.trim().trim_end_matches('?');
    for marker in [
        " changed to ",
        " change to ",
        " became ",
        " become ",
        " switched to ",
        " switch to ",
    ] {
        let Some((_, target)) = split_once_case_insensitive(trimmed, marker) else {
            continue;
        };
        let clean = target.trim();
        if let Some(option) = build_temporal_event_option(clean) {
            return Some(option);
        }
    }
    None
}

fn extract_temporal_as_of_point(task: &str) -> Option<String> {
    let (_, rest) = split_once_case_insensitive(task, "as of ")?;
    normalize_temporal_query_point(rest)
}

fn normalize_temporal_query_point(text: &str) -> Option<String> {
    if let Some(point) = extract_iso_temporal_point(text) {
        if point.len() == 10 {
            return Some(format!("{point}T23:59:59Z"));
        }
        return Some(point);
    }

    let (year, month, day) = extract_explicit_date(text, None)?;
    Some(format!("{year:04}-{month:02}-{day:02}T23:59:59Z"))
}

fn extract_iso_temporal_point(text: &str) -> Option<String> {
    for raw in text.split_whitespace() {
        let clean =
            raw.trim_matches(|c: char| matches!(c, ',' | '.' | '?' | '!' | ';' | '(' | ')'));
        if clean.len() < 10 || !is_iso_date_fragment(&clean[..10]) {
            continue;
        }
        if clean.len() >= 19 && clean.as_bytes().get(10) == Some(&b'T') {
            return Some(format!("{}Z", &clean[..19].trim_end_matches('Z')));
        }
        return Some(clean[..10].to_string());
    }
    None
}

fn is_iso_date_fragment(text: &str) -> bool {
    let bytes = text.as_bytes();
    bytes.len() == 10
        && bytes[4] == b'-'
        && bytes[7] == b'-'
        && bytes
            .iter()
            .enumerate()
            .all(|(idx, byte)| matches!(idx, 4 | 7) || byte.is_ascii_digit())
}

fn temporal_state_candidate_score(
    task_terms: &[String],
    retrieval_score: f32,
    entity: &kg::KgEntity,
    predicate: &str,
) -> f32 {
    let predicate_context = kg_predicate_query_terms(predicate).join(" ");
    if predicate_context.is_empty() || task_overlap_count(&predicate_context, task_terms) == 0 {
        return 0.0;
    }

    let entity_context = kg_entity_query_terms(&entity.entity).join(" ");
    let entity_overlap = if entity_context.is_empty() {
        0.0
    } else {
        task_overlap_count(&entity_context, task_terms) as f32 * 6.0
    };
    let combined_context = if entity_context.is_empty() {
        predicate_context
    } else {
        format!("{predicate_context} {entity_context}")
    };

    candidate_weight(&combined_context, task_terms, retrieval_score, false) + entity_overlap
}

pub(super) fn kg_predicate_query_terms(predicate: &str) -> Vec<String> {
    let mut terms = predicate
        .split('_')
        .map(str::trim)
        .filter(|token| token.len() >= 3)
        .map(ToString::to_string)
        .collect::<Vec<_>>();
    let extras: &[&str] = match predicate {
        "status" => &["state", "progress"],
        "blocker" => &["blocked", "blocking", "stuck", "waiting"],
        "next_step" => &["next", "step", "follow", "action"],
        "goal" => &["objective", "target", "aim"],
        "outcome" => &["result", "finding", "decision", "conclusion"],
        "title" => &["task", "focus", "work"],
        "action" => &["doing", "working", "investigating", "reviewing"],
        "location" => &["live", "lived", "home", "city", "based", "move", "moved"],
        "occupation" => &["job", "work", "career", "role", "employed"],
        "education" => &[
            "degree",
            "study",
            "studied",
            "graduate",
            "graduated",
            "school",
        ],
        "major" => &["study", "studied", "degree", "school"],
        "book" => &["reading", "read", "novel"],
        "partner" => &["wife", "husband", "boyfriend", "girlfriend", "spouse"],
        "pet" => &["dog", "cat", "pets"],
        "phone" => &["number", "call"],
        "project_name" => &["project", "name", "called"],
        "instagram_followers" => &["instagram", "follower", "followers"],
        "commute_time" => &["commute", "travel", "minutes", "time"],
        "fitness_record" => &["record", "best", "personal"],
        "vehicle_model" => &["vehicle", "car", "truck", "drive", "model"],
        "family_trip_location" => &["family", "trip", "vacation", "travel", "where"],
        "related_entity" => &[
            "entity",
            "entities",
            "file",
            "files",
            "module",
            "modules",
            "component",
        ],
        _ => &[],
    };
    terms.extend(extras.iter().map(|term| (*term).to_string()));
    terms.sort();
    terms.dedup();
    terms
}

pub(super) fn kg_entity_query_terms(entity: &str) -> Vec<String> {
    let mut terms = entity
        .split('_')
        .filter(|token| token.len() >= 3)
        .filter(|token| !matches!(*token, "agent" | "entity"))
        .map(ToString::to_string)
        .collect::<Vec<_>>();
    terms.sort();
    terms.dedup();
    terms
}

fn render_temporal_state_kg_answer(
    query: &TemporalStateQuery,
    entity: &kg::KgEntity,
    predicate: &str,
) -> Option<String> {
    match query {
        TemporalStateQuery::CurrentValue => format_kg_values(current_kg_values(entity, predicate)),
        TemporalStateQuery::AsOfValue { as_of } => {
            format_kg_values(kg_values_for_predicate_as_of(entity, predicate, as_of))
        },
        TemporalStateQuery::LastChange { target_value } => {
            kg_last_change_for_predicate(entity, predicate, target_value.as_ref())
        },
    }
}

pub(super) fn current_kg_values(entity: &kg::KgEntity, predicate: &str) -> Vec<String> {
    collect_kg_values(
        entity
            .facts
            .iter()
            .filter(|fact| fact.predicate == predicate && fact.ended.is_empty())
            .collect(),
    )
}

fn kg_values_for_predicate_as_of(
    entity: &kg::KgEntity,
    predicate: &str,
    as_of: &str,
) -> Vec<String> {
    collect_kg_values(
        entity
            .facts
            .iter()
            .filter(|fact| fact.predicate == predicate && kg_fact_is_active_as_of(fact, as_of))
            .collect(),
    )
}

fn kg_fact_is_active_as_of(fact: &kg::KgFact, as_of: &str) -> bool {
    let as_of = as_of.trim();
    if as_of.is_empty() {
        return fact.ended.is_empty();
    }
    if !fact.valid_from.is_empty() && fact.valid_from.as_str() > as_of {
        return false;
    }
    if !fact.ended.is_empty() && as_of >= fact.ended.as_str() {
        return false;
    }
    true
}

fn collect_kg_values(mut facts: Vec<&kg::KgFact>) -> Vec<String> {
    facts.sort_by(|a, b| {
        a.valid_from
            .cmp(&b.valid_from)
            .then_with(|| a.value.cmp(&b.value))
    });

    let mut values = Vec::new();
    for fact in facts {
        let value = render_kg_value(&fact.value);
        if value.is_empty() || values.iter().any(|existing| existing == &value) {
            continue;
        }
        values.push(value);
    }
    values
}

fn render_kg_value(value: &str) -> String {
    sanitize_inline(&value.replace('_', " "))
}

fn format_kg_values(values: Vec<String>) -> Option<String> {
    (!values.is_empty()).then(|| values.join(", "))
}

fn kg_last_change_for_predicate(
    entity: &kg::KgEntity,
    predicate: &str,
    target_value: Option<&ChoiceOption>,
) -> Option<String> {
    let mut timeline = entity.timeline_for(predicate);
    if let Some(target_value) = target_value {
        timeline.retain(|fact| kg_value_matches_target(&fact.value, target_value));
    }
    timeline.retain(|fact| !fact.valid_from.trim().is_empty());
    timeline.sort_by(|a, b| a.valid_from.cmp(&b.valid_from));
    timeline.last().map(|fact| fact.valid_from.clone())
}

fn kg_value_matches_target(value: &str, target_value: &ChoiceOption) -> bool {
    let lower = value.to_ascii_lowercase().replace('_', " ");
    target_value
        .tokens
        .iter()
        .all(|token| line_matches_event_token(&lower, token))
}

pub(super) fn parse_temporal_gap_query(task: &str) -> Option<TemporalGapQuery> {
    if task.to_ascii_lowercase().starts_with("how many days") {
        if let Some((start, end)) = parse_temporal_duration_events(task) {
            return Some(TemporalGapQuery {
                start,
                end: TemporalGapEndpoint::Event(end),
                answer_style: TemporalGapAnswerStyle::FixedUnit {
                    unit: "day".to_string(),
                },
            });
        }
    }
    parse_temporal_explicit_unit_gap_query(task)
        .or_else(|| parse_temporal_how_long_gap_query(task))
        .or_else(|| {
            let (start, end) = parse_temporal_duration_events(task)?;
            Some(TemporalGapQuery {
                start,
                end: TemporalGapEndpoint::Event(end),
                answer_style: TemporalGapAnswerStyle::FixedUnit {
                    unit: "day".to_string(),
                },
            })
        })
}

fn parse_temporal_explicit_unit_gap_query(task: &str) -> Option<TemporalGapQuery> {
    let trimmed = task.trim().trim_end_matches('?');
    let lower = trimmed.to_ascii_lowercase();
    for unit in ["day", "week", "month", "year"] {
        let prefixes = [
            format!("how many {unit} "),
            format!("how many {unit}s "),
            format!("how many {unit}"),
            format!("how many {unit}s"),
        ];
        if !prefixes
            .iter()
            .any(|prefix| lower.starts_with(prefix.trim_end()))
        {
            continue;
        }

        if let Some(rest) =
            strip_prefix_case_insensitive(trimmed, &format!("How many {unit}s had passed between "))
                .or_else(|| {
                    strip_prefix_case_insensitive(
                        trimmed,
                        &format!("How many {unit} had passed between "),
                    )
                })
                .or_else(|| {
                    strip_prefix_case_insensitive(
                        trimmed,
                        &format!("How many {unit}s passed between "),
                    )
                })
                .or_else(|| {
                    strip_prefix_case_insensitive(
                        trimmed,
                        &format!("How many {unit} passed between "),
                    )
                })
                .or_else(|| {
                    strip_prefix_case_insensitive(
                        trimmed,
                        &format!("How many {unit}s were there between "),
                    )
                })
                .or_else(|| {
                    strip_prefix_case_insensitive(
                        trimmed,
                        &format!("How many {unit} were there between "),
                    )
                })
                .or_else(|| {
                    strip_prefix_case_insensitive(
                        trimmed,
                        &format!("How many {unit}s passed between the time "),
                    )
                })
                .or_else(|| {
                    strip_prefix_case_insensitive(
                        trimmed,
                        &format!("How many {unit} passed between the time "),
                    )
                })
        {
            let (left, right) = split_once_case_insensitive(rest, " and ")?;
            return Some(TemporalGapQuery {
                start: build_temporal_event_option(left)?,
                end: TemporalGapEndpoint::Event(build_temporal_event_option(right)?),
                answer_style: TemporalGapAnswerStyle::FixedUnit {
                    unit: unit.to_string(),
                },
            });
        }

        if let Some(rest) =
            strip_prefix_case_insensitive(trimmed, &format!("How many {unit}s before ")).or_else(
                || strip_prefix_case_insensitive(trimmed, &format!("How many {unit} before ")),
            )
        {
            let (reference, target) = split_once_case_insensitive(rest, " did ")?;
            return Some(TemporalGapQuery {
                start: build_temporal_event_option(target)?,
                end: TemporalGapEndpoint::Event(build_temporal_event_option(reference)?),
                answer_style: TemporalGapAnswerStyle::FixedUnit {
                    unit: unit.to_string(),
                },
            });
        }

        if let Some(rest) =
            strip_prefix_case_insensitive(trimmed, &format!("How many {unit}s after ")).or_else(
                || strip_prefix_case_insensitive(trimmed, &format!("How many {unit} after ")),
            )
        {
            let (reference, target) = split_once_case_insensitive(rest, " did ")?;
            return Some(TemporalGapQuery {
                start: build_temporal_event_option(reference)?,
                end: TemporalGapEndpoint::Event(build_temporal_event_option(target)?),
                answer_style: TemporalGapAnswerStyle::FixedUnit {
                    unit: unit.to_string(),
                },
            });
        }

        let take_markers = [format!("did it take for "), format!("did it take me to ")];
        for marker in take_markers {
            if let Some(idx) = lower.find(&marker) {
                let rest = &trimmed[idx + marker.len()..];
                let (target, start) = split_once_case_insensitive(rest, " after ")?;
                return Some(TemporalGapQuery {
                    start: build_temporal_event_option(start)?,
                    end: TemporalGapEndpoint::Event(build_temporal_event_option(target)?),
                    answer_style: TemporalGapAnswerStyle::FixedUnit {
                        unit: unit.to_string(),
                    },
                });
            }
        }

        if let Some(rest) =
            strip_prefix_case_insensitive(trimmed, &format!("How many {unit}s had passed since "))
                .or_else(|| {
                    strip_prefix_case_insensitive(
                        trimmed,
                        &format!("How many {unit} had passed since "),
                    )
                })
        {
            if let Some((start, end)) = split_once_case_insensitive(rest, " when ") {
                return Some(TemporalGapQuery {
                    start: build_temporal_event_option(start)?,
                    end: TemporalGapEndpoint::Event(build_temporal_event_option(end)?),
                    answer_style: TemporalGapAnswerStyle::FixedUnit {
                        unit: unit.to_string(),
                    },
                });
            }
        }

        if let Some(rest) =
            strip_prefix_case_insensitive(trimmed, &format!("How many {unit}s have passed since "))
                .or_else(|| {
                    strip_prefix_case_insensitive(
                        trimmed,
                        &format!("How many {unit} have passed since "),
                    )
                })
                .or_else(|| {
                    strip_prefix_case_insensitive(
                        trimmed,
                        &format!("How many {unit}s has passed since "),
                    )
                })
                .or_else(|| {
                    strip_prefix_case_insensitive(
                        trimmed,
                        &format!("How many {unit} has passed since "),
                    )
                })
        {
            return Some(TemporalGapQuery {
                start: build_temporal_event_option(rest)?,
                end: TemporalGapEndpoint::CurrentMoment,
                answer_style: TemporalGapAnswerStyle::FixedUnit {
                    unit: unit.to_string(),
                },
            });
        }

        if let Some(rest) =
            strip_prefix_case_insensitive(trimmed, &format!("How many {unit}s have I been "))
                .or_else(|| {
                    strip_prefix_case_insensitive(trimmed, &format!("How many {unit} have I been "))
                })
                .or_else(|| {
                    strip_prefix_case_insensitive(trimmed, &format!("How many {unit}s had I been "))
                })
                .or_else(|| {
                    strip_prefix_case_insensitive(trimmed, &format!("How many {unit} had I been "))
                })
        {
            let (start, end) = split_once_case_insensitive(rest, " when ")?;
            return Some(TemporalGapQuery {
                start: build_temporal_event_option(start)?,
                end: TemporalGapEndpoint::Event(build_temporal_event_option(end)?),
                answer_style: TemporalGapAnswerStyle::FixedUnit {
                    unit: unit.to_string(),
                },
            });
        }
    }
    None
}

fn parse_temporal_how_long_gap_query(task: &str) -> Option<TemporalGapQuery> {
    let trimmed = task.trim().trim_end_matches('?');
    if trimmed
        .to_ascii_lowercase()
        .starts_with("how long have i been working before i started my current job at ")
    {
        return None;
    }
    let rest = strip_prefix_case_insensitive(trimmed, "How long ")?;
    let (start, end) = if let Some((left, right)) = split_once_case_insensitive(rest, " when ") {
        (left, right)
    } else if let Some((left, right)) = split_once_case_insensitive(rest, " before ") {
        (left, right)
    } else if let Some((left, right)) = split_once_case_insensitive(rest, " after ") {
        (right, left)
    } else {
        return None;
    };
    let start = strip_prefix_case_insensitive(start.trim(), "had I been ")
        .or_else(|| strip_prefix_case_insensitive(start.trim(), "have I been "))
        .or_else(|| strip_prefix_case_insensitive(start.trim(), "did I "))
        .unwrap_or(start)
        .trim();
    Some(TemporalGapQuery {
        start: build_temporal_event_option(start)?,
        end: TemporalGapEndpoint::Event(build_temporal_event_option(end)?),
        answer_style: TemporalGapAnswerStyle::NaturalLanguage,
    })
}

fn parse_temporal_duration_events(task: &str) -> Option<(ChoiceOption, ChoiceOption)> {
    let trimmed = task.trim().trim_end_matches('?');
    let lower = trimmed.to_ascii_lowercase();
    if !lower.contains("how many days") {
        return None;
    }

    if let Some(rest) = trimmed
        .strip_prefix("How many days had passed between ")
        .or_else(|| trimmed.strip_prefix("How many days passed between "))
        .or_else(|| trimmed.strip_prefix("How many days were there between "))
    {
        let (left, right) = split_once_case_insensitive(rest, " and ")?;
        return Some((
            build_temporal_event_option(left)?,
            build_temporal_event_option(right)?,
        ));
    }

    if let Some(rest) = trimmed.strip_prefix("How many days before ") {
        let (reference, target) = split_once_case_insensitive(rest, " did ")?;
        return Some((
            build_temporal_event_option(target)?,
            build_temporal_event_option(reference)?,
        ));
    }

    if let Some(rest) = trimmed.strip_prefix("How many days after ") {
        let (reference, target) = split_once_case_insensitive(rest, " did ")?;
        return Some((
            build_temporal_event_option(reference)?,
            build_temporal_event_option(target)?,
        ));
    }

    let take_marker = "did it take for ";
    if let Some(idx) = lower.find(take_marker) {
        let rest = &trimmed[idx + take_marker.len()..];
        let (target, start) = split_once_case_insensitive(rest, " after ")?;
        return Some((
            build_temporal_event_option(start)?,
            build_temporal_event_option(target)?,
        ));
    }

    None
}

fn build_temporal_event_option(text: &str) -> Option<ChoiceOption> {
    let display = strip_leading_temporal_actor(text);
    let mut tokens = display
        .split(|c: char| !c.is_alphanumeric() && c != '\'')
        .filter_map(|token| {
            let lower = token
                .trim_matches(|c: char| !c.is_ascii_alphanumeric())
                .trim_matches('\'')
                .to_ascii_lowercase();
            if lower.len() < 3
                || (QUESTION_STOPWORDS.contains(&lower.as_str())
                    && !matches!(lower.as_str(), "book" | "booked" | "booking"))
                || parse_count_token(&lower).is_some()
            {
                None
            } else {
                Some(lower)
            }
        })
        .collect::<Vec<_>>();
    if tokens.iter().filter(|token| token.len() >= 4).count() >= 2 {
        tokens.retain(|token| token.len() >= 4);
    }
    tokens.sort();
    tokens.dedup();
    if display.is_empty() || tokens.is_empty() {
        return None;
    }
    Some(ChoiceOption { display, tokens })
}

fn strip_leading_temporal_actor(text: &str) -> String {
    let mut clean = sanitize_answer_text(text);
    loop {
        let mut stripped = false;
        for prefix in [
            "the day i ",
            "the time i ",
            "the day ",
            "the time ",
            "day i ",
            "time i ",
            "when i ",
            "i ",
            "me ",
            "my ",
            "we ",
            "our ",
            "he ",
            "his ",
            "she ",
            "her ",
            "they ",
            "their ",
            "to ",
        ] {
            if clean.to_ascii_lowercase().starts_with(prefix) {
                clean = clean[prefix.len()..].trim().to_string();
                stripped = true;
                break;
            }
        }
        if !stripped {
            break;
        }
    }
    clean
}

pub(super) fn split_once_case_insensitive<'a>(
    text: &'a str,
    needle: &str,
) -> Option<(&'a str, &'a str)> {
    let lower_text = text.to_ascii_lowercase();
    let lower_needle = needle.to_ascii_lowercase();
    let idx = lower_text.find(&lower_needle)?;
    Some((&text[..idx], &text[idx + needle.len()..]))
}

fn strip_prefix_case_insensitive<'a>(text: &'a str, prefix: &str) -> Option<&'a str> {
    if text.len() >= prefix.len() && text[..prefix.len()].eq_ignore_ascii_case(prefix) {
        Some(&text[prefix.len()..])
    } else {
        None
    }
}

pub(super) fn parse_temporal_elapsed_query(task: &str) -> Option<(String, ChoiceOption)> {
    let trimmed = task.trim().trim_end_matches('?');
    if !trimmed.to_ascii_lowercase().starts_with("how many ") {
        return None;
    }
    for unit in ["day", "week", "month", "year"] {
        for marker in [format!("{unit} ago did "), format!("{unit}s ago did ")] {
            let Some((_, event)) = split_once_case_insensitive(trimmed, &marker) else {
                continue;
            };
            return Some((unit.to_string(), build_temporal_event_option(event)?));
        }
    }
    None
}

fn parse_temporal_order_direction(task: &str) -> Option<TemporalDirection> {
    let lower = task.to_ascii_lowercase();
    if lower.contains(" first")
        || lower.starts_with("first ")
        || lower.contains(" earliest")
        || lower.contains(" older")
    {
        Some(TemporalDirection::Earlier)
    } else if lower.contains(" last")
        || lower.contains(" latest")
        || lower.contains(" most recent")
        || lower.contains(" newest")
    {
        Some(TemporalDirection::Later)
    } else {
        None
    }
}

pub(super) fn is_temporal_sequence_query(task: &str) -> bool {
    let lower = task.to_ascii_lowercase();
    lower.starts_with("what is the order of")
        || lower.starts_with("what's the order of")
        || lower.contains("from first to last")
        || lower.contains("from last to first")
        || lower.contains("order from first to last")
        || lower.contains("from earliest to latest")
        || lower.contains("from latest to earliest")
        || lower.contains("starting from the earliest")
        || lower.contains("starting from the latest")
        || lower.contains("first, second")
        || lower.contains("first second")
}

pub(super) fn temporal_focus_terms(text: &str) -> Vec<String> {
    let lower = text.to_ascii_lowercase();
    let mut terms = salient_query_terms(text);
    terms.retain(|term| {
        !matches!(
            term.as_str(),
            "many"
                | "ago"
                | "first"
                | "last"
                | "latest"
                | "earliest"
                | "recent"
                | "months"
                | "month"
                | "weeks"
                | "week"
                | "days"
                | "day"
                | "years"
                | "year"
                | "happened"
                | "happen"
                | "current"
                | "currently"
        )
    });

    if lower.contains("shoe") {
        terms.extend(
            ["sneakers", "boots", "sandals", "trainers"]
                .iter()
                .map(|term| (*term).to_string()),
        );
    }
    if lower.contains("issue") || lower.contains("problem") {
        terms.extend(
            ["issue", "problem", "malfunction", "broken"]
                .iter()
                .map(|term| (*term).to_string()),
        );
    }
    if lower.contains("clean") || lower.contains("wash") {
        terms.extend(
            ["clean", "cleaned", "washed", "washing"]
                .iter()
                .map(|term| (*term).to_string()),
        );
    }
    if lower.contains("event") || lower.contains("participat") || lower.contains("charity") {
        terms.extend(
            [
                "event",
                "events",
                "attended",
                "attend",
                "participated",
                "participate",
                "volunteered",
                "volunteer",
                "joined",
                "join",
                "walk",
                "walked",
                "run",
                "ran",
                "tournament",
                "gala",
                "marathon",
                "triathlon",
                "cleanup",
                "drive",
            ]
            .iter()
            .map(|term| (*term).to_string()),
        );
    }
    if lower.contains("buy")
        || lower.contains("bought")
        || lower.contains("purchase")
        || lower.contains("purchased")
        || lower.contains(" got ")
        || lower.starts_with("got ")
    {
        terms.extend(
            ["buy", "bought", "purchase", "purchased", "got"]
                .iter()
                .map(|term| (*term).to_string()),
        );
    }
    if lower.contains("set up") || lower.contains("setup") || lower.contains("install") {
        terms.extend(
            ["setup", "installed", "installing", "upgraded"]
                .iter()
                .map(|term| (*term).to_string()),
        );
    }
    if lower.contains("work") || lower.contains("job") {
        terms.extend(
            ["working", "job", "career", "professionally"]
                .iter()
                .map(|term| (*term).to_string()),
        );
    }
    terms.sort();
    terms.dedup();
    terms
}

fn temporal_sequence_focus_terms(task: &str) -> Vec<String> {
    let lower = task.to_ascii_lowercase();
    let mut terms = temporal_focus_terms(task);
    terms.retain(|term| {
        !matches!(
            term.as_str(),
            "order"
                | "past"
                | "latest"
                | "earliest"
                | "starting"
                | "first"
                | "second"
                | "third"
                | "fourth"
                | "three"
                | "four"
                | "five"
                | "six"
                | "among"
        )
    });

    if lower.contains("trip") {
        terms.extend(
            [
                "trip", "trips", "road", "hike", "camping", "travel", "vacation",
            ]
            .iter()
            .map(|term| (*term).to_string()),
        );
    }
    if lower.contains("concert") || lower.contains("music") {
        terms.extend(
            ["concert", "music", "festival", "jazz", "show"]
                .iter()
                .map(|term| (*term).to_string()),
        );
    }
    if lower.contains("museum") {
        terms.extend(
            ["museum", "art", "history", "science"]
                .iter()
                .map(|term| (*term).to_string()),
        );
    }
    if lower.contains("event") {
        terms.extend(
            ["event", "events", "attended", "participated", "visited"]
                .iter()
                .map(|term| (*term).to_string()),
        );
    }
    if lower.contains("graduat") {
        terms.extend(
            ["graduated", "graduation", "graduate"]
                .iter()
                .map(|term| (*term).to_string()),
        );
    }
    terms.sort();
    terms.dedup();
    terms
}

pub(super) fn is_temporal_reasoning_query(task: &str) -> bool {
    let lower = task.to_ascii_lowercase();
    let how_many_temporal = lower.starts_with("how many ")
        && (lower.contains(" day")
            || lower.contains(" week")
            || lower.contains(" month")
            || lower.contains(" year")
            || lower.contains(" ago")
            || lower.contains(" before ")
            || lower.contains(" after ")
            || lower.contains(" first")
            || lower.contains(" last"));
    lower.starts_with("when ")
        || lower.contains(" what date")
        || lower.contains(" which date")
        || lower.contains(" which day")
        || how_many_temporal
        || lower.starts_with("how long ")
        || lower.contains(" first")
        || lower.contains(" last")
        || lower.contains(" earliest")
        || lower.contains(" latest")
        || lower.contains(" most recent")
        || lower.contains(" before ")
        || lower.contains(" after ")
        || lower.contains("order of")
}

pub(super) fn collect_temporal_candidates(
    evidence: &[EvidenceItem],
    stage: &str,
) -> Vec<TemporalCandidate> {
    let mut candidates = Vec::new();
    let mut seen: HashSet<(String, Option<(i32, u32, u32)>, bool)> = HashSet::new();

    for (item_index, item) in evidence.iter().enumerate() {
        let Some(content) = read_context_text(&item.path, stage) else {
            continue;
        };
        let base_date = extract_session_base_date(&content);
        let turns = parse_dialogue_turns(&content);
        let mut local_sequence = 0usize;
        if !turns.is_empty() {
            let has_user_turns = turns.iter().any(is_user_dialogue_turn);
            for turn in turns {
                let user_authored = is_user_dialogue_turn(&turn);
                if has_user_turns && !user_authored {
                    continue;
                }
                let turn_base = turn.session_date.or(base_date);
                push_temporal_candidate(
                    &mut candidates,
                    &mut seen,
                    &turn.text,
                    turn_base,
                    item.score,
                    user_authored,
                    temporal_candidate_sequence_rank(&item.path, item_index, local_sequence),
                );
                local_sequence += 1;
                for fragment in split_candidate_fragments(&turn.text) {
                    push_temporal_candidate(
                        &mut candidates,
                        &mut seen,
                        &fragment,
                        turn_base,
                        item.score,
                        user_authored,
                        temporal_candidate_sequence_rank(&item.path, item_index, local_sequence),
                    );
                    local_sequence += 1;
                }
            }
            continue;
        }

        for line in answer_candidate_lines(&content) {
            push_temporal_candidate(
                &mut candidates,
                &mut seen,
                &line,
                base_date,
                item.score,
                false,
                temporal_candidate_sequence_rank(&item.path, item_index, local_sequence),
            );
            local_sequence += 1;
        }
    }

    candidates
}

fn is_user_dialogue_turn(turn: &DialogueTurn) -> bool {
    turn.speaker
        .as_deref()
        .map(|speaker| speaker.eq_ignore_ascii_case("user"))
        .unwrap_or(false)
}

fn push_temporal_candidate(
    candidates: &mut Vec<TemporalCandidate>,
    seen: &mut HashSet<(String, Option<(i32, u32, u32)>, bool)>,
    text: &str,
    base_date: Option<(i32, u32, u32)>,
    retrieval_score: f32,
    user_authored: bool,
    sequence_rank: Option<i32>,
) {
    let clean = sanitize_temporal_candidate_text(text);
    if clean.split_whitespace().count() < 3 {
        return;
    }
    let key = (clean.to_ascii_lowercase(), base_date, user_authored);
    if !seen.insert(key) {
        return;
    }
    candidates.push(TemporalCandidate {
        text: clean,
        base_date,
        retrieval_score,
        user_authored,
        ordinal: candidates.len(),
        sequence_rank,
    });
}

fn sanitize_temporal_candidate_text(text: &str) -> String {
    collapse_inline_whitespace(text).chars().take(600).collect()
}

fn candidate_temporal_rank(candidate: &TemporalCandidate) -> Option<i32> {
    extract_temporal_rank(&candidate.text, candidate.base_date)
}

fn candidate_temporal_base_rank(candidate: &TemporalCandidate) -> Option<i32> {
    candidate_temporal_rank(candidate).or_else(|| candidate_temporal_state_origin_rank(candidate))
}

fn temporal_state_origin_rank_from_text(
    text: &str,
    base_date: Option<(i32, u32, u32)>,
    user_authored: bool,
) -> Option<i32> {
    let lower = text.to_ascii_lowercase();
    let first_person = lower.starts_with("i ")
        || lower.starts_with("i'")
        || lower.contains(" i've ")
        || lower.contains(" i have ")
        || lower.contains(" i am ")
        || lower.contains(" i'm ");
    if !user_authored && !first_person {
        return None;
    }
    let looks_ongoing = lower.contains("i've been")
        || lower.contains("i have been")
        || lower.contains("i'm")
        || lower.contains("i am")
        || lower.contains(" now")
        || lower.contains(" currently")
        || lower.contains(" so far");
    if !looks_ongoing {
        return None;
    }

    let current_rank = base_date
        .map(|(year, month, day)| ymd_to_days(year, month, day))
        .unwrap_or(0);
    extract_duration_days_near_phrases(text, &["for", "now", "currently", "so far", "already"])
        .map(|days| current_rank - days)
}

fn candidate_temporal_state_origin_rank(candidate: &TemporalCandidate) -> Option<i32> {
    temporal_state_origin_rank_from_text(
        &candidate.text,
        candidate.base_date,
        candidate.user_authored,
    )
}

fn candidate_temporal_order_rank(
    candidate: &TemporalCandidate,
    candidates: &[TemporalCandidate],
) -> Option<i32> {
    {
        let (direction, offset_days) = extract_relative_reference_offset_days(&candidate.text)?;
        let reference_rank = resolve_temporal_reference_rank(candidate, candidates)?;
        Some(match direction {
            TemporalDirection::Earlier => reference_rank - offset_days,
            TemporalDirection::Later => reference_rank + offset_days,
        })
    }
    .or_else(|| candidate_temporal_base_rank(candidate))
}

fn temporal_current_rank_from_text(
    text: &str,
    base_date: Option<(i32, u32, u32)>,
    user_authored: bool,
) -> Option<i32> {
    let lower = text.to_ascii_lowercase();
    let first_person = lower.starts_with("i ")
        || lower.starts_with("i'")
        || lower.contains(" i've ")
        || lower.contains(" i have ")
        || lower.contains(" i am ")
        || lower.contains(" i'm ");
    if !user_authored && !first_person {
        return None;
    }
    let has_current_marker = lower.contains("today")
        || lower.contains("right now")
        || lower.contains("currently")
        || lower.contains("this week")
        || lower.contains("this month")
        || lower.contains("this year")
        || contains_standalone_token(&lower, "now");
    if !has_current_marker {
        return None;
    }
    Some(
        base_date
            .map(|(year, month, day)| ymd_to_days(year, month, day))
            .unwrap_or(0),
    )
}

fn candidate_temporal_current_rank(candidate: &TemporalCandidate) -> Option<i32> {
    temporal_current_rank_from_text(
        &candidate.text,
        candidate.base_date,
        candidate.user_authored,
    )
}

fn candidate_temporal_event_rank(
    candidate: &TemporalCandidate,
    candidates: &[TemporalCandidate],
) -> Option<i32> {
    candidate_temporal_order_rank(candidate, candidates)
        .or_else(|| candidate_temporal_current_rank(candidate))
}

fn temporal_option_focus_tail<'a>(text: &'a str, option: &ChoiceOption) -> Option<&'a str> {
    let lower = text.to_ascii_lowercase();
    let start = option
        .tokens
        .iter()
        .flat_map(|token| lower.match_indices(token).map(|(idx, _)| idx))
        .max()?;
    Some(&text[start..])
}

fn resolve_from_temporal_option_focus<T>(
    text: &str,
    option: &ChoiceOption,
    mut resolver: impl FnMut(&str) -> Option<T>,
) -> Option<T> {
    if let Some(tail) = temporal_option_focus_tail(text, option) {
        if let Some(value) = resolver(tail) {
            return Some(value);
        }
    }
    resolver(text)
}

fn candidate_temporal_order_rank_for_option(
    candidate: &TemporalCandidate,
    option: &ChoiceOption,
    candidates: &[TemporalCandidate],
) -> Option<i32> {
    temporal_option_focus_tail(&candidate.text, option)
        .and_then(|tail| {
            {
                let (direction, offset_days) = extract_relative_reference_offset_days(tail)?;
                let reference_rank = resolve_temporal_reference_rank(candidate, candidates)?;
                Some(match direction {
                    TemporalDirection::Earlier => reference_rank - offset_days,
                    TemporalDirection::Later => reference_rank + offset_days,
                })
            }
            .or_else(|| extract_temporal_rank(tail, candidate.base_date))
            .or_else(|| {
                temporal_state_origin_rank_from_text(
                    tail,
                    candidate.base_date,
                    candidate.user_authored,
                )
            })
        })
        .or_else(|| candidate_temporal_order_rank(candidate, candidates))
}

fn candidate_temporal_event_rank_for_option(
    candidate: &TemporalCandidate,
    option: &ChoiceOption,
    candidates: &[TemporalCandidate],
) -> Option<i32> {
    candidate_temporal_order_rank_for_option(candidate, option, candidates).or_else(|| {
        temporal_option_focus_tail(&candidate.text, option)
            .and_then(|tail| {
                temporal_current_rank_from_text(tail, candidate.base_date, candidate.user_authored)
            })
            .or_else(|| candidate_temporal_current_rank(candidate))
    })
}

fn best_temporal_candidate_rank(
    option: &ChoiceOption,
    candidates: &[TemporalCandidate],
) -> Option<i32> {
    best_temporal_candidate_rank_for_comparison(option, candidates, false)
}

fn best_temporal_candidate_rank_for_comparison(
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

fn best_calendar_grounded_temporal_candidate_rank(
    option: &ChoiceOption,
    candidates: &[TemporalCandidate],
) -> Option<CalendarGroundedRank> {
    best_calendar_grounded_temporal_candidate_rank_with_competing_option(option, None, candidates)
}

fn best_calendar_grounded_temporal_candidate_rank_with_competing_option(
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

fn best_temporal_candidate_strict_rank(
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

pub(super) fn best_calendar_grounded_current_anchor_rank(
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

fn best_temporal_candidate_sequence_rank(
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

fn best_temporal_candidate_loose_sequence_rank(
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

fn best_temporal_duration_pair(
    start: &ChoiceOption,
    end: &ChoiceOption,
    evidence: &[EvidenceItem],
) -> Option<(i32, i32)> {
    let candidates = collect_temporal_candidates(evidence, "temporal duration answer selection");
    best_temporal_duration_pair_from_candidates(start, end, &candidates)
}

fn best_temporal_duration_pair_from_candidates(
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

fn best_temporal_duration_pair_strict(
    start: &ChoiceOption,
    end: &ChoiceOption,
    evidence: &[EvidenceItem],
) -> Option<(i32, i32)> {
    best_temporal_duration_pair(start, end, evidence)
}

fn temporal_specificity_score(
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

fn temporal_candidate_score(candidate: &TemporalCandidate, option: &ChoiceOption) -> f32 {
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

fn temporal_target_score(candidate: &TemporalCandidate, target_terms: &[String]) -> f32 {
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

fn extract_relative_unit_amount(text: &str, unit: &str) -> Option<i32> {
    let lower = text.to_ascii_lowercase();
    match unit {
        "day" => {
            if lower.contains("yesterday") {
                return Some(1);
            }
        },
        "week" => {
            if lower.contains("last week") {
                return Some(1);
            }
        },
        "month" => {
            if lower.contains("last month") {
                return Some(1);
            }
        },
        "year" => {
            if lower.contains("last year") {
                return Some(1);
            }
        },
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

fn extract_relative_reference_offset_days(text: &str) -> Option<(TemporalDirection, i32)> {
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

fn resolve_temporal_reference_rank(
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

fn temporal_reference_terms(text: &str) -> Vec<String> {
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

fn extract_duration_months(text: &str) -> Option<i32> {
    let years = extract_count_before_unit(text, "year").unwrap_or(0);
    let months = extract_count_before_unit(text, "month").unwrap_or(0);
    ((years > 0) || (months > 0)).then_some(years * 12 + months)
}

fn extract_duration_months_near_phrases(text: &str, phrases: &[&str]) -> Option<i32> {
    let tokens = duration_candidate_tokens(text);
    let spans = extract_duration_month_spans(&tokens);
    if spans.is_empty() {
        return None;
    }

    let mut positions = phrases
        .iter()
        .flat_map(|phrase| phrase_token_positions(&tokens, phrase))
        .collect::<Vec<_>>();
    positions.sort();
    positions.dedup();
    if positions.is_empty() {
        return extract_duration_months(text);
    }

    positions
        .into_iter()
        .flat_map(|position| {
            spans.iter().map(move |(start, end, months)| {
                let distance = if *start >= position {
                    *start - position
                } else {
                    position.saturating_sub(*end)
                };
                (distance, *start, *months)
            })
        })
        .min_by(|left, right| left.cmp(right))
        .map(|(_, _, months)| months)
        .or_else(|| extract_duration_months(text))
}

fn extract_duration_days(text: &str) -> Option<i32> {
    let tokens = duration_candidate_tokens(text);
    extract_duration_day_spans(&tokens)
        .first()
        .map(|(_, _, days)| *days)
}

fn extract_duration_days_near_phrases(text: &str, phrases: &[&str]) -> Option<i32> {
    let tokens = duration_candidate_tokens(text);
    let spans = extract_duration_day_spans(&tokens);
    if spans.is_empty() {
        return None;
    }

    let mut positions = phrases
        .iter()
        .flat_map(|phrase| phrase_token_positions(&tokens, phrase))
        .collect::<Vec<_>>();
    positions.sort();
    positions.dedup();
    if positions.is_empty() {
        return extract_duration_days(text);
    }

    positions
        .into_iter()
        .flat_map(|position| {
            spans.iter().map(move |(start, end, days)| {
                let distance = if *start >= position {
                    *start - position
                } else {
                    position.saturating_sub(*end)
                };
                (distance, *start, *days)
            })
        })
        .min_by(|left, right| left.cmp(right))
        .map(|(_, _, days)| days)
        .or_else(|| extract_duration_days(text))
}

fn duration_candidate_tokens(text: &str) -> Vec<String> {
    text.to_ascii_lowercase()
        .split_whitespace()
        .map(|token| token.trim_matches(|c: char| !c.is_ascii_alphanumeric() && c != '+'))
        .filter(|token| !token.is_empty())
        .map(|token| token.to_string())
        .collect()
}

fn extract_duration_month_spans(tokens: &[String]) -> Vec<(usize, usize, i32)> {
    let mut spans = Vec::new();
    let mut index = 0usize;
    while index + 1 < tokens.len() {
        let Some(amount) = parse_count_token(tokens[index].as_str()) else {
            index += 1;
            continue;
        };
        let unit = tokens[index + 1].as_str();
        if matches!(unit, "month" | "months") {
            spans.push((index, index + 2, amount));
            index += 2;
            continue;
        }
        if !matches!(unit, "year" | "years") {
            index += 1;
            continue;
        }

        let mut months = amount * 12;
        let mut end = index + 2;
        let month_start = if tokens.get(end).map(|token| token.as_str()) == Some("and") {
            end + 1
        } else {
            end
        };
        if let (Some(month_amount), Some(month_unit)) =
            (tokens.get(month_start), tokens.get(month_start + 1))
        {
            if matches!(month_unit.as_str(), "month" | "months") {
                if let Some(extra_months) = parse_count_token(month_amount) {
                    months += extra_months;
                    end = month_start + 2;
                }
            }
        }
        spans.push((index, end, months));
        index = end;
    }
    spans
}

fn duration_unit_days(unit: &str) -> Option<i32> {
    match unit {
        "day" | "days" => Some(1),
        "week" | "weeks" => Some(7),
        "month" | "months" => Some(30),
        "year" | "years" => Some(365),
        _ => None,
    }
}

fn extract_duration_day_spans(tokens: &[String]) -> Vec<(usize, usize, i32)> {
    let mut spans = Vec::new();
    let mut index = 0usize;
    while index + 1 < tokens.len() {
        let Some(amount) = parse_count_token(tokens[index].as_str()) else {
            index += 1;
            continue;
        };
        let Some(scale) = duration_unit_days(tokens[index + 1].as_str()) else {
            index += 1;
            continue;
        };

        let mut total_days = amount * scale;
        let mut end = index + 2;
        let next_start = if tokens.get(end).map(|token| token.as_str()) == Some("and") {
            end + 1
        } else {
            end
        };
        if let (Some(extra_amount), Some(extra_unit)) =
            (tokens.get(next_start), tokens.get(next_start + 1))
        {
            if let (Some(extra_count), Some(extra_scale)) = (
                parse_count_token(extra_amount.as_str()),
                duration_unit_days(extra_unit.as_str()),
            ) {
                total_days += extra_count * extra_scale;
                end = next_start + 2;
            }
        }

        spans.push((index, end, total_days));
        index = end;
    }
    spans
}

fn phrase_token_positions(tokens: &[String], phrase: &str) -> Vec<usize> {
    let phrase_tokens = duration_candidate_tokens(phrase);
    if phrase_tokens.is_empty() || phrase_tokens.len() > tokens.len() {
        return Vec::new();
    }

    (0..=tokens.len() - phrase_tokens.len())
        .filter(|start| tokens[*start..*start + phrase_tokens.len()] == phrase_tokens[..])
        .collect()
}

fn merge_duration_max(slot: &mut Option<i32>, candidate: Option<i32>) {
    if let Some(value) = candidate {
        *slot = Some(slot.map_or(value, |existing| existing.max(value)));
    }
}

fn extract_count_before_unit(text: &str, unit: &str) -> Option<i32> {
    let lower = text.to_ascii_lowercase();
    let unit_plural = format!("{unit}s");
    let tokens = lower
        .split_whitespace()
        .map(|token| token.trim_matches(|c: char| !c.is_ascii_alphanumeric() && c != '+'))
        .collect::<Vec<_>>();
    for idx in 1..tokens.len() {
        if tokens[idx] == unit || tokens[idx] == unit_plural {
            if let Some(value) = parse_count_token(tokens[idx - 1]) {
                return Some(value);
            }
        }
    }
    None
}

fn format_duration_months(total_months: i32) -> String {
    let years = total_months / 12;
    let months = total_months % 12;
    match (years, months) {
        (0, months) => format!("{months} months"),
        (years, 0) => format!("{years} years"),
        (years, months) => format!("{years} years and {months} months"),
    }
}

fn render_temporal_gap_answer(days: i32, style: &TemporalGapAnswerStyle) -> Option<String> {
    match style {
        TemporalGapAnswerStyle::FixedUnit { unit } => {
            let amount = convert_days_to_gap_unit(days, unit)?;
            let suffix = if amount == 1 {
                unit.as_str()
            } else {
                match unit.as_str() {
                    "day" => "days",
                    "week" => "weeks",
                    "month" => "months",
                    "year" => "years",
                    _ => return None,
                }
            };
            Some(format!("{amount} {suffix}"))
        },
        TemporalGapAnswerStyle::NaturalLanguage => Some(render_natural_duration(days)),
    }
}

fn convert_days_to_gap_unit(days: i32, unit: &str) -> Option<i32> {
    let amount = match unit {
        "day" => days,
        "week" => (days + 3) / 7,
        "month" => (days + 15) / 30,
        "year" => (days + 182) / 365,
        _ => return None,
    };
    Some(amount.max(0))
}

fn render_natural_duration(days: i32) -> String {
    if days >= 365 {
        let years = days / 365;
        let months = ((days % 365) + 15) / 30;
        if months == 0 {
            return render_small_duration_quantity(years, "year");
        }
        return format!(
            "{} and {}",
            render_small_duration_quantity(years, "year"),
            render_small_duration_quantity(months, "month")
        );
    }

    if days >= 45 {
        return render_small_duration_quantity(((days + 15) / 30).max(1), "month");
    }

    if days >= 7 {
        return render_small_duration_quantity(((days + 3) / 7).max(1), "week");
    }

    render_small_duration_quantity(days.max(1), "day")
}

fn render_small_duration_quantity(amount: i32, unit: &str) -> String {
    let quantity = small_number_word(amount).unwrap_or_else(|| amount.to_string());
    let suffix = if amount == 1 {
        unit.to_string()
    } else {
        format!("{unit}s")
    };
    format!("{quantity} {suffix}")
}

fn elapsed_days_since_anchor(anchor_rank: Option<i32>, rank: i32) -> Option<i32> {
    if rank < 0 {
        return Some(-rank);
    }
    Some((anchor_rank? - rank).abs())
}

fn convert_days_to_elapsed_unit(days: i32, unit: &str) -> Option<i32> {
    let amount = match unit {
        "day" => days,
        "week" => (days + 3) / 7,
        "month" => (days + 15) / 30,
        "year" => (days + 182) / 365,
        _ => return None,
    };
    Some(amount.max(1))
}

fn render_relative_elapsed(unit: &str, amount: i32) -> String {
    let quantity = small_number_word(amount).unwrap_or_else(|| amount.to_string());
    let suffix = if amount == 1 {
        unit.to_string()
    } else {
        format!("{unit}s")
    };
    format!("{quantity} {suffix} ago")
}

fn small_number_word(value: i32) -> Option<String> {
    let word = match value {
        0 => "zero",
        1 => "one",
        2 => "two",
        3 => "three",
        4 => "four",
        5 => "five",
        6 => "six",
        7 => "seven",
        8 => "eight",
        9 => "nine",
        10 => "ten",
        11 => "eleven",
        12 => "twelve",
        _ => return None,
    };
    Some(word.to_string())
}

fn parse_requested_sequence_count(task: &str) -> usize {
    let lower = task.to_ascii_lowercase();
    if lower.contains("first, second and third") || lower.contains("first second and third") {
        return 3;
    }

    duration_candidate_tokens(task)
        .into_iter()
        .filter_map(|token| parse_count_token(&token))
        .find(|value| (2..=8).contains(value))
        .map(|value| value as usize)
        .unwrap_or(3)
}

fn parse_temporal_sequence_options(task: &str) -> Option<Vec<ChoiceOption>> {
    let trimmed = task.trim().trim_end_matches('?');
    let quoted = extract_all_quoted_spans(trimmed)
        .into_iter()
        .filter_map(|span| build_temporal_event_option(&span))
        .collect::<Vec<_>>();
    if quoted.len() >= 2 {
        return Some(quoted);
    }

    let (_, tail) = split_once_case_insensitive(trimmed, ": ")?;
    let mut parts = if let Some((head, last)) = split_once_case_insensitive(tail, ", and ") {
        let mut pieces = head
            .split(", ")
            .map(str::trim)
            .filter(|part| !part.is_empty())
            .map(ToString::to_string)
            .collect::<Vec<_>>();
        pieces.push(last.trim().to_string());
        pieces
    } else if let Some((head, last)) = split_once_case_insensitive(tail, " and ") {
        let mut pieces = head
            .split(", ")
            .map(str::trim)
            .filter(|part| !part.is_empty())
            .map(ToString::to_string)
            .collect::<Vec<_>>();
        pieces.push(last.trim().to_string());
        pieces
    } else {
        Vec::new()
    };
    parts.retain(|part| part.split_whitespace().count() >= 3);
    let options = parts
        .into_iter()
        .filter_map(|part| build_temporal_event_option(&part))
        .collect::<Vec<_>>();
    (options.len() >= 2).then_some(options)
}

fn extract_all_quoted_spans(text: &str) -> Vec<String> {
    let mut spans = extract_quoted_spans(text);
    spans.sort();
    spans.dedup();
    spans
}

fn looks_like_completed_temporal_event(text: &str) -> bool {
    let lower = text.to_ascii_lowercase();
    if lower.starts_with("i'm planning")
        || lower.starts_with("i am planning")
        || lower.starts_with("i'm thinking")
        || lower.starts_with("i am thinking")
        || lower.starts_with("i'm looking")
        || lower.starts_with("i am looking")
        || lower.starts_with("i'm wondering")
        || lower.starts_with("i am wondering")
        || lower.contains(" later this year")
        || lower.contains(" upcoming ")
        || lower.contains("similar to the one i attended")
    {
        return false;
    }

    [
        "just got back",
        "recently got back",
        "got back from",
        "attended",
        "participated",
        "volunteered",
        "joined",
        "helped",
        "ran",
        "went on",
        "went to",
        "started",
        "finished",
        "completed",
        "graduated",
        "visited",
    ]
    .iter()
    .any(|marker| lower.contains(marker))
}

fn compact_temporal_event_summary(text: &str) -> String {
    let lower = text.to_ascii_lowercase();
    if lower.contains("similar to the one i attended") {
        return String::new();
    }
    if let Some(quoted) = extract_first_quoted_span(text) {
        return quoted;
    }

    let mut candidate = strip_temporal_discourse_prefix(text);
    for marker in [
        "just got back from ",
        "recently got back from ",
        "got back from ",
        "just participated in ",
        "participated in ",
        "volunteered at ",
        "volunteering at ",
        "attended ",
        "went on ",
        "went to ",
        "started my ",
        "helped with ",
    ] {
        if let Some(idx) = candidate.to_ascii_lowercase().find(marker) {
            candidate = candidate[idx + marker.len()..].trim().to_string();
            break;
        }
    }
    let compact = trim_answer_tail(&candidate, false);
    if compact.is_empty() {
        sanitize_answer_text(text)
    } else {
        compact
    }
}

fn extract_first_quoted_span(text: &str) -> Option<String> {
    extract_quoted_spans(text)
        .into_iter()
        .find(|candidate| candidate.split_whitespace().count() >= 2)
}

fn extract_quoted_spans(text: &str) -> Vec<String> {
    let mut spans = Vec::new();
    spans.extend(extract_quoted_spans_for(text, '"'));
    spans.extend(extract_quoted_spans_for(text, '\''));
    spans
}

fn extract_quoted_spans_for(text: &str, quote: char) -> Vec<String> {
    let mut indices = Vec::new();
    for (idx, ch) in text.char_indices() {
        if ch != quote {
            continue;
        }
        if quote == '\'' {
            let prev = text[..idx].chars().next_back();
            let next = text[idx + ch.len_utf8()..].chars().next();
            if prev.map(|c| c.is_ascii_alphanumeric()).unwrap_or(false)
                || next.map(|c| c.is_ascii_alphanumeric()).unwrap_or(false)
            {
                continue;
            }
        }
        indices.push(idx);
    }

    let mut spans = Vec::new();
    let mut iter = indices.into_iter();
    while let (Some(start), Some(end)) = (iter.next(), iter.next()) {
        if end <= start + quote.len_utf8() {
            continue;
        }
        let candidate = text[start + quote.len_utf8()..end].trim();
        if candidate.split_whitespace().count() >= 2 {
            spans.push(candidate.to_string());
        }
    }
    spans
}

fn render_temporal_sequence_answer(items: &[String]) -> Option<String> {
    if items.len() < 2 {
        return None;
    }
    let mut out = String::new();
    for (index, item) in items.iter().enumerate() {
        if index > 0 {
            out.push_str(if index + 1 == items.len() {
                ", and finally "
            } else {
                ", then "
            });
        } else {
            out.push_str("First, ");
        }
        out.push_str(item);
    }
    Some(out)
}

fn temporal_candidate_sequence_rank(
    path: &Path,
    item_index: usize,
    local_index: usize,
) -> Option<i32> {
    let file_name = path.file_name()?.to_string_lossy().to_ascii_lowercase();
    if file_name.contains("_summary") {
        return None;
    }
    let base = file_name
        .find("_chunk")
        .and_then(|idx| {
            file_name[..idx]
                .rsplit('_')
                .next()
                .filter(|digits| !digits.is_empty() && digits.chars().all(|c| c.is_ascii_digit()))
                .and_then(|digits| digits.parse::<i32>().ok())
        })
        .unwrap_or(item_index as i32);
    Some(base.saturating_mul(1000) + local_index as i32)
}

fn render_temporal_candidate_answer(
    task: &str,
    candidate: &TemporalCandidate,
    task_terms: &[String],
) -> String {
    compact_answer(task, &candidate.text, task_terms)
        .unwrap_or_else(|| summarize_turn_text(&candidate.text, task_terms))
}

fn temporal_event_match_score(line: &str, option: &ChoiceOption, retrieval_score: f32) -> f32 {
    let lower = line.to_ascii_lowercase();
    let required_tokens = temporal_required_option_tokens(option);
    if !required_tokens.is_empty()
        && !required_tokens
            .iter()
            .all(|token| line_matches_event_token(&lower, token))
    {
        return 0.0;
    }
    if !required_tokens.is_empty() && option.tokens.len() > required_tokens.len() {
        let has_non_tail_match = option
            .tokens
            .iter()
            .filter(|token| !required_tokens.iter().any(|required| required == *token))
            .any(|token| line_matches_event_token(&lower, token));
        if !has_non_tail_match {
            return 0.0;
        }
    }
    let overlap = option
        .tokens
        .iter()
        .filter(|token| line_matches_event_token(&lower, token))
        .count() as f32;
    if overlap == 0.0 {
        return 0.0;
    }
    let coverage = overlap / option.tokens.len().max(1) as f32;
    candidate_weight(line, &option.tokens, retrieval_score, false) + overlap * 6.0 + coverage * 10.0
}

fn line_matches_event_token(lower_line: &str, token: &str) -> bool {
    if lower_line.contains(token) {
        return true;
    }

    match token {
        "find" => lower_line.contains("found"),
        "found" => lower_line.contains("find"),
        "buy" => lower_line.contains("bought"),
        "bought" => lower_line.contains("buy"),
        "get" => lower_line.contains("got"),
        "got" => lower_line.contains("get"),
        "go" => lower_line.contains("went"),
        "went" => lower_line.contains("go"),
        "take" => lower_line.contains("taking") || lower_line.contains("took"),
        "taking" => lower_line.contains("take") || lower_line.contains("took"),
        _ => {
            let stem = token
                .trim_end_matches("ing")
                .trim_end_matches("ed")
                .trim_end_matches('s');
            stem.len() >= 3 && lower_line.contains(stem)
        },
    }
}

fn temporal_required_option_tokens(option: &ChoiceOption) -> Vec<String> {
    required_tail_anchor_tokens(&option.display)
}

pub(super) fn required_tail_anchor_tokens(text: &str) -> Vec<String> {
    let display_lower = text.to_ascii_lowercase();
    let mut best_tail = None;
    let mut best_idx = 0usize;
    for marker in [" from ", " in ", " at "] {
        if let Some(idx) = display_lower.rfind(marker) {
            if best_tail.is_none() || idx > best_idx {
                best_idx = idx;
                best_tail = Some(&text[idx + marker.len()..]);
            }
        }
    }

    let Some(tail) = best_tail else {
        return Vec::new();
    };

    let mut tokens = Vec::new();
    for raw in tail.split(|c: char| !c.is_alphanumeric() && c != '\'') {
        let lower = raw
            .trim_matches(|c: char| !c.is_ascii_alphanumeric() && c != '\'')
            .trim_matches('\'')
            .to_ascii_lowercase();
        if lower.is_empty()
            || lower.len() < 3
            || parse_count_token(&lower).is_some()
            || QUESTION_STOPWORDS.contains(&lower.as_str())
            || GENERIC_ANCHOR_TERMS.contains(&lower.as_str())
            || matches!(
                lower.as_str(),
                "again" | "after" | "before" | "because" | "later" | "so" | "then" | "there"
            )
        {
            continue;
        }
        if !tokens.iter().any(|existing| existing == &lower) {
            tokens.push(lower);
        }
    }
    tokens
}
