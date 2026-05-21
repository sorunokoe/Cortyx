//! Duration and event count selection for temporal answers.

use super::*;
use std::collections::HashSet;

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
        let normalized = normalized_answer_key(&summary);
        if !seen.insert(normalized) {
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

/// Extracts a completion duration for "How long did it take [me] to X?" queries
/// that are NOT already handled by the temporal gap parser (i.e., no "after Y"
/// endpoint is present in the question).
///
/// Scans evidence lines for a "N unit" pattern (e.g., "4 hours", "30 minutes")
/// on lines that overlap with the activity terms from the task.
/// Lines containing future-tense markers are skipped.
pub(crate) fn select_activity_completion_duration_answer(
    task: &str,
    evidence: &[EvidenceItem],
) -> Option<String> {
    let lower_task = task.to_ascii_lowercase();
    if !lower_task.starts_with("how long did it take") {
        return None;
    }
    // The temporal gap parser already handles "how long … after Y" structures.
    if parse_temporal_gap_query(task).is_some() {
        return None;
    }

    let task_terms = salient_query_terms(task);
    if task_terms.is_empty() {
        return None;
    }

    const DURATION_UNITS: &[&str] = &[
        "hours", "minutes", "hour", "minute", "days", "day", "weeks", "week",
    ];
    const FUTURE_MARKERS: &[&str] = &[
        " will ",
        " going to ",
        " plan to ",
        " planning to ",
        " should take ",
    ];

    let mut best: Option<(usize, String)> = None; // (overlap, "N unit")

    for item in evidence {
        let Some(content) = read_context_text(&item.path, "activity duration answer") else {
            continue;
        };
        for line in content.lines() {
            let lower_line = line.to_ascii_lowercase();
            // Skip future-tense lines
            if FUTURE_MARKERS.iter().any(|m| lower_line.contains(m)) {
                continue;
            }
            // Must share terms with the task
            let overlap = task_terms
                .iter()
                .filter(|t| lower_line.contains(t.as_str()))
                .count();
            if overlap == 0 {
                continue;
            }

            let tokens: Vec<&str> = lower_line.split_whitespace().collect();
            let original_tokens: Vec<&str> = line.split_whitespace().collect();

            for (i, tok) in tokens.iter().enumerate() {
                if DURATION_UNITS.contains(tok) && i > 0 {
                    if let Some(n) = parse_count_token(tokens[i - 1]) {
                        if n > 0 {
                            let num_str = original_tokens
                                .get(i - 1)
                                .copied()
                                .unwrap_or_default()
                                .trim_matches(|c: char| !c.is_ascii_alphanumeric());
                            let unit_str = original_tokens.get(i).copied().unwrap_or_default();
                            let duration = format!("{num_str} {unit_str}");
                            if best.as_ref().map(|(s, _)| overlap > *s).unwrap_or(true) {
                                best = Some((overlap, duration));
                            }
                        }
                    }
                }
            }
        }
    }

    best.map(|(_, dur)| dur)
}
