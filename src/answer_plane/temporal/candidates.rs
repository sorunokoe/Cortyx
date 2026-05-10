//! Temporal candidate collection and ranking.

use super::*;
use std::collections::HashSet;

pub fn is_temporal_sequence_query(task: &str) -> bool {
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

pub fn temporal_focus_terms(text: &str) -> Vec<String> {
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

pub fn temporal_sequence_focus_terms(task: &str) -> Vec<String> {
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

pub fn is_temporal_reasoning_query(task: &str) -> bool {
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

pub fn collect_temporal_candidates(
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

pub fn candidate_temporal_rank(candidate: &TemporalCandidate) -> Option<i32> {
    extract_temporal_rank(&candidate.text, candidate.base_date)
}

pub fn candidate_temporal_base_rank(candidate: &TemporalCandidate) -> Option<i32> {
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

pub fn candidate_temporal_order_rank(
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

pub fn candidate_temporal_event_rank(
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

pub fn resolve_from_temporal_option_focus<T>(
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

pub fn candidate_temporal_event_rank_for_option(
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

pub fn best_temporal_candidate_rank(
    option: &ChoiceOption,
    candidates: &[TemporalCandidate],
) -> Option<i32> {
    best_temporal_candidate_rank_for_comparison(option, candidates, false)
}

pub fn best_temporal_candidate_rank_for_comparison(
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

pub fn best_calendar_grounded_temporal_candidate_rank(
    option: &ChoiceOption,
    candidates: &[TemporalCandidate],
) -> Option<CalendarGroundedRank> {
    best_calendar_grounded_temporal_candidate_rank_with_competing_option(option, None, candidates)
}

pub fn best_calendar_grounded_temporal_candidate_rank_with_competing_option(
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

pub fn best_temporal_candidate_strict_rank(
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

pub fn best_calendar_grounded_current_anchor_rank(candidates: &[TemporalCandidate]) -> Option<i32> {
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

pub fn best_temporal_candidate_sequence_rank(
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

pub fn best_temporal_candidate_loose_sequence_rank(
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

pub fn temporal_candidate_score(candidate: &TemporalCandidate, option: &ChoiceOption) -> f32 {
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

pub fn temporal_target_score(candidate: &TemporalCandidate, target_terms: &[String]) -> f32 {
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

pub fn extract_relative_reference_offset_days(text: &str) -> Option<(TemporalDirection, i32)> {
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
