//! Temporal candidate collection and core temporal helpers.

use super::*;
use std::collections::HashSet;

pub(crate) fn collect_temporal_candidates(
    evidence: &[EvidenceItem],
    stage: &str,
) -> Vec<TemporalCandidate> {
    let mut candidates = Vec::new();
    #[allow(clippy::type_complexity)]
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

#[allow(clippy::type_complexity)]
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

pub(crate) fn candidate_temporal_rank(candidate: &TemporalCandidate) -> Option<i32> {
    extract_temporal_rank(&candidate.text, candidate.base_date)
}

pub(crate) fn candidate_temporal_base_rank(candidate: &TemporalCandidate) -> Option<i32> {
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

pub(crate) fn candidate_temporal_order_rank(
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

pub(crate) fn temporal_current_rank_from_text(
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

pub(crate) fn candidate_temporal_event_rank(
    candidate: &TemporalCandidate,
    candidates: &[TemporalCandidate],
) -> Option<i32> {
    candidate_temporal_order_rank(candidate, candidates)
        .or_else(|| candidate_temporal_current_rank(candidate))
}

pub(crate) fn temporal_option_focus_tail<'a>(
    text: &'a str,
    option: &ChoiceOption,
) -> Option<&'a str> {
    let lower = text.to_ascii_lowercase();
    let start = option
        .tokens
        .iter()
        .flat_map(|token| lower.match_indices(token).map(|(idx, _)| idx))
        .max()?;
    Some(&text[start..])
}

pub(crate) fn resolve_from_temporal_option_focus<T>(
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

pub(crate) fn candidate_temporal_order_rank_for_option(
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

pub(crate) fn candidate_temporal_event_rank_for_option(
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
