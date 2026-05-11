//! Temporal event extraction and matching: parse events, render sequences, and score event matches.
//!
//! This module handles event extraction from text, sequence parsing,
//! and scoring event matches against choice options.

use super::*;
use std::path::Path;

pub(crate) fn parse_requested_sequence_count(task: &str) -> usize {
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

pub(crate) fn parse_temporal_sequence_options(task: &str) -> Option<Vec<ChoiceOption>> {
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
        let mut pieces: Vec<String> = head
            .split(", ")
            .map(|s| s.trim())
            .filter(|part| !part.is_empty())
            .map(|part| part.to_string())
            .collect();
        pieces.push(last.trim().to_string());
        pieces
    } else if let Some((head, last)) = split_once_case_insensitive(tail, " and ") {
        let mut pieces: Vec<String> = head
            .split(", ")
            .map(|s| s.trim())
            .filter(|part| !part.is_empty())
            .map(|part| part.to_string())
            .collect();
        pieces.push(last.trim().to_string());
        pieces
    } else {
        Vec::new()
    };
    parts.retain(|part: &String| part.split_whitespace().count() >= 3);
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

pub(crate) fn looks_like_completed_temporal_event(text: &str) -> bool {
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

pub(crate) fn compact_temporal_event_summary(text: &str) -> String {
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

pub(crate) fn render_temporal_sequence_answer(items: &[String]) -> Option<String> {
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

pub(crate) fn temporal_candidate_sequence_rank(
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

pub(crate) fn render_temporal_candidate_answer(
    task: &str,
    candidate: &TemporalCandidate,
    task_terms: &[String],
) -> String {
    compact_answer(task, &candidate.text, task_terms)
        .unwrap_or_else(|| summarize_turn_text(&candidate.text, task_terms))
}

pub(crate) fn temporal_event_match_score(
    line: &str,
    option: &ChoiceOption,
    retrieval_score: f32,
) -> f32 {
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

pub(crate) fn line_matches_event_token(lower_line: &str, token: &str) -> bool {
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

pub(crate) fn required_tail_anchor_tokens(text: &str) -> Vec<String> {
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

fn duration_candidate_tokens(text: &str) -> Vec<String> {
    text.to_ascii_lowercase()
        .split_whitespace()
        .map(|token| token.trim_matches(|c: char| !c.is_ascii_alphanumeric() && c != '+'))
        .filter(|token| !token.is_empty())
        .map(|token| token.to_string())
        .collect()
}
