use super::*;
use crate::index::{compile_regex, compile_regex_static};

pub(in crate::index) fn temporal_from_now_overlap_count(
    lower_line: &str,
    terms: &[String],
) -> usize {
    terms
        .iter()
        .filter(|term| temporal_from_now_line_matches_term(lower_line, term))
        .count()
}

pub(in crate::index) fn temporal_from_now_line_matches_term(lower_line: &str, term: &str) -> bool {
    if lower_line.contains(term) {
        return true;
    }
    match term {
        "find" | "found" => {
            lower_line.contains("find")
                || lower_line.contains("found")
                || lower_line.contains("saw")
        },
        "launch" | "launched" => lower_line.contains("launch"),
        "sign" | "signed" => lower_line.contains("sign"),
        "go" | "went" => lower_line.contains("go") || lower_line.contains("went"),
        "take" | "taking" | "took" => {
            lower_line.contains("take")
                || lower_line.contains("taking")
                || lower_line.contains("took")
        },
        _ => {
            let stem = term
                .trim_end_matches("ing")
                .trim_end_matches("ed")
                .trim_end_matches('s');
            stem.len() >= 3 && lower_line.contains(stem)
        },
    }
}

pub(in crate::index) fn temporal_from_now_focus_terms(terms: &[String]) -> Vec<String> {
    const LEADING_FOCUS_STOP: &[&str] = &[
        "attend", "visit", "go", "join", "make", "buy", "take", "run", "last", "i", "me", "my",
    ];

    let mut start = 0usize;
    while start + 1 < terms.len() {
        let key = synthetic_answer_surface_term_key(&terms[start]);
        if LEADING_FOCUS_STOP.contains(&key.as_str()) {
            start += 1;
            continue;
        }
        break;
    }

    let focus = terms[start..]
        .iter()
        .filter(|term| {
            let key = synthetic_answer_surface_term_key(term);
            !matches!(key.as_str(), "i" | "me" | "my" | "last")
        })
        .cloned()
        .collect::<Vec<_>>();
    if focus.is_empty() {
        terms.to_vec()
    } else {
        focus
    }
}

pub(in crate::index) fn extract_temporal_rank_value(line: &str) -> Option<i32> {
    if let Some(day) = extract_explicit_date_rank(line) {
        return Some(day);
    }
    let days_ago = extract_temporal_relative_days(line)?;
    let adjusted = match extract_relative_reference_offset_days(line) {
        Some((SyntheticTemporalDirection::Earlier, offset)) => days_ago + offset,
        Some((SyntheticTemporalDirection::Later, offset)) => days_ago.saturating_sub(offset),
        None => days_ago,
    };
    Some(-adjusted)
}

pub(in crate::index) fn extract_current_duration_days(line: &str) -> Option<i32> {
    duration_answer_magnitude(&extract_duration_answer_from_line(line)?)
        .map(|days| days.round() as i32)
}

pub(in crate::index) fn temporal_base_day_at_line(
    lines: &[String],
    line_idx: usize,
) -> Option<i32> {
    lines
        .iter()
        .take(line_idx + 1)
        .rev()
        .find_map(|line| extract_explicit_date_rank(line))
}

pub(in crate::index) fn best_temporal_current_anchor_line(
    lines: &[String],
) -> Option<(usize, usize, String)> {
    let mut best: Option<(usize, usize, usize, String)> = None;
    let mut user_turn = 0usize;
    for (line_idx, line) in lines.iter().enumerate() {
        let lower = line.to_ascii_lowercase();
        if !lower.starts_with("user:") {
            continue;
        }
        user_turn += 1;
        if !has_temporal_current_marker(&lower) {
            continue;
        }
        let score = 10 + user_turn;
        let should_replace = best
            .as_ref()
            .map(|(best_score, best_turn, best_line_idx, best_line)| {
                score > *best_score
                    || (score == *best_score
                        && (user_turn > *best_turn
                            || (user_turn == *best_turn
                                && (line_idx > *best_line_idx
                                    || (line_idx == *best_line_idx && line < best_line)))))
            })
            .unwrap_or(true);
        if should_replace {
            best = Some((score, user_turn, line_idx, line.clone()));
        }
    }
    best.map(|(score, _, line_idx, line)| (score, line_idx, line))
}

pub(in crate::index) fn has_temporal_current_marker(lower: &str) -> bool {
    lower.contains("today")
        || lower.contains("right now")
        || lower.contains("currently")
        || lower.contains("this week")
        || lower.contains("this month")
        || lower.contains("this year")
        || lower
            .split(|c: char| !c.is_ascii_alphanumeric())
            .any(|token| token == "now")
}

pub(in crate::index) fn extract_temporal_relative_days(text: &str) -> Option<i32> {
    let lower = text.to_ascii_lowercase();
    if lower.contains("today") {
        return Some(0);
    }
    if lower.contains("yesterday") {
        return Some(1);
    }
    if lower.contains("a couple of days ago") {
        return Some(2);
    }
    if lower.contains("a few days ago") || lower.contains("few days ago") {
        return Some(3);
    }
    if lower.contains("last weekend") || lower.contains("last week") {
        return Some(7);
    }
    if lower.contains("last month") {
        return Some(30);
    }
    for (unit, scale) in [("day", 1), ("week", 7), ("month", 30), ("year", 365)] {
        for marker in [format!("{unit} ago"), format!("{unit}s ago")] {
            if !lower.contains(&marker) {
                continue;
            }
            let prefix = lower.split(&marker).next()?;
            let amount = extract_temporal_trailing_count(prefix)?;
            return Some(amount * scale);
        }
    }
    None
}

pub(in crate::index) fn extract_relative_reference_offset_days(
    text: &str,
) -> Option<(SyntheticTemporalDirection, i32)> {
    let lower = text.to_ascii_lowercase();
    for (unit, scale) in [("day", 1), ("week", 7), ("month", 30), ("year", 365)] {
        for (marker, direction) in [
            (
                format!("{unit} in advance"),
                SyntheticTemporalDirection::Earlier,
            ),
            (
                format!("{unit}s in advance"),
                SyntheticTemporalDirection::Earlier,
            ),
            (
                format!("{unit} before"),
                SyntheticTemporalDirection::Earlier,
            ),
            (
                format!("{unit}s before"),
                SyntheticTemporalDirection::Earlier,
            ),
            (format!("{unit} after"), SyntheticTemporalDirection::Later),
            (format!("{unit}s after"), SyntheticTemporalDirection::Later),
            (format!("{unit} later"), SyntheticTemporalDirection::Later),
            (format!("{unit}s later"), SyntheticTemporalDirection::Later),
        ] {
            if !lower.contains(&marker) {
                continue;
            }
            let prefix = lower.split(&marker).next()?;
            let amount = extract_temporal_trailing_count(prefix)?;
            return Some((direction, amount * scale));
        }
    }
    None
}

pub(in crate::index) fn extract_temporal_trailing_count(prefix: &str) -> Option<i32> {
    let token = prefix
        .split_whitespace()
        .rev()
        .find(|token| !token.is_empty())?;
    parse_temporal_count_token(token)
}

pub(in crate::index) fn parse_temporal_count_token(token: &str) -> Option<i32> {
    let clean = token
        .trim_matches(|c: char| !c.is_ascii_alphanumeric() && c != '+')
        .trim_end_matches('+');
    if let Ok(value) = clean.parse::<i32>() {
        return Some(value);
    }
    match clean {
        "a" | "an" | "one" => Some(1),
        "two" => Some(2),
        "three" => Some(3),
        "four" => Some(4),
        "five" => Some(5),
        "six" => Some(6),
        "seven" => Some(7),
        "eight" => Some(8),
        "nine" => Some(9),
        "ten" => Some(10),
        "eleven" => Some(11),
        "twelve" => Some(12),
        "couple" => Some(2),
        "few" => Some(3),
        _ => None,
    }
}

pub(in crate::index) fn extract_duration_months_from_text(text: &str) -> Option<i32> {
    let lower = text.to_ascii_lowercase();
    let years = compile_regex_static(
        r"(?i)\b(a|an|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|\d+)\s+years?\b",
    )
    .captures(&lower)
    .and_then(|caps| caps.get(1))
    .and_then(|value| parse_temporal_count_token(value.as_str()));
    let months = compile_regex_static(
        r"(?i)\b(a|an|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|\d+)\s+months?\b",
    )
    .captures(&lower)
    .and_then(|caps| caps.get(1))
    .and_then(|value| parse_temporal_count_token(value.as_str()));
    match (years, months) {
        (None, None) => None,
        (Some(years), None) => Some(years * 12),
        (None, Some(months)) => Some(months),
        (Some(years), Some(months)) => Some(years * 12 + months),
    }
}
