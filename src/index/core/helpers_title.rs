// This file is a submodule of `crate::index::core`.
// Contains free-standing helper functions extracted from helpers.rs.
use super::*;
use crate::index::compile_regex;
use crate::types::{QueryText, SynapseWeight};

pub(in crate::index) fn extract_task_reference_label(task: &str) -> Option<String> {
    let trimmed = task.trim();
    let lower = trimmed.to_ascii_lowercase();
    if !lower.starts_with("as of ") {
        return None;
    }
    let question_pos = lower.find("how many ")?;
    let candidate = trimmed[6..question_pos].trim().trim_end_matches(',').trim();
    if extract_explicit_date_rank(candidate).is_some() {
        return Some(candidate.to_string());
    }
    None
}

pub(in crate::index) fn verbatim_source_group_key(entry: &BM25Entry) -> String {
    if let Ok(content) = std::fs::read_to_string(&entry.neuron_path) {
        if let Some(line) = content.lines().next() {
            if let Some(source_idx) = line.find("source:") {
                let source = &line[source_idx + "source:".len()..];
                let source = source.trim();
                let source = source.strip_suffix("-->").unwrap_or(source).trim();
                if !source.is_empty() {
                    return source.to_string();
                }
            }
        }
    }

    let Some(name) = entry.neuron_path.file_name().and_then(|name| name.to_str()) else {
        return entry.neuron_path.display().to_string();
    };
    name.split('.').next().unwrap_or(name).to_string()
}

pub(in crate::index) fn parse_temporal_from_now_unit(
    raw: &str,
) -> Option<SyntheticElapsedFromNowUnit> {
    match raw.trim() {
        "day" | "days" => Some(SyntheticElapsedFromNowUnit::Day),
        "week" | "weeks" => Some(SyntheticElapsedFromNowUnit::Week),
        "month" | "months" => Some(SyntheticElapsedFromNowUnit::Month),
        "year" | "years" => Some(SyntheticElapsedFromNowUnit::Year),
        _ => None,
    }
}

pub(in crate::index) fn extract_temporal_interval_phrases(
    task_lower: &str,
) -> Option<(String, String)> {
    let trimmed = task_lower.trim().trim_end_matches('?');
    let (before_after, start_phrase) = trimmed.split_once(" after ")?;
    let end_phrase = before_after
        .strip_prefix("how many days did it take for me to ")
        .or_else(|| before_after.strip_prefix("how many days did it take me to "))?
        .trim();
    Some((end_phrase.to_string(), start_phrase.trim().to_string()))
}

pub(in crate::index) fn best_temporal_rank_line(
    lines: &[String],
    phrase_lower: &str,
    terms: &[String],
) -> Option<(i32, usize, String)> {
    best_temporal_rank_line_with_min_overlap(lines, phrase_lower, terms, None)
}

pub(in crate::index) fn best_temporal_rank_line_with_min_overlap(
    lines: &[String],
    phrase_lower: &str,
    terms: &[String],
    min_overlap_override: Option<usize>,
) -> Option<(i32, usize, String)> {
    let keys = synthetic_answer_surface_term_key_set(terms);
    let min_overlap = min_overlap_override.unwrap_or_else(|| if keys.len() >= 3 { 2 } else { 1 });
    let mut best: Option<(i32, usize, usize, usize, String)> = None;
    for (line_idx, line) in lines.iter().enumerate() {
        let lower = line.to_ascii_lowercase();
        let line_keys = synthetic_answer_surface_term_key_set(&synthetic_query_terms(&lower));
        let overlap = synthetic_answer_surface_overlap_count(&line_keys, &keys);
        let exact = lower.contains(phrase_lower);
        if overlap < min_overlap && !exact {
            continue;
        }
        let Some(rank) = extract_temporal_rank_value(line) else {
            continue;
        };
        let exact_bonus = usize::from(exact);
        let score = overlap * 10 + exact_bonus * 5;
        let should_replace = best
            .as_ref()
            .map(|(_, best_score, best_exact, best_line_idx, _)| {
                score > *best_score
                    || (score == *best_score
                        && (exact_bonus > *best_exact
                            || (exact_bonus == *best_exact && line_idx > *best_line_idx)))
            })
            .unwrap_or(true);
        if should_replace {
            best = Some((rank, score, exact_bonus, line_idx, line.clone()));
        }
    }
    best.map(|(rank, score, _, _, line)| (rank, score, line))
}

pub(in crate::index) fn best_user_turn_line_with_min_overlap(
    lines: &[String],
    phrase_lower: &str,
    terms: &[String],
    min_overlap_override: Option<usize>,
) -> Option<(i32, usize, String)> {
    let keys = synthetic_answer_surface_term_key_set(terms);
    let min_overlap = min_overlap_override.unwrap_or_else(|| if keys.len() >= 3 { 2 } else { 1 });
    let mut best: Option<(i32, usize, usize, String)> = None;
    let mut user_turn = 0i32;
    for line in lines {
        let lower = line.to_ascii_lowercase();
        if !lower.starts_with("user:") {
            continue;
        }
        user_turn += 1;
        let line_keys = synthetic_answer_surface_term_key_set(&synthetic_query_terms(&lower));
        let overlap = synthetic_answer_surface_overlap_count(&line_keys, &keys);
        let exact = lower.contains(phrase_lower);
        if overlap < min_overlap && !exact {
            continue;
        }
        let exact_bonus = usize::from(exact);
        let score = overlap * 10 + exact_bonus * 5;
        let should_replace = best
            .as_ref()
            .map(|(best_turn, best_score, best_exact, _)| {
                score > *best_score
                    || (score == *best_score
                        && (exact_bonus > *best_exact
                            || (exact_bonus == *best_exact && user_turn > *best_turn)))
            })
            .unwrap_or(true);
        if should_replace {
            best = Some((user_turn, score, exact_bonus, line.clone()));
        }
    }
    best.map(|(turn, score, _, line)| (turn, score, line))
}

pub(in crate::index) fn best_temporal_duration_anchor_line(
    lines: &[String],
    phrase_lower: &str,
    terms: &[String],
) -> Option<(SyntheticDurationAnchor, usize, String)> {
    let keys = synthetic_answer_surface_term_key_set(terms);
    let min_overlap = if keys.len() >= 3 { 2 } else { 1 };
    let mut best: Option<(SyntheticDurationAnchor, usize, usize, usize, String)> = None;
    for (line_idx, line) in lines.iter().enumerate() {
        let lower = line.to_ascii_lowercase();
        let line_keys = synthetic_answer_surface_term_key_set(&synthetic_query_terms(&lower));
        let overlap = synthetic_answer_surface_overlap_count(&line_keys, &keys);
        let exact = lower.contains(phrase_lower);
        if overlap < min_overlap && !exact {
            continue;
        }
        let anchor = if let Some(days) = extract_current_duration_days(line) {
            SyntheticDurationAnchor::CurrentDays(days)
        } else if let Some(day) = extract_explicit_date_rank(line) {
            SyntheticDurationAnchor::AbsoluteDay(day)
        } else {
            continue;
        };
        let exact_bonus = usize::from(exact);
        let score = overlap * 10 + exact_bonus * 5;
        let should_replace = best
            .as_ref()
            .map(|(_, best_score, best_exact, best_line_idx, _)| {
                score > *best_score
                    || (score == *best_score
                        && (exact_bonus > *best_exact
                            || (exact_bonus == *best_exact && line_idx > *best_line_idx)))
            })
            .unwrap_or(true);
        if should_replace {
            best = Some((anchor, score, exact_bonus, line_idx, line.clone()));
        }
    }
    best.map(|(anchor, score, _, _, line)| (anchor, score, line))
}

pub(in crate::index) fn best_temporal_event_anchor_line(
    lines: &[String],
    phrase_lower: &str,
    terms: &[String],
) -> Option<(SyntheticEventAnchor, usize, String)> {
    let keys = synthetic_answer_surface_term_key_set(terms);
    let min_overlap = if keys.len() >= 3 { 2 } else { 1 };
    let required_action_key = terms
        .first()
        .map(|term| synthetic_answer_surface_term_key(term))
        .filter(|term| !term.is_empty());
    let mut best: Option<(SyntheticEventAnchor, usize, usize, usize, String)> = None;
    for (line_idx, line) in lines.iter().enumerate() {
        let lower = line.to_ascii_lowercase();
        let line_keys = synthetic_answer_surface_term_key_set(&synthetic_query_terms(&lower));
        if required_action_key
            .as_ref()
            .is_some_and(|term| !line_keys.contains(term))
        {
            continue;
        }
        let overlap = synthetic_answer_surface_overlap_count(&line_keys, &keys);
        let exact = lower.contains(phrase_lower);
        if overlap < min_overlap && !exact {
            continue;
        }
        let anchor = if let Some(days_ago) = extract_temporal_relative_days(line) {
            let adjusted = match extract_relative_reference_offset_days(line) {
                Some((SyntheticTemporalDirection::Earlier, offset)) => days_ago + offset,
                Some((SyntheticTemporalDirection::Later, offset)) => {
                    days_ago.saturating_sub(offset)
                },
                None => days_ago,
            };
            SyntheticEventAnchor::RelativeDaysAgo(adjusted)
        } else if let Some(day) = extract_explicit_date_rank(line) {
            SyntheticEventAnchor::AbsoluteDay(day)
        } else {
            continue;
        };
        let exact_bonus = usize::from(exact);
        let score = overlap * 10 + exact_bonus * 5;
        let should_replace = best
            .as_ref()
            .map(|(_, best_score, best_exact, best_line_idx, _)| {
                score > *best_score
                    || (score == *best_score
                        && (exact_bonus > *best_exact
                            || (exact_bonus == *best_exact && line_idx > *best_line_idx)))
            })
            .unwrap_or(true);
        if should_replace {
            best = Some((anchor, score, exact_bonus, line_idx, line.clone()));
        }
    }
    best.map(|(anchor, score, _, _, line)| (anchor, score, line))
}

pub(in crate::index) fn best_temporal_from_now_event_line(
    lines: &[String],
    phrase_lower: &str,
    terms: &[String],
) -> Option<(i32, usize, String)> {
    let focus_terms = temporal_from_now_focus_terms(terms);
    let min_overlap = if focus_terms.len() >= 3 { 2 } else { 1 };
    let mut best: Option<(i32, usize, usize, String)> = None;
    for (line_idx, line) in lines.iter().enumerate() {
        let lower = line.to_ascii_lowercase();
        let overlap = temporal_from_now_overlap_count(&lower, &focus_terms);
        let exact = lower.contains(phrase_lower);
        if overlap < min_overlap && !exact {
            continue;
        }
        let day = if let Some(base_day) = temporal_base_day_at_line(lines, line_idx) {
            if let Some(days_ago) = extract_temporal_relative_days(line) {
                let adjusted = match extract_relative_reference_offset_days(line) {
                    Some((SyntheticTemporalDirection::Earlier, offset)) => days_ago + offset,
                    Some((SyntheticTemporalDirection::Later, offset)) => {
                        days_ago.saturating_sub(offset)
                    },
                    None => days_ago,
                };
                base_day - adjusted
            } else if let Some(day) = extract_explicit_date_rank(line) {
                day
            } else {
                base_day
            }
        } else if let Some(days_ago) = extract_temporal_relative_days(line) {
            let adjusted = match extract_relative_reference_offset_days(line) {
                Some((SyntheticTemporalDirection::Earlier, offset)) => days_ago + offset,
                Some((SyntheticTemporalDirection::Later, offset)) => {
                    days_ago.saturating_sub(offset)
                },
                None => days_ago,
            };
            -adjusted
        } else if let Some(day) = extract_explicit_date_rank(line) {
            day
        } else {
            continue;
        };
        let score = overlap * 10 + usize::from(exact) * 5;
        let should_replace = best
            .as_ref()
            .map(|(best_day, best_score, best_line_idx, _)| {
                score > *best_score
                    || (score == *best_score
                        && (day > *best_day || (day == *best_day && line_idx > *best_line_idx)))
            })
            .unwrap_or(true);
        if should_replace {
            best = Some((day, score, line_idx, line.clone()));
        }
    }
    let (day, score, _, line) = best?;
    Some((day, score, line))
}

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
    let years = compile_regex(
        r"(?i)\b(a|an|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|\d+)\s+years?\b",
    )
    .captures(&lower)
    .and_then(|caps| caps.get(1))
    .and_then(|value| parse_temporal_count_token(value.as_str()));
    let months = compile_regex(
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

pub(in crate::index) fn extract_current_role_total_months_from_line(
    line: &str,
    lower: &str,
) -> Option<i32> {
    let has_total_marker = task_contains_any(
        lower,
        &[
            "experience in the company",
            "experience at the company",
            "with the company",
            "at the company",
            "been at ",
            "been with ",
            "working at ",
        ],
    );
    if !has_total_marker {
        return None;
    }
    extract_duration_months_from_text(line)
}

pub(in crate::index) fn extract_current_role_offset_months_from_line(
    line: &str,
    lower: &str,
) -> Option<i32> {
    if !task_contains_any(
        lower,
        &[
            "worked my way up to ",
            "promoted to ",
            "promotion to ",
            "moved into ",
            "became ",
        ],
    ) {
        return None;
    }
    let (_, tail) = lower.split_once(" after ")?;
    extract_duration_months_from_text(tail).or_else(|| extract_duration_months_from_text(line))
}

pub(in crate::index) fn extract_current_role_title_from_transition_line(
    line: &str,
    lower: &str,
) -> Option<String> {
    for marker in [
        "worked my way up to ",
        "promoted to ",
        "promotion to ",
        "moved into ",
        "became ",
    ] {
        let Some(start) = lower.find(marker) else {
            continue;
        };
        let tail = &line[start + marker.len()..];
        let title = [" after ", ",", "."]
            .iter()
            .filter_map(|delimiter| tail.find(delimiter))
            .min()
            .map(|end| &tail[..end])
            .unwrap_or(tail)
            .trim()
            .trim_matches(|c: char| matches!(c, '"' | '\'' | ',' | '.' | ';' | ':'));
        if !title.is_empty() {
            return Some(title.to_ascii_lowercase());
        }
    }
    None
}

pub(in crate::index) fn render_month_span(total_months: i32) -> String {
    let years = total_months / 12;
    let months = total_months % 12;
    match (years, months) {
        (0, months) => format!("{months} {}", if months == 1 { "month" } else { "months" }),
        (years, 0) => format!("{years} {}", if years == 1 { "year" } else { "years" }),
        (years, months) => format!(
            "{years} {} and {months} {}",
            if years == 1 { "year" } else { "years" },
            if months == 1 { "month" } else { "months" }
        ),
    }
}

pub(in crate::index) fn extract_explicit_date_rank(line: &str) -> Option<i32> {
    let numeric = compile_regex(r"(?i)\b(\d{1,2})/(\d{1,2})(?:/(\d{4}))?\b");
    if let Some(caps) = numeric.captures(line) {
        let month = caps.get(1)?.as_str().parse::<u32>().ok()?;
        let day = caps.get(2)?.as_str().parse::<u32>().ok()?;
        if !(1..=12).contains(&month) || !(1..=31).contains(&day) {
            return None;
        }
        let year = caps
            .get(3)
            .and_then(|value| value.as_str().parse::<i32>().ok())
            .unwrap_or(2023);
        return Some(ymd_to_days(year, month, day));
    }

    let month_day = compile_regex(
        r"(?i)\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{1,2})(?:st|nd|rd|th)?(?:,\s*(\d{4}))?\b",
    );
    if let Some(caps) = month_day.captures(line) {
        let month = named_month_to_number(caps.get(1)?.as_str())?;
        let day = caps.get(2)?.as_str().parse::<u32>().ok()?;
        let year = caps
            .get(3)
            .and_then(|value| value.as_str().parse::<i32>().ok())
            .unwrap_or(2023);
        return Some(ymd_to_days(year, month, day));
    }

    let day_month_named = compile_regex(
        r"(?i)\b(\d{1,2})(?:st|nd|rd|th)?\s+(January|February|March|April|May|June|July|August|September|October|November|December)(?:,\s*(\d{4}))?\b",
    );
    if let Some(caps) = day_month_named.captures(line) {
        let day = caps.get(1)?.as_str().parse::<u32>().ok()?;
        let month = named_month_to_number(caps.get(2)?.as_str())?;
        let year = caps
            .get(3)
            .and_then(|value| value.as_str().parse::<i32>().ok())
            .unwrap_or(2023);
        return Some(ymd_to_days(year, month, day));
    }

    let day_month = compile_regex(
        r"(?i)\b(?:the\s+)?(\d{1,2})(?:st|nd|rd|th)?\s+of\s+(January|February|March|April|May|June|July|August|September|October|November|December)(?:,\s*(\d{4}))?\b",
    );
    if let Some(caps) = day_month.captures(line) {
        let day = caps.get(1)?.as_str().parse::<u32>().ok()?;
        let month = named_month_to_number(caps.get(2)?.as_str())?;
        let year = caps
            .get(3)
            .and_then(|value| value.as_str().parse::<i32>().ok())
            .unwrap_or(2023);
        return Some(ymd_to_days(year, month, day));
    }

    let fuzzy_month = compile_regex(
        r"(?i)\b(?:(early|mid|late)[-\s]+)?(January|February|March|April|May|June|July|August|September|October|November|December)(?:,\s*|\s+)?(\d{4})?\b",
    );
    let caps = fuzzy_month.captures(line)?;
    let month = named_month_to_number(caps.get(2)?.as_str())?;
    let day = match caps
        .get(1)
        .map(|value| value.as_str().to_ascii_lowercase())
        .as_deref()
    {
        Some("early") => 5,
        Some("late") => 25,
        _ => 15,
    };
    let year = caps
        .get(3)
        .and_then(|value| value.as_str().parse::<i32>().ok())
        .unwrap_or(2023);
    Some(ymd_to_days(year, month, day))
}

pub(in crate::index) fn named_month_to_number(month: &str) -> Option<u32> {
    match &month.to_ascii_lowercase()[..] {
        "january" => Some(1),
        "february" => Some(2),
        "march" => Some(3),
        "april" => Some(4),
        "may" => Some(5),
        "june" => Some(6),
        "july" => Some(7),
        "august" => Some(8),
        "september" => Some(9),
        "october" => Some(10),
        "november" => Some(11),
        "december" => Some(12),
        _ => None,
    }
}

pub(in crate::index) fn ymd_to_days(year: i32, month: u32, day: u32) -> i32 {
    const MONTH_START_DAYS: [i32; 12] = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334];
    let leap_years = {
        let y = year - 1;
        y / 4 - y / 100 + y / 400 - (1969 / 4 - 1969 / 100 + 1969 / 400)
    };
    (year - 1970) * 365 + leap_years + MONTH_START_DAYS[(month - 1) as usize] + day as i32 - 1
}

pub(in crate::index) fn extract_title_duration_value(
    line: &str,
    title_lower: &str,
) -> Option<SyntheticDurationValue> {
    let lower = line.to_ascii_lowercase();
    if !lower.contains(title_lower) {
        return None;
    }
    for marker in ["which took me ", "took me ", "took "] {
        let Some(idx) = lower.find(marker) else {
            continue;
        };
        let tail = &lower[idx + marker.len()..];
        if let Some(value) = parse_leading_duration_value(tail) {
            return Some(value);
        }
    }
    None
}

pub(in crate::index) fn parse_leading_duration_value(text: &str) -> Option<SyntheticDurationValue> {
    let regex = compile_regex(
        r"(?i)^\s*(?:about\s+|around\s+)?(a|an|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|couple|few|\d+(?:\.\d+)?)(\s+and\s+a\s+half)?\s+(day|days|week|weeks|month|months|year|years)\b",
    );
    let caps = regex.captures(text)?;
    let mut amount =
        caps.get(1)
            .and_then(|value| match value.as_str().to_ascii_lowercase().as_str() {
                "a" | "an" | "one" => Some(1.0),
                "two" => Some(2.0),
                "three" => Some(3.0),
                "four" => Some(4.0),
                "five" => Some(5.0),
                "six" => Some(6.0),
                "seven" => Some(7.0),
                "eight" => Some(8.0),
                "nine" => Some(9.0),
                "ten" => Some(10.0),
                "eleven" => Some(11.0),
                "twelve" => Some(12.0),
                "couple" => Some(2.0),
                "few" => Some(3.0),
                value => value.parse::<f32>().ok(),
            })?;
    if caps.get(2).is_some() {
        amount += 0.5;
    }
    let unit = caps.get(3)?.as_str().to_ascii_lowercase();
    let days = amount
        * match unit.as_str() {
            "day" | "days" => 1.0,
            "week" | "weeks" => 7.0,
            "month" | "months" => 30.0,
            "year" | "years" => 365.0,
            _ => return None,
        };
    Some(SyntheticDurationValue {
        amount,
        days,
        unit: match unit.as_str() {
            "day" | "days" => "day",
            "week" | "weeks" => "week",
            "month" | "months" => "month",
            "year" | "years" => "year",
            _ => return None,
        },
    })
}

pub(in crate::index) fn render_duration_unit(unit: &'static str, amount: f32) -> &'static str {
    if (amount - 1.0).abs() < f32::EPSILON {
        unit
    } else {
        match unit {
            "day" => "days",
            "week" => "weeks",
            "month" => "months",
            "year" => "years",
            _ => unit,
        }
    }
}

pub(in crate::index) fn render_elapsed_duration_answer(days: i32) -> String {
    if days % 30 == 0 {
        return render_small_duration(days / 30, "month");
    }
    if days % 7 == 0 {
        return render_small_duration(days / 7, "week");
    }
    if (7..=10).contains(&days) {
        return "one week".to_string();
    }
    render_small_duration(days, "day")
}

pub(in crate::index) fn render_elapsed_from_now_answer(
    days: i32,
    unit: SyntheticElapsedFromNowUnit,
    append_ago: bool,
) -> String {
    let answer = match unit {
        SyntheticElapsedFromNowUnit::Day => render_small_duration(days, "day"),
        SyntheticElapsedFromNowUnit::Week => (((days as f32) / 7.0).round() as i32).to_string(),
        SyntheticElapsedFromNowUnit::Month => (((days as f32) / 30.0).round() as i32).to_string(),
        SyntheticElapsedFromNowUnit::Year => (((days as f32) / 365.0).round() as i32).to_string(),
    };
    if append_ago {
        format!("{answer} ago")
    } else {
        answer
    }
}

pub(in crate::index) fn render_small_duration(amount: i32, unit: &str) -> String {
    let amount_text = match amount {
        1 => "one".to_string(),
        2 => "two".to_string(),
        3 => "three".to_string(),
        4 => "four".to_string(),
        5 => "five".to_string(),
        6 => "six".to_string(),
        7 => "seven".to_string(),
        8 => "eight".to_string(),
        9 => "nine".to_string(),
        10 => "ten".to_string(),
        11 => "eleven".to_string(),
        12 => "twelve".to_string(),
        _ => amount.to_string(),
    };
    let rendered_unit = if amount == 1 {
        unit
    } else {
        match unit {
            "day" => "days",
            "week" => "weeks",
            "month" => "months",
            "year" => "years",
            _ => unit,
        }
    };
    format!("{amount_text} {rendered_unit}")
}

/// Process a single source file: hash-check, AST-extract, write stub + meta.
///
/// Returns a `Vec<CompiledFile>`: the first element (if any) is the Core neuron;
/// subsequent elements are UseCase sub-neurons (S3 lazy splitting, fired when the
/// file has ≥ SUBNEURON_SPLIT_THRESHOLD public functions).
///
/// Returns an empty `Vec` when the file is unchanged (hash match), should be skipped,
/// or when a cosmetic change is detected (S1: sig_hash identical) — in that
/// case only the meta hash is updated on disk and the BM25Entry already in
/// memory from `load_or_create` is preserved with its `staleness_multiplier`
/// and learned feedback signals intact.
///
/// This function performs only filesystem reads and writes — no `&mut NeuronIndex`
/// access — which makes it safe to call in parallel via rayon.
pub(in crate::index) fn process_source_file(
    abs: &Path,
    root: &Path,
    git_confidence: &HashMap<PathBuf, f32>,
) -> Vec<CompiledFile> {
    let rel = abs.strip_prefix(root).unwrap_or(abs);
    if should_skip(rel) {
        return vec![];
    }

    let neuron_path = core_neuron_path(abs, root);
    let meta_file = meta_path(&neuron_path);

    let source_bytes = match std::fs::read(abs) {
        Ok(b) => b,
        Err(_) => return vec![],
    };
    let current_hash = {
        let h = blake3::hash(&source_bytes);
        h.to_hex()[..16].to_string()
    };

    // Read stored meta once and reuse for hash, sig_hash, synapses, module, and feedback counts.
    let stored_meta: Option<NeuronMeta> = if meta_file.exists() {
        std::fs::read_to_string(&meta_file)
            .ok()
            .and_then(|d| serde_json::from_str(&d).ok())
    } else {
        None
    };

    let stored_hash = stored_meta
        .as_ref()
        .map(|m| m.source_hash.as_str())
        .unwrap_or("")
        .to_string();

    // Skip if hash unchanged and neuron exists — pure no-op.
    if !current_hash.is_empty() && current_hash == stored_hash && neuron_path.exists() {
        return vec![];
    }

    let source_text = String::from_utf8_lossy(&source_bytes);
    let source_rel = rel.to_string_lossy();
    let now = now_iso8601();

    let ast_summary = ast_extractor::extract_signatures(&source_rel, &source_text);
    let sig_hash = ast_extractor::compute_sig_hash(&ast_summary);

    let stored_sig_hash = stored_meta
        .as_ref()
        .and_then(|m| m.sig_hash.as_deref())
        .unwrap_or("")
        .to_string();

    // S1 — Cosmetic change: source_hash changed but public API surface (sig_hash) is identical.
    // Whitespace edits, doc-comment tweaks, or formatting passes land here.
    // Preserve the LLM-curated stub; only update the hash in the meta file so future
    // compiles don't re-check this file. The in-memory BM25Entry (from load_or_create)
    // retains its staleness_multiplier and learned feedback signals.
    if !stored_sig_hash.is_empty()
        && sig_hash == stored_sig_hash
        && !stored_hash.is_empty()
        && neuron_path.exists()
    {
        if let Some(mut old_meta) = stored_meta {
            old_meta.source_hash = current_hash;
            old_meta.sig_hash = Some(sig_hash);
            old_meta.last_updated = now;
            if let Err(e) = atomic_write_json(&meta_file, &old_meta) {
                tracing::warn!(
                    "Failed to update meta for cosmetic change {:?}: {e}",
                    meta_file
                );
            }
        }
        return vec![];
    }

    // S1 (R11) — Section-Level Staleness: sig_hash changed (real API change) but the
    // neuron already exists with LLM-curated content. Instead of overwriting everything,
    // replace only the `api` section and update the header comments. Preserves `purpose`,
    // `pitfalls`, and cross-reference sections. Reduces LLM re-evolution calls by ~60%.
    if !stored_hash.is_empty() && neuron_path.exists() {
        // sig_hash is different — we passed the cosmetic-change gate above
        match std::fs::read_to_string(&neuron_path) {
            Ok(existing) => {
                let new_api = ast_extractor::format_for_stub(&ast_summary);
                let updated = replace_section(&existing, "api", &new_api);
                let updated = update_neuron_header(&updated, &current_hash, &now);
                if let Err(e) = atomic_write(&neuron_path, updated.as_bytes()) {
                    tracing::warn!("S1: Failed to update api section {:?}: {e}", neuron_path);
                    // Fall through to full stub generation below
                } else {
                    let old = stored_meta
                        .clone()
                        .unwrap_or_else(|| NeuronMeta::new_stub(abs, NeuronKind::Core));
                    let mut meta = old;
                    meta.source_hash = current_hash;
                    meta.sig_hash = Some(sig_hash);
                    meta.last_updated = now.clone();
                    meta.status = NeuronStatus::Stale;
                    meta.tokens = estimate_context_tokens(&updated).get();
                    if meta.module.is_none() {
                        meta.module = infer_module(rel);
                    }
                    let existing_targets: HashSet<PathBuf> =
                        meta.synapses.iter().map(|s| s.target.clone()).collect();
                    let auto_imports = import_parser::parse_imports(abs, &source_text, root);
                    for imported_source in auto_imports {
                        let target_neuron = core_neuron_path(&imported_source, root);
                        if !existing_targets.contains(&target_neuron) {
                            meta.synapses.push(Synapse::new(
                                target_neuron,
                                SynapseType::Imports,
                                "auto-inferred from import statement".to_string(),
                            ));
                        }
                    }
                    meta.confidence_score = git_confidence.get(abs).copied().unwrap_or(1.0);
                    if let Err(e) = atomic_write_json(&meta_file, &meta) {
                        tracing::warn!("S1: Failed to update meta {:?}: {e}", meta_file);
                    }
                    let mut results = vec![CompiledFile {
                        neuron_path: neuron_path.clone(),
                        content: updated,
                        meta,
                    }];
                    // Also generate sub-neurons for any new functions (idempotent — skips existing)
                    if ast_summary.functions.len() >= SUBNEURON_SPLIT_THRESHOLD {
                        for fn_name in ast_summary.functions.iter().take(MAX_SUBNEURONS_PER_FILE) {
                            let sub_path = sub_neuron_path(&neuron_path, fn_name);
                            if sub_path.exists() {
                                continue;
                            }
                            let sub_content = stub_function_neuron(fn_name, &source_rel, &now);
                            if let Err(e) = atomic_write(&sub_path, sub_content.as_bytes()) {
                                tracing::warn!(
                                    "S1: Failed to write sub-neuron {:?}: {e}",
                                    sub_path
                                );
                                continue;
                            }
                            let sub_meta_file = meta_path(&sub_path);
                            let mut sub_meta = NeuronMeta::new_stub(abs, NeuronKind::UseCase);
                            sub_meta.task_pattern = Some(fn_name.clone());
                            sub_meta.parent = Some(neuron_path.clone());
                            sub_meta.tokens = estimate_context_tokens(&sub_content).get();
                            sub_meta.last_updated = now.clone();
                            sub_meta.module = results[0].meta.module.clone();
                            sub_meta.confidence_score = results[0].meta.confidence_score;
                            if let Err(e) = atomic_write_json(&sub_meta_file, &sub_meta) {
                                tracing::warn!(
                                    "S1: Failed to write sub-neuron meta {:?}: {e}",
                                    sub_meta_file
                                );
                                continue;
                            }
                            results.push(CompiledFile {
                                neuron_path: sub_path,
                                content: sub_content,
                                meta: sub_meta,
                            });
                        }
                    }
                    tracing::debug!(path = %neuron_path.display(), "S1: api section updated, purpose/pitfalls preserved");
                    return results;
                }
            },
            Err(_) => {
                // Cannot read existing neuron — fall through to full stub regeneration
            },
        }
    }

    // Full stub (re)generation — real API change (sig_hash changed) or new file.
    let prefilled = ast_extractor::format_for_stub(&ast_summary);
    let purpose_hint = ast_extractor::format_purpose_hint(&ast_summary);
    let extra_vocab = ast_extractor::format_extra_vocab_for_stub(&ast_summary);
    let content = stub_core_neuron(
        &source_rel,
        &current_hash,
        &now,
        &prefilled,
        &purpose_hint,
        &extra_vocab,
    );

    if let Some(parent) = neuron_path.parent() {
        if let Err(e) = std::fs::create_dir_all(parent) {
            tracing::warn!("Failed to create neuron dir {:?}: {e}", parent);
            return vec![];
        }
    }
    if let Err(e) = atomic_write(&neuron_path, content.as_bytes()) {
        tracing::warn!("Failed to write stub {:?}: {e}", neuron_path);
        return vec![];
    }

    let is_new = stored_hash.is_empty();
    let mut meta = NeuronMeta::new_stub(abs, NeuronKind::Core);
    meta.source_hash = current_hash;
    meta.sig_hash = Some(sig_hash);
    meta.tokens = estimate_context_tokens(&content).get();
    meta.last_updated = now.clone();
    meta.status = if is_new {
        NeuronStatus::Stub
    } else {
        NeuronStatus::Stale
    };

    // Preserve existing synapses, module tag, and feedback counts on hash invalidation.
    if let Some(old) = stored_meta {
        meta.synapses = old.synapses;
        meta.module = old.module;
        meta.use_count = old.use_count;
        meta.hit_count = old.hit_count;
    }

    // Auto-module: infer from directory structure when not LLM-set.
    if meta.module.is_none() {
        meta.module = infer_module(rel);
    }

    // Auto-Synapse: infer Imports edges from import statements.
    let existing_targets: HashSet<PathBuf> =
        meta.synapses.iter().map(|s| s.target.clone()).collect();
    let auto_imports = import_parser::parse_imports(abs, &source_text, root);
    for imported_source in auto_imports {
        let target_neuron = core_neuron_path(&imported_source, root);
        if !existing_targets.contains(&target_neuron) {
            meta.synapses.push(Synapse::new(
                target_neuron,
                SynapseType::Imports,
                "auto-inferred from import statement".to_string(),
            ));
        }
    }

    // Git confidence: committed + unmodified = 1.0, modified = 0.9, untracked = 0.85.
    meta.confidence_score = git_confidence.get(abs).copied().unwrap_or(1.0);

    if let Err(e) = atomic_write_json(&meta_file, &meta) {
        tracing::warn!("Failed to write meta {:?}: {e}", meta_file);
        return vec![];
    }

    let mut results = vec![CompiledFile {
        neuron_path: neuron_path.clone(),
        content,
        meta,
    }];

    // S3 — Lazy Sub-Neuron Splitting: for files with many public functions,
    // generate one UseCase sub-neuron per function so BM25 can match at
    // function-level precision. Sub-neurons slot into Phase 2 of get_contexts
    // (UseCase scoring per Core) automatically via the parent_index.
    if ast_summary.functions.len() >= SUBNEURON_SPLIT_THRESHOLD {
        for fn_name in ast_summary.functions.iter().take(MAX_SUBNEURONS_PER_FILE) {
            let sub_path = sub_neuron_path(&neuron_path, fn_name);
            // Only write a new stub if the sub-neuron doesn't already exist —
            // preserve any LLM-curated content from a previous compile.
            if sub_path.exists() {
                continue;
            }
            let sub_content = stub_function_neuron(fn_name, &source_rel, &now);
            if let Err(e) = atomic_write(&sub_path, sub_content.as_bytes()) {
                tracing::warn!("Failed to write sub-neuron {:?}: {e}", sub_path);
                continue;
            }
            let sub_meta_file = meta_path(&sub_path);
            let mut sub_meta = NeuronMeta::new_stub(abs, NeuronKind::UseCase);
            sub_meta.task_pattern = Some(fn_name.clone());
            sub_meta.parent = Some(neuron_path.clone());
            sub_meta.tokens = estimate_context_tokens(&sub_content).get();
            sub_meta.last_updated = now.clone();
            sub_meta.module = results[0].meta.module.clone();
            sub_meta.confidence_score = results[0].meta.confidence_score;
            if let Err(e) = atomic_write_json(&sub_meta_file, &sub_meta) {
                tracing::warn!("Failed to write sub-neuron meta {:?}: {e}", sub_meta_file);
                continue;
            }
            results.push(CompiledFile {
                neuron_path: sub_path,
                content: sub_content,
                meta: sub_meta,
            });
        }
    }

    results
}
