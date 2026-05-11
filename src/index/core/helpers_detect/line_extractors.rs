//! Content extraction from individual lines.

use super::super::*;
use crate::index::compile_regex;

pub fn small_count_word_lower(value: i32) -> Option<&'static str> {
    match value {
        0 => Some("zero"),
        1 => Some("one"),
        2 => Some("two"),
        3 => Some("three"),
        4 => Some("four"),
        5 => Some("five"),
        6 => Some("six"),
        7 => Some("seven"),
        8 => Some("eight"),
        9 => Some("nine"),
        10 => Some("ten"),
        11 => Some("eleven"),
        12 => Some("twelve"),
        _ => None,
    }
}

pub fn supporting_word_count_surface(
    lines: &[String],
    value: i32,
    focus_terms: &[String],
) -> Option<String> {
    let word = small_count_word_lower(value)?;
    let focus_keys = synthetic_answer_surface_term_key_set(focus_terms);
    for line in lines {
        let lower = line.to_ascii_lowercase();
        if !lower.contains(word) {
            continue;
        }
        let line_keys = synthetic_answer_surface_term_key_set(&synthetic_query_terms(&lower));
        if synthetic_answer_surface_overlap_count(&line_keys, &focus_keys) >= 1 {
            return Some(word.to_string());
        }
    }
    None
}

pub fn parse_frequency_count_token(token: &str) -> Option<i32> {
    match token
        .trim_matches(|c: char| !c.is_ascii_alphanumeric())
        .to_ascii_lowercase()
        .as_str()
    {
        "once" => Some(1),
        "twice" => Some(2),
        "thrice" => Some(3),
        other => parse_count_token_value(other),
    }
}

pub fn extract_meetup_count_surface_from_line(line: &str, lower: &str) -> Option<String> {
    if !lower.contains("met up")
        || task_contains_any(
            lower,
            &[
                "planning to meet up",
                "plan to meet up",
                "we're planning to meet up",
                "going to meet up",
            ],
        )
    {
        return None;
    }
    let raw = compile_regex(
        r"(?i)\bmet up\s+(once|twice|thrice|one|two|three|four|five|six|seven|eight|nine|ten|\d+)(?:\s+times?)?\b",
    )
    .captures(line)
    .and_then(|caps| caps.get(1))
    .map(|m| m.as_str().trim())?;
    let normalized = raw.to_ascii_lowercase();
    Some(if normalized.chars().all(|c| c.is_ascii_digit()) {
        format!("We've met up {} times.", normalized)
    } else {
        format!("We've met up {}.", normalized)
    })
}

pub fn extract_meetup_count_from_line(line: &str, lower: &str) -> Option<i32> {
    if !lower.contains("met up")
        || task_contains_any(
            lower,
            &[
                "planning to meet up",
                "plan to meet up",
                "we're planning to meet up",
                "going to meet up",
            ],
        )
    {
        return None;
    }
    let raw = compile_regex(
        r"(?i)\bmet up\s+(once|twice|thrice|one|two|three|four|five|six|seven|eight|nine|ten|\d+)(?:\s+times?)?\b",
    )
    .captures(line)
    .and_then(|caps| caps.get(1))
    .map(|m| m.as_str())?;
    parse_frequency_count_token(raw)
}

pub fn extract_item_usage_count_surface_from_line(
    line: &str,
    lower: &str,
    usage_kind: &str,
) -> Option<String> {
    let raw = match usage_kind {
        "wear" => {
            if !(task_contains_any(lower, &["worn", "wore"]) && lower.contains("times")) {
                return None;
            }
            compile_regex(
                r"(?i)\b(one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|\d+)\s+times?\b",
            )
            .captures(line)
            .and_then(|caps| caps.get(1))
            .map(|m| m.as_str().trim())?
        },
        "trip" => {
            if !(lower.contains("trip") || lower.contains("adventure")) {
                return None;
            }
            compile_regex(
                r"(?i)\b(one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|\d+)\s+(?:trip|trips|adventures)\b",
            )
            .captures(line)
            .and_then(|caps| caps.get(1))
            .map(|m| m.as_str().trim())?
        },
        _ => return None,
    };
    Some(raw.to_ascii_lowercase())
}

pub fn extract_item_usage_count_from_line(
    line: &str,
    lower: &str,
    usage_kind: &str,
) -> Option<i32> {
    let surface = extract_item_usage_count_surface_from_line(line, lower, usage_kind)?;
    parse_count_token_value(&surface)
}

pub fn extract_women_count_from_line(line: &str, lower: &str) -> Option<i32> {
    if !lower.contains("women") {
        return None;
    }
    let raw = compile_regex(
        r"(?i)\b(one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|\d+)\s+women\b",
    )
    .captures(line)
    .and_then(|caps| caps.get(1))
    .map(|m| m.as_str().trim())?;
    parse_count_token_value(raw)
}

pub fn extract_weight_loss_answer_from_line(line: &str, lower: &str) -> Option<(i32, String)> {
    if !lower.contains("lost") || !lower.contains("pound") {
        return None;
    }
    let captures = compile_regex(
        r"(?i)\b(?:lost|down)\s+(about\s+)?(one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|\d+)\s+pounds?\b",
    )
    .captures(line)?;
    let about = captures
        .get(1)
        .map(|m| !m.as_str().trim().is_empty())
        .unwrap_or(false);
    let raw = captures.get(2)?.as_str().trim().to_ascii_lowercase();
    let value = parse_count_token_value(&raw)?;
    let surface = if about {
        format!("about {raw} pounds")
    } else {
        format!("{raw} pounds")
    };
    Some((value, surface))
}

pub fn extract_frequency_surface_from_line(line: &str, lower: &str) -> Option<String> {
    if lower.contains("every other week") {
        return Some("every other week".to_string());
    }
    if lower.contains("every two weeks") {
        return Some("every two weeks".to_string());
    }
    if lower.contains("every week") || lower.contains("weekly") {
        return Some("every week".to_string());
    }
    if lower.contains("every day") || lower.contains("daily") {
        return Some("every day".to_string());
    }
    compile_regex(
        r"(?i)\b(once|twice|thrice|one|two|three|four|five|\d+)\s+times?\s+(?:a|per)\s+(day|week|month|year)\b",
    )
    .captures(line)
    .and_then(|caps| {
        let raw = caps.get(1)?.as_str().trim().to_ascii_lowercase();
        let unit = caps.get(2)?.as_str().trim().to_ascii_lowercase();
        Some(format!("{raw} times a {unit}"))
    })
}

pub fn extract_time_answer_from_line(line: &str) -> Option<String> {
    [
        r"(?i)\b(\d{1,2}:\d{2}\s?(?:AM|PM))\b",
        r"(?i)\b(\d{1,2}\s?(?:AM|PM))\b",
    ]
    .into_iter()
    .find_map(|pattern| {
        compile_regex(pattern)
            .captures(line)
            .and_then(|caps| caps.get(1))
            .map(|m| m.as_str().trim().to_string())
    })
}

pub fn extract_focus_aligned_time_answer_from_line(
    line: &str,
    lower: &str,
    focus_terms: &[String],
) -> Option<String> {
    let pattern = compile_regex(r"(?i)\b(\d{1,2}(?::\d{2})?\s?(?:AM|PM))\b");
    let matches = pattern
        .captures_iter(line)
        .filter_map(|caps| caps.get(1))
        .map(|m| (m.start(), m.as_str().trim().to_string()))
        .collect::<Vec<_>>();
    if matches.is_empty() {
        return None;
    }
    if matches.len() == 1 {
        return extract_time_answer_from_line(line);
    }
    let focus_positions = focus_terms
        .iter()
        .filter_map(|term| lower.find(term))
        .collect::<Vec<_>>();
    if focus_positions.is_empty() {
        return matches.last().map(|(_, value)| value.clone());
    }
    matches
        .into_iter()
        .min_by_key(|(time_idx, _)| {
            focus_positions
                .iter()
                .map(|focus_idx| focus_idx.abs_diff(*time_idx))
                .min()
                .unwrap_or(usize::MAX)
        })
        .map(|(_, value)| value)
}

pub fn extract_schedule_slot_focus_phrase(task_lower: &str) -> Option<String> {
    for marker in [
        "what day of the week do i ",
        "which day do i ",
        "what time do i ",
    ] {
        let Some((_, tail)) = task_lower.split_once(marker) else {
            continue;
        };
        let phrase = tail.trim().trim_end_matches('?');
        if !phrase.is_empty() {
            return Some(phrase.to_string());
        }
    }
    None
}

pub fn extract_points_answer_from_line(line: &str, lower: &str) -> Option<String> {
    if !(lower.contains("score") || lower.contains("points")) {
        return None;
    }
    let raw = compile_regex(r"(?i)\b(\d+)\s+points\b")
        .captures(line)
        .and_then(|caps| caps.get(1))
        .map(|m| m.as_str().trim())?;
    Some(format!("{raw} points"))
}

pub fn extract_record_answer_from_line(line: &str, lower: &str) -> Option<String> {
    if !(lower.contains("record") || lower.contains("we're") || lower.contains("we are")) {
        return None;
    }
    compile_regex(r"\b(\d+\s*-\s*\d+)\b")
        .captures(line)
        .and_then(|caps| caps.get(1))
        .map(|m| m.as_str().trim().replace(' ', ""))
}

pub fn extract_status_answer_from_line(line: &str, lower: &str) -> Option<String> {
    if !lower.contains("status") {
        return None;
    }
    compile_regex(r"(?i)\b(Premier\s+(?:Silver|Gold|Platinum|Bronze|Diamond|1K))\s+status\b")
        .captures(line)
        .and_then(|caps| caps.get(1))
        .map(|m| m.as_str().trim().to_string())
}

pub fn extract_level_goal_answer_from_line(line: &str, lower: &str) -> Option<String> {
    if !lower.contains("level")
        || !(line_has_future_goal_marker(lower)
            || lower.contains("determined to reach")
            || lower.contains("aiming to hit")
            || lower.contains("current goal"))
    {
        return None;
    }
    compile_regex(r"(?i)\b(level\s+\d+)\b")
        .captures(line)
        .and_then(|caps| caps.get(1))
        .map(|m| m.as_str().trim().to_ascii_lowercase())
}

pub fn extract_state_transition_surface_from_line(
    line: &str,
    lower: &str,
    state_kind: &str,
) -> Option<String> {
    match state_kind {
        "score" => extract_points_answer_from_line(line, lower),
        "record" => extract_record_answer_from_line(line, lower),
        "status" => extract_status_answer_from_line(line, lower),
        "goal" => extract_level_goal_answer_from_line(line, lower),
        _ => None,
    }
}

pub fn extract_relative_purchase_current_item(task_lower: &str) -> Option<String> {
    [
        "before getting the ",
        "before getting ",
        "before i got the ",
        "before i got ",
        "before buying the ",
        "before buying ",
        "before i bought the ",
        "before i bought ",
        "before purchasing the ",
        "before purchasing ",
        "before i purchased the ",
        "before i purchased ",
    ]
    .into_iter()
    .find_map(|marker| {
        let (_, tail) = task_lower.split_once(marker)?;
        let item = normalize_query_item_surface(tail);
        (!item.is_empty()).then_some(item)
    })
}

pub fn normalize_query_item_surface(value: &str) -> String {
    let trimmed = value
        .trim()
        .trim_end_matches('?')
        .trim_end_matches('.')
        .trim();
    for prefix in ["the ", "a ", "an "] {
        if let Some(stripped) = trimmed.strip_prefix(prefix) {
            return stripped.trim().to_string();
        }
    }
    trimmed.to_string()
}

pub fn extract_purchase_family_item_from_line(
    line: &str,
    lower: &str,
    family: &str,
) -> Option<String> {
    match family {
        "gadget" => extract_gadget_purchase_item_from_line(line, lower),
        "lens" => extract_lens_purchase_item_from_line(line, lower),
        _ => None,
    }
}

pub fn extract_gadget_purchase_item_from_line(line: &str, lower: &str) -> Option<String> {
    if !task_contains_any(
        lower,
        &[
            "my new ",
            "i got",
            "got yesterday",
            "bought",
            "purchased",
            "gift",
            "using the ",
            "using my ",
        ],
    ) {
        return None;
    }
    compile_regex(
        r"(?i)\b(?:my\s+new\s+|my\s+|the\s+)?((?:[a-z0-9][a-z0-9+-]*)(?:\s+[a-z0-9][a-z0-9+-]*){0,2}\s(?:pot|fryer|mixer|blender|processor|maker|oven|grill|toaster|microwave|cooker|skillet))\b",
    )
    .captures_iter(line)
    .filter_map(|caps| caps.get(1))
    .map(|m| m.as_str().trim().to_string())
    .last()
}

pub fn extract_lens_purchase_item_from_line(line: &str, lower: &str) -> Option<String> {
    let has_ownership_marker = task_contains_any(
        lower,
        &[
            "i got",
            "got my ",
            "recently got",
            "just got",
            "bought my ",
            "bought a ",
            "bought an ",
            "purchased",
            "picked up",
            "my new ",
        ],
    );
    if !lower.contains("lens") || !has_ownership_marker {
        return None;
    }
    if task_contains_any(lower, &["haven't bought", "have not bought", "might buy"])
        && !task_contains_any(lower, &["got my ", "recently got", "just got", "my new "])
    {
        return None;
    }
    let phrase = compile_regex(
        r"(?i)\b(?:old\s+|new\s+)?((?:\d{1,3}(?:-\d{1,3})?mm|[a-z]+(?:-[a-z]+)?)(?:\s+[a-z]+(?:-[a-z]+)?){0,2}\s+lens)\b",
    )
    .captures_iter(line)
    .filter_map(|caps| caps.get(1))
    .map(|m| m.as_str().trim().to_string())
    .last()?;
    Some(render_with_indefinite_article(&phrase))
}

pub fn render_with_indefinite_article(value: &str) -> String {
    let trimmed = value.trim();
    let lower = trimmed.to_ascii_lowercase();
    if lower.starts_with("a ") || lower.starts_with("an ") {
        return trimmed.to_string();
    }
    let article = match lower.chars().next() {
        Some('a' | 'e' | 'i' | 'o' | 'u') => "an",
        _ => "a",
    };
    format!("{article} {trimmed}")
}

pub fn extract_trip_destination_from_query(task_lower: &str) -> Option<String> {
    for marker in ["trip to ", "vacation to ", "visit to "] {
        let Some((_, tail)) = task_lower.split_once(marker) else {
            continue;
        };
        let destination = tail.trim().trim_end_matches('?').trim().to_string();
        if !destination.is_empty() {
            return Some(destination);
        }
    }
    None
}

pub fn extract_planned_stay_location_from_line(line: &str, lower: &str) -> Option<String> {
    let value = extract_phrase_after_any_index(
        line,
        lower,
        &[
            "planning to stay on ",
            "planning to stay in ",
            "planning to stay at ",
            "plan to stay on ",
            "plan to stay in ",
            "plan to stay at ",
            "staying on ",
            "staying in ",
            "staying at ",
            "stay on ",
            "stay in ",
            "stay at ",
        ],
        &[
            " for ",
            " because ",
            " and ",
            " but ",
            " while ",
            ".",
            ",",
            ";",
            " instead",
            " during ",
        ],
        1,
    )?;
    (value.split_whitespace().count() <= 6).then(|| normalize_location_kg_value(&value))
}

pub fn line_has_current_company_marker(lower: &str) -> bool {
    task_contains_any(
        lower,
        &[
            "currently working at ",
            "currently at ",
            "current company is ",
            "works at ",
            "working at ",
            "employed at ",
        ],
    )
}
