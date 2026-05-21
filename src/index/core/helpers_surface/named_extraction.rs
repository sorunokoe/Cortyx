//! Named entity extraction: locations, occupations, money, dates, colors, purchase items.

use super::super::*;
use crate::index::{compile_regex, compile_regex_static};

pub fn extract_percent_answer_from_line(line: &str) -> Option<String> {
    compile_regex_static(r"(?i)(\d+(?:\.\d+)?%)")
        .captures(line)
        .and_then(|caps| caps.get(1))
        .map(|m| m.as_str().trim().to_string())
}

pub fn extract_speed_answer_from_line(line: &str) -> Option<String> {
    compile_regex_static(r"(?i)(\d+(?:\.\d+)?\s*(?:mbps|gbps))")
        .captures(line)
        .and_then(|caps| caps.get(1))
        .map(|m| m.as_str().trim().to_string())
}

pub fn extract_university_name_from_line(line: &str) -> Option<String> {
    compile_regex_static(r"([A-Z][A-Za-z&.'-]*(?:\s+[A-Z][A-Za-z&.'-]*)*\s+University)")
        .captures(line)
        .and_then(|caps| caps.get(1))
        .map(|m| m.as_str().trim().to_string())
}

pub fn extract_query_month_name(lower: &str) -> Option<&'static str> {
    [
        "january",
        "february",
        "march",
        "april",
        "may",
        "june",
        "july",
        "august",
        "september",
        "october",
        "november",
        "december",
    ]
    .into_iter()
    .find(|month| lower.contains(month))
}

pub fn next_month_name(month: &str) -> Option<&'static str> {
    match month {
        "january" => Some("february"),
        "february" => Some("march"),
        "march" => Some("april"),
        "april" => Some("may"),
        "may" => Some("june"),
        "june" => Some("july"),
        "july" => Some("august"),
        "august" => Some("september"),
        "september" => Some("october"),
        "october" => Some("november"),
        "november" => Some("december"),
        "december" => Some("january"),
        _ => None,
    }
}

pub fn line_matches_query_month_window(lower: &str, month: &str) -> bool {
    if lower.contains(month) {
        return true;
    }

    lower.contains("this month")
        && next_month_name(month)
            .map(|next_month| lower.contains(&format!("before {next_month}")))
            .unwrap_or(false)
}

pub fn line_describes_actual_doctor_visit(lower: &str) -> bool {
    let positive = task_contains_any(
        lower,
        &[
            "follow-up appointment",
            "appointment with",
            "went to see",
            "got back from",
            "diagnosed me with",
            "diagnosed with",
            "was prescribed",
            "prescribed antibiotics",
            "prescribed a nasal spray",
            "recently had",
            "just got diagnosed",
        ],
    );
    if !positive {
        return false;
    }

    if task_contains_any(
        lower,
        &[
            "thinking about",
            "considering",
            "i'll schedule",
            "i will schedule",
            "schedule an appointment",
            "scheduling an appointment",
            "talk to dr.",
            "ask dr.",
            "follow up with dr.",
            "consult with",
        ],
    ) {
        return false;
    }

    true
}

pub fn extract_doctor_role_from_line(_line: &str, lower: &str) -> Option<String> {
    [
        ("primary care physician", "a primary care physician"),
        ("ent specialist", "an ENT specialist"),
        ("dermatologist", "a dermatologist"),
        ("orthopedic surgeon", "an orthopedic surgeon"),
        ("neurologist", "a neurologist"),
        ("gastroenterologist", "a gastroenterologist"),
    ]
    .into_iter()
    .find(|(needle, _)| lower.contains(needle))
    .map(|(_, rendered)| rendered.to_string())
}

pub fn doctor_role_sort_key(role: &str) -> usize {
    match role {
        "a primary care physician" => 0,
        "an ENT specialist" => 1,
        "a dermatologist" => 2,
        "an orthopedic surgeon" => 3,
        "a neurologist" => 4,
        "a gastroenterologist" => 5,
        _ => 99,
    }
}

pub fn doctor_visit_event_key(role: &str, lower: &str) -> String {
    let day = compile_regex_static(r"\b(?:january|february|march|april|may|june|july|august|september|october|november|december)\s+(\d{1,2})(?:st|nd|rd|th)?\b")
        .captures(lower)
        .and_then(|caps| caps.get(1))
        .map(|m| m.as_str().to_string());
    match day {
        Some(day) => format!("{role}|{day}"),
        None => role.to_string(),
    }
}

pub fn extract_duration_answer_from_line(line: &str) -> Option<String> {
    compile_regex_static(
        r"(?i)\b((?:about\s+)?(?:an?|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|\d+(?:\.\d+)?(?:\s*-\s*\d+(?:\.\d+)?)?)\s+(?:days?|weeks?|months?|years?|hours?|minutes?)(?:\s+(?:ago|now|each way))?)\b",
    )
    .captures(line)
    .and_then(|caps| caps.get(1))
    .map(|m| m.as_str().trim().to_string())
}

pub fn normalize_current_duration_answer(duration: &str) -> String {
    duration
        .trim()
        .trim_start_matches("about ")
        .trim_end_matches(" now")
        .trim_end_matches(" ago")
        .trim_start_matches("an ")
        .trim_start_matches("a ")
        .to_string()
        .replacen("one ", "1 ", 1)
}

pub fn duration_answer_magnitude(duration: &str) -> Option<f32> {
    let lower = duration.to_ascii_lowercase();
    let caps = compile_regex_static(
        r"\b(\d+(?:\.\d+)?|an?|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)(?:\s*-\s*(\d+(?:\.\d+)?))?\s+(day|week|month|year|hour|minute)s?\b",
    )
    .captures(&lower)?;
    let quantity = match caps.get(2).map(|m| m.as_str()) {
        Some(value) => value.parse::<f32>().ok()?,
        None => match caps.get(1)?.as_str() {
            "a" | "an" => 1.0,
            "one" => 1.0,
            "two" => 2.0,
            "three" => 3.0,
            "four" => 4.0,
            "five" => 5.0,
            "six" => 6.0,
            "seven" => 7.0,
            "eight" => 8.0,
            "nine" => 9.0,
            "ten" => 10.0,
            "eleven" => 11.0,
            "twelve" => 12.0,
            value => value.parse::<f32>().ok()?,
        },
    };
    let unit_days = match caps.get(3)?.as_str() {
        "minute" => 1.0 / (24.0 * 60.0),
        "hour" => 1.0 / 24.0,
        "day" => 1.0,
        "week" => 7.0,
        "month" => 30.0,
        "year" => 365.0,
        _ => return None,
    };
    Some(quantity * unit_days)
}

pub fn is_ongoing_duration_query(task_lower: &str) -> bool {
    task_lower.starts_with("how long have ")
        && !task_contains_any(
            task_lower,
            &[" before ", " after ", " until ", "left to", "remaining"],
        )
}

pub fn extract_ongoing_duration_anchor_terms(terms: &[String]) -> Vec<String> {
    const STOP: &[&str] = &[
        "long",
        "been",
        "being",
        "using",
        "living",
        "sticking",
        "staying",
        "working",
        "collecting",
        "keeping",
        "having",
        "doing",
        "going",
        "current",
        "daily",
        "about",
        "around",
        "there",
        "here",
    ];
    let anchors: Vec<String> = terms
        .iter()
        .filter(|term| term.len() >= 4)
        .filter(|term| !STOP.contains(&term.as_str()))
        .cloned()
        .collect();
    if anchors.is_empty() {
        terms
            .iter()
            .filter(|term| term.len() >= 3)
            .filter(|term| !STOP.contains(&term.as_str()))
            .cloned()
            .collect()
    } else {
        anchors
    }
}

pub fn extract_tablespoon_water_ounces(line: &str) -> Option<f32> {
    let lower = line.to_ascii_lowercase();
    if !(lower.contains("tablespoon")
        && lower.contains("coffee")
        && lower.contains("ounces")
        && lower.contains("water"))
    {
        return None;
    }
    compile_regex_static(r"(?i)\b(\d+(?:\.\d+)?)\s+ounces?\s+of\s+water\b")
        .captures(line)
        .and_then(|caps| caps.get(1))
        .and_then(|m| m.as_str().parse::<f32>().ok())
}

pub fn compact_decimal_string(value: f32) -> String {
    let mut rendered = value.to_string();
    if rendered.ends_with(".0") {
        rendered.truncate(rendered.len() - 2);
    }
    rendered
}

pub fn extract_date_or_time_answer_from_line(line: &str) -> Option<String> {
    for pattern in [
        r"(?i)\b((?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}(?:st|nd|rd|th)?(?:-\d{1,2}(?:st|nd|rd|th)?)?)\b",
        r"(?i)\b(\d{1,2}:\d{2}\s?(?:AM|PM))\b",
        r"(?i)\b(\d{1,2}\s?(?:AM|PM))\b",
        r"(?i)\b(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)\b",
    ] {
        if let Some(value) = compile_regex_static(pattern)
            .captures(line)
            .and_then(|caps| caps.get(1))
            .map(|m| m.as_str().trim().to_string())
        {
            return Some(value);
        }
    }
    None
}

pub fn extract_color_answer_from_line(line: &str) -> Option<String> {
    for pattern in [
        r"(?i)\b((?:a\s+)?(?:lighter|darker|light|dark|soft|pale|bright|deep)\s+shade of\s+(?:gray|grey|blue|green|pink|purple|yellow|red|orange|white|black|beige|brown))\b",
        r"(?i)\b((?:light|dark|pale|bright|deep|soft)\s+(?:gray|grey|blue|green|pink|purple|yellow|red|orange|white|black|beige|brown))\b",
        r"(?i)\b(gray|grey|blue|green|pink|purple|yellow|red|orange|white|black|beige|brown)\b",
    ] {
        if let Some(value) = compile_regex_static(pattern)
            .captures(line)
            .and_then(|caps| caps.get(1))
            .map(|m| m.as_str().trim().to_string())
        {
            return Some(value);
        }
    }
    None
}

pub fn extract_query_aligned_numeric_answer(task_lower: &str, line: &str) -> Option<String> {
    let mut terms = synthetic_query_terms(task_lower)
        .into_iter()
        .filter(|term| term.len() >= 4)
        .filter(|term| {
            ![
                "current",
                "currently",
                "recently",
                "specific",
                "previous",
                "conversation",
                "recommended",
            ]
            .contains(&term.as_str())
        })
        .collect::<Vec<_>>();
    if task_lower.contains("times") {
        terms.extend(
            ["game", "games", "match", "matches", "meeting", "meetings"]
                .into_iter()
                .map(str::to_string),
        );
    }
    terms.sort();
    terms.dedup();
    let line_lower = line.to_ascii_lowercase();
    let anchor_terms = assistant_followup_anchor_terms(task_lower);
    let mut best_anchor_match: Option<(usize, usize, String)> = None;
    for term in &terms {
        let pattern = compile_regex(&format!(
            r"(?i)\b((?:\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred))\s+{}s?\b",
            regex::escape(term)
        ))
        .unwrap_or_else(|err| panic!("escaped numeric-answer regex failed to compile: {err}"));
        for capture in pattern.captures_iter(line) {
            let Some(full_match) = capture.get(0) else {
                continue;
            };
            let Some(value_match) = capture.get(1) else {
                continue;
            };
            let Some(distance) =
                assistant_followup_anchor_distance(&line_lower, full_match.end(), &anchor_terms)
            else {
                continue;
            };
            let value = value_match.as_str().trim().to_string();
            if best_anchor_match
                .as_ref()
                .map(|(best_distance, best_start, _)| {
                    distance < *best_distance
                        || (distance == *best_distance && full_match.start() > *best_start)
                })
                .unwrap_or(true)
            {
                best_anchor_match = Some((distance, full_match.start(), value));
            }
        }
    }
    if let Some((_, _, value)) = best_anchor_match {
        return Some(value);
    }
    for term in terms {
        let pattern = compile_regex(&format!(
            r"(?i)\b((?:\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred))\s+{}s?\b",
            regex::escape(&term)
        ))
        .unwrap_or_else(|err| panic!("escaped numeric-answer regex failed to compile: {err}"));
        if let Some(value) = pattern
            .captures(line)
            .and_then(|caps| caps.get(1))
            .map(|m| m.as_str().trim().to_string())
        {
            return Some(value);
        }
    }
    None
}

pub fn extract_session_purchase_item(line: &str, lower: &str) -> Option<String> {
    extract_phrase_after_any_index(
        line,
        lower,
        &[
            "purchased a ",
            "purchased an ",
            "bought a ",
            "bought an ",
            "picked up a ",
            "picked up an ",
            "got a ",
            "got an ",
        ],
        &[" for ", " with ", " because ", " and ", " but ", "."],
        1,
    )
}

pub fn extract_title_like_phrases(text: &str) -> Vec<String> {
    const CONNECTORS: &[&str] = &[
        "of", "the", "and", "at", "in", "on", "to", "for", "dei", "del", "di", "du", "&", "+",
    ];
    let mut phrases = Vec::new();
    let mut current = Vec::new();
    let mut seen_title = false;

    for raw in text.split_whitespace() {
        let cleaned = raw.trim_matches(|c: char| {
            !c.is_ascii_alphanumeric() && !matches!(c, '&' | '+' | '\'' | '-')
        });
        if cleaned.is_empty() {
            continue;
        }
        let lower = cleaned.to_ascii_lowercase();
        let starts_upper = cleaned
            .chars()
            .next()
            .map(|c| c.is_ascii_uppercase())
            .unwrap_or(false);
        let short_acronym = cleaned.len() <= 5
            && cleaned
                .chars()
                .all(|c| c.is_ascii_uppercase() || c.is_ascii_digit() || matches!(c, '&' | '+'));
        let is_title = starts_upper || short_acronym;

        if is_title || (seen_title && CONNECTORS.contains(&lower.as_str())) {
            current.push(cleaned.to_string());
            if is_title {
                seen_title = true;
            }
            continue;
        }

        if seen_title && !current.is_empty() {
            let phrase = current.join(" ");
            if phrase.split_whitespace().count() <= 8 {
                phrases.push(phrase);
            }
        }
        current.clear();
        seen_title = false;
    }

    if seen_title && !current.is_empty() {
        let phrase = current.join(" ");
        if phrase.split_whitespace().count() <= 8 {
            phrases.push(phrase);
        }
    }

    phrases
}
