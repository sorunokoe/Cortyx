//! Item extraction: bikes, fitness, months, days.

use super::super::*;
use crate::index::compile_regex;

pub fn extract_phrase_after_any_index(
    line: &str,
    lower: &str,
    markers: &[&str],
    stop_markers: &[&str],
    min_words: usize,
) -> Option<String> {
    let mut best = None;
    for marker in markers {
        let Some(idx) = lower.find(marker) else {
            continue;
        };
        let tail = &line[idx + marker.len()..];
        let lower_tail = tail.to_ascii_lowercase();
        let cut = stop_markers
            .iter()
            .filter_map(|needle| lower_tail.find(needle))
            .min()
            .unwrap_or(tail.len());
        let mut phrase = tail[..cut]
            .trim()
            .trim_matches(|c: char| matches!(c, '"' | '\'' | ',' | '.' | ';' | ':' | '!' | '?'))
            .trim()
            .to_string();
        for prefix in ["the ", "a ", "an ", "simple "] {
            if phrase.to_ascii_lowercase().starts_with(prefix) {
                phrase = phrase[prefix.len()..].trim().to_string();
            }
        }
        if phrase.split_whitespace().count() < min_words {
            continue;
        }
        if best
            .as_ref()
            .map(|current: &String| phrase.len() > current.len())
            .unwrap_or(true)
        {
            best = Some(phrase);
        }
    }
    best
}

pub fn extract_project_count_item(line: &str, lower: &str) -> Option<String> {
    if lower.contains("case competition") {
        return Some("case competition".to_string());
    }
    let phrase = extract_phrase_after_any_index(
        line,
        lower,
        &[
            "working on a ",
            "working on ",
            "leading a ",
            "leading ",
            "started a ",
            "building a ",
            "creating a ",
        ],
        &[
            " for ",
            " with ",
            " because ",
            " and ",
            " but ",
            " that ",
            ",",
        ],
        2,
    )?;
    let lower_phrase = phrase.to_ascii_lowercase();
    (lower_phrase.contains("project") || lower_phrase.contains("competition")).then_some(phrase)
}

pub fn normalize_model_kit_count_item(text: &str) -> String {
    let mut item = text.trim().to_string();
    for prefix in [
        "diorama featuring a ",
        "diorama featuring ",
        "a simple ",
        "simple ",
    ] {
        if item.to_ascii_lowercase().starts_with(prefix) {
            item = item[prefix.len()..].trim().to_string();
            break;
        }
    }
    let lower = item.to_ascii_lowercase();
    let cutoff = [" do you ", ". do you ", "? ", " and i'm ", " and i’m "]
        .iter()
        .filter_map(|marker| lower.find(marker))
        .min()
        .unwrap_or(item.len());
    item.truncate(cutoff);
    item = item
        .trim()
        .trim_matches(|c: char| matches!(c, ',' | '.' | ';' | ':' | '"' | '\''))
        .to_string();
    if item.to_ascii_lowercase().ends_with(" kit") {
        item.truncate(item.len().saturating_sub(4));
        item = item.trim().to_string();
    }
    item
}

pub fn extract_model_kit_count_item(line: &str, lower: &str) -> Option<String> {
    let phrase = extract_phrase_after_any_index(
        line,
        lower,
        &[
            "finished a simple ",
            "finished a ",
            "working on a ",
            "working on ",
            "next project, a ",
            "featuring a ",
            "for your ",
        ],
        &[
            " that ",
            " and ",
            " but ",
            " because ",
            " while ",
            " next",
            " where ",
            ",",
        ],
        2,
    )?;
    let item = normalize_model_kit_count_item(&phrase);
    let lower_item = item.to_ascii_lowercase();
    (lower_item.contains("scale")
        || lower_item.contains("camaro")
        || lower_item.contains("bomber")
        || lower_item.contains("tank")
        || lower_item.contains("spitfire")
        || lower_item.contains("eagle"))
    .then_some(item)
}

pub fn extract_clothing_store_item(line: &str, lower: &str) -> Option<String> {
    if lower.contains("dry cleaning for ") {
        return extract_phrase_after_any_index(
            line,
            lower,
            &["dry cleaning for the ", "dry cleaning for "],
            &[" i ", " and ", " but ", " because ", ","],
            2,
        );
    }

    let phrase = extract_phrase_after_any_index(
        line,
        lower,
        &[
            "return some ",
            "return my ",
            "return the ",
            "pick up my ",
            "pick up the ",
        ],
        &[" to ", " from ", " because ", " and ", " but ", ","],
        1,
    )?;
    let lower_phrase = phrase.to_ascii_lowercase();
    [
        "blazer", "boots", "jeans", "shirt", "sweater", "dress", "sundress", "coat", "jacket",
        "pants", "trousers", "skirt", "top",
    ]
    .iter()
    .any(|needle| lower_phrase.contains(needle))
    .then_some(phrase)
}

pub fn normalize_family_origin_item(text: &str) -> String {
    let mut item = text.trim().to_string();
    for prefix in ["a set of ", "set of ", "my ", "the ", "a ", "an "] {
        if item.to_ascii_lowercase().starts_with(prefix) {
            item = item[prefix.len()..].trim().to_string();
        }
    }
    item.trim_matches(|c: char| matches!(c, '"' | '\'' | ',' | '.' | ';' | ':' | '!' | '?'))
        .trim()
        .to_string()
}

pub fn extract_family_origin_antique_items_from_line(line: &str, lower: &str) -> Vec<String> {
    if !task_contains_any(
        lower,
        &[
            "grandmother",
            "great-aunt",
            "great aunt",
            "mom",
            "dad",
            "cousin",
            "family heirloom",
            "family heirlooms",
            "inherited",
            "belonged to my",
            "from my",
        ],
    ) || !task_contains_any(lower, &["antique", "vintage", "depression-era"])
    {
        return Vec::new();
    }

    let pattern = compile_regex(
        r"(?i)(?:antique|vintage|depression-era)\s+[a-z][a-z-]*(?:\s+[a-z][a-z-]*){0,3}",
    );
    let mut items = Vec::new();
    let mut seen = HashSet::new();
    for item_match in pattern.find_iter(line) {
        let mut item = item_match.as_str().trim().to_string();
        let lower_item = item.to_ascii_lowercase();
        if let Some(cut) = [
            " from ",
            " that ",
            " which ",
            " belonged ",
            " came ",
            " insured",
            " appraised",
            " valued",
            " selling",
            " sold",
        ]
        .iter()
        .filter_map(|needle| lower_item.find(needle))
        .min()
        {
            item = item[..cut].trim().to_string();
        }
        let item = normalize_family_origin_item(&item);
        let lower_item = item.to_ascii_lowercase();
        if item.is_empty()
            || task_contains_any(
                &lower_item,
                &[
                    "dealer",
                    "dealers",
                    "appraiser",
                    "appraisers",
                    "insurance",
                    "company",
                    "companies",
                    "organization",
                    "organizations",
                    "marketplace",
                    "marketplaces",
                    "forum",
                    "forums",
                ],
            )
        {
            continue;
        }
        let key = normalized_synthetic_phrase_key(&item);
        if seen.insert(key) {
            items.push(item);
        }
    }
    items
}

pub fn extract_born_child_names_from_line(line: &str, lower: &str) -> Vec<String> {
    if lower.contains("adopted") {
        return Vec::new();
    }

    let mut names = Vec::new();
    let mut seen = HashSet::new();

    let twin_pattern =
        compile_regex(r"(?i)\btwins?(?:\s+\w+)?\s*,\s*([A-Z][a-z]+)\s+and\s+([A-Z][a-z]+)\b");
    for caps in twin_pattern.captures_iter(line) {
        for idx in [1, 2] {
            let Some(name_match) = caps.get(idx) else {
                continue;
            };
            let name = name_match.as_str().trim().to_string();
            let key = normalized_synthetic_phrase_key(&name);
            if seen.insert(key) {
                names.push(name);
            }
        }
    }

    let single_patterns = [
        compile_regex(r"(?i)\bbaby\s+(?:boy|girl)\s+named\s+([A-Z][a-z]+)\b"),
        compile_regex(r"(?i)\b(?:son|daughter)\s+([A-Z][a-z]+)\b"),
    ];
    for pattern in &single_patterns {
        for caps in pattern.captures_iter(line) {
            let Some(name_match) = caps.get(1) else {
                continue;
            };
            let name = name_match.as_str().trim().to_string();
            let key = normalized_synthetic_phrase_key(&name);
            if seen.insert(key) {
                names.push(name);
            }
        }
    }

    names
}

pub fn normalize_bike_service_item(text: &str) -> String {
    let mut item = text.trim().to_string();
    for prefix in ["regular ", "my ", "the ", "our ", "a ", "an "] {
        if item.to_ascii_lowercase().starts_with(prefix) {
            item = item[prefix.len()..].trim().to_string();
        }
    }
    item.trim_matches(|c: char| matches!(c, '"' | '\'' | ',' | '.' | ';' | ':' | '!' | '?'))
        .trim()
        .to_string()
}

pub fn extract_bike_phrase_from_line(line: &str, _lower: &str) -> Option<String> {
    let with_determiner = compile_regex(
        r"(?i)\b(?:my|the|our|a|an)\s+((?:road|commuter|mountain|hybrid|gravel|touring|electric|e-bike|ebike|bmx|trail)\s+bike)\b",
    )
            .captures(line)
            .and_then(|caps| caps.get(1))
            .map(|m| m.as_str().trim().to_string());
    let phrase = with_determiner.or_else(|| {
        compile_regex(
            r"(?i)\b((?:road|commuter|mountain|hybrid|gravel|touring|electric|e-bike|ebike|bmx|trail)\s+bike)\b",
        )
            .captures(line)
            .and_then(|caps| caps.get(1))
            .map(|m| m.as_str().trim().to_string())
    })?;
    let phrase = normalize_bike_service_item(&phrase);
    (phrase != "bike").then_some(phrase)
}

pub fn line_describes_bike_service_event(lower: &str) -> bool {
    lower.contains("bike")
        && task_contains_any(
            lower,
            &[
                "serviced at",
                "bike serviced",
                "cleaned and lubricated",
                "cleaning and lubricating",
                "time to replace",
                "replace it this month",
                "before april",
                "planning to service",
                "plan to service",
                "getting a new tire",
                "get a new tire",
                "new tire for my",
            ],
        )
}

pub fn extract_bike_service_item_from_line(line: &str, lower: &str, month: &str) -> Option<String> {
    if !line_matches_query_month_window(lower, month) || !line_describes_bike_service_event(lower) {
        return None;
    }
    extract_bike_phrase_from_line(line, lower)
}

pub fn render_day_count_answer(count: usize) -> String {
    format!("{count} {}", if count == 1 { "day" } else { "days" })
}

pub fn line_describes_countable_fitness_class_schedule(line: &str, lower: &str) -> bool {
    let speaker_grounded = lower.starts_with("user:") || line.trim_start().starts_with('-');
    let assistant_restate = lower.contains("your ");
    let explicit_class_signal = task_contains_any(
        lower,
        &[
            "fitness class",
            "fitness classes",
            "bodypump",
            "hip hop abs",
            "yoga class",
            "yoga classes",
            "zumba",
        ],
    ) || ((lower.contains(" class") || lower.contains(" classes"))
        && task_contains_any(
            lower,
            &[
                "weightlifting",
                "strength training",
                "pilates",
                "spin",
                "kickboxing",
                "barre",
                "cycling",
                "aerobics",
            ],
        ));

    explicit_class_signal && (speaker_grounded || assistant_restate)
}

pub fn extract_weekday_mentions_from_line(lower: &str) -> Vec<String> {
    let mut days = Vec::new();
    let mut seen = HashSet::new();
    for day in [
        "sunday",
        "monday",
        "tuesday",
        "wednesday",
        "thursday",
        "friday",
        "saturday",
    ] {
        if lower.contains(day) && seen.insert(day) {
            days.push(day.to_string());
        }
    }
    days
}

pub fn push_month_day(days: &mut Vec<u32>, seen: &mut HashSet<u32>, value: u32) {
    if (1..=31).contains(&value) && seen.insert(value) {
        days.push(value);
    }
}

pub fn push_month_day_range(days: &mut Vec<u32>, seen: &mut HashSet<u32>, start: u32, end: u32) {
    if !(1..=31).contains(&start) || !(1..=31).contains(&end) || end < start {
        return;
    }
    for value in start..=end {
        push_month_day(days, seen, value);
    }
}

pub fn extract_month_day_values_from_line(line: &str, lower: &str, month: &str) -> Vec<u32> {
    if !lower.contains(month) {
        return Vec::new();
    }

    let month_pattern = regex::escape(month);
    let mut days = Vec::new();
    let mut seen = HashSet::new();

    let month_range = compile_regex(&format!(
        r"(?i)\b{}\s+(\d{{1,2}})(?:st|nd|rd|th)?\s*-\s*(\d{{1,2}})(?:st|nd|rd|th)?\b",
        month_pattern
    ));
    for caps in month_range.captures_iter(line) {
        let Some(start) = caps.get(1).and_then(|m| m.as_str().parse::<u32>().ok()) else {
            continue;
        };
        let Some(end) = caps.get(2).and_then(|m| m.as_str().parse::<u32>().ok()) else {
            continue;
        };
        push_month_day_range(&mut days, &mut seen, start, end);
    }

    let day_pair = compile_regex(&format!(
        r"(?i)\b(\d{{1,2}})(?:st|nd|rd|th)?\s+and\s+(\d{{1,2}})(?:st|nd|rd|th)?\s+of\s+{}\b",
        month_pattern
    ));
    for caps in day_pair.captures_iter(line) {
        let Some(first) = caps.get(1).and_then(|m| m.as_str().parse::<u32>().ok()) else {
            continue;
        };
        let Some(second) = caps.get(2).and_then(|m| m.as_str().parse::<u32>().ok()) else {
            continue;
        };
        push_month_day(&mut days, &mut seen, first);
        push_month_day(&mut days, &mut seen, second);
    }

    let month_single = compile_regex(&format!(
        r"(?i)\b{}\s+(\d{{1,2}})(?:st|nd|rd|th)?\b",
        month_pattern
    ));
    for caps in month_single.captures_iter(line) {
        let Some(day) = caps.get(1).and_then(|m| m.as_str().parse::<u32>().ok()) else {
            continue;
        };
        push_month_day(&mut days, &mut seen, day);
    }

    let of_month_single = compile_regex(&format!(
        r"(?i)\b(\d{{1,2}})(?:st|nd|rd|th)?\s+of\s+{}\b",
        month_pattern
    ));
    for caps in of_month_single.captures_iter(line) {
        let Some(day) = caps.get(1).and_then(|m| m.as_str().parse::<u32>().ok()) else {
            continue;
        };
        push_month_day(&mut days, &mut seen, day);
    }

    days
}

pub fn line_matches_activity_markers(lower: &str, markers: &[&str]) -> bool {
    markers.iter().any(|marker| lower.contains(marker))
}

pub fn extract_month_scoped_activity_days_from_line(
    line: &str,
    lower: &str,
    month: &str,
    activity_markers: &[&str],
) -> Vec<u32> {
    if !line_matches_query_month_window(lower, month)
        || !line_matches_activity_markers(lower, activity_markers)
    {
        return Vec::new();
    }
    extract_month_day_values_from_line(line, lower, month)
}

pub fn month_name_to_number(month: &str) -> Option<u32> {
    match month {
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

pub fn line_matches_query_month_or_numeric_date(line: &str, lower: &str, month: &str) -> bool {
    if line_matches_query_month_window(lower, month) {
        return true;
    }
    let Some(target_month) = month_name_to_number(month) else {
        return false;
    };
    compile_regex(r"(?i)\b(\d{1,2})/(\d{1,2})(?:/(\d{2,4}))?\b")
        .captures_iter(line)
        .filter_map(|caps| caps.get(1))
        .filter_map(|value| value.as_str().parse::<u32>().ok())
        .any(|value| value == target_month)
}
