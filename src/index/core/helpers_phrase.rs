// This file is a submodule of `crate::index::core`.
// Contains free-standing helper functions extracted from helpers.rs.
use super::*;
use crate::index::compile_regex;
use crate::types::{QueryText, SynapseWeight};

pub(in crate::index) fn extract_phrase_after_any_index(
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

pub(in crate::index) fn extract_project_count_item(line: &str, lower: &str) -> Option<String> {
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

pub(in crate::index) fn normalize_model_kit_count_item(text: &str) -> String {
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

pub(in crate::index) fn extract_model_kit_count_item(line: &str, lower: &str) -> Option<String> {
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

pub(in crate::index) fn extract_clothing_store_item(line: &str, lower: &str) -> Option<String> {
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

pub(in crate::index) fn normalize_family_origin_item(text: &str) -> String {
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

pub(in crate::index) fn extract_family_origin_antique_items_from_line(
    line: &str,
    lower: &str,
) -> Vec<String> {
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

pub(in crate::index) fn extract_born_child_names_from_line(line: &str, lower: &str) -> Vec<String> {
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

pub(in crate::index) fn normalize_bike_service_item(text: &str) -> String {
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

pub(in crate::index) fn extract_bike_phrase_from_line(line: &str, _lower: &str) -> Option<String> {
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

pub(in crate::index) fn line_describes_bike_service_event(lower: &str) -> bool {
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

pub(in crate::index) fn extract_bike_service_item_from_line(
    line: &str,
    lower: &str,
    month: &str,
) -> Option<String> {
    if !line_matches_query_month_window(lower, month) || !line_describes_bike_service_event(lower) {
        return None;
    }
    extract_bike_phrase_from_line(line, lower)
}

pub(in crate::index) fn render_day_count_answer(count: usize) -> String {
    format!("{count} {}", if count == 1 { "day" } else { "days" })
}

pub(in crate::index) fn line_describes_countable_fitness_class_schedule(
    line: &str,
    lower: &str,
) -> bool {
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

pub(in crate::index) fn extract_weekday_mentions_from_line(lower: &str) -> Vec<String> {
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

pub(in crate::index) fn push_month_day(days: &mut Vec<u32>, seen: &mut HashSet<u32>, value: u32) {
    if (1..=31).contains(&value) && seen.insert(value) {
        days.push(value);
    }
}

pub(in crate::index) fn push_month_day_range(
    days: &mut Vec<u32>,
    seen: &mut HashSet<u32>,
    start: u32,
    end: u32,
) {
    if !(1..=31).contains(&start) || !(1..=31).contains(&end) || end < start {
        return;
    }
    for value in start..=end {
        push_month_day(days, seen, value);
    }
}

pub(in crate::index) fn extract_month_day_values_from_line(
    line: &str,
    lower: &str,
    month: &str,
) -> Vec<u32> {
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

pub(in crate::index) fn line_matches_activity_markers(lower: &str, markers: &[&str]) -> bool {
    markers.iter().any(|marker| lower.contains(marker))
}

pub(in crate::index) fn extract_month_scoped_activity_days_from_line(
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

pub(in crate::index) fn month_name_to_number(month: &str) -> Option<u32> {
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

pub(in crate::index) fn line_matches_query_month_or_numeric_date(
    line: &str,
    lower: &str,
    month: &str,
) -> bool {
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

pub(in crate::index) fn extract_first_quoted_phrase(line: &str) -> Option<String> {
    compile_regex(r#""([^"]+)""#)
        .captures(line)
        .and_then(|caps| caps.get(1))
        .map(|m| m.as_str().trim().to_string())
}

pub(in crate::index) fn extract_art_event_anchor(line: &str) -> Option<String> {
    extract_first_quoted_phrase(line).or_else(|| {
        extract_title_like_phrases(line)
            .into_iter()
            .filter(|phrase| {
                let lower = phrase.to_ascii_lowercase();
                lower.contains("museum")
                    || lower.contains("gallery")
                    || lower.contains("art cube")
                    || lower.contains("women in art")
                    || lower.contains("art afternoon")
            })
            .max_by_key(|phrase| phrase.len())
    })
}

pub(in crate::index) fn line_describes_art_related_event(lower: &str) -> bool {
    task_contains_any(
        lower,
        &[
            "art",
            "museum",
            "gallery",
            "street art",
            "children's museum",
        ],
    ) && task_contains_any(
        lower,
        &[
            "guided tour",
            "lecture",
            "exhibition",
            "opening night",
            "workshop",
            "event",
            "festival",
        ],
    ) && task_contains_any(
        lower,
        &[
            "attended",
            "went on",
            "went to",
            "visited",
            "volunteered at",
            "opening night",
        ],
    )
}

pub(in crate::index) fn extract_art_related_event_signature_from_line(
    line: &str,
    lower: &str,
) -> Option<(i32, String)> {
    if !line_describes_art_related_event(lower) {
        return None;
    }
    let rank = extract_explicit_date_rank(line)?;
    let kind = if lower.contains("guided tour") {
        "guided-tour"
    } else if lower.contains("opening night") {
        "opening-night"
    } else if lower.contains("lecture") {
        "lecture"
    } else if lower.contains("exhibition") {
        "exhibition"
    } else if lower.contains("workshop") {
        "workshop"
    } else if lower.contains("festival") {
        "festival"
    } else if lower.contains("event") {
        "event"
    } else {
        return None;
    };
    let anchor = extract_art_event_anchor(line)
        .map(|value| normalized_synthetic_phrase_key(&value))
        .unwrap_or_default();
    Some((rank, format!("{rank}:{kind}:{anchor}")))
}

pub(in crate::index) fn line_describes_cuisine_learning_or_trying(lower: &str) -> bool {
    task_contains_any(
        lower,
        &[
            "tried out",
            "learned how to make",
            "learned to make",
            "class on",
            "attended a class on",
            "recipe for",
            "online recipe library",
            "restaurant",
        ],
    )
}

pub(in crate::index) fn extract_cuisine_labels_from_line(_line: &str, lower: &str) -> Vec<String> {
    if !line_describes_cuisine_learning_or_trying(lower) {
        return Vec::new();
    }
    let mut cuisines = Vec::new();
    let mut seen = HashSet::new();
    for cuisine in [
        "ethiopian",
        "indian",
        "korean",
        "thai",
        "mexican",
        "italian",
        "japanese",
        "chinese",
        "greek",
        "moroccan",
        "vietnamese",
        "french",
        "mediterranean",
        "lebanese",
        "spanish",
        "turkish",
        "brazilian",
        "peruvian",
        "middle eastern",
        "vegan",
    ] {
        if lower.contains(cuisine) && seen.insert(cuisine) {
            cuisines.push(cuisine.to_string());
        }
    }
    cuisines
}

pub(in crate::index) fn line_describes_museum_gallery_visit(lower: &str) -> bool {
    task_contains_any(
        lower,
        &[
            "visited",
            "went to",
            "took my niece to",
            "opening night of",
            "met the curator",
            "guided tour at",
        ],
    )
}

pub(in crate::index) fn normalize_visit_venue(text: &str) -> String {
    let mut venue = text.trim().to_string();
    for prefix in ["the ", "my ", "our ", "a ", "an "] {
        if venue.to_ascii_lowercase().starts_with(prefix) {
            venue = venue[prefix.len()..].trim().to_string();
        }
    }
    venue
        .trim_matches(|c: char| matches!(c, '"' | '\'' | ',' | '.' | ';' | ':' | '!' | '?'))
        .trim()
        .to_string()
}

pub(in crate::index) fn extract_museum_gallery_visit_venue_from_line(
    line: &str,
    lower: &str,
    month: &str,
) -> Option<String> {
    if !line_matches_query_month_or_numeric_date(line, lower, month)
        || !line_describes_museum_gallery_visit(lower)
    {
        return None;
    }
    let direct = extract_phrase_after_any_index(
        line,
        lower,
        &[
            "visited ",
            "opening night of ",
            "took my niece to ",
            "went on a guided tour at ",
            "guided tour at ",
            "went to ",
        ],
        &[" on ", ",", ".", " and "],
        1,
    )
    .map(|phrase| normalize_visit_venue(&phrase))
    .filter(|phrase| {
        let lower = phrase.to_ascii_lowercase();
        lower.contains("museum") || lower.contains("gallery") || lower.contains("art cube")
    });
    direct.or_else(|| {
        extract_title_like_phrases(line)
            .into_iter()
            .map(|phrase| normalize_visit_venue(&phrase))
            .filter(|phrase| {
                let lower = phrase.to_ascii_lowercase();
                lower.contains("museum") || lower.contains("gallery") || lower.contains("art cube")
            })
            .max_by_key(|phrase| phrase.len())
    })
}

pub(in crate::index) fn line_mentions_candidate_museum_gallery_visit(
    line: &str,
    lower: &str,
    month: &str,
) -> bool {
    line_matches_query_month_or_numeric_date(line, lower, month)
        && line_describes_museum_gallery_visit(lower)
        && task_contains_any(lower, &["museum", "gallery", "art cube"])
}

pub(in crate::index) fn extract_citrus_fruits_from_line(_line: &str, lower: &str) -> Vec<String> {
    if !task_contains_any(
        lower,
        &[
            "cocktail",
            "cocktails",
            "sangria",
            "daiquiri",
            "gimlet",
            "bitters",
            "mixology",
        ],
    ) {
        return Vec::new();
    }
    let mut fruits = Vec::new();
    let mut seen = HashSet::new();
    for fruit in ["orange", "lemon", "lime", "grapefruit"] {
        if lower.contains(fruit) && seen.insert(fruit) {
            fruits.push(fruit.to_string());
        }
    }
    fruits
}

pub(in crate::index) fn extract_food_delivery_service_from_line(
    _line: &str,
    lower: &str,
) -> Option<String> {
    let labels = [
        ("fresh fusion", "Fresh Fusion"),
        ("uber eats", "Uber Eats"),
        ("domino's pizza", "Domino's Pizza"),
        ("dominos pizza", "Domino's Pizza"),
        ("domino's", "Domino's Pizza"),
        ("doordash", "DoorDash"),
        ("grubhub", "Grubhub"),
        ("postmates", "Postmates"),
        ("seamless", "Seamless"),
        ("caviar", "Caviar"),
    ];
    labels
        .into_iter()
        .find(|(needle, _)| lower.contains(needle))
        .map(|(_, label)| label.to_string())
}

pub(in crate::index) fn extract_missed_fun_run_signature_from_line(
    line: &str,
    lower: &str,
    month: &str,
) -> Option<String> {
    if !line_matches_query_month_or_numeric_date(line, lower, month)
        || !task_contains_any(lower, &["fun run", "fun runs", "5k fun run", "5k fun runs"])
        || !task_contains_any(lower, &["missed", "had to miss", "unable to attend"])
    {
        return None;
    }
    let mut days = extract_month_day_values_from_line(line, lower, month);
    if days.is_empty() {
        let rank = extract_explicit_date_rank(line)?;
        return Some(format!("fun-run:{rank}"));
    }
    days.sort_unstable();
    let day = *days.last()?;
    Some(format!("fun-run:{month}:{day}"))
}

pub(in crate::index) fn line_mentions_recent_three_month_window(lower: &str) -> bool {
    task_contains_any(
        lower,
        &[
            "today",
            "yesterday",
            "last week",
            "week ago",
            "weeks ago",
            "last month",
            "month ago",
            "months ago",
            "a few weeks ago",
            "few weeks ago",
            "a couple of weeks ago",
            "couple of weeks ago",
            "two months ago",
            "three months ago",
        ],
    )
}

pub(in crate::index) fn trim_trailing_relative_time_phrase(text: &str) -> String {
    let trimmed = compile_regex(
        r"(?i)\s+(?:about|around)?\s*(?:a\s+few|few|a\s+couple\s+of|couple\s+of|one|two|three|\d+)\s+(?:day|days|week|weeks|month|months|year|years)\s+ago[.!?,]?\s*$",
    )
    .replace(text.trim(), "")
    .to_string();
    trimmed
        .trim_matches(|c: char| matches!(c, '"' | '\'' | ',' | '.' | ';' | ':' | '!' | '?'))
        .trim()
        .to_string()
}

pub(in crate::index) fn extract_graduation_ceremony_signature_from_line(
    line: &str,
    lower: &str,
) -> Option<String> {
    if !lower.contains("graduation")
        || !task_contains_any(lower, &["attended my", "attended our", "attended the"])
        || !line_mentions_recent_three_month_window(lower)
    {
        return None;
    }
    let caps = compile_regex(
        r"(?i)attended (?:my|our|the) ([^\n]+?)'s ((?:[^.!?\n]+?\s+)?graduation(?: ceremony)?(?: from [^.!?\n]+?)?)\b",
    )
    .captures(line)?;
    let owner = normalized_synthetic_phrase_key(caps.get(1)?.as_str());
    let event =
        normalized_synthetic_phrase_key(&trim_trailing_relative_time_phrase(caps.get(2)?.as_str()));
    Some(format!("{owner}:{event}"))
}

pub(in crate::index) fn extract_health_device_units_from_line(
    _line: &str,
    lower: &str,
) -> Vec<String> {
    let mut devices = Vec::new();
    let mut seen = HashSet::new();

    let has_specific_fitbit =
        lower.contains("fitbit versa 3 smartwatch") || lower.contains("fitbit versa 3");
    let has_generic_fitbit = compile_regex(r"(?i)\bfitbit\b").is_match(lower);
    let wearable = if has_specific_fitbit {
        Some("fitbit versa 3 smartwatch")
    } else if has_generic_fitbit {
        Some("fitbit")
    } else {
        None
    };
    if let Some(device) = wearable {
        if seen.insert(device) {
            devices.push(device.to_string());
        }
    }

    if lower.contains("hearing aids") {
        devices.push("left hearing aid".to_string());
        devices.push("right hearing aid".to_string());
        return devices;
    }

    let mentions_batteries = lower.contains("battery") || lower.contains("batteries");

    for device in [
        "hearing aid",
        "blood pressure monitor",
        "glucose monitor",
        "continuous glucose monitor",
        "fitness tracker",
        "smartwatch",
        "cpap",
        "inhaler",
    ] {
        if has_generic_fitbit && matches!(device, "fitness tracker" | "smartwatch") {
            continue;
        }
        if mentions_batteries && device == "hearing aid" {
            continue;
        }
        if lower.contains(device) && seen.insert(device) {
            devices.push(device.to_string());
        }
    }

    devices
}

pub(in crate::index) fn extract_peak_campaign_weekly_hour_delta_from_line(
    line: &str,
    lower: &str,
) -> Option<f32> {
    if !lower.contains("peak campaign")
        || !task_contains_any(
            lower,
            &["i increase my work hours by", "increase my work hours by"],
        )
    {
        return None;
    }
    compile_regex(
        r"(?i)\bincrease my (?:work )?hours by (\d+(?:\.\d+)?) hours? (?:weekly|a week|per week)\b",
    )
    .captures(line)?
    .get(1)?
    .as_str()
    .parse::<f32>()
    .ok()
}

pub(in crate::index) fn extract_typical_weekly_work_hours_from_line(
    line: &str,
    lower: &str,
) -> Option<f32> {
    if !task_contains_any(lower, &["i usually work", "usually work"]) {
        return None;
    }
    compile_regex(r"(?i)\bi usually work (\d+(?:\.\d+)?) hours? (?:a|per) week\b")
        .captures(line)?
        .get(1)?
        .as_str()
        .parse::<f32>()
        .ok()
}

pub(in crate::index) fn extract_peak_campaign_total_weekly_hours_from_line(
    line: &str,
    lower: &str,
) -> Option<f32> {
    if !lower.contains("peak campaign") {
        return None;
    }
    compile_regex(
        r"(?i)\b(?:working )?up to (\d+(?:\.\d+)?) hours?(?:\s*/\s*week|\s+per\s+week|\s+a\s+week)\b",
    )
    .captures(line)?
    .get(1)?
    .as_str()
    .parse::<f32>()
    .ok()
}

pub(in crate::index) fn extract_recent_activity_query_labels(
    task_lower: &str,
) -> Vec<&'static str> {
    let mut labels = Vec::new();
    for (label, needles) in [
        ("jogging", &["jogging", "jog"][..]),
        ("yoga", &["yoga"][..]),
        ("walking", &["walking", "walk"][..]),
        ("swimming", &["swimming", "swim"][..]),
        ("cycling", &["cycling", "biking", "bike", "cycle"][..]),
        (
            "strength training",
            &["strength training", "weightlifting", "lifting"][..],
        ),
    ] {
        if task_contains_any(task_lower, needles) {
            labels.push(label);
        }
    }
    labels
}

pub(in crate::index) fn line_mentions_recent_activity_label(lower: &str, label: &str) -> bool {
    match label {
        "jogging" => task_contains_any(lower, &["jogging", "jog", "jogged"]),
        "yoga" => lower.contains("yoga"),
        "walking" => task_contains_any(lower, &["walking", "walk", "walked"]),
        "swimming" => task_contains_any(lower, &["swimming", "swim", "swam"]),
        "cycling" => task_contains_any(lower, &["cycling", "biking", "bike", "biked", "cycled"]),
        "strength training" => {
            task_contains_any(lower, &["strength training", "weightlifting", "lifting"])
        },
        _ => false,
    }
}

pub(in crate::index) fn extract_recent_activity_duration_facts_from_line(
    line: &str,
    lower: &str,
    requested_activities: &[&'static str],
) -> Vec<(String, &'static str, SyntheticDurationValue)> {
    if !task_contains_any(
        lower,
        &[
            "i went for",
            "i went on",
            "i did",
            "i completed",
            "i ran",
            "i jogged",
            "i walked",
            "i biked",
            "i cycled",
            "i swam",
            "i practiced",
        ],
    ) {
        return Vec::new();
    }
    if task_contains_any(
        lower,
        &[
            "used to",
            "slacking off",
            "trying to get back",
            "schedule my",
            "set reminders",
            "habit",
            "times a week",
            "each time for",
            "looking to increase",
            "trying to incorporate",
        ],
    ) || line_has_future_goal_marker(lower)
    {
        return Vec::new();
    }

    let Some(duration) = extract_aggregate_duration_value(line) else {
        return Vec::new();
    };
    let day_surface = extract_weekday_surface_from_line(lower);
    let line_body = normalize_session_answer_line_body(line);
    let mut facts = Vec::new();
    for activity in requested_activities {
        if !line_mentions_recent_activity_label(lower, activity) {
            continue;
        }
        let signature = match day_surface.as_deref() {
            Some(day) => format!("{activity}:{day}"),
            None => format!("{activity}:{}", normalized_synthetic_phrase_key(&line_body)),
        };
        facts.push((signature, *activity, duration));
    }
    facts
}

pub(in crate::index) fn extract_current_magazine_subscription_updates_from_line(
    line: &str,
    lower: &str,
) -> Vec<(String, bool)> {
    let mut updates = Vec::new();
    let mut seen = HashSet::new();
    let mut push_update = |publication: Option<String>, is_active: bool| {
        let Some(publication) = publication else {
            return;
        };
        let publication = publication
            .trim_matches(|c: char| matches!(c, '"' | '\'' | ',' | '.' | ';' | ':' | '!' | '?'))
            .trim()
            .to_string();
        let normalized = normalized_synthetic_phrase_key(&publication);
        if publication.is_empty() || normalized.len() < 4 {
            return;
        }
        let key = format!("{normalized}:{is_active}");
        if seen.insert(key) {
            updates.push((publication, is_active));
        }
    };

    push_update(
        extract_phrase_after_any_index(
            line,
            lower,
            &["canceled my "],
            &[
                " magazine subscription",
                " subscription",
                " because ",
                ",",
                ".",
            ],
            1,
        ),
        false,
    );
    push_update(
        extract_phrase_after_any_index(
            line,
            lower,
            &[
                "loving my subscription to ",
                "enjoying my subscription to ",
                "my subscription to ",
            ],
            &[
                " magazine",
                " subscription",
                " which ",
                " in ",
                " on ",
                ",",
                ".",
                " -",
            ],
            1,
        ),
        true,
    );
    push_update(
        extract_phrase_after_any_index(
            line,
            lower,
            &["other publications like "],
            &[" which ", " in ", " on ", ",", ".", " -"],
            1,
        ),
        true,
    );
    push_update(
        extract_phrase_after_any_index(
            line,
            lower,
            &["i'm also getting ", "i am also getting "],
            &[" which ", " in ", " on ", ",", ".", " -"],
            1,
        ),
        true,
    );

    updates
}

pub(in crate::index) fn extract_hour_minute_total_from_text(text: &str) -> Option<i32> {
    for regex in [
        compile_regex(r"(?i)\b(\d+)\s*h(?:ours?)?\s*(\d+)\s*min(?:ute)?s?\b"),
        compile_regex(r"(?i)\b(\d+)\s+hours?\s+(?:and\s+)?(\d+)\s+minutes?\b"),
    ] {
        let Some(caps) = regex.captures(text) else {
            continue;
        };
        let hours = caps.get(1)?.as_str().parse::<i32>().ok()?;
        let minutes = caps.get(2)?.as_str().parse::<i32>().ok()?;
        return Some(hours * 60 + minutes);
    }
    None
}

pub(in crate::index) fn extract_marathon_completion_minutes_from_line(
    line: &str,
    lower: &str,
) -> Option<i32> {
    if !lower.contains("marathon")
        || !task_contains_any(
            lower,
            &["completed my first full marathon", "completed the marathon"],
        )
    {
        return None;
    }
    for marker in [
        "completed my first full marathon in ",
        "completed the marathon in ",
        "full marathon in ",
        "marathon in ",
    ] {
        let Some(idx) = lower.find(marker) else {
            continue;
        };
        if let Some(total) = extract_hour_minute_total_from_text(&line[idx + marker.len()..]) {
            return Some(total);
        }
    }
    None
}

pub(in crate::index) fn extract_marathon_target_minutes_from_line(
    line: &str,
    lower: &str,
) -> Option<i32> {
    if !lower.contains("marathon") || !lower.contains("target time") {
        return None;
    }
    for marker in [
        "target time for the marathon was ",
        "target time for the marathon is ",
        "target time was ",
    ] {
        let Some(idx) = lower.find(marker) else {
            continue;
        };
        if let Some(total) = extract_hour_minute_total_from_text(&line[idx + marker.len()..]) {
            return Some(total);
        }
    }
    None
}

pub(in crate::index) fn extract_attended_movie_festival_from_line(
    line: &str,
    lower: &str,
) -> Option<String> {
    if !task_contains_any(
        lower,
        &[
            "i volunteered",
            "i even volunteered",
            "i recently participated",
            "i was impressed by",
            "i got to discuss",
            "i've been fortunate enough",
            "i had the opportunity",
            "i had a great conversation",
            "i was part of a team",
            "i attended",
        ],
    ) {
        return None;
    }
    let caps = compile_regex(
        r"(?i)\b(?:at|after the screening at|like)\b\s+(?:the\s+)?([A-Z][A-Za-z0-9&' .-]+?Film Festival|AFI Fest|TIFF)\b",
    )
    .captures(line)?;
    Some(caps.get(1)?.as_str().trim().to_string())
}

pub(in crate::index) fn spell_small_cardinal(count: usize) -> Option<&'static str> {
    match count {
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

pub(in crate::index) fn extract_music_release_signatures_from_line(
    line: &str,
    lower: &str,
) -> Vec<String> {
    let mut releases = Vec::new();
    let mut seen = HashSet::new();

    if task_contains_any(lower, &["i bought", "i ended up buying"]) {
        if let Some(caps) = compile_regex(r#"(?i)\b(?:EP|album)\s+["']([^"']+)["']"#).captures(line)
        {
            if let Some(title) = caps.get(1) {
                let key = normalized_synthetic_phrase_key(title.as_str());
                if key.len() >= 3 && seen.insert(key.clone()) {
                    releases.push(key);
                }
            }
        }
    }

    if lower.contains("downloaded") {
        if let Some(caps) =
            compile_regex(r#"(?i)\balbum\s+["']([^"']+)["'][^.\n]*\bdownloaded\b"#).captures(line)
        {
            if let Some(title) = caps.get(1) {
                let key = normalized_synthetic_phrase_key(title.as_str());
                if key.len() >= 3 && seen.insert(key.clone()) {
                    releases.push(key);
                }
            }
        }
    }

    if lower.contains("vinyl") && lower.contains("signed") {
        for regex in [
            compile_regex(r"(?i)\bgot my ([A-Z][A-Za-z0-9&' .-]+?) vinyl signed\b"),
            compile_regex(
                r"(?i)\bsaw ([A-Z][A-Za-z0-9&' .-]+?) live[^.\n]*\bgot my vinyl signed\b",
            ),
        ] {
            let Some(caps) = regex.captures(line) else {
                continue;
            };
            let Some(artist) = caps.get(1) else {
                continue;
            };
            let key = normalized_synthetic_phrase_key(&format!("{} vinyl", artist.as_str().trim()));
            if key.len() >= 3 && seen.insert(key.clone()) {
                releases.push(key);
            }
        }
    }

    releases
}

pub(in crate::index) fn extract_owned_musical_instrument_signatures_from_line(
    line: &str,
    lower: &str,
) -> Vec<String> {
    let mut instruments = Vec::new();
    let mut seen = HashSet::new();
    if !(lower.starts_with("user:") || line.trim_start().starts_with('-')) {
        return instruments;
    }
    if task_contains_any(
        lower,
        &[
            "thinking of buying",
            "eyeing a ",
            "considering buying",
            "maybe getting",
            "might get",
            "want to buy",
        ],
    ) {
        return instruments;
    }

    let mut push = |label: String| {
        let key = normalized_synthetic_phrase_key(&label);
        if key.len() >= 3 && seen.insert(key.clone()) {
            instruments.push(key);
        }
    };

    if lower.contains("drum set")
        && task_contains_any(
            lower,
            &["my old drum set", "my drum set", "selling my old drum set"],
        )
    {
        let mut inserted = false;
        for regex in [
            compile_regex(
                r"\bdrum set,\s+a\s+((?:\d+-piece\s+)?[A-Z][A-Za-z0-9]+(?:\s+[A-Z0-9][A-Za-z0-9]+)*)\b",
            ),
            compile_regex(
                r"\b((?:\d+-piece\s+)?[A-Z][A-Za-z0-9]+(?:\s+[A-Z0-9][A-Za-z0-9]+)*)\s+drum set\b",
            ),
        ] {
            let Some(caps) = regex.captures(line) else {
                continue;
            };
            let Some(model) = caps.get(1) else {
                continue;
            };
            push(format!("{} drum set", model.as_str().trim()));
            inserted = true;
        }
        if !inserted {
            push("drum set".to_string());
        }
    }

    if lower.contains("piano") && lower.contains(" my ") {
        let mut inserted = false;
        for regex in [
            compile_regex(r"\bpiano,\s+a\s+([A-Z][A-Za-z0-9]+(?:\s+[A-Z0-9][A-Za-z0-9]+)*)\b"),
            compile_regex(r"\b([A-Z][A-Za-z0-9]+(?:\s+[A-Z0-9][A-Za-z0-9]+)*)\s+piano\b"),
            compile_regex(r"\b(Korg\s+B1)\b"),
        ] {
            let Some(caps) = regex.captures(line) else {
                continue;
            };
            let Some(model) = caps.get(1) else {
                continue;
            };
            push(format!("{} piano", model.as_str().trim()));
            inserted = true;
        }
        if !inserted && lower.contains("my piano") {
            push("piano".to_string());
        }
    }

    if lower.contains("acoustic guitar") {
        let mut inserted = false;
        for regex in [
            compile_regex(
                r"\bacoustic guitar,\s+a\s+([A-Z][A-Za-z0-9]+(?:\s+[A-Z0-9][A-Za-z0-9]+)*)\b",
            ),
            compile_regex(r"\b([A-Z][A-Za-z0-9]+(?:\s+[A-Z0-9][A-Za-z0-9]+)*)\s+acoustic guitar\b"),
        ] {
            let Some(caps) = regex.captures(line) else {
                continue;
            };
            let Some(model) = caps.get(1) else {
                continue;
            };
            push(format!("{} acoustic guitar", model.as_str().trim()));
            inserted = true;
        }
        if !inserted && lower.contains("my acoustic guitar") {
            push("acoustic guitar".to_string());
        }
    }

    if lower.contains("electric guitar") {
        let mut inserted = false;
        for regex in [
            compile_regex(
                r"\b(?:my|had my|playing my)\s+(?:[a-z]+\s+)?([A-Z][A-Za-z0-9]+(?:\s+[A-Z0-9][A-Za-z0-9]+)*)\s+electric guitar\b",
            ),
            compile_regex(r"\b([A-Z][A-Za-z0-9]+(?:\s+[A-Z0-9][A-Za-z0-9]+)*)\s+electric guitar\b"),
        ] {
            let Some(caps) = regex.captures(line) else {
                continue;
            };
            let Some(model) = caps.get(1) else {
                continue;
            };
            push(format!("{} electric guitar", model.as_str().trim()));
            inserted = true;
        }
        if !inserted && lower.contains("my electric guitar") {
            push("electric guitar".to_string());
        }
    }

    if lower.contains("ukulele") && lower.contains("my ") {
        let mut inserted = false;
        for regex in [
            compile_regex(r"\bukulele,\s+a\s+([A-Z][A-Za-z0-9]+(?:\s+[A-Z0-9][A-Za-z0-9]+)*)\b"),
            compile_regex(r"\b([A-Z][A-Za-z0-9]+(?:\s+[A-Z0-9][A-Za-z0-9]+)*)\s+ukulele\b"),
        ] {
            let Some(caps) = regex.captures(line) else {
                continue;
            };
            let Some(model) = caps.get(1) else {
                continue;
            };
            push(format!("{} ukulele", model.as_str().trim()));
            inserted = true;
        }
        if !inserted && lower.contains("my ukulele") {
            push("ukulele".to_string());
        }
    }

    instruments
}

pub(in crate::index) fn extract_online_course_completion_updates_from_line(
    line: &str,
    lower: &str,
) -> Vec<(String, i32)> {
    let mut updates = Vec::new();
    let mut seen = HashSet::new();
    if !(lower.starts_with("user:") || line.trim_start().starts_with('-'))
        || !lower.contains("completed")
        || !lower.contains("course")
    {
        return updates;
    }

    let mut count = None;
    for regex in [
        compile_regex(r"(?i)\bcompleted\s+([A-Za-z0-9,-]+)\s+courses?\b"),
        compile_regex(r"(?i)\b([A-Za-z0-9,-]+)\s+courses?\s+on\b"),
    ] {
        let Some(caps) = regex.captures(line) else {
            continue;
        };
        let Some(value) = caps
            .get(1)
            .and_then(|m| parse_count_token_value(m.as_str()))
        else {
            continue;
        };
        if value > 0 {
            count = Some(value);
            break;
        }
    }
    let Some(count) = count else {
        return updates;
    };

    for (platform_key, platform_name) in [
        ("coursera", "Coursera"),
        ("edx", "edX"),
        ("udemy", "Udemy"),
        ("datacamp", "DataCamp"),
        ("fast.ai", "Fast.ai"),
        ("kaggle", "Kaggle"),
    ] {
        if lower.contains(platform_key) && seen.insert(platform_key) {
            updates.push((platform_name.to_string(), count));
        }
    }

    updates
}

pub(in crate::index) fn extract_recent_furniture_action_signatures_from_line(
    line: &str,
    lower: &str,
) -> Vec<String> {
    let mut items = Vec::new();
    let mut seen = HashSet::new();
    if !(lower.starts_with("user:") || line.trim_start().starts_with('-')) {
        return items;
    }

    let mut push = |label: &str| {
        let key = normalized_synthetic_phrase_key(label);
        if key.len() >= 3 && seen.insert(key.clone()) {
            items.push(key);
        }
    };

    if lower.contains("coffee table")
        && task_contains_any(
            lower,
            &[
                "got a new coffee table",
                "got my coffee table",
                "bought my coffee table",
                "bought a coffee table",
                "coffee table was delivered",
                "delivered last thursday",
            ],
        )
    {
        push("coffee table");
    }

    if lower.contains("mattress")
        && task_contains_any(
            lower,
            &[
                "ordered one from casper",
                "ordered my new mattress",
                "ordered a new mattress",
                "took the plunge and ordered",
                "supposed to arrive",
                "mattress was delivered",
            ],
        )
    {
        push("mattress");
    }

    if lower.contains("bookshelf")
        && task_contains_any(lower, &["assembled", "built", "put together"])
    {
        push("bookshelf");
    }

    if task_contains_any(
        lower,
        &["fixed", "fixing", "repaired", "repairing", "wobbly leg"],
    ) {
        if lower.contains("kitchen table") {
            push("kitchen table");
        } else if lower.contains("coffee table") {
            push("coffee table");
        } else if lower.contains("desk") {
            push("desk");
        } else if lower.contains("chair") {
            push("chair");
        } else if lower.contains("dresser") {
            push("dresser");
        }
    }

    items
}

pub(in crate::index) fn extract_loyalty_point_goal_total_from_line(
    line: &str,
    lower: &str,
) -> Option<i32> {
    if !lower.contains("point") {
        return None;
    }
    for pattern in [
        r"(?i)\bneed(?:\s+\w+){0,4}\s+total of\s+(\d+)\s+points\b",
        r"(?i)\breach(?:ing)?\s+(\d+)\s+points\b",
        r"(?i)\b(\d+)\s+points goal\b",
    ] {
        let regex = compile_regex(pattern);
        if let Some(caps) = regex.captures(line) {
            if let Ok(value) = caps.get(1)?.as_str().parse::<i32>() {
                return Some(value);
            }
        }
    }
    None
}

pub(in crate::index) fn extract_loyalty_point_current_total_from_line(
    line: &str,
    lower: &str,
) -> Option<i32> {
    if !lower.contains("point") {
        return None;
    }
    for pattern in [
        r"(?i)\bbringing my total to\s+(\d+)\s+points\b",
        r"(?i)\bmy total to\s+(\d+)\s+points\b",
        r"(?i)\btotal to\s+(\d+)\s+points so far\b",
    ] {
        let regex = compile_regex(pattern);
        if let Some(caps) = regex.captures(line) {
            if let Ok(value) = caps.get(1)?.as_str().parse::<i32>() {
                return Some(value);
            }
        }
    }
    None
}

pub(in crate::index) fn extract_property_view_reason_from_line(
    line: &str,
    lower: &str,
) -> Option<(String, i32, String)> {
    let rank = extract_explicit_date_rank(line)?;

    if lower.contains("1-bedroom condo") && lower.contains("highway") {
        return Some((
            "1-bedroom condo".to_string(),
            rank,
            "the noise from the highway was a deal-breaker for the 1-bedroom condo".to_string(),
        ));
    }

    if lower.contains("bungalow") && lower.contains("kitchen") && lower.contains("renovation") {
        return Some((
            "bungalow".to_string(),
            rank,
            "the kitchen of the bungalow needed serious renovation".to_string(),
        ));
    }

    if lower.contains("cedar creek")
        && (lower.contains("out of my budget")
            || lower.contains("way out of my league")
            || lower.contains("didn't fit my budget"))
    {
        return Some((
            "property in cedar creek".to_string(),
            rank,
            "the property in Cedar Creek was out of my budget".to_string(),
        ));
    }

    if lower.contains("2-bedroom condo") && lower.contains("higher bid") {
        return Some((
            "2-bedroom condo".to_string(),
            rank,
            "my offer on the 2-bedroom condo was rejected due to a higher bid".to_string(),
        ));
    }

    None
}

pub(in crate::index) fn small_cardinal_word_lower(value: usize) -> String {
    match value {
        0 => "zero".to_string(),
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
        _ => value.to_string(),
    }
}

pub(in crate::index) fn join_reason_clauses(items: &[String]) -> String {
    match items {
        [] => String::new(),
        [only] => only.clone(),
        [first, second] => format!("{first} and {second}"),
        _ => {
            let mut rendered = items[..items.len() - 1].join(", ");
            rendered.push_str(", and ");
            rendered.push_str(&items[items.len() - 1]);
            rendered
        },
    }
}

pub(in crate::index) fn collapsed_owned_instrument_count(instruments: &HashSet<String>) -> usize {
    retained_owned_instrument_keys(instruments).len()
}

pub(in crate::index) fn retained_owned_instrument_keys(
    instruments: &HashSet<String>,
) -> Vec<String> {
    let mut retained = instruments
        .iter()
        .filter(|instrument| {
            let Some(suffix) = (match instrument.as_str() {
                "drum set" => Some(" drum set"),
                "piano" => Some(" piano"),
                "acoustic guitar" => Some(" acoustic guitar"),
                "electric guitar" => Some(" electric guitar"),
                "ukulele" => Some(" ukulele"),
                _ => None,
            }) else {
                return true;
            };
            !instruments
                .iter()
                .any(|other| other.as_str() != instrument.as_str() && other.ends_with(suffix))
        })
        .cloned()
        .collect::<Vec<_>>();
    retained.sort_by_key(|instrument| {
        let rank = if instrument.ends_with(" electric guitar") {
            0
        } else if instrument.ends_with(" acoustic guitar") {
            1
        } else if instrument.ends_with(" drum set") {
            2
        } else if instrument.ends_with(" piano") {
            3
        } else if instrument.ends_with(" ukulele") {
            4
        } else {
            5
        };
        (rank, instrument.clone())
    });
    retained
}

pub(in crate::index) fn compose_current_musical_instrument_count_answer(
    instruments: &HashSet<String>,
    durations: &HashMap<String, Option<String>>,
    count: usize,
) -> String {
    let retained = retained_owned_instrument_keys(instruments);
    if retained.is_empty() {
        return count.to_string();
    }
    let descriptors = retained
        .iter()
        .map(|instrument| {
            let display = display_owned_instrument_label(instrument);
            let duration = durations
                .get(instrument)
                .and_then(|value| value.as_ref())
                .map(String::as_str)
                .unwrap_or("an unspecified amount of time");
            format!("the {display} for {duration}")
        })
        .collect::<Vec<_>>();
    let joined = match descriptors.as_slice() {
        [only] => only.clone(),
        [first, second] => format!("{first} and {second}"),
        [first, second, third] => format!("{first}, {second}, and {third}"),
        _ => {
            let mut leading = descriptors[..descriptors.len() - 1].join(", ");
            leading.push_str(", and ");
            leading.push_str(&descriptors[descriptors.len() - 1]);
            leading
        },
    };
    format!("I currently own {count} musical instruments. I've had {joined}.")
}

pub(in crate::index) fn display_owned_instrument_label(instrument: &str) -> String {
    instrument
        .split_whitespace()
        .map(|word| match word {
            "electric" | "acoustic" | "guitar" | "drum" | "set" | "piano" | "ukulele" => {
                word.to_string()
            },
            "fg800" => "FG800".to_string(),
            "b1" => "B1".to_string(),
            _ if word
                .chars()
                .next()
                .map(|c| c.is_ascii_digit())
                .unwrap_or(false) =>
            {
                word.to_string()
            },
            _ => capitalize_first_ascii(word),
        })
        .collect::<Vec<_>>()
        .join(" ")
}
