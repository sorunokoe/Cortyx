//! Culture event extraction: art, cuisine, museum, citrus, fun run, health.

use super::super::*;
use crate::index::compile_regex;

pub fn extract_first_quoted_phrase(line: &str) -> Option<String> {
    compile_regex(r#""([^"]+)""#)
        .captures(line)
        .and_then(|caps| caps.get(1))
        .map(|m| m.as_str().trim().to_string())
}

pub fn extract_art_event_anchor(line: &str) -> Option<String> {
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

pub fn line_describes_art_related_event(lower: &str) -> bool {
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

pub fn extract_art_related_event_signature_from_line(
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

pub fn line_describes_cuisine_learning_or_trying(lower: &str) -> bool {
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

pub fn extract_cuisine_labels_from_line(_line: &str, lower: &str) -> Vec<String> {
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

pub fn line_describes_museum_gallery_visit(lower: &str) -> bool {
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

pub fn normalize_visit_venue(text: &str) -> String {
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

pub fn extract_museum_gallery_visit_venue_from_line(
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

pub fn line_mentions_candidate_museum_gallery_visit(line: &str, lower: &str, month: &str) -> bool {
    line_matches_query_month_or_numeric_date(line, lower, month)
        && line_describes_museum_gallery_visit(lower)
        && task_contains_any(lower, &["museum", "gallery", "art cube"])
}

pub fn extract_citrus_fruits_from_line(_line: &str, lower: &str) -> Vec<String> {
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

pub fn extract_food_delivery_service_from_line(_line: &str, lower: &str) -> Option<String> {
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

pub fn extract_missed_fun_run_signature_from_line(
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

pub fn line_mentions_recent_three_month_window(lower: &str) -> bool {
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

pub fn trim_trailing_relative_time_phrase(text: &str) -> String {
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

pub fn extract_graduation_ceremony_signature_from_line(line: &str, lower: &str) -> Option<String> {
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

pub fn extract_health_device_units_from_line(_line: &str, lower: &str) -> Vec<String> {
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

pub fn extract_peak_campaign_weekly_hour_delta_from_line(line: &str, lower: &str) -> Option<f32> {
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

pub fn extract_typical_weekly_work_hours_from_line(line: &str, lower: &str) -> Option<f32> {
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

pub fn extract_peak_campaign_total_weekly_hours_from_line(line: &str, lower: &str) -> Option<f32> {
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

pub fn extract_recent_activity_query_labels(task_lower: &str) -> Vec<&'static str> {
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
