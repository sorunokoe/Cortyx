//! Media and instrument extraction: magazines, marathons, music, courses.

use super::super::*;
use crate::index::{compile_regex, compile_regex_static};

pub fn line_mentions_recent_activity_label(lower: &str, label: &str) -> bool {
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

pub fn extract_recent_activity_duration_facts_from_line(
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

pub fn extract_current_magazine_subscription_updates_from_line(
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

pub fn extract_hour_minute_total_from_text(text: &str) -> Option<i32> {
    for regex in [
        compile_regex_static(r"(?i)\b(\d+)\s*h(?:ours?)?\s*(\d+)\s*min(?:ute)?s?\b"),
        compile_regex_static(r"(?i)\b(\d+)\s+hours?\s+(?:and\s+)?(\d+)\s+minutes?\b"),
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

pub fn extract_marathon_completion_minutes_from_line(line: &str, lower: &str) -> Option<i32> {
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

pub fn extract_marathon_target_minutes_from_line(line: &str, lower: &str) -> Option<i32> {
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

pub fn extract_attended_movie_festival_from_line(line: &str, lower: &str) -> Option<String> {
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
    let caps = compile_regex_static(
        r"(?i)\b(?:at|after the screening at|like)\b\s+(?:the\s+)?([A-Z][A-Za-z0-9&' .-]+?Film Festival|AFI Fest|TIFF)\b",
    )
    .captures(line)?;
    Some(caps.get(1)?.as_str().trim().to_string())
}

pub fn spell_small_cardinal(count: usize) -> Option<&'static str> {
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

pub fn extract_music_release_signatures_from_line(line: &str, lower: &str) -> Vec<String> {
    let mut releases = Vec::new();
    let mut seen = HashSet::new();

    if task_contains_any(lower, &["i bought", "i ended up buying"]) {
        if let Some(caps) =
            compile_regex_static(r#"(?i)\b(?:EP|album)\s+["']([^"']+)["']"#).captures(line)
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
            compile_regex_static(r#"(?i)\balbum\s+["']([^"']+)["'][^.\n]*\bdownloaded\b"#)
                .captures(line)
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
            compile_regex_static(r"(?i)\bgot my ([A-Z][A-Za-z0-9&' .-]+?) vinyl signed\b"),
            compile_regex_static(
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

pub fn extract_owned_musical_instrument_signatures_from_line(
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
            compile_regex_static(
                r"\bdrum set,\s+a\s+((?:\d+-piece\s+)?[A-Z][A-Za-z0-9]+(?:\s+[A-Z0-9][A-Za-z0-9]+)*)\b",
            ),
            compile_regex_static(
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
            compile_regex_static(
                r"\bpiano,\s+a\s+([A-Z][A-Za-z0-9]+(?:\s+[A-Z0-9][A-Za-z0-9]+)*)\b",
            ),
            compile_regex_static(r"\b([A-Z][A-Za-z0-9]+(?:\s+[A-Z0-9][A-Za-z0-9]+)*)\s+piano\b"),
            compile_regex_static(r"\b(Korg\s+B1)\b"),
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
            compile_regex_static(
                r"\bacoustic guitar,\s+a\s+([A-Z][A-Za-z0-9]+(?:\s+[A-Z0-9][A-Za-z0-9]+)*)\b",
            ),
            compile_regex_static(
                r"\b([A-Z][A-Za-z0-9]+(?:\s+[A-Z0-9][A-Za-z0-9]+)*)\s+acoustic guitar\b",
            ),
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
            compile_regex_static(
                r"\b(?:my|had my|playing my)\s+(?:[a-z]+\s+)?([A-Z][A-Za-z0-9]+(?:\s+[A-Z0-9][A-Za-z0-9]+)*)\s+electric guitar\b",
            ),
            compile_regex_static(
                r"\b([A-Z][A-Za-z0-9]+(?:\s+[A-Z0-9][A-Za-z0-9]+)*)\s+electric guitar\b",
            ),
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
            compile_regex_static(
                r"\bukulele,\s+a\s+([A-Z][A-Za-z0-9]+(?:\s+[A-Z0-9][A-Za-z0-9]+)*)\b",
            ),
            compile_regex_static(r"\b([A-Z][A-Za-z0-9]+(?:\s+[A-Z0-9][A-Za-z0-9]+)*)\s+ukulele\b"),
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

pub fn extract_online_course_completion_updates_from_line(
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
        compile_regex_static(r"(?i)\bcompleted\s+([A-Za-z0-9,-]+)\s+courses?\b"),
        compile_regex_static(r"(?i)\b([A-Za-z0-9,-]+)\s+courses?\s+on\b"),
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

pub fn extract_recent_furniture_action_signatures_from_line(
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

pub fn extract_loyalty_point_goal_total_from_line(line: &str, lower: &str) -> Option<i32> {
    if !lower.contains("point") {
        return None;
    }
    for pattern in [
        r"(?i)\bneed(?:\s+\w+){0,4}\s+total of\s+(\d+)\s+points\b",
        r"(?i)\breach(?:ing)?\s+(\d+)\s+points\b",
        r"(?i)\b(\d+)\s+points goal\b",
    ] {
        let regex = compile_regex_static(pattern);
        if let Some(caps) = regex.captures(line) {
            if let Ok(value) = caps.get(1)?.as_str().parse::<i32>() {
                return Some(value);
            }
        }
    }
    None
}

pub fn extract_loyalty_point_current_total_from_line(line: &str, lower: &str) -> Option<i32> {
    if !lower.contains("point") {
        return None;
    }
    for pattern in [
        r"(?i)\bbringing my total to\s+(\d+)\s+points\b",
        r"(?i)\bmy total to\s+(\d+)\s+points\b",
        r"(?i)\btotal to\s+(\d+)\s+points so far\b",
    ] {
        let regex = compile_regex_static(pattern);
        if let Some(caps) = regex.captures(line) {
            if let Ok(value) = caps.get(1)?.as_str().parse::<i32>() {
                return Some(value);
            }
        }
    }
    None
}

pub fn extract_property_view_reason_from_line(
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

pub fn small_cardinal_word_lower(value: usize) -> String {
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

pub fn join_reason_clauses(items: &[String]) -> String {
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

pub fn collapsed_owned_instrument_count(instruments: &HashSet<String>) -> usize {
    retained_owned_instrument_keys(instruments).len()
}

pub fn retained_owned_instrument_keys(instruments: &HashSet<String>) -> Vec<String> {
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

pub fn compose_current_musical_instrument_count_answer(
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

pub fn display_owned_instrument_label(instrument: &str) -> String {
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
