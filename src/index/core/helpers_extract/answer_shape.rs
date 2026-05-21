use super::*;
use crate::index::{compile_regex, compile_regex_static};

pub(in crate::index) fn looks_like_answer_surface_date(answer_span: &str) -> bool {
    const MONTHS: &[&str] = &[
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
    ];
    let lower = answer_span.to_ascii_lowercase();
    compile_regex_static(r"\b(?:19|20)\d{2}\b").is_match(&lower)
        || MONTHS.iter().any(|month| lower.contains(month))
        || task_contains_any(
            &lower,
            &[
                "yesterday",
                "today",
                "tonight",
                "tomorrow",
                "last week",
                "last month",
                "last year",
                "next week",
                "next month",
                "week before",
                "month before",
                "year before",
                "last saturday",
                "last sunday",
                "last monday",
                "last tuesday",
                "last wednesday",
                "last thursday",
                "last friday",
            ],
        )
}

pub(in crate::index) fn looks_like_answer_surface_duration(answer_span: &str) -> bool {
    let lower = answer_span.to_ascii_lowercase();
    lower.starts_with("since ")
        || compile_regex_static(
            r"\b(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\s+(?:day|week|month|year)s?\b",
        )
        .is_match(&lower)
        || compile_regex_static(
            r"\b(?:day|week|month|year)s?\s+(?:ago|already|now)\b",
        )
        .is_match(&lower)
}

pub(in crate::index) fn looks_like_answer_surface_count(answer_span: &str) -> bool {
    if looks_like_answer_surface_date(answer_span) {
        return false;
    }
    let lower = answer_span.to_ascii_lowercase();
    compile_regex_static(
        r"^(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|twice|thrice)(?:\s+(?:times?|kids?|children|dogs?|cats?|followers?|issues?|books?|letters?))?$",
    )
    .is_match(lower.trim())
}

pub(in crate::index) fn looks_like_answer_surface_person(answer_span: &str) -> bool {
    let lower = answer_span.to_ascii_lowercase();
    if task_contains_any(
        &lower,
        &[
            "family",
            "friends",
            "friend",
            "mentor",
            "mentors",
            "mother",
            "mom",
            "father",
            "dad",
            "aunt",
            "uncle",
            "sister",
            "brother",
            "husband",
            "wife",
            "partner",
            "spouse",
            "colleague",
            "colleagues",
            "teammates",
            "children",
            "kids",
        ],
    ) {
        return true;
    }
    let words = answer_span.split_whitespace().collect::<Vec<_>>();
    words.len() <= 8
        && words.iter().any(|word| {
            word.chars()
                .next()
                .map(|c| c.is_ascii_uppercase())
                .unwrap_or(false)
        })
}

pub(in crate::index) fn looks_like_answer_surface_name_like(answer_span: &str) -> bool {
    if answer_span.contains('?')
        || answer_span.contains(". ")
        || looks_like_answer_surface_date(answer_span)
        || looks_like_answer_surface_duration(answer_span)
        || looks_like_answer_surface_count(answer_span)
    {
        return false;
    }
    let words = answer_span.split_whitespace().collect::<Vec<_>>();
    words.len() <= 10
        && words.iter().any(|word| {
            word.chars()
                .next()
                .map(|c| c.is_ascii_uppercase())
                .unwrap_or(false)
        })
}

pub(in crate::index) fn looks_like_answer_surface_list_item(answer_span: &str) -> bool {
    if answer_span.contains('?')
        || answer_span.contains(". ")
        || looks_like_answer_surface_date(answer_span)
        || looks_like_answer_surface_duration(answer_span)
        || looks_like_answer_surface_count(answer_span)
    {
        return false;
    }
    let words = answer_span.split_whitespace().collect::<Vec<_>>();
    !words.is_empty()
        && words.len() <= 8
        && !task_contains_any(
            &answer_span.to_ascii_lowercase(),
            &[" because ", " although ", " however ", " but "],
        )
}

pub(in crate::index) fn looks_like_answer_surface_location(answer_span: &str) -> bool {
    if looks_like_answer_surface_date(answer_span) || looks_like_answer_surface_count(answer_span) {
        return false;
    }
    let lower = answer_span.to_ascii_lowercase();
    if task_contains_any(
        &lower,
        &[
            "beach",
            "mountain",
            "mountains",
            "forest",
            "woods",
            "lake",
            "park",
            "city",
            "country",
            "state",
            "suburbs",
            "downtown",
            "village",
            "town",
            "island",
        ],
    ) {
        return true;
    }
    let words = answer_span.split_whitespace().collect::<Vec<_>>();
    words.len() <= 6
        && words.iter().any(|word| {
            word.chars()
                .next()
                .map(|c| c.is_ascii_uppercase())
                .unwrap_or(false)
        })
}
