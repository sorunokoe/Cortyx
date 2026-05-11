//! Query classification for temporal reasoning.

use super::*;

pub(crate) fn is_temporal_sequence_query(task: &str) -> bool {
    let lower = task.to_ascii_lowercase();
    lower.starts_with("what is the order of")
        || lower.starts_with("what's the order of")
        || lower.contains("from first to last")
        || lower.contains("from last to first")
        || lower.contains("order from first to last")
        || lower.contains("from earliest to latest")
        || lower.contains("from latest to earliest")
        || lower.contains("starting from the earliest")
        || lower.contains("starting from the latest")
        || lower.contains("first, second")
        || lower.contains("first second")
}

pub(crate) fn temporal_focus_terms(text: &str) -> Vec<String> {
    let lower = text.to_ascii_lowercase();
    let mut terms = salient_query_terms(text);
    terms.retain(|term| {
        !matches!(
            term.as_str(),
            "many"
                | "ago"
                | "first"
                | "last"
                | "latest"
                | "earliest"
                | "recent"
                | "months"
                | "month"
                | "weeks"
                | "week"
                | "days"
                | "day"
                | "years"
                | "year"
                | "happened"
                | "happen"
                | "current"
                | "currently"
        )
    });

    if lower.contains("shoe") {
        terms.extend(
            ["sneakers", "boots", "sandals", "trainers"]
                .iter()
                .map(|term| (*term).to_string()),
        );
    }
    if lower.contains("issue") || lower.contains("problem") {
        terms.extend(
            ["issue", "problem", "malfunction", "broken"]
                .iter()
                .map(|term| (*term).to_string()),
        );
    }
    if lower.contains("clean") || lower.contains("wash") {
        terms.extend(
            ["clean", "cleaned", "washed", "washing"]
                .iter()
                .map(|term| (*term).to_string()),
        );
    }
    if lower.contains("event") || lower.contains("participat") || lower.contains("charity") {
        terms.extend(
            [
                "event",
                "events",
                "attended",
                "attend",
                "participated",
                "participate",
                "volunteered",
                "volunteer",
                "joined",
                "join",
                "walk",
                "walked",
                "run",
                "ran",
                "tournament",
                "gala",
                "marathon",
                "triathlon",
                "cleanup",
                "drive",
            ]
            .iter()
            .map(|term| (*term).to_string()),
        );
    }
    if lower.contains("buy")
        || lower.contains("bought")
        || lower.contains("purchase")
        || lower.contains("purchased")
        || lower.contains(" got ")
        || lower.starts_with("got ")
    {
        terms.extend(
            ["buy", "bought", "purchase", "purchased", "got"]
                .iter()
                .map(|term| (*term).to_string()),
        );
    }
    if lower.contains("set up") || lower.contains("setup") || lower.contains("install") {
        terms.extend(
            ["setup", "installed", "installing", "upgraded"]
                .iter()
                .map(|term| (*term).to_string()),
        );
    }
    if lower.contains("work") || lower.contains("job") {
        terms.extend(
            ["working", "job", "career", "professionally"]
                .iter()
                .map(|term| (*term).to_string()),
        );
    }
    terms.sort();
    terms.dedup();
    terms
}

pub(crate) fn temporal_sequence_focus_terms(task: &str) -> Vec<String> {
    let lower = task.to_ascii_lowercase();
    let mut terms = temporal_focus_terms(task);
    terms.retain(|term| {
        !matches!(
            term.as_str(),
            "order"
                | "past"
                | "latest"
                | "earliest"
                | "starting"
                | "first"
                | "second"
                | "third"
                | "fourth"
                | "three"
                | "four"
                | "five"
                | "six"
                | "among"
        )
    });

    if lower.contains("trip") {
        terms.extend(
            [
                "trip", "trips", "road", "hike", "camping", "travel", "vacation",
            ]
            .iter()
            .map(|term| (*term).to_string()),
        );
    }
    if lower.contains("concert") || lower.contains("music") {
        terms.extend(
            ["concert", "music", "festival", "jazz", "show"]
                .iter()
                .map(|term| (*term).to_string()),
        );
    }
    if lower.contains("museum") {
        terms.extend(
            ["museum", "art", "history", "science"]
                .iter()
                .map(|term| (*term).to_string()),
        );
    }
    if lower.contains("event") {
        terms.extend(
            ["event", "events", "attended", "participated", "visited"]
                .iter()
                .map(|term| (*term).to_string()),
        );
    }
    if lower.contains("graduat") {
        terms.extend(
            ["graduated", "graduation", "graduate"]
                .iter()
                .map(|term| (*term).to_string()),
        );
    }
    terms.sort();
    terms.dedup();
    terms
}

pub(crate) fn is_temporal_reasoning_query(task: &str) -> bool {
    let lower = task.to_ascii_lowercase();
    let how_many_temporal = lower.starts_with("how many ")
        && (lower.contains(" day")
            || lower.contains(" week")
            || lower.contains(" month")
            || lower.contains(" year")
            || lower.contains(" ago")
            || lower.contains(" before ")
            || lower.contains(" after ")
            || lower.contains(" first")
            || lower.contains(" last"));
    lower.starts_with("when ")
        || lower.contains(" what date")
        || lower.contains(" which date")
        || lower.contains(" which day")
        || how_many_temporal
        || lower.starts_with("how long ")
        || lower.contains(" first")
        || lower.contains(" last")
        || lower.contains(" earliest")
        || lower.contains(" latest")
        || lower.contains(" most recent")
        || lower.contains(" before ")
        || lower.contains(" after ")
        || lower.contains("order of")
}
