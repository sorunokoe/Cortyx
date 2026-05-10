//! KG-backed temporal helpers: state queries, date arithmetic, and KG entity/predicate lookups.
//!
//! This module handles temporal state query parsing, knowledge graph entity queries,
//! and date/time arithmetic for temporal reasoning.

use super::*;
use crate::kg;

pub fn shift_date_by_days(base: (i32, u32, u32), delta_days: i32) -> (i32, u32, u32) {
    let rank = ymd_to_days(base.0, base.1, base.2) + delta_days;
    days_to_ymd(rank)
}

pub fn shift_month(year: i32, month: u32, delta_months: i32) -> (i32, u32) {
    let base = year * 12 + month as i32 - 1 + delta_months;
    let shifted_year = base.div_euclid(12);
    let shifted_month = base.rem_euclid(12) + 1;
    (shifted_year, shifted_month as u32)
}

fn days_to_ymd(mut days: i32) -> (i32, u32, u32) {
    let mut year = 1970;
    loop {
        let year_days = if is_leap_year(year) { 366 } else { 365 };
        if days < year_days {
            break;
        }
        days -= year_days;
        year += 1;
    }

    let month_lengths = [
        31,
        if is_leap_year(year) { 29 } else { 28 },
        31,
        30,
        31,
        30,
        31,
        31,
        30,
        31,
        30,
        31,
    ];
    let mut month = 1u32;
    for month_length in month_lengths {
        if days < month_length {
            break;
        }
        days -= month_length;
        month += 1;
    }
    (year, month, days as u32 + 1)
}

fn is_leap_year(year: i32) -> bool {
    (year % 4 == 0 && year % 100 != 0) || year % 400 == 0
}

pub fn month_name(month: u32) -> &'static str {
    match month {
        1 => "January",
        2 => "February",
        3 => "March",
        4 => "April",
        5 => "May",
        6 => "June",
        7 => "July",
        8 => "August",
        9 => "September",
        10 => "October",
        11 => "November",
        12 => "December",
        _ => "Unknown",
    }
}

pub(super) fn parse_temporal_state_query(task: &str) -> Option<TemporalStateQuery> {
    if let Some(as_of) = extract_temporal_as_of_point(task) {
        return Some(TemporalStateQuery::AsOfValue { as_of });
    }

    let lower = task.to_ascii_lowercase();
    let asks_when = lower.starts_with("when ")
        || lower.contains(" what date")
        || lower.contains(" which date")
        || lower.contains(" at what time");
    let asks_change_time = asks_when
        && (lower.contains(" last change")
            || lower.contains(" last changed")
            || lower.contains(" latest change")
            || lower.contains(" most recent change")
            || lower.contains(" change ")
            || lower.contains(" changed ")
            || lower.contains(" become ")
            || lower.contains(" became "));
    if asks_change_time {
        return Some(TemporalStateQuery::LastChange {
            target_value: parse_temporal_change_target(task),
        });
    }

    let asks_current_state = lower.contains(" current ")
        || lower.contains(" currently ")
        || lower.contains(" right now")
        || lower.contains(" now ")
        || lower.ends_with(" now?")
        || lower.contains(" latest ")
        || lower.starts_with("what is my latest")
        || lower.contains(" still ")
        || lower.contains(" present ");
    asks_current_state.then_some(TemporalStateQuery::CurrentValue)
}

fn parse_temporal_change_target(task: &str) -> Option<ChoiceOption> {
    let trimmed = task.trim().trim_end_matches('?');
    for marker in [
        " changed to ",
        " change to ",
        " became ",
        " become ",
        " switched to ",
        " switch to ",
    ] {
        let Some((_, target)) = split_once_case_insensitive(trimmed, marker) else {
            continue;
        };
        let clean = target.trim();
        if let Some(option) = build_temporal_event_option(clean) {
            return Some(option);
        }
    }
    None
}

fn extract_temporal_as_of_point(task: &str) -> Option<String> {
    let (_, rest) = split_once_case_insensitive(task, "as of ")?;
    normalize_temporal_query_point(rest)
}

fn normalize_temporal_query_point(text: &str) -> Option<String> {
    if let Some(point) = extract_iso_temporal_point(text) {
        if point.len() == 10 {
            return Some(format!("{point}T23:59:59Z"));
        }
        return Some(point);
    }
    let lower = text.trim().to_ascii_lowercase();
    let now_phrases = ["now", "today", "currently", "present", "at the moment"];
    if now_phrases.iter().any(|phrase| lower == *phrase) {
        return Some("now".to_string());
    }
    None
}

fn extract_iso_temporal_point(text: &str) -> Option<String> {
    let trimmed = text.trim();
    for token in trimmed.split_whitespace() {
        if is_iso_date_fragment(token) {
            return Some(token.to_string());
        }
    }
    if is_iso_date_fragment(trimmed) {
        return Some(trimmed[..10.min(trimmed.len())].to_string());
    }
    None
}

fn is_iso_date_fragment(text: &str) -> bool {
    (text.len() >= 10 || text.starts_with("202") || text.starts_with("201"))
        && text
            .chars()
            .take(10)
            .zip(['-', '-', '-'].iter().cycle())
            .all(|(c, expected)| c.is_ascii_digit() || *expected == '-')
}

pub(super) fn temporal_state_candidate_score(
    task_terms: &[String],
    retrieval_score: f32,
    entity: &kg::KgEntity,
    predicate: &str,
) -> f32 {
    let predicate_context = kg_predicate_query_terms(predicate).join(" ");
    if predicate_context.is_empty() || task_overlap_count(&predicate_context, task_terms) == 0 {
        return 0.0;
    }

    let entity_context = kg_entity_query_terms(&entity.entity).join(" ");
    let entity_overlap = if entity_context.is_empty() {
        0.0
    } else {
        task_overlap_count(&entity_context, task_terms) as f32 * 6.0
    };
    let combined_context = if entity_context.is_empty() {
        predicate_context
    } else {
        format!("{predicate_context} {entity_context}")
    };

    candidate_weight(&combined_context, task_terms, retrieval_score, false) + entity_overlap
}

pub fn kg_predicate_query_terms(predicate: &str) -> Vec<String> {
    let mut terms = predicate
        .split('_')
        .map(str::trim)
        .filter(|token| token.len() >= 3)
        .map(ToString::to_string)
        .collect::<Vec<_>>();
    let extras: &[&str] = match predicate {
        "status" => &["state", "progress"],
        "blocker" => &["blocked", "blocking", "stuck", "waiting"],
        "next_step" => &["next", "step", "follow", "action"],
        "goal" => &["objective", "target", "aim"],
        "outcome" => &["result", "finding", "decision", "conclusion"],
        "title" => &["task", "focus", "work"],
        "action" => &["doing", "working", "investigating", "reviewing"],
        "location" => &["live", "lived", "home", "city", "based", "move", "moved"],
        "occupation" => &["job", "work", "career", "role", "employed"],
        "education" => &[
            "degree",
            "study",
            "studied",
            "graduate",
            "graduated",
            "school",
        ],
        "major" => &["study", "studied", "degree", "school"],
        "book" => &["reading", "read", "novel"],
        "partner" => &["wife", "husband", "boyfriend", "girlfriend", "spouse"],
        "pet" => &["dog", "cat", "pets"],
        "phone" => &["number", "call"],
        "project_name" => &["project", "name", "called"],
        "instagram_followers" => &["instagram", "follower", "followers"],
        "commute_time" => &["commute", "travel", "minutes", "time"],
        "fitness_record" => &["record", "best", "personal"],
        "vehicle_model" => &["vehicle", "car", "truck", "drive", "model"],
        "family_trip_location" => &["family", "trip", "vacation", "travel", "where"],
        "related_entity" => &[
            "entity",
            "entities",
            "file",
            "files",
            "module",
            "modules",
            "component",
        ],
        _ => &[],
    };
    terms.extend(extras.iter().map(|term| (*term).to_string()));
    terms.sort();
    terms.dedup();
    terms
}

pub fn kg_entity_query_terms(entity: &str) -> Vec<String> {
    let mut terms = entity
        .split('_')
        .filter(|token| token.len() >= 3)
        .filter(|token| !matches!(*token, "agent" | "entity"))
        .map(ToString::to_string)
        .collect::<Vec<_>>();
    terms.sort();
    terms.dedup();
    terms
}

pub(super) fn render_temporal_state_kg_answer(
    query: &TemporalStateQuery,
    entity: &kg::KgEntity,
    predicate: &str,
) -> Option<String> {
    match query {
        TemporalStateQuery::CurrentValue => format_kg_values(current_kg_values(entity, predicate)),
        TemporalStateQuery::AsOfValue { as_of } => {
            format_kg_values(kg_values_for_predicate_as_of(entity, predicate, as_of))
        },
        TemporalStateQuery::LastChange { target_value } => {
            kg_last_change_for_predicate(entity, predicate, target_value.as_ref())
        },
    }
}

pub fn current_kg_values(entity: &kg::KgEntity, predicate: &str) -> Vec<String> {
    collect_kg_values(
        entity
            .facts
            .iter()
            .filter(|fact| fact.predicate == predicate && fact.ended.is_empty())
            .collect(),
    )
}

fn kg_values_for_predicate_as_of(
    entity: &kg::KgEntity,
    predicate: &str,
    as_of: &str,
) -> Vec<String> {
    collect_kg_values(
        entity
            .facts
            .iter()
            .filter(|fact| fact.predicate == predicate && kg_fact_is_active_as_of(fact, as_of))
            .collect(),
    )
}

fn kg_fact_is_active_as_of(fact: &kg::KgFact, as_of: &str) -> bool {
    let as_of = as_of.trim();
    if as_of.is_empty() {
        return fact.ended.is_empty();
    }
    if !fact.valid_from.is_empty() && fact.valid_from.as_str() > as_of {
        return false;
    }
    if !fact.ended.is_empty() && as_of >= fact.ended.as_str() {
        return false;
    }
    true
}

fn collect_kg_values(mut facts: Vec<&kg::KgFact>) -> Vec<String> {
    facts.sort_by(|a, b| {
        a.valid_from
            .cmp(&b.valid_from)
            .then_with(|| a.value.cmp(&b.value))
    });

    let mut values = Vec::new();
    for fact in facts {
        let value = render_kg_value(&fact.value);
        if value.is_empty() || values.iter().any(|existing| existing == &value) {
            continue;
        }
        values.push(value);
    }
    values
}

fn render_kg_value(value: &str) -> String {
    sanitize_inline(&value.replace('_', " "))
}

fn format_kg_values(values: Vec<String>) -> Option<String> {
    (!values.is_empty()).then(|| values.join(", "))
}

fn kg_last_change_for_predicate(
    entity: &kg::KgEntity,
    predicate: &str,
    target_value: Option<&ChoiceOption>,
) -> Option<String> {
    let mut timeline = entity.timeline_for(predicate);
    if let Some(target_value) = target_value {
        timeline.retain(|fact| kg_value_matches_target(&fact.value, target_value));
    }
    timeline.retain(|fact| !fact.valid_from.trim().is_empty());
    timeline.sort_by(|a, b| a.valid_from.cmp(&b.valid_from));
    timeline.last().map(|fact| fact.valid_from.clone())
}

fn kg_value_matches_target(value: &str, target_value: &ChoiceOption) -> bool {
    let lower = value.to_ascii_lowercase().replace('_', " ");
    target_value
        .tokens
        .iter()
        .all(|token| line_matches_event_token(&lower, token))
}
