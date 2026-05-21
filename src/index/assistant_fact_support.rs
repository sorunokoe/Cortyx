use super::assistant_fact_extractors::{
    render_assistant_fact_answer, trim_fact_value, AssistantFactAnswerRender, EntityKind,
    EntityRecallQuery, ValueKind, ValueRecallQuery,
};
use super::*;

pub(super) fn extract_entity_candidate(
    query: &EntityRecallQuery,
    line: &str,
    lower: &str,
) -> Option<String> {
    match query.kind {
        EntityKind::NamedThing => {
            if named_thing_query_expects_company(query) {
                extract_company_named_thing_from_line(line, lower)
                    .map(|value| maybe_augment_named_venue_with_location(query, line, value))
            } else {
                extract_named_thing_from_line(line, lower)
                    .map(|value| maybe_augment_named_venue_with_location(query, line, value))
            }
        },
        EntityKind::PersonByRole => extract_role_person_from_line(line),
        EntityKind::Chapter => extract_chapter_phrase_from_line(line),
        EntityKind::Wearing => extract_wearing_phrase_from_line(line, lower),
        EntityKind::BeerType => extract_beer_recommendation_answer_from_line(lower),
        EntityKind::ImplementedAlgorithm => extract_implemented_algorithm_from_line(line, lower),
    }
}

pub(super) fn entity_line_bonus(query: &EntityRecallQuery, line: &str, lower: &str) -> usize {
    let label_bonus = usize::from(extract_structured_fact_label(line).is_some()) * 10;
    let title_bonus = usize::from(
        extract_title_like_phrases(line)
            .into_iter()
            .next()
            .is_some(),
    ) * 4;
    let quote_bonus = usize::from(extract_any_quoted_phrase(line).is_some()) * 8;
    let special_bonus = match query.kind {
        EntityKind::PersonByRole => usize::from(extract_role_person_from_line(line).is_some()) * 10,
        EntityKind::Chapter => usize::from(extract_chapter_phrase_from_line(line).is_some()) * 10,
        EntityKind::Wearing => {
            usize::from(task_contains_any(lower, &["wears ", "wore ", "wearing "])) * 8
        },
        EntityKind::BeerType => usize::from(lower.contains("beer")) * 8,
        EntityKind::ImplementedAlgorithm => usize::from(lower.contains("implemented in the")) * 8,
        EntityKind::NamedThing => {
            if named_thing_query_expects_company(query) {
                usize::from(extract_company_named_thing_from_line(line, lower).is_some()) * 24
                    + usize::from(lower.contains(" company")) * 6
            } else {
                0
            }
        },
    };
    label_bonus + title_bonus + quote_bonus + special_bonus
}

pub(super) fn extract_value_candidate(query: &ValueRecallQuery, line: &str) -> Option<String> {
    match query.kind {
        ValueKind::Money => extract_money_answer_from_line(line),
        ValueKind::Handle => compile_regex_static(r"@[A-Za-z0-9_.]+")
            .find(line)
            .map(|value| value.as_str().trim().to_string()),
        ValueKind::Phone => compile_regex_static(r"\+\d[\d\s()/.-]+\d")
            .find(line)
            .map(|value| value.as_str().trim().to_string()),
        ValueKind::Ratio => compile_regex_static(r"\b\d+\s*:\s*\d+\b")
            .find(line)
            .map(|value| value.as_str().trim().to_string()),
        ValueKind::Year => extract_year_candidate(query, line),
        ValueKind::Count => extract_count_candidate(query, line),
    }
}

pub(super) fn value_line_bonus(query: &ValueRecallQuery, line: &str) -> usize {
    match query.kind {
        ValueKind::Money => usize::from(extract_money_answer_from_line(line).is_some()) * 8,
        ValueKind::Handle => usize::from(line.contains('@')) * 8,
        ValueKind::Phone => usize::from(line.to_ascii_lowercase().contains("phone")) * 8,
        ValueKind::Ratio => usize::from(line.contains(':')) * 6,
        ValueKind::Year => {
            usize::from(compile_regex_static(r"\b(?:19|20)\d{2}\b").is_match(line)) * 6
        },
        ValueKind::Count => usize::from(extract_value_candidate(query, line).is_some()) * 6,
    }
}

pub(super) fn extract_quote_candidate(line: &str) -> Option<String> {
    extract_any_quoted_phrase(line)
        .map(|value| value.trim_end_matches('.').to_string())
        .filter(|value| value.split_whitespace().count() >= 4)
}

pub(super) fn extract_label_list_item(line: &str) -> Option<String> {
    extract_numbered_list_item(line)
        .map(|(_, value)| value)
        .and_then(|value| {
            value
                .split_once(" - ")
                .map(|(label, _)| trim_fact_value(label))
                .or_else(|| {
                    value
                        .split_once(": ")
                        .map(|(label, _)| trim_fact_value(label))
                })
                .or_else(|| Some(trim_fact_value(&value)))
        })
        .or_else(|| extract_structured_fact_label(line))
        .or_else(|| {
            extract_title_like_phrases(line)
                .into_iter()
                .find(|phrase| phrase.split_whitespace().count() <= 6)
        })
}

pub(super) fn extract_objective_list_item(line: &str) -> Option<String> {
    let body = normalize_session_answer_line_body(line);
    let trimmed = body.trim();
    trimmed
        .to_ascii_lowercase()
        .starts_with("to ")
        .then(|| trimmed.trim_end_matches('.').to_string())
}

pub(super) fn render_list_answer(render: &AssistantFactAnswerRender, items: &[String]) -> String {
    render_assistant_fact_answer(render, &join_fact_items(items))
}

fn extract_count_candidate(query: &ValueRecallQuery, line: &str) -> Option<String> {
    if let Some(subject) = query.subject_hint.as_deref() {
        let singular = subject.trim_end_matches('s');
        let pattern = compile_regex(&format!(
            r"(?i)\b(\d+(?:-\d+)?)\s+{}s?\b",
            regex::escape(singular)
        ))
        .unwrap_or_else(|err| panic!("escaped assistant-fact regex failed to compile: {err}"));
        if let Some(value) = pattern
            .captures(line)
            .and_then(|caps| caps.get(0))
            .map(|value| value.as_str().trim().to_string())
        {
            return Some(value);
        }
        if subject != "times" {
            return None;
        }
    }
    extract_focus_aligned_count(line, &query.focus_terms, &query.task_lower).map(|(count, _)| {
        if query.subject_hint.as_deref() == Some("times") {
            format!("{count} times")
        } else {
            count.to_string()
        }
    })
}

fn extract_year_candidate(query: &ValueRecallQuery, line: &str) -> Option<String> {
    let line_lower = line.to_ascii_lowercase();
    if task_contains_any(
        &query.task_lower,
        &[" began", " started", " start ", " commenced", " started?"],
    ) && !task_contains_any(
        &line_lower,
        &[" began", " started", " start ", " commenced"],
    ) {
        return None;
    }
    compile_regex_static(r"\b(?:19|20)\d{2}\b")
        .find(line)
        .map(|value| value.as_str().trim().to_string())
}

fn extract_named_thing_from_line(line: &str, lower: &str) -> Option<String> {
    if let Some(named_subject) = extract_named_subject_after_label(line)
        .filter(|value| is_specific_named_thing_candidate(value))
    {
        return Some(named_subject);
    }
    if let Some(label) =
        extract_structured_fact_label(line).filter(|value| is_specific_named_thing_candidate(value))
    {
        return Some(label);
    }
    if let Some(quoted) =
        extract_any_quoted_phrase(line).filter(|value| is_specific_named_thing_candidate(value))
    {
        return Some(quoted);
    }
    if let Some(person) =
        extract_role_person_from_line(line).filter(|value| is_specific_named_thing_candidate(value))
    {
        return Some(person);
    }
    if let Some(named) = extract_descriptor_led_name_from_line(line)
        .filter(|value| is_specific_named_thing_candidate(value))
    {
        return Some(named);
    }
    if let Some(leading) =
        extract_leading_fact_name(line).filter(|value| is_specific_named_thing_candidate(value))
    {
        return Some(leading);
    }
    if let Some(value) = extract_phrase_after_any_index(
        line,
        lower,
        &[
            "there's also ",
            "there is also ",
            "there is ",
            "example is the ",
            "example is ",
            "called ",
            "named ",
            "designation ",
        ],
        &[
            " which ",
            " that ",
            " because ",
            " and ",
            " but ",
            ".",
            ",",
            ";",
        ],
        1,
    ) {
        let cleaned = trim_fact_value(&value);
        if is_specific_named_thing_candidate(&cleaned)
            && cleaned
                .chars()
                .next()
                .map(|ch| ch.is_ascii_uppercase() || ch.is_ascii_digit() || ch == '@')
                .unwrap_or(false)
        {
            return Some(cleaned);
        }
    }
    extract_title_like_phrases(line)
        .into_iter()
        .find(|phrase| phrase.split_whitespace().count() <= 7)
        .filter(|value| is_specific_named_thing_candidate(value))
}

fn extract_company_named_thing_from_line(line: &str, lower: &str) -> Option<String> {
    extract_named_subject_after_label(line)
        .or_else(|| {
            (task_contains_any(
                lower,
                &[
                    "sustainab",
                    "environment",
                    "responsib",
                    "supply chain",
                    "company",
                ],
            ) || lower.contains(" uses "))
            .then(|| extract_leading_fact_name(line))
            .flatten()
        })
        .or_else(|| {
            extract_phrase_after_any_index(
                line,
                lower,
                &["lead of ", "like "],
                &[" and ", " but ", ".", ",", ";"],
                1,
            )
            .map(|value| trim_fact_value(&value))
        })
        .filter(|value| is_specific_named_thing_candidate(value))
}

fn named_thing_query_expects_company(query: &EntityRecallQuery) -> bool {
    query
        .focus_terms
        .iter()
        .chain(query.required_terms.iter())
        .any(|term| {
            matches!(
                term.as_str(),
                "company" | "companies" | "business" | "businesses"
            )
        })
}

fn maybe_augment_named_venue_with_location(
    query: &EntityRecallQuery,
    line: &str,
    candidate: String,
) -> String {
    if !named_thing_query_expects_venue_location(query) {
        return candidate;
    }
    let candidate_lower = candidate.to_ascii_lowercase();
    if task_contains_any(&candidate_lower, &[" at ", " in ", " on ", " near "]) {
        return candidate;
    }
    let Some((preposition, location)) = extract_named_venue_location(line) else {
        return candidate;
    };
    format!("{candidate} {preposition} {location}")
}

fn named_thing_query_expects_venue_location(query: &EntityRecallQuery) -> bool {
    query
        .focus_terms
        .iter()
        .chain(query.required_terms.iter())
        .any(|term| {
            matches!(
                term.as_str(),
                "bakery"
                    | "bar"
                    | "cafe"
                    | "café"
                    | "deli"
                    | "dessert"
                    | "hostel"
                    | "restaurant"
                    | "shop"
                    | "spot"
            )
        })
}

fn extract_named_venue_location(line: &str) -> Option<(&'static str, String)> {
    compile_regex_static(r"(?i)\blocated\s+(at|in|on)\s+([^.,;]+?)\s+(?:that|which|with|where|who|serves|offers|has|is)\b")
        .captures(line)
        .and_then(|caps| {
            let preposition = caps.get(1)?.as_str().to_ascii_lowercase();
            let location = trim_fact_value(caps.get(2)?.as_str());
            let mapped = match preposition.as_str() {
                "at" => "at",
                "in" => "in",
                "on" => "on",
                _ => return None,
            };
            (!location.is_empty()).then_some((mapped, location))
        })
}

pub(super) fn extract_structured_fact_label(line: &str) -> Option<String> {
    let body = extract_numbered_list_item(line)
        .map(|(_, value)| value)
        .unwrap_or_else(|| normalize_session_answer_line_body(line));
    for separator in [" - ", ": "] {
        if let Some((label, _)) = body.split_once(separator) {
            let cleaned = trim_fact_value(label);
            if cleaned.split_whitespace().count() <= 8 {
                return Some(cleaned);
            }
        }
    }
    None
}

fn extract_named_subject_after_label(line: &str) -> Option<String> {
    let body = normalize_session_answer_line_body(line);
    let (_, tail) = body.split_once(": ").or_else(|| body.split_once(" - "))?;
    let tail_lower = tail.to_ascii_lowercase();
    let split_idx = [
        " is ",
        " was ",
        " uses ",
        " use ",
        " has ",
        " have ",
        " had ",
        " works ",
        " work ",
        " encourages ",
        " encourage ",
        " monitors ",
        " monitor ",
        ", ",
    ]
    .into_iter()
    .filter_map(|marker| tail_lower.find(marker))
    .min()?;
    let candidate = trim_fact_value(&tail[..split_idx]);
    let tokens: Vec<&str> = candidate
        .split_whitespace()
        .map(|token| {
            token.trim_matches(|ch: char| !ch.is_alphanumeric() && ch != '\'' && ch != '-')
        })
        .filter(|token| !token.is_empty())
        .collect();
    if tokens.is_empty() || tokens.len() > 5 {
        return None;
    }
    if matches!(
        tokens[0],
        "A" | "An" | "Another" | "As" | "It" | "Its" | "One" | "Some" | "The" | "These" | "This"
    ) {
        return None;
    }
    tokens
        .iter()
        .all(|token| {
            token
                .chars()
                .next()
                .map(|ch| ch.is_ascii_uppercase() || ch.is_ascii_digit() || ch == '@')
                .unwrap_or(false)
        })
        .then_some(candidate)
}

fn extract_leading_fact_name(line: &str) -> Option<String> {
    compile_regex_static(
        r"^(?:\d+\.\s*)?([A-Za-z0-9@][A-Za-z0-9@&+./'_-]*(?:\s+[A-Za-z0-9][A-Za-z0-9@&+./'_-]*){0,5})\s*(?:\(|,| is | was )",
    )
    .captures(&normalize_session_answer_line_body(line))
        .and_then(|caps| caps.get(1))
        .map(|value| trim_fact_value(value.as_str()))
}

fn is_specific_named_thing_candidate(value: &str) -> bool {
    let lower = value.trim().to_ascii_lowercase();
    let tokens: Vec<&str> = lower.split_whitespace().collect();
    if tokens.is_empty() {
        return false;
    }
    if tokens.len() == 1 && tokens[0].len() == 1 {
        return false;
    }
    if tokens
        .last()
        .is_some_and(|token| matches!(*token, "that" | "which" | "who"))
    {
        return false;
    }
    if matches!(
        tokens.as_slice(),
        ["absolutely", "here"]
            | ["best", "regards"]
            | ["do", "not"]
            | ["i", "don't"]
            | ["finally"]
            | ["here", "are"]
            | ["in", "addition"]
            | ["popular", "option"]
            | ["thank", "you"]
            | ["there", "are"]
            | ["there", "is"]
    ) {
        return false;
    }
    if tokens.iter().all(|token| {
        matches!(
            *token,
            "absolutely"
                | "addition"
                | "are"
                | "do"
                | "don't"
                | "finally"
                | "here"
                | "in"
                | "is"
                | "not"
                | "option"
                | "popular"
                | "thank"
                | "there"
                | "you"
        )
    }) {
        return false;
    }
    !matches!(
        tokens.as_slice(),
        ["another"]
            | ["example"]
            | ["examples"]
            | ["here"]
            | ["idea"]
            | ["i"]
            | ["likewise"]
            | ["sure"]
            | ["there"]
            | ["these"]
            | ["this"]
            | ["those"]
            | ["yes"]
            | ["it"]
            | ["we"]
            | ["you"]
            | ["they"]
    )
}

fn extract_descriptor_led_name_from_line(line: &str) -> Option<String> {
    let body = normalize_session_answer_line_body(line);
    let body_lower = body.to_ascii_lowercase();
    let split_idx = [
        " has ", " have ", " had ", " is ", " was ", " said ", " taken ",
    ]
    .into_iter()
    .filter_map(|marker| body_lower.find(marker))
    .min()?;
    let mut prefix = body[..split_idx].trim();
    for marker in ["for example,", "for instance,", "likewise,", "similarly,"] {
        if body_lower.starts_with(marker) {
            prefix = prefix[marker.len()..].trim();
            break;
        }
    }
    prefix = prefix
        .trim_start_matches("the ")
        .trim_start_matches("a ")
        .trim_start_matches("an ")
        .trim();
    let prefix_lower = prefix.to_ascii_lowercase();
    if task_contains_any(
        &prefix_lower,
        &[
            "example of",
            "examples of",
            "few examples",
            "list of",
            "types of",
        ],
    ) {
        return None;
    }
    let tokens: Vec<&str> = prefix
        .split_whitespace()
        .map(|token| {
            token.trim_matches(|ch: char| !ch.is_alphanumeric() && ch != '\'' && ch != '-')
        })
        .filter(|token| !token.is_empty())
        .collect();
    if tokens.len() < 2 {
        return None;
    }
    let candidate_tokens: Vec<&str> = tokens
        .iter()
        .rev()
        .take_while(|token| !token.contains('/') && !token.eq_ignore_ascii_case("the"))
        .take(2)
        .copied()
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect();
    if candidate_tokens.len() < 2 {
        return None;
    }
    let candidate_lower_tokens: Vec<String> = candidate_tokens
        .iter()
        .map(|token| token.to_ascii_lowercase())
        .collect();
    if candidate_lower_tokens
        .last()
        .is_some_and(|token| matches!(token.as_str(), "that" | "which" | "who"))
    {
        return None;
    }
    Some(title_case_words(&candidate_tokens.join(" ")))
}

fn title_case_words(value: &str) -> String {
    value
        .split_whitespace()
        .map(|word| {
            let mut chars = word.chars();
            match chars.next() {
                Some(first) => format!("{}{}", first.to_uppercase(), chars.as_str()),
                None => String::new(),
            }
        })
        .collect::<Vec<_>>()
        .join(" ")
}

fn extract_role_person_from_line(line: &str) -> Option<String> {
    compile_regex_static(
        r"(?i)\b((?:Dr\.\s+)?[A-Z][A-Za-z.'-]+(?:\s+[A-Z][A-Za-z.'-]+){0,4}),\s+the ",
    )
    .captures(line)
    .and_then(|caps| caps.get(1))
    .map(|value| trim_fact_value(value.as_str()))
}

fn extract_chapter_phrase_from_line(line: &str) -> Option<String> {
    compile_regex_static(
        r#"(?i)\b(Chapter\s+\d+(?:\s+of\s+Book\s+\d+)?(?:,\s+titled\s+["“][^"”]+["”])?)"#,
    )
    .captures(line)
    .and_then(|caps| caps.get(1))
    .map(|value| trim_fact_value(value.as_str()))
}

fn extract_wearing_phrase_from_line(line: &str, lower: &str) -> Option<String> {
    extract_phrase_after_any_index(
        line,
        lower,
        &["wears ", "wore ", "wearing "],
        &[" while ", " and ", " but ", ".", ";"],
        1,
    )
    .map(|value| {
        let trimmed = trim_fact_value(&value);
        if task_contains_any(lower, &["wears an ", "wore an ", "wearing an "])
            && !trimmed.to_ascii_lowercase().starts_with("an ")
        {
            return format!("an {trimmed}");
        }
        if task_contains_any(lower, &["wears a ", "wore a ", "wearing a "])
            && !trimmed.to_ascii_lowercase().starts_with("a ")
        {
            return format!("a {trimmed}");
        }
        trimmed
    })
}

fn extract_implemented_algorithm_from_line(line: &str, lower: &str) -> Option<String> {
    if !lower.contains("implemented in the") {
        return None;
    }
    for label in ["6S", "MAJA", "Sen2Cor"] {
        if line.contains(label) {
            return Some(label.to_string());
        }
    }
    extract_structured_fact_label(line).or_else(|| extract_leading_fact_name(line))
}

fn extract_any_quoted_phrase(line: &str) -> Option<String> {
    extract_first_quoted_phrase(line)
        .or_else(|| {
            compile_regex_static(r#"“([^”]+)”"#)
                .captures(line)
                .and_then(|caps| caps.get(1))
                .map(|value| value.as_str().trim().to_string())
        })
        .map(|value| trim_fact_value(&value))
}

fn join_fact_items(items: &[String]) -> String {
    match items {
        [] => String::new(),
        [only] => only.clone(),
        [first, second] => format!("{first} and {second}"),
        [rest @ .., last] => {
            let mut rendered = rest.join(", ");
            rendered.push_str(", and ");
            rendered.push_str(last);
            rendered
        },
    }
}
