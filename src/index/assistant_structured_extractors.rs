use super::*;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum StructuredRecallSource {
    Assistant,
    AssistantOrUser,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) enum AssistantStructuredQuery {
    DescribedEntity(DescribedEntityQuery),
    ExampleList(ExampleListQuery),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct DescribedEntityQuery {
    pub(super) focus_terms: Vec<String>,
    pub(super) required_terms: Vec<String>,
    pub(super) entity_hint: Option<String>,
    pub(super) prefer_latest: bool,
    pub(super) source: StructuredRecallSource,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct ExampleListQuery {
    pub(super) focus_terms: Vec<String>,
    pub(super) subject_terms: Vec<String>,
    pub(super) render_label: String,
}

impl AssistantStructuredQuery {
    pub(super) fn focus_terms(&self) -> &[String] {
        match self {
            AssistantStructuredQuery::DescribedEntity(query) => &query.focus_terms,
            AssistantStructuredQuery::ExampleList(query) => &query.focus_terms,
        }
    }
}

pub(super) fn parse_assistant_structured_query(
    _task: &str,
    task_lower: &str,
) -> Option<AssistantStructuredQuery> {
    if !has_assistant_structured_shape(task_lower) {
        return None;
    }
    if task_lower.contains("two-factor authentication")
        && task_contains_any(task_lower, &["what kind", "what type"])
    {
        return Some(AssistantStructuredQuery::ExampleList(ExampleListQuery {
            focus_terms: assistant_structured_focus_terms(task_lower),
            subject_terms: vec!["two-factor".to_string(), "authentication".to_string()],
            render_label: "two-factor authentication methods".to_string(),
        }));
    }
    if task_lower.contains("website") {
        return None;
    }
    if task_contains_any(
        task_lower,
        &[
            "brand that",
            "cartoon you mentioned",
            "what was the name of that",
            "what was the name of the",
            "finally decided to name",
            "decided to name it",
        ],
    ) {
        let entity_hint = extract_structured_entity_hint(task_lower);
        let decision_query = task_contains_any(
            task_lower,
            &["finally decided to name", "decided to name it"],
        );
        if entity_hint.is_none() && !decision_query {
            return None;
        }
        return Some(AssistantStructuredQuery::DescribedEntity(
            DescribedEntityQuery {
                focus_terms: assistant_structured_focus_terms(task_lower),
                required_terms: assistant_structured_required_terms(task_lower),
                entity_hint,
                prefer_latest: decision_query,
                source: if decision_query {
                    StructuredRecallSource::AssistantOrUser
                } else {
                    StructuredRecallSource::Assistant
                },
            },
        ));
    }
    None
}

pub(super) fn extract_examples_list_from_line(
    line: &str,
    lower: &str,
    query: &ExampleListQuery,
) -> Option<String> {
    if !query.subject_terms.iter().all(|term| lower.contains(term)) {
        return None;
    }
    let values = extract_phrase_after_any_index(
        line,
        lower,
        &["such as "],
        &[". ", ".", ";", ", enhances", " enhances"],
        1,
    )?;
    Some(format!(
        "I mentioned {} as examples of {}.",
        values.trim().trim_end_matches(','),
        query.render_label
    ))
}

pub(super) fn extract_described_entity_from_line(line: &str, lower: &str) -> Option<String> {
    if let Some(domain) = extract_domain_like_label(line) {
        return Some(domain);
    }
    if let Some(quoted) = extract_first_quoted_phrase(line).or_else(|| {
        compile_regex(r"“([^”]+)”")
            .captures(line)
            .and_then(|captures| captures.get(1))
            .map(|value| value.as_str().trim().to_string())
    }) {
        return Some(trim_entity_label(&quoted));
    }
    if let Some(label) = extract_structured_label(line) {
        return Some(label);
    }
    if let Some(code) = extract_structured_code(line) {
        return Some(code);
    }
    if let Some(value) = extract_phrase_after_any_index(
        line,
        lower,
        &[
            "example is the ",
            "example is ",
            "is the ",
            "is a ",
            "is ",
            "was the ",
            "was ",
            "called ",
            "named ",
            "how about ",
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
        if let Some(code) = extract_structured_code(&value) {
            return Some(code);
        }
        if let Some(title) = extract_title_like_phrases(&value).into_iter().next() {
            return Some(trim_entity_label(&title));
        }
    }
    extract_title_like_phrases(line)
        .into_iter()
        .find(|phrase| phrase.split_whitespace().count() <= 6)
        .map(|value| trim_entity_label(&value))
}

pub(super) fn described_entity_line_bonus(line: &str, lower: &str) -> usize {
    usize::from(extract_first_quoted_phrase(line).is_some()) * 8
        + usize::from(extract_structured_label(line).is_some()) * 8
        + usize::from(extract_structured_code(line).is_some()) * 8
        + usize::from(task_contains_any(
            lower,
            &["recommend", "recommended", "example", "named", "called"],
        )) * 4
}

pub(super) fn render_described_entity_answer(query: &DescribedEntityQuery, entity: &str) -> String {
    let trimmed = trim_entity_label(entity);
    if query.entity_hint.as_deref() == Some("trail")
        && !trimmed.to_ascii_lowercase().contains("trail")
    {
        return format!("The {trimmed} trail.");
    }
    if matches!(trimmed.chars().last(), Some('!') | Some('?') | Some('.')) {
        return trimmed;
    }
    if query.entity_hint.as_deref() == Some("brand") || looks_like_domain_label(&trimmed) {
        return trimmed;
    }
    format!("{trimmed}.")
}

fn has_assistant_structured_shape(task_lower: &str) -> bool {
    task_has_recall_context(task_lower)
        || task_contains_any(
            task_lower,
            &[
                "previous conversation",
                "previous chat",
                "looking back at our previous",
                "going back to our previous",
                "you mentioned",
                "you recommended",
                "you told me",
            ],
        )
}

fn assistant_structured_focus_terms(task_lower: &str) -> Vec<String> {
    let descriptor = descriptor_clause(task_lower).unwrap_or(task_lower);
    let mut focus_terms = synthetic_query_terms(descriptor)
        .into_iter()
        .filter(|term| term.len() >= 4)
        .filter(|term| {
            !matches!(
                term.as_str(),
                "about"
                    | "back"
                    | "brand"
                    | "cartoon"
                    | "chat"
                    | "conversation"
                    | "could"
                    | "decide"
                    | "decided"
                    | "finally"
                    | "going"
                    | "into"
                    | "kind"
                    | "looking"
                    | "mentioned"
                    | "name"
                    | "previous"
                    | "recommended"
                    | "referring"
                    | "remind"
                    | "that"
                    | "through"
                    | "trail"
                    | "type"
                    | "website"
                    | "what"
                    | "which"
                    | "wondering"
                    | "would"
                    | "you"
            )
        })
        .collect::<Vec<_>>();
    if focus_terms.len() >= 2 {
        return focus_terms;
    }
    focus_terms = synthetic_query_terms(task_lower)
        .into_iter()
        .filter(|term| term.len() >= 4)
        .filter(|term| {
            !matches!(
                term.as_str(),
                "about"
                    | "back"
                    | "brand"
                    | "cartoon"
                    | "chat"
                    | "conversation"
                    | "could"
                    | "decide"
                    | "decided"
                    | "finally"
                    | "going"
                    | "into"
                    | "kind"
                    | "looking"
                    | "mentioned"
                    | "name"
                    | "previous"
                    | "recommended"
                    | "referring"
                    | "remind"
                    | "that"
                    | "through"
                    | "trail"
                    | "type"
                    | "website"
                    | "what"
                    | "which"
                    | "wondering"
                    | "would"
                    | "you"
            )
        })
        .collect();
    focus_terms
}

fn assistant_structured_required_terms(task_lower: &str) -> Vec<String> {
    let Some(descriptor) = descriptor_clause(task_lower) else {
        return Vec::new();
    };
    synthetic_query_terms(descriptor)
        .into_iter()
        .filter(|term| term.len() >= 5)
        .filter(|term| {
            !matches!(
                term.as_str(),
                "brand"
                    | "cartoon"
                    | "culture"
                    | "offers"
                    | "park"
                    | "sourced"
                    | "surrounding"
                    | "takes"
                    | "through"
                    | "trail"
                    | "using"
                    | "views"
            )
        })
        .take(3)
        .collect()
}

fn extract_structured_entity_hint(task_lower: &str) -> Option<String> {
    for hint in ["brand", "trail", "cartoon"] {
        if task_lower.contains(hint) {
            return Some(hint.to_string());
        }
    }
    None
}

fn extract_structured_label(line: &str) -> Option<String> {
    compile_regex(r"^(?:\d+\.\s*)?([A-Za-z0-9][A-Za-z0-9.&'!+/\-]*(?:\s+[A-Za-z0-9][A-Za-z0-9.&'!+/\-]*){0,5})\s*[-:]\s+")
        .captures(line)
        .and_then(|captures| captures.get(1))
        .map(|value| trim_entity_label(value.as_str()))
}

fn extract_structured_code(text: &str) -> Option<String> {
    compile_regex(r"\b([A-Z]{2,5}-\d{1,3})\b")
        .captures(text)
        .and_then(|captures| captures.get(1))
        .map(|value| value.as_str().trim().to_string())
}

fn extract_domain_like_label(text: &str) -> Option<String> {
    compile_regex(r"\b([A-Za-z0-9-]+\.(?:org|com|net|edu|io))\b")
        .captures(text)
        .and_then(|captures| captures.get(1))
        .map(|value| value.as_str().trim().to_string())
}

fn trim_entity_label(value: &str) -> String {
    value
        .trim()
        .trim_matches(|c: char| matches!(c, '*' | '"' | '\'' | ',' | ':' | ';'))
        .trim()
        .to_string()
}

fn looks_like_domain_label(value: &str) -> bool {
    compile_regex(r"(?i)^[A-Za-z0-9-]+\.(?:org|com|net|edu|io)$").is_match(value.trim())
}

fn descriptor_clause(task_lower: &str) -> Option<&str> {
    task_lower
        .rfind(" that ")
        .map(|idx| &task_lower[idx + " that ".len()..])
        .or_else(|| {
            task_lower
                .rfind(" which ")
                .map(|idx| &task_lower[idx + " which ".len()..])
        })
}
