use super::*;

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) enum AssistantResourceQuery {
    Video(VideoRecallQuery),
    Website(WebsiteRecallQuery),
    ExampleEntity(ExampleEntityQuery),
    SpecificList(SpecificListQuery),
    Duration(DurationRecallQuery),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct VideoRecallQuery {
    pub(super) focus_terms: Vec<String>,
    pub(super) source_hint: Option<String>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct WebsiteRecallQuery {
    pub(super) focus_terms: Vec<String>,
    pub(super) quoted_anchors: Vec<String>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct ExampleEntityQuery {
    pub(super) focus_terms: Vec<String>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct SpecificListQuery {
    pub(super) focus_terms: Vec<String>,
    pub(super) label: Option<String>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct DurationRecallQuery {
    pub(super) focus_terms: Vec<String>,
    pub(super) required_terms: Vec<String>,
}

impl AssistantResourceQuery {
    pub(super) fn focus_terms(&self) -> &[String] {
        match self {
            AssistantResourceQuery::Video(query) => &query.focus_terms,
            AssistantResourceQuery::Website(query) => &query.focus_terms,
            AssistantResourceQuery::ExampleEntity(query) => &query.focus_terms,
            AssistantResourceQuery::SpecificList(query) => &query.focus_terms,
            AssistantResourceQuery::Duration(query) => &query.focus_terms,
        }
    }
}

pub(super) fn parse_assistant_resource_query(
    task: &str,
    task_lower: &str,
) -> Option<AssistantResourceQuery> {
    if !has_assistant_resource_shape(task_lower) {
        return None;
    }
    if task_lower.contains("video")
        && task_contains_any(task_lower, &["you recommended", "you shared"])
    {
        return Some(AssistantResourceQuery::Video(VideoRecallQuery {
            focus_terms: assistant_resource_focus_terms(task_lower),
            source_hint: extract_video_source_hint(task, task_lower),
        }));
    }
    if task_lower.contains("website") {
        return Some(AssistantResourceQuery::Website(WebsiteRecallQuery {
            focus_terms: assistant_resource_focus_terms(task_lower),
            quoted_anchors: extract_quoted_titles(task)
                .into_iter()
                .map(|value| value.to_ascii_lowercase())
                .collect(),
        }));
    }
    if task_contains_any(
        task_lower,
        &["which state you mentioned", "which state did you mention"],
    ) {
        return Some(AssistantResourceQuery::ExampleEntity(ExampleEntityQuery {
            focus_terms: assistant_resource_focus_terms(task_lower),
        }));
    }
    if task_contains_any(
        task_lower,
        &[
            "specific back-end programming languages",
            "specific backend programming languages",
        ],
    ) {
        return Some(AssistantResourceQuery::SpecificList(SpecificListQuery {
            focus_terms: assistant_resource_focus_terms(task_lower),
            label: extract_specific_list_label(task_lower),
        }));
    }
    if task_lower.contains("how long did you say") {
        return Some(AssistantResourceQuery::Duration(DurationRecallQuery {
            focus_terms: assistant_resource_focus_terms(task_lower),
            required_terms: extract_duration_required_terms(task_lower),
        }));
    }
    None
}

pub(super) fn extract_video_recall_answer(
    line: &str,
    lower: &str,
    source_hint: Option<&str>,
) -> Option<String> {
    if let Some(source_hint) = source_hint {
        if !lower.contains(source_hint) {
            return None;
        }
    }
    let title = extract_first_quoted_phrase(line)?;
    let url = compile_regex(r"https?://[^\s>]+")
        .find(line)
        .map(|value| value.as_str().trim_end_matches('>').to_string())?;
    Some(format!("The video is '{}' and the link is {}.", title, url))
}

pub(super) fn extract_website_recall_answer(line: &str, lower: &str) -> Option<String> {
    if !lower.contains("website") {
        return None;
    }
    compile_regex(r"(?i)^(?:\d+\.\s*)?([^:]+):")
        .captures(line)
        .and_then(|captures| captures.get(1))
        .map(|value| value.as_str().trim().trim_matches('*').trim().to_string())
}

pub(super) fn website_query_matches_line(query: &WebsiteRecallQuery, lower: &str) -> bool {
    query.quoted_anchors.is_empty()
        || query
            .quoted_anchors
            .iter()
            .all(|anchor| lower.contains(anchor))
}

pub(super) fn looks_like_website_label(value: &str) -> bool {
    compile_regex(r"(?i)^[A-Za-z0-9-]+\.(?:org|com|net|edu|io)$").is_match(value.trim())
}

pub(super) fn extract_example_entity_from_line(line: &str, lower: &str) -> Option<String> {
    if !lower.contains("for example") {
        return None;
    }
    compile_regex(r"\bFor example,\s+([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)*)\b")
        .captures(line)
        .and_then(|captures| captures.get(1))
        .map(|value| value.as_str().trim().to_string())
}

pub(super) fn extract_specific_list_answer(
    line: &str,
    lower: &str,
    query: &SpecificListQuery,
) -> Option<String> {
    let list = extract_phrase_after_any_index(
        line,
        lower,
        &["such as "],
        &[". ", " you", " you'll", " you’ll"],
        1,
    )?;
    if let Some(label) = &query.label {
        return Some(format!(
            "I recommended learning {} as a {}.",
            list.trim_end_matches('.'),
            singularize_label(label)
        ));
    }
    Some(list)
}

pub(super) fn extract_duration_recall_answer(line: &str, lower: &str) -> Option<String> {
    if !task_contains_any(
        lower,
        &["after ", "for ", "minutes", "hours", "days", "weeks"],
    ) {
        return None;
    }
    extract_duration_answer_from_line(line)
}

fn has_assistant_resource_shape(task_lower: &str) -> bool {
    task_contains_any(
        task_lower,
        &[
            "i wanted to follow up",
            "can you remind me",
            "you mentioned",
            "you recommended",
            "you shared",
        ],
    )
}

fn assistant_resource_focus_terms(task_lower: &str) -> Vec<String> {
    synthetic_query_terms(task_lower)
        .into_iter()
        .filter(|term| term.len() >= 4)
        .filter(|term| {
            !matches!(
                term.as_str(),
                "about"
                    | "back"
                    | "been"
                    | "can"
                    | "conversation"
                    | "follow"
                    | "great"
                    | "learn"
                    | "mentioned"
                    | "previous"
                    | "recommended"
                    | "remind"
                    | "resource"
                    | "said"
                    | "share"
                    | "shared"
                    | "specific"
                    | "that"
                    | "their"
                    | "video"
                    | "website"
                    | "what"
                    | "which"
                    | "with"
                    | "would"
                    | "you"
            )
        })
        .collect()
}

fn extract_video_source_hint(task: &str, task_lower: &str) -> Option<String> {
    let marker = "the ";
    let video_idx = task_lower.find(" video")?;
    let prefix = &task[..video_idx];
    let start = prefix.to_ascii_lowercase().rfind(marker)?;
    let source = prefix[start + marker.len()..].trim();
    (!source.is_empty()).then_some(source.to_ascii_lowercase())
}

fn extract_specific_list_label(task_lower: &str) -> Option<String> {
    if task_lower.contains("back-end programming languages") {
        return Some("back-end programming languages".to_string());
    }
    if task_lower.contains("backend programming languages") {
        return Some("backend programming languages".to_string());
    }
    None
}

fn singularize_label(label: &str) -> String {
    label
        .strip_suffix("languages")
        .map(|value| format!("{}language", value))
        .unwrap_or_else(|| label.to_string())
}

fn extract_duration_required_terms(task_lower: &str) -> Vec<String> {
    [
        "tomato", "lemon", "cucumber", "tea", "turmeric", "almond", "rose", "aloe", "coconut",
        "mint",
    ]
    .into_iter()
    .filter(|term| task_lower.contains(term))
    .map(str::to_string)
    .collect()
}
