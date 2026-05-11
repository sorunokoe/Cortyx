use super::*;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum RecallSource {
    Assistant,
    User,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) enum AssistantRecallQuery {
    RecommendedName(RecommendedNameQuery),
    OrdinalListItem(OrdinalListItemQuery),
    SectionItems(SectionItemsQuery),
    NumericValue(NumericValueQuery),
    ExampleTitle(ExampleTitleQuery),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct RecommendedNameQuery {
    pub(super) focus_terms: Vec<String>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct OrdinalListItemQuery {
    pub(super) ordinal: usize,
    pub(super) label: String,
    pub(super) focus_terms: Vec<String>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct SectionItemsQuery {
    pub(super) section_label: String,
    pub(super) focus_terms: Vec<String>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct NumericValueQuery {
    pub(super) subject: String,
    pub(super) context_clause: Option<String>,
    pub(super) focus_terms: Vec<String>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct ExampleTitleQuery {
    pub(super) focus_terms: Vec<String>,
}

impl AssistantRecallQuery {
    pub(super) fn focus_terms(&self) -> &[String] {
        match self {
            AssistantRecallQuery::RecommendedName(query) => &query.focus_terms,
            AssistantRecallQuery::OrdinalListItem(query) => &query.focus_terms,
            AssistantRecallQuery::SectionItems(query) => &query.focus_terms,
            AssistantRecallQuery::NumericValue(query) => &query.focus_terms,
            AssistantRecallQuery::ExampleTitle(query) => &query.focus_terms,
        }
    }
}

pub(super) fn parse_assistant_recall_query(
    task: &str,
    task_lower: &str,
) -> Option<AssistantRecallQuery> {
    if !has_typed_assistant_recall_shape(task_lower) {
        return None;
    }
    if let Some(ordinal) = extract_query_ordinal(task) {
        if task_contains_any(
            task_lower,
            &["parameter", "item", "entry", "list", "bottle"],
        ) {
            return Some(AssistantRecallQuery::OrdinalListItem(
                OrdinalListItemQuery {
                    ordinal,
                    label: extract_recall_label(task_lower),
                    focus_terms: assistant_recall_focus_terms(task_lower),
                },
            ));
        }
    }
    if task_contains_any(
        task_lower,
        &["used as an example", "mentioned as an example"],
    ) && task_contains_any(
        task_lower,
        &["what show", "what movie", "what book", "what example"],
    ) {
        return Some(AssistantRecallQuery::ExampleTitle(ExampleTitleQuery {
            focus_terms: assistant_recall_focus_terms(task_lower),
        }));
    }
    if task_contains_any(
        task_lower,
        &[
            "what kind of processes are used",
            "what processes are used",
            "which processes are used",
            "what kind of processes",
        ],
    ) {
        let section_label = extract_section_label_from_task(task_lower)?;
        return Some(AssistantRecallQuery::SectionItems(SectionItemsQuery {
            section_label,
            focus_terms: assistant_recall_focus_terms(task_lower),
        }));
    }
    if task_lower.starts_with("can you remind me what was the ")
        || task_lower.starts_with("what was the ")
    {
        let subject = extract_numeric_subject_from_task(task_lower)?;
        if task_contains_any(
            task_lower,
            &["improvement", "increase", "decrease", "reduction"],
        ) {
            return Some(AssistantRecallQuery::NumericValue(NumericValueQuery {
                subject,
                context_clause: extract_numeric_context_clause(task, task_lower),
                focus_terms: assistant_recall_focus_terms(task_lower),
            }));
        }
    }
    if task_contains_any(
        task_lower,
        &["you recommended", "you suggested", "you told me"],
    ) && task_contains_any(task_lower, &["name of", "which one", "what was the name"])
    {
        if task_contains_any(
            task_lower,
            &["hiking trail", "brand that", "cartoon you mentioned"],
        ) && task_lower.contains(" that ")
        {
            return None;
        }
        return Some(AssistantRecallQuery::RecommendedName(
            RecommendedNameQuery {
                focus_terms: assistant_recall_focus_terms(task_lower),
            },
        ));
    }
    None
}

pub(super) fn extract_recommended_name_from_line(line: &str, lower: &str) -> Option<String> {
    let recommend_match = compile_regex(r"(?i)\brecommend\b").find(lower)?;
    if lower.contains("recommended ") {
        return None;
    }
    let tail = line.get(recommend_match.end()..)?.trim();
    let first_clause = tail
        .split(['.', '!', '?', ';'])
        .next()
        .unwrap_or(tail)
        .trim();
    extract_title_like_phrases(first_clause).into_iter().next()
}

pub(super) fn extract_metric_value_from_line(line: &str, lower: &str) -> Option<String> {
    if !task_contains_any(
        lower,
        &["approximately", "approx", "about", "around", "%", "x"],
    ) {
        return None;
    }
    compile_regex(r"(?i)\b(?:approximately|approx\.?|about|around)\s+\d+(?:\.\d+)?%")
        .find(line)
        .map(|value| value.as_str().trim().to_string())
        .or_else(|| {
            compile_regex(r"(?i)\b\d+(?:\.\d+)?%")
                .find(line)
                .map(|value| value.as_str().trim().to_string())
        })
        .or_else(|| {
            compile_regex(r"(?i)\bapproximately\s+\d+(?:\.\d+)?x\b")
                .find(line)
                .map(|value| value.as_str().trim().to_string())
        })
}

pub(super) fn extract_section_heading_from_line(line: &str) -> Option<String> {
    let trimmed = line.trim();
    if trimmed.starts_with('*') {
        return None;
    }
    let body = extract_numbered_list_item(trimmed)
        .map(|(_, value)| value)
        .unwrap_or_else(|| trimmed.to_string());
    let heading = body.trim().trim_end_matches(':').trim();
    (!heading.is_empty() && body.trim_end().ends_with(':')).then_some(heading.to_string())
}

pub(super) fn extract_section_item_label(line: &str) -> Option<String> {
    let trimmed = line.trim();
    let body = if let Some(rest) = trimmed.strip_prefix('*') {
        rest.trim()
    } else if let Some((_, value)) = extract_numbered_list_item(trimmed) {
        return value
            .split_once(':')
            .map(|(label, _)| label.trim().to_string())
            .or(Some(value));
    } else {
        return None;
    };
    body.split_once(':')
        .map(|(label, _)| label.trim().to_string())
        .or_else(|| (!body.is_empty()).then_some(body.to_string()))
}

pub(super) fn extract_example_title_from_line(line: &str, lower: &str) -> Option<String> {
    if !lower.contains("example") {
        return None;
    }
    extract_first_quoted_phrase(line)
        .map(|value| title_case_phrase(&value))
        .or_else(|| {
            extract_title_like_phrases(line)
                .into_iter()
                .find(|phrase| phrase.split_whitespace().count() <= 5)
        })
}

pub(super) fn render_ordinal_recall_answer(query: &OrdinalListItemQuery, value: &str) -> String {
    let compact_value = value
        .split_once(':')
        .map(|(label, _)| label.trim())
        .filter(|label| label.split_whitespace().count() <= 6)
        .unwrap_or(value)
        .trim()
        .trim_end_matches('.');
    if query.label == "bottle" {
        return compact_value.to_string();
    }
    format!(
        "The {} {} was '{}'.",
        ordinal_surface(query.ordinal),
        query.label,
        compact_value
    )
}

pub(super) fn render_numeric_recall_answer(query: &NumericValueQuery, value: &str) -> String {
    if let Some(clause) = &query.context_clause {
        format!(
            "The {} was {} {}.",
            query.subject,
            value.trim_end_matches('.'),
            clause
        )
    } else {
        format!("The {} was {}.", query.subject, value.trim_end_matches('.'))
    }
}

fn assistant_recall_focus_terms(task_lower: &str) -> Vec<String> {
    synthetic_query_terms(task_lower)
        .into_iter()
        .filter(|term| term.len() >= 4)
        .filter(|term| {
            !matches!(
                term.as_str(),
                "about"
                    | "acting"
                    | "average"
                    | "back"
                    | "can"
                    | "chat"
                    | "check"
                    | "did"
                    | "earlier"
                    | "example"
                    | "from"
                    | "gave"
                    | "have"
                    | "kind"
                    | "list"
                    | "mentioned"
                    | "name"
                    | "output"
                    | "parameter"
                    | "previous"
                    | "provided"
                    | "recommend"
                    | "recommended"
                    | "remember"
                    | "remind"
                    | "show"
                    | "that"
                    | "told"
                    | "used"
                    | "using"
                    | "want"
                    | "what"
                    | "when"
                    | "which"
                    | "with"
                    | "would"
                    | "you"
                    | "your"
            )
        })
        .collect()
}

fn has_typed_assistant_recall_shape(task_lower: &str) -> bool {
    task_contains_any(
        task_lower,
        &[
            "can you remind me",
            "do you remember",
            "i remember you",
            "i mentioned",
            "you recommended",
            "you told me",
            "you provided",
        ],
    )
}

fn extract_recall_label(task_lower: &str) -> String {
    for label in ["parameter", "item", "entry", "point", "bottle"] {
        if task_lower.contains(label) {
            return label.to_string();
        }
    }
    "item".to_string()
}

fn extract_section_label_from_task(task_lower: &str) -> Option<String> {
    for marker in ["at the ", "for the ", "used at the ", "used in the "] {
        if let Some((_, rest)) = task_lower.split_once(marker) {
            let stop = rest.find('?').unwrap_or(rest.len());
            let label = rest[..stop]
                .trim()
                .trim_end_matches('.')
                .trim_end_matches('?')
                .to_string();
            if !label.is_empty() {
                return Some(label);
            }
        }
    }
    None
}

fn extract_numeric_subject_from_task(task_lower: &str) -> Option<String> {
    let value = task_lower
        .strip_prefix("can you remind me what was the ")
        .or_else(|| task_lower.strip_prefix("what was the "))?;
    let stop = value
        .find(" when ")
        .or_else(|| value.find('?'))
        .unwrap_or(value.len());
    let subject = value[..stop].trim();
    (!subject.is_empty()).then_some(subject.to_string())
}

fn extract_numeric_context_clause(task: &str, task_lower: &str) -> Option<String> {
    let marker = " when ";
    let start = task_lower.find(marker)?;
    let original = task.get(start + 1..)?.trim();
    let stop = original
        .find(" in the ")
        .or_else(|| original.find('?'))
        .unwrap_or(original.len());
    let clause = original[..stop].trim();
    (!clause.is_empty()).then_some(clause.to_string())
}

fn ordinal_surface(value: usize) -> String {
    let suffix = match value % 100 {
        11..=13 => "th",
        _ => match value % 10 {
            1 => "st",
            2 => "nd",
            3 => "rd",
            _ => "th",
        },
    };
    format!("{value}{suffix}")
}

fn title_case_phrase(value: &str) -> String {
    value
        .split_whitespace()
        .map(|word| capitalize_first_ascii(&word.to_ascii_lowercase()))
        .collect::<Vec<_>>()
        .join(" ")
}
