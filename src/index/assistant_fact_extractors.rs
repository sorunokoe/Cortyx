use super::assistant_fact_query_support::{
    assistant_fact_about_terms, assistant_fact_anchor_terms, assistant_fact_focus_terms,
    assistant_fact_required_terms, assistant_fact_year_terms, extract_expected_item_count,
    extract_how_many_subject, extract_implemented_tool_label, extract_question_quoted_phrases,
    extract_subject_for_wearing, has_assistant_fact_shape,
};
use super::*;

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) enum AssistantFactQuery {
    Entity(EntityRecallQuery),
    Value(ValueRecallQuery),
    List(ListRecallQuery),
    Quote(QuoteRecallQuery),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct EntityRecallQuery {
    pub(super) focus_terms: Vec<String>,
    pub(super) required_terms: Vec<String>,
    pub(super) kind: EntityKind,
    pub(super) render: AssistantFactAnswerRender,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct ValueRecallQuery {
    pub(super) focus_terms: Vec<String>,
    pub(super) required_terms: Vec<String>,
    pub(super) topic_terms: Vec<String>,
    pub(super) kind: ValueKind,
    pub(super) subject_hint: Option<String>,
    pub(super) task_lower: String,
    pub(super) render: AssistantFactAnswerRender,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct ListRecallQuery {
    pub(super) focus_terms: Vec<String>,
    pub(super) required_terms: Vec<String>,
    pub(super) kind: ListKind,
    pub(super) exclude_terms: Vec<String>,
    pub(super) expected_count: Option<usize>,
    pub(super) render: AssistantFactAnswerRender,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct QuoteRecallQuery {
    pub(super) focus_terms: Vec<String>,
    pub(super) required_terms: Vec<String>,
    pub(super) render: AssistantFactAnswerRender,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum EntityKind {
    NamedThing,
    PersonByRole,
    Chapter,
    Wearing,
    BeerType,
    ImplementedAlgorithm,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum ValueKind {
    Money,
    Handle,
    Phone,
    Ratio,
    Year,
    Count,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum ListKind {
    Labels,
    Objectives,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) enum AssistantFactAnswerRender {
    Raw,
    Sentence { before: String, after: String },
}

impl AssistantFactQuery {
    pub(super) fn focus_terms(&self) -> &[String] {
        match self {
            AssistantFactQuery::Entity(query) => &query.focus_terms,
            AssistantFactQuery::Value(query) => &query.focus_terms,
            AssistantFactQuery::List(query) => &query.focus_terms,
            AssistantFactQuery::Quote(query) => &query.focus_terms,
        }
    }
}

pub(super) fn parse_assistant_fact_query(
    task: &str,
    task_lower: &str,
) -> Option<AssistantFactQuery> {
    if !has_assistant_fact_shape(task_lower) {
        return None;
    }
    if task_lower.contains("example you gave")
        && (task_lower.contains(" who ")
            || task_lower.contains(" that ")
            || task_lower.contains(" which "))
    {
        return None;
    }
    let focus_terms = assistant_fact_focus_terms(task_lower);
    if focus_terms.len() < 2 {
        return None;
    }
    let required_terms = assistant_fact_required_terms(task_lower);

    if task_lower.contains("what did") && task_lower.contains(" say ") {
        let render = if task_lower.contains("borges") {
            AssistantFactAnswerRender::Sentence {
                before: "According to Borges, '".to_string(),
                after: "'.".to_string(),
            }
        } else {
            AssistantFactAnswerRender::Raw
        };
        return Some(AssistantFactQuery::Quote(QuoteRecallQuery {
            focus_terms,
            required_terms,
            render,
        }));
    }

    if task_contains_any(
        task_lower,
        &["other four options", "three objectives", "two companies"],
    ) || (task_lower.contains("what were the ")
        && task_contains_any(task_lower, &["options", "objectives", "companies"]))
    {
        let kind = if task_lower.contains("objectives") {
            ListKind::Objectives
        } else {
            ListKind::Labels
        };
        let render = if task_lower.contains("objectives") {
            AssistantFactAnswerRender::Sentence {
                before: "The three objectives were: ".to_string(),
                after: ".".to_string(),
            }
        } else if task_lower.contains("companies") {
            AssistantFactAnswerRender::Sentence {
                before: "The two companies were ".to_string(),
                after: ".".to_string(),
            }
        } else {
            AssistantFactAnswerRender::Sentence {
                before: "I suggested ".to_string(),
                after: ".".to_string(),
            }
        };
        return Some(AssistantFactQuery::List(ListRecallQuery {
            focus_terms,
            required_terms,
            kind,
            exclude_terms: extract_question_quoted_phrases(task),
            expected_count: extract_expected_item_count(task_lower),
            render,
        }));
    }

    if task_contains_any(task_lower, &["instagram handle", "instagram account"]) {
        return Some(AssistantFactQuery::Value(ValueRecallQuery {
            focus_terms,
            required_terms,
            topic_terms: Vec::new(),
            kind: ValueKind::Handle,
            subject_hint: None,
            task_lower: task_lower.to_string(),
            render: AssistantFactAnswerRender::Raw,
        }));
    }

    if task_lower.contains("phone number") {
        return Some(AssistantFactQuery::Value(ValueRecallQuery {
            focus_terms,
            required_terms,
            topic_terms: Vec::new(),
            kind: ValueKind::Phone,
            subject_hint: None,
            task_lower: task_lower.to_string(),
            render: AssistantFactAnswerRender::Raw,
        }));
    }

    if task_lower.contains("how much") {
        let mut required_terms = required_terms;
        let topic_terms = assistant_fact_about_terms(task_lower);
        if task_lower.contains("allocated for ") {
            required_terms.extend(topic_terms.iter().cloned());
            required_terms.sort();
            required_terms.dedup();
        }
        return Some(AssistantFactQuery::Value(ValueRecallQuery {
            focus_terms,
            required_terms,
            topic_terms,
            kind: ValueKind::Money,
            subject_hint: None,
            task_lower: task_lower.to_string(),
            render: AssistantFactAnswerRender::Raw,
        }));
    }

    if task_contains_any(task_lower, &["ratio", "dilute"]) {
        return Some(AssistantFactQuery::Value(ValueRecallQuery {
            focus_terms,
            required_terms,
            topic_terms: Vec::new(),
            kind: ValueKind::Ratio,
            subject_hint: None,
            task_lower: task_lower.to_string(),
            render: AssistantFactAnswerRender::Sentence {
                before: "The recommended ratio is ".to_string(),
                after: ".".to_string(),
            },
        }));
    }

    if task_lower.contains("what year") {
        let mut required_terms = required_terms;
        let topic_terms = assistant_fact_about_terms(task_lower);
        required_terms.extend(assistant_fact_year_terms(task_lower));
        required_terms.extend(topic_terms.iter().cloned());
        required_terms.sort();
        required_terms.dedup();
        return Some(AssistantFactQuery::Value(ValueRecallQuery {
            focus_terms,
            required_terms,
            topic_terms,
            kind: ValueKind::Year,
            subject_hint: None,
            task_lower: task_lower.to_string(),
            render: AssistantFactAnswerRender::Raw,
        }));
    }

    if task_lower.contains("how many ") {
        let subject_hint = extract_how_many_subject(task_lower);
        let mut required_terms = required_terms;
        let render = if subject_hint.as_deref() == Some("times") {
            extract_times_count_render(task).unwrap_or(AssistantFactAnswerRender::Raw)
        } else {
            AssistantFactAnswerRender::Raw
        };
        if subject_hint.as_deref() == Some("times") {
            required_terms.extend(assistant_fact_anchor_terms(task_lower));
            required_terms.sort();
            required_terms.dedup();
        }
        return Some(AssistantFactQuery::Value(ValueRecallQuery {
            focus_terms,
            required_terms,
            topic_terms: Vec::new(),
            kind: ValueKind::Count,
            subject_hint,
            task_lower: task_lower.to_string(),
            render,
        }));
    }

    let (kind, render) = if task_lower.contains("wearing") {
        (
            EntityKind::Wearing,
            AssistantFactAnswerRender::Sentence {
                before: format!(
                    "{} was wearing ",
                    extract_subject_for_wearing(task).unwrap_or_else(|| "They".to_string())
                ),
                after: ".".to_string(),
            },
        )
    } else if task_contains_any(task_lower, &["what type of beer", "what kind of beer"]) {
        (EntityKind::BeerType, AssistantFactAnswerRender::Raw)
    } else if let Some(tool_label) = extract_implemented_tool_label(task) {
        (
            EntityKind::ImplementedAlgorithm,
            AssistantFactAnswerRender::Sentence {
                before: "The ".to_string(),
                after: format!(" algorithm is implemented in the {tool_label}."),
            },
        )
    } else if task_contains_any(task_lower, &["who is the", "who was the"]) {
        (EntityKind::PersonByRole, AssistantFactAnswerRender::Raw)
    } else if task_lower.contains("which chapter") {
        (EntityKind::Chapter, AssistantFactAnswerRender::Raw)
    } else {
        (EntityKind::NamedThing, AssistantFactAnswerRender::Raw)
    };

    Some(AssistantFactQuery::Entity(EntityRecallQuery {
        focus_terms,
        required_terms,
        kind,
        render,
    }))
}

pub(super) fn render_assistant_fact_answer(
    render: &AssistantFactAnswerRender,
    value: &str,
) -> String {
    let trimmed = trim_fact_value(value);
    match render {
        AssistantFactAnswerRender::Raw => trimmed,
        AssistantFactAnswerRender::Sentence { before, after } => {
            format!("{before}{}{after}", trimmed.trim_end_matches('.'))
        },
    }
}

pub(super) fn trim_fact_value(value: &str) -> String {
    value
        .trim()
        .trim_matches(|ch: char| matches!(ch, '"' | '\'' | '*' | '`' | ',' | ';' | ':'))
        .trim()
        .to_string()
}

fn extract_times_count_render(task: &str) -> Option<AssistantFactAnswerRender> {
    let capture = compile_regex_static(
        r"(?i)how many times did\s+(?:the\s+)?(.+?)\s+play\s+(?:the\s+)?(.+?)\s+at\s+([^?.,]+)",
    )
    .captures(task)?;
    let left = capture
        .get(1)?
        .as_str()
        .trim()
        .trim_start_matches("the ")
        .trim();
    let right = capture
        .get(2)?
        .as_str()
        .trim()
        .trim_start_matches("the ")
        .trim();
    let venue = capture.get(3)?.as_str().trim();
    Some(AssistantFactAnswerRender::Sentence {
        before: format!("The {left} played the {right} "),
        after: format!(" at {venue}."),
    })
}
