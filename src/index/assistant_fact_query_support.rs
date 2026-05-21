use super::assistant_fact_extractors::trim_fact_value;
use super::*;

pub(super) fn has_assistant_fact_shape(task_lower: &str) -> bool {
    task_contains_any(
        task_lower,
        &[
            "remind me",
            "previous conversation",
            "previous chat",
            "last time",
            "follow up",
            "follow-up",
            "you mentioned",
            "you recommended",
            "you suggested",
            "you told me",
            "you provided",
        ],
    )
}

pub(super) fn assistant_fact_focus_terms(task_lower: &str) -> Vec<String> {
    synthetic_query_terms(task_lower)
        .into_iter()
        .filter(|term| term.len() >= 4 || term.chars().any(|ch| ch.is_ascii_digit()))
        .filter(|term| {
            !matches!(
                term.as_str(),
                "about"
                    | "again"
                    | "back"
                    | "chat"
                    | "clarify"
                    | "confirm"
                    | "conversation"
                    | "could"
                    | "going"
                    | "last"
                    | "looking"
                    | "mentioned"
                    | "name"
                    | "planning"
                    | "previous"
                    | "provided"
                    | "recommended"
                    | "remember"
                    | "remind"
                    | "suggested"
                    | "talked"
                    | "through"
                    | "time"
                    | "told"
                    | "want"
                    | "wanted"
                    | "what"
                    | "which"
                    | "wondering"
                    | "would"
                    | "you"
            )
        })
        .collect()
}

pub(super) fn assistant_fact_required_terms(task_lower: &str) -> Vec<String> {
    for source in [
        named_subject_clause(task_lower),
        descriptor_clause(task_lower),
        clause_after(task_lower, "allocated for "),
        clause_after(task_lower, "for the "),
        clause_after(task_lower, "who is the "),
        clause_after(task_lower, "who was the "),
        clause_after(task_lower, "about "),
        Some(task_lower),
    ] {
        let Some(source) = source else {
            continue;
        };
        let terms = required_terms_from_source(source);
        if !terms.is_empty() {
            return terms;
        }
    }
    Vec::new()
}

pub(super) fn assistant_fact_about_terms(task_lower: &str) -> Vec<String> {
    clause_after(task_lower, "about ")
        .map(required_terms_from_source)
        .unwrap_or_default()
}

pub(super) fn assistant_fact_year_terms(task_lower: &str) -> Vec<String> {
    clause_after(task_lower, "what year the ")
        .map(required_terms_from_source)
        .unwrap_or_default()
}

pub(super) fn extract_expected_item_count(task_lower: &str) -> Option<usize> {
    compile_regex_static(r"\b(one|two|three|four|five|six|\d+)\b")
        .captures(task_lower)
        .and_then(|caps| caps.get(1))
        .and_then(|value| match value.as_str() {
            "one" => Some(1),
            "two" => Some(2),
            "three" => Some(3),
            "four" => Some(4),
            "five" => Some(5),
            "six" => Some(6),
            digits => digits.parse::<usize>().ok(),
        })
}

pub(super) fn extract_question_quoted_phrases(task: &str) -> Vec<String> {
    compile_regex_static(r#"["“']([^"”']+)["”']"#)
        .captures_iter(task)
        .filter_map(|caps| caps.get(1))
        .map(|value| trim_fact_value(value.as_str()))
        .collect()
}

pub(super) fn extract_how_many_subject(task_lower: &str) -> Option<String> {
    compile_regex_static(r"how many\s+([a-z-]+)")
        .captures(task_lower)
        .and_then(|caps| caps.get(1))
        .map(|value| value.as_str().trim().to_string())
}

pub(super) fn assistant_fact_anchor_terms(task_lower: &str) -> Vec<String> {
    let Some((_, tail)) = task_lower.rsplit_once(" at ") else {
        return Vec::new();
    };
    let segment = tail.split(['.', '?', '!', ',']).next().unwrap_or("").trim();
    let terms: Vec<String> = synthetic_query_terms(segment)
        .into_iter()
        .filter(|term| term.len() >= 4)
        .collect();
    if (1..=4).contains(&terms.len()) {
        terms
    } else {
        Vec::new()
    }
}

pub(super) fn extract_subject_for_wearing(task: &str) -> Option<String> {
    compile_regex_static(r"(?i)what was ([A-Za-z][A-Za-z' -]+?) wearing")
        .captures(task)
        .and_then(|caps| caps.get(1))
        .map(|value| value.as_str().trim().to_string())
}

pub(super) fn extract_implemented_tool_label(task: &str) -> Option<String> {
    compile_regex_static(r"(?i)implemented in the ([^?.,]+)")
        .captures(task)
        .and_then(|caps| caps.get(1))
        .map(|value| value.as_str().trim().trim_end_matches('.').to_string())
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
        .or_else(|| {
            task_lower
                .rfind(" who ")
                .map(|idx| &task_lower[idx + " who ".len()..])
        })
}

fn named_subject_clause(task_lower: &str) -> Option<&str> {
    for marker in ["name of that ", "name of the "] {
        let Some((_, tail)) = task_lower.split_once(marker) else {
            continue;
        };
        let stop = [" that ", " who ", " which ", " we ", " you "]
            .into_iter()
            .filter_map(|delimiter| tail.find(delimiter))
            .min()
            .unwrap_or(tail.len());
        let clause = tail[..stop].trim();
        if !clause.is_empty() {
            return Some(clause);
        }
    }
    None
}

fn clause_after<'a>(task_lower: &'a str, marker: &str) -> Option<&'a str> {
    let (_, rest) = task_lower.split_once(marker)?;
    let stop = rest
        .char_indices()
        .find_map(|(idx, ch)| matches!(ch, '?' | '.' | ',').then_some(idx))
        .unwrap_or(rest.len());
    let clause = rest[..stop].trim();
    (!clause.is_empty()).then_some(clause)
}

fn required_terms_from_source(source: &str) -> Vec<String> {
    synthetic_query_terms(source)
        .into_iter()
        .filter(|term| term.len() >= 5 || term.chars().any(|ch| ch.is_ascii_digit()))
        .filter(|term| {
            !matches!(
                term.as_str(),
                "about"
                    | "again"
                    | "allocation"
                    | "chat"
                    | "could"
                    | "conversation"
                    | "earlier"
                    | "mentioned"
                    | "movie"
                    | "number"
                    | "planning"
                    | "previous"
                    | "provided"
                    | "question"
                    | "recommended"
                    | "remember"
                    | "remind"
                    | "revisit"
                    | "scene"
                    | "script"
                    | "suggested"
                    | "through"
                    | "wanted"
                    | "which"
                    | "wondering"
                    | "would"
                    | "you"
            )
        })
        .take(5)
        .collect()
}
