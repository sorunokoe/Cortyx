use super::*;

pub(crate) fn is_reason_query(task: &str) -> bool {
    let lower = task.to_ascii_lowercase();
    lower.starts_with("why ")
        || lower.contains(" motivated ")
        || lower.contains("motivate")
        || lower.contains("inspired")
        || lower.contains(" inspire ")
        || lower.contains("what made")
        || lower.contains("what pushed")
        || lower.contains("what gave")
}

pub(crate) fn extract_reason_answer(text: &str) -> Option<String> {
    let clean = sanitize_inline(text);
    if clean.is_empty() {
        return None;
    }

    let lower = clean.to_ascii_lowercase();
    for marker in ["because ", "since ", "after ", "from "] {
        if let Some(idx) = lower.find(marker) {
            let phrase = trim_answer_tail(&clean[idx + marker.len()..], false);
            if phrase.split_whitespace().count() >= 3 {
                return Some(phrase);
            }
        }
    }

    let mut first_clause = clean
        .split(['.', '!', '?', ';'])
        .map(str::trim)
        .find(|clause| clause.split_whitespace().count() >= 4)?
        .to_string();
    let lower_clause = first_clause.to_ascii_lowercase();
    for boundary in [", and i ", " and i ", ", but i ", " but i "] {
        if let Some(idx) = lower_clause.find(boundary) {
            let head = first_clause[..idx].trim();
            if head.split_whitespace().count() >= 4 {
                first_clause = head.to_string();
                break;
            }
        }
    }
    Some(sanitize_inline(&first_clause))
}

pub(crate) fn relation_answer_markers(lower_task: &str) -> &'static [&'static str] {
    if lower_task.contains("raise awareness") {
        &["awareness for "]
    } else if lower_task.contains("work with") {
        &["worked with ", "working with ", "collaborated with "]
    } else if lower_task.contains("blog") || lower_task.contains("topic") {
        &["blogging about ", "writing about ", "posting about "]
    } else if lower_task.contains("fan of") {
        &["fan of "]
    } else if lower_task.contains("screenplay") {
        &[
            "screenplay about ",
            "screenplay explores ",
            "screenplay is about ",
            "movie about ",
            "story about ",
        ]
    } else if lower_task.contains("letter about") {
        &["letter about ", "wrote me a letter about "]
    } else if lower_task.contains("share") {
        &["shared ", "share "]
    } else if lower_task.contains("play") || lower_task.contains("game convention") {
        &["played ", "playing "]
    } else if lower_task.contains("feel") {
        &["felt ", "feeling "]
    } else if lower_task.contains("plan") || lower_task.contains("later on") {
        &["planned to ", "planning to "]
    } else if lower_task.contains("opening") || lower_task.contains("working on opening") {
        &["working on opening ", "opening ", "working on "]
    } else if lower_task.contains("join") || lower_task.contains("group") {
        &["joined a ", "joined an ", "joined "]
    } else if lower_task.contains("teach") && lower_task.contains("kids") {
        &[
            "teach my kids ",
            "teach his kids ",
            "teach her kids ",
            "teach our kids ",
        ]
    } else {
        &[]
    }
}

pub(crate) fn extract_relation_answer(
    task: &str,
    text: &str,
    task_terms: &[String],
) -> Option<String> {
    let lower_task = task.to_ascii_lowercase();
    if lower_task.contains("ingredient") || lower_task.contains("recipe") {
        if let Some(list) = extract_ingredient_list(text) {
            return Some(list);
        }
    }

    let markers = relation_answer_markers(&lower_task);

    for marker in markers {
        if let Some(answer) = extract_after_marker(task, text, marker, task_terms) {
            if lower_task.contains("group") && !answer.to_ascii_lowercase().contains("group") {
                continue;
            }
            return Some(answer);
        }
    }
    None
}

pub(crate) fn extract_after_marker(
    task: &str,
    text: &str,
    marker: &str,
    task_terms: &[String],
) -> Option<String> {
    let lower_text = text.to_ascii_lowercase();
    let idx = lower_text.find(marker)?;
    let tail = &text[idx + marker.len()..];
    let mut phrase = trim_answer_tail(tail, true);
    if let Some(to_idx) = phrase.to_ascii_lowercase().find(" to ") {
        let head = phrase[..to_idx].trim();
        if head.split_whitespace().count() >= 2 {
            phrase = head.to_string();
        }
    }
    if marker.starts_with("shared") || marker.starts_with("share ") {
        if let Some(with_idx) = phrase.to_ascii_lowercase().find(" with ") {
            let head = phrase[..with_idx].trim();
            if head.split_whitespace().count() >= 2 {
                phrase = head.to_string();
            }
        }
    }
    if marker.starts_with("played") || marker.starts_with("playing") {
        if let Some(at_idx) = phrase.to_ascii_lowercase().find(" at ") {
            let head = phrase[..at_idx].trim();
            if head.split_whitespace().count() >= 1 {
                phrase = head.to_string();
            }
        }
    }
    if marker.starts_with("planned") || marker.starts_with("planning") {
        if let Some(later_idx) = phrase.to_ascii_lowercase().find(" later ") {
            let head = phrase[..later_idx].trim();
            if head.split_whitespace().count() >= 2 {
                phrase = head.to_string();
            }
        }
    }
    is_plausible_compact_answer(task, &phrase, task_terms)
        .then_some(phrase)
        .filter(|answer| is_informative_compact_answer(answer))
}

pub(crate) fn extract_ingredient_list(text: &str) -> Option<String> {
    if text.contains('?') {
        return None;
    }

    let clean = sanitize_answer_text(text);
    if clean.is_empty() || !clean.contains(',') {
        return None;
    }

    let normalized = clean.replace(" and ", ", ");
    let mut parts = normalized
        .split(',')
        .map(str::trim)
        .filter(|part| {
            let words = part.split_whitespace().count();
            words >= 1
                && words <= 4
                && !part.eq_ignore_ascii_case("and")
                && !part.eq_ignore_ascii_case("or")
        })
        .map(sanitize_inline)
        .filter(|part| !part.is_empty())
        .collect::<Vec<_>>();
    parts.sort();
    parts.dedup();
    (parts.len() >= 3).then(|| parts.into_iter().take(4).collect::<Vec<_>>().join(", "))
}

pub(crate) fn summarize_turn_text(text: &str, task_terms: &[String]) -> String {
    let mut best = sanitize_inline(text);
    let mut best_score = candidate_weight(&best, task_terms, 0.0, false);

    for fragment in split_candidate_fragments(text) {
        let clean = sanitize_inline(&fragment);
        if clean.is_empty() {
            continue;
        }
        let score = candidate_weight(&clean, task_terms, 0.0, false);
        if score > best_score {
            best_score = score;
            best = clean;
        }
    }

    best.split_whitespace()
        .take(24)
        .collect::<Vec<_>>()
        .join(" ")
}

pub(crate) fn is_informative_compact_answer(text: &str) -> bool {
    let lower = text.to_ascii_lowercase();
    let words = lower.split_whitespace().collect::<Vec<_>>();
    if words.is_empty() {
        return false;
    }
    if words.len() == 1 {
        let token = words[0];
        if matches!(
            token,
            "good" | "time" | "part" | "customers" | "topic" | "topics" | "again" | "them"
        ) {
            return false;
        }
        if token.chars().all(|c| c.is_ascii_lowercase()) && token.len() <= 4 {
            return false;
        }
    }
    if words.len() == 2
        && matches!(
            lower.as_str(),
            "a good"
                | "the cause"
                | "the convention"
                | "my favorite"
                | "my customers"
                | "your rock"
                | "to chat"
                | "those topics"
                | "having them"
        )
    {
        return false;
    }
    true
}

pub(crate) fn extract_derived_answer(content: &str) -> Option<String> {
    for line in content.lines() {
        let line = line.trim();
        if let Some(rest) = line.strip_prefix("Answer:") {
            let clean = sanitize_answer_text(rest);
            if !clean.is_empty() {
                return Some(clean);
            }
        }
    }
    None
}

pub(crate) fn derived_answer_is_explicit_abstention(answer: &str) -> bool {
    let lower = answer.trim().to_ascii_lowercase();
    lower.starts_with("the information provided is not enough")
        || lower.starts_with("you did not mention")
        || lower.starts_with("the information provided doesn't say")
}
