use super::*;

pub(crate) fn compact_answer(task: &str, text: &str, task_terms: &[String]) -> Option<String> {
    let lower_task = task.to_ascii_lowercase();

    if let Some(answer) = extract_after_action_marker(task, text, &lower_task, task_terms) {
        return Some(answer);
    }

    if let Some(answer) = extract_after_preposition(task, text, &lower_task, task_terms) {
        return Some(answer);
    }

    if let Some(answer) = extract_after_anchor_copula(task, text, task_terms) {
        return Some(answer);
    }

    None
}

pub(crate) fn extract_after_action_marker(
    task: &str,
    text: &str,
    lower_task: &str,
    task_terms: &[String],
) -> Option<String> {
    let lower_text = text.to_ascii_lowercase();
    let markers: &[&str] = if lower_task.contains("blog") || lower_task.contains("topic") {
        &["blogging about ", "writing about ", "posting about "]
    } else if lower_task.contains("research") {
        &[
            "researched ",
            "researching ",
            "looking into ",
            "look into ",
            "checking out ",
            "check out ",
        ]
    } else if lower_task.contains("join") || lower_task.contains("group") {
        &["joined ", "join "]
    } else if lower_task.contains("open")
        || lower_task.contains("working on")
        || lower_task.contains("start")
        || lower_task.contains("business")
    {
        &[
            "starting ",
            "opening ",
            "building ",
            "launching ",
            "working on ",
            "planning ",
            "creating ",
        ]
    } else if is_education_field_query(lower_task) {
        &[
            "keen on ",
            "interested in ",
            "thinking of ",
            "thinking about ",
            "working in ",
            "looking into ",
            "look into ",
        ]
    } else {
        &[]
    };

    for marker in markers {
        if let Some(idx) = lower_text.find(marker) {
            let tail = &text[idx + marker.len()..];
            let mut phrase = trim_answer_tail(tail, true);
            if is_education_field_query(lower_task) {
                if let Some((head, rest)) = split_once_case_insensitive(&phrase, " or working in ")
                {
                    phrase = format!("{} or {}", head.trim(), rest.trim());
                } else if let Some(rest) = phrase.strip_prefix("working in ") {
                    phrase = rest.trim().to_string();
                }
            }
            if is_plausible_compact_answer(task, &phrase, task_terms) {
                return Some(phrase);
            }
        }
    }
    None
}

pub(crate) fn extract_after_preposition(
    task: &str,
    text: &str,
    lower_task: &str,
    task_terms: &[String],
) -> Option<String> {
    let lower_text = text.to_ascii_lowercase();
    for prep in PREPOSITION_HINTS {
        let prep_marker = format!("{prep} ");
        if !contains_standalone_token(lower_task, prep) {
            continue;
        }
        let mut search_start = 0usize;
        let mut best: Option<(usize, String)> = None;
        while let Some(rel_idx) = lower_text[search_start..].find(&prep_marker) {
            let idx = search_start + rel_idx;
            let tail = &text[idx + prep_marker.len()..];
            let phrase = trim_answer_tail(tail, true);
            if is_plausible_compact_answer(task, &phrase, task_terms) {
                let window_start = idx.saturating_sub(96);
                let context = safe_slice(&lower_text, window_start, idx);
                let overlap = task_terms
                    .iter()
                    .filter(|term| context.contains(term.as_str()))
                    .count();
                let score = overlap * 10 + phrase.split_whitespace().count().min(8);
                if best
                    .as_ref()
                    .map(|(best_score, _)| score > *best_score)
                    .unwrap_or(true)
                {
                    best = Some((score, phrase));
                }
            }
            search_start = idx + prep_marker.len();
        }
        if let Some((_, phrase)) = best {
            return Some(phrase);
        }
    }
    None
}

pub(crate) fn extract_after_anchor_copula(
    task: &str,
    text: &str,
    task_terms: &[String],
) -> Option<String> {
    let lower_text = text.to_ascii_lowercase();
    let mut anchors: Vec<&str> = task_terms.iter().map(String::as_str).collect();
    anchors.sort_by_key(|term| std::cmp::Reverse(term.len()));

    for anchor in anchors {
        if let Some(anchor_idx) = lower_text.find(anchor) {
            let after_anchor = &lower_text[anchor_idx + anchor.len()..];
            for marker in [" is ", " was ", " are ", " were ", ": "] {
                if let Some(marker_idx) = after_anchor.find(marker) {
                    let raw_tail = &text[anchor_idx + anchor.len() + marker_idx + marker.len()..];
                    let phrase = trim_answer_tail(raw_tail, marker != ": ");
                    if is_plausible_compact_answer(task, &phrase, task_terms) {
                        return Some(phrase);
                    }
                }
            }
        }
    }
    None
}

pub(crate) fn trim_answer_tail(tail: &str, stop_on_copula: bool) -> String {
    let mut cleaned = sanitize_inline(tail);
    let lower = cleaned.to_ascii_lowercase();
    let mut cut = cleaned.len();

    for boundary in TAIL_BOUNDARIES {
        if let Some(idx) = lower.find(boundary) {
            cut = cut.min(idx);
        }
    }
    if stop_on_copula {
        for boundary in COPULA_BOUNDARIES {
            if let Some(idx) = lower.find(boundary) {
                cut = cut.min(idx);
            }
        }
    }
    cleaned.truncate(cut);

    cleaned = cleaned
        .trim_matches(|c: char| {
            matches!(
                c,
                '"' | '\'' | '(' | ')' | '[' | ']' | '{' | '}' | ',' | '.' | ';' | ':' | '!' | '?'
            )
        })
        .trim()
        .to_string();

    for prefix in ["the ", "a ", "an "] {
        if cleaned.to_ascii_lowercase().starts_with(prefix)
            && cleaned.split_whitespace().count() > 2
        {
            cleaned = cleaned[prefix.len()..].trim().to_string();
            break;
        }
    }

    cleaned
}

pub(crate) fn contains_standalone_token(text: &str, token: &str) -> bool {
    text.split(|c: char| !c.is_ascii_alphanumeric())
        .any(|part| !part.is_empty() && part == token)
}

pub(crate) fn safe_slice(text: &str, start: usize, end: usize) -> &str {
    fn floor_char_boundary(text: &str, mut idx: usize) -> usize {
        idx = idx.min(text.len());
        while idx > 0 && !text.is_char_boundary(idx) {
            idx -= 1;
        }
        idx
    }

    fn ceil_char_boundary(text: &str, mut idx: usize) -> usize {
        idx = idx.min(text.len());
        while idx < text.len() && !text.is_char_boundary(idx) {
            idx += 1;
        }
        idx
    }

    let start = floor_char_boundary(text, start);
    let end = ceil_char_boundary(text, end);
    if start >= end {
        ""
    } else {
        &text[start..end]
    }
}

pub(crate) fn split_candidate_fragments(line: &str) -> Vec<String> {
    let mut fragments = vec![line.to_string()];
    for separator in ['.', '!', '?', ';'] {
        fragments = fragments
            .into_iter()
            .flat_map(|fragment| {
                fragment
                    .split(separator)
                    .map(str::trim)
                    .filter(|part| part.split_whitespace().count() >= 3)
                    .map(|part| part.to_string())
                    .collect::<Vec<_>>()
            })
            .collect();
    }

    let mut expanded = Vec::new();
    for fragment in fragments {
        expanded.push(fragment.clone());
        let discourse = strip_temporal_discourse_prefix(&fragment);
        if discourse != fragment && discourse.split_whitespace().count() >= 3 {
            expanded.push(discourse);
        }
        for marker in [
            " and got ",
            " and bought ",
            " and ordered ",
            " and attended ",
            " and joined ",
            " and redeemed ",
            " and signed up ",
            " and used ",
            " and received ",
            " and started ",
            " and finished ",
            " and discovered ",
            " and found ",
            " and took ",
            " and realized ",
        ] {
            if let Some((_, tail)) = split_once_case_insensitive(&fragment, marker) {
                let head = marker.trim().trim_start_matches("and ").to_string();
                let clause = format!("{head} {tail}").trim().to_string();
                if clause.split_whitespace().count() >= 3 {
                    expanded.push(clause);
                }
            }
        }
        for marker in [" - ", " — "] {
            for part in fragment.split(marker).map(str::trim) {
                if part.split_whitespace().count() >= 3 {
                    expanded.push(part.to_string());
                }
            }
        }
    }
    expanded.sort();
    expanded.dedup();
    expanded
}

pub(crate) fn strip_temporal_discourse_prefix(text: &str) -> String {
    let mut clean = sanitize_inline(text);
    loop {
        let lower = clean.to_ascii_lowercase();
        if lower.starts_with("by the way, ") {
            clean = clean["by the way, ".len()..].trim().to_string();
            continue;
        }
        if lower.starts_with("by the way ") {
            clean = clean["by the way ".len()..].trim().to_string();
            continue;
        }
        if lower.starts_with("and by the way, ") {
            clean = clean["and by the way, ".len()..].trim().to_string();
            continue;
        }
        if lower.starts_with("oh, and by the way, ") {
            clean = clean["oh, and by the way, ".len()..].trim().to_string();
            continue;
        }
        if lower.starts_with("speaking of ") {
            if let Some((_, rest)) = clean.split_once(',') {
                clean = rest.trim().to_string();
                continue;
            }
        }
        if lower.starts_with("also, ") {
            clean = clean["also, ".len()..].trim().to_string();
            continue;
        }
        if lower.starts_with("oh, ") {
            clean = clean["oh, ".len()..].trim().to_string();
            continue;
        }
        break;
    }
    clean
}

pub(crate) fn is_plausible_compact_answer(task: &str, text: &str, task_terms: &[String]) -> bool {
    if text.is_empty() {
        return false;
    }
    let word_count = text.split_whitespace().count();
    if word_count == 0 || word_count > 8 {
        return false;
    }
    if !text.chars().any(|c| c.is_alphanumeric()) {
        return false;
    }
    let lower = normalized_validation_text(text).to_ascii_lowercase();
    if !task.is_empty()
        && !is_temporal_reasoning_query(task)
        && answer_form_confidence(task, text, task_terms) <= 0.0
    {
        return false;
    }
    let overlap = task_terms
        .iter()
        .filter(|term| task_overlap_count(&lower, &[(*term).clone()]) > 0)
        .count();
    if overlap < task_terms.len().min(2) {
        return true;
    }

    let novel_tokens = lower
        .split(|c: char| !c.is_ascii_alphanumeric())
        .filter(|token| token.len() >= 3)
        .filter(|token| {
            !task_terms
                .iter()
                .any(|term| query_term_matches_token(term, token))
        })
        .count();
    novel_tokens > 0
}
