use super::*;

pub(crate) fn extract_fact_after_any(
    line: &str,
    lower_line: &str,
    markers: &[&str],
    stop_tokens: &[&str],
    max_words: usize,
) -> Option<String> {
    for marker in markers {
        if let Some(idx) = lower_line.find(marker) {
            let tail = line[idx + marker.len()..].trim();
            let lower_tail = tail.to_ascii_lowercase();
            let cutoff = stop_tokens
                .iter()
                .filter_map(|token| lower_tail.find(token))
                .min()
                .unwrap_or(tail.len());
            let bounded_tail = tail[..cutoff].trim();
            if let Some(value) = extract_phrase_fact_value(bounded_tail, &[], max_words) {
                let clean = value.trim_matches(|c: char| matches!(c, '"' | '\'' | ',' | ';' | ':'));
                if !clean.is_empty() {
                    return Some(clean.to_string());
                }
            }
        }
    }
    None
}

pub(in super::super) fn extract_clause_after_any(
    line: &str,
    lower_line: &str,
    markers: &[&str],
    stop_markers: &[&str],
    max_words: usize,
) -> Option<String> {
    for marker in markers {
        if let Some(idx) = lower_line.find(marker) {
            let tail = line[idx + marker.len()..].trim();
            if let Some(value) = extract_clause_fact_value(tail, stop_markers, max_words) {
                return Some(value);
            }
        }
    }
    None
}

fn extract_clause_fact_value(
    after: &str,
    stop_markers: &[&str],
    max_words: usize,
) -> Option<String> {
    let lower = after.to_ascii_lowercase();
    let cutoff = stop_markers
        .iter()
        .filter_map(|marker| lower.find(marker))
        .min()
        .unwrap_or(after.len());
    let trimmed = after[..cutoff].trim();
    if trimmed.is_empty() {
        return None;
    }
    let words = trimmed
        .split_whitespace()
        .take(max_words)
        .collect::<Vec<_>>()
        .join(" ");
    let clean = words.trim_matches(|c: char| matches!(c, '"' | '\'' | ',' | ';' | ':' | '-' | '.'));
    (!clean.is_empty()).then(|| clean.to_string())
}

pub(crate) fn normalize_dialogue_reason_phrase(value: &str) -> String {
    let lower = value.to_ascii_lowercase();
    for prefix in [
        "i want to ",
        "i'd love to ",
        "i would love to ",
        "i wanna ",
        "my goal is to ",
        "goal is to ",
    ] {
        if lower.starts_with(prefix) {
            let rest = value[prefix.len()..].trim();
            return normalize_answer_surface_span(rest);
        }
    }
    normalize_answer_surface_span(value)
}

pub(in super::super) fn normalize_dialogue_support_effect_phrase(value: &str) -> String {
    let mut clean = normalize_answer_surface_span(value);
    clean = clean.replace("and given me ", "and have ");
    clean = clean.replace("And given me ", "and have ");
    if clean.to_ascii_lowercase().starts_with("accepted ") {
        clean = format!("feel {clean}");
    }
    clean
}

pub(crate) fn extract_issue_surface_value(line: &str) -> Option<String> {
    let lower = line.to_ascii_lowercase();
    if let Some(issue) = extract_fact_after_any(
        line,
        &lower,
        &["first issue was ", "issue was ", "problem was "],
        &[" and ", " but ", " because "],
        8,
    ) {
        return Some(issue);
    }

    for marker in [
        " wasn't functioning",
        " not functioning",
        " stopped working",
    ] {
        if let Some(idx) = lower.find(marker) {
            let tail = &line[idx..];
            let lower_tail = tail.to_ascii_lowercase();
            let cutoff = [" after ", " because ", " but ", " and "]
                .iter()
                .filter_map(|stop| lower_tail.find(stop))
                .min()
                .unwrap_or(tail.len());
            let start = line[..idx]
                .rfind(['.', '!', '?', ';'])
                .map(|pos| pos + 1)
                .unwrap_or(0);
            let clause = format!(
                "{}{}",
                line[start..idx]
                    .trim()
                    .trim_matches(|c: char| matches!(c, ',' | ';' | ':' | '"' | '\'')),
                &tail[..cutoff]
            );
            let clean = normalize_answer_surface_span(&clause);
            if !clean.is_empty() {
                return Some(clean);
            }
        }
    }
    None
}

pub(super) fn extract_relax_activity_surface_value(line: &str, lower: &str) -> Option<String> {
    if let Some(idx) = lower.find("went on a ") {
        let tail = &line[idx..];
        return extract_phrase_fact_value(
            tail,
            &[" and ", " but ", " because ", " after ", " with "],
            5,
        )
        .map(|value| normalize_answer_surface_span(&value));
    }
    if let Some(idx) = lower.find("went on ") {
        let tail = &line[idx..];
        return extract_phrase_fact_value(
            tail,
            &[" and ", " but ", " because ", " after ", " with "],
            5,
        )
        .map(|value| normalize_answer_surface_span(&value));
    }
    if let Some(idx) = lower.find("went hiking") {
        let tail = &line[idx..];
        return extract_phrase_fact_value(
            tail,
            &[" and ", " but ", " because ", " after ", " with "],
            3,
        )
        .map(|value| normalize_answer_surface_span(&value));
    }
    None
}

pub(in super::super) fn extract_research_surface_value(line: &str, lower: &str) -> Option<String> {
    if !lower.contains("research")
        && !lower.contains("looking into")
        && !lower.contains("investigating")
    {
        return None;
    }
    extract_fact_after_any(
        line,
        lower,
        &[
            "researching ",
            "researched ",
            "been researching ",
            "been looking into ",
            "looking into ",
            "investigating ",
            "research into ",
        ],
        &[
            "because", "and", "but", "so", "lately", "recently", "online", "after", "before",
            "it's", "it", "i'm", "im", "more",
        ],
        6,
    )
}

pub(super) fn extract_fact_before_any(
    line: &str,
    lower_line: &str,
    markers: &[&str],
    max_words: usize,
) -> Option<String> {
    for marker in markers {
        if let Some(idx) = lower_line.find(marker) {
            let mut words = Vec::new();
            for raw in line[..idx].split_whitespace().rev() {
                let cleaned = raw.trim_matches(|c: char| {
                    !c.is_alphanumeric() && c != '-' && c != '&' && c != '\''
                });
                if cleaned.is_empty() {
                    continue;
                }
                words.push(cleaned.to_string());
                if words.len() >= max_words {
                    break;
                }
            }
            if !words.is_empty() {
                words.reverse();
                return Some(words.join(" "));
            }
        }
    }
    None
}

pub(super) fn looks_like_job_surface_value(value: &str) -> bool {
    let first = value
        .split_whitespace()
        .next()
        .unwrap_or_default()
        .to_ascii_lowercase();
    !matches!(
        first.as_str(),
        "huge" | "big" | "small" | "massive" | "little" | "fan" | "bit"
    )
}
