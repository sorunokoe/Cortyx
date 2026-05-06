use super::temporal_anchor_extractors::{
    RelativeTemporalRecallAnswerKind, RelativeTemporalRecallQuery,
};
use super::temporal_anchor_families::{dedupe_temporal_anchor_evidence, grouped_verbatim_lines};
use super::*;

#[derive(Clone, Debug)]
struct RelativeRecallSession {
    header: Option<String>,
    day_rank: Option<i32>,
    lines: Vec<String>,
}

pub(super) fn best_grouped_relative_temporal_recall(
    idx: &NeuronIndex,
    query: &RelativeTemporalRecallQuery,
) -> Option<(String, Vec<String>, usize)> {
    grouped_verbatim_lines(idx)
        .into_values()
        .flat_map(split_relative_recall_sessions)
        .filter_map(|session| resolve_relative_temporal_recall_session(session, query))
        .max_by_key(|(_, _, score)| *score)
}

fn split_relative_recall_sessions(lines: Vec<String>) -> Vec<RelativeRecallSession> {
    let mut sessions = Vec::new();
    let mut current = RelativeRecallSession {
        header: None,
        day_rank: None,
        lines: Vec::new(),
    };
    for line in lines {
        if line.starts_with("[Session ") {
            if !current.lines.is_empty() || current.header.is_some() {
                sessions.push(current);
            }
            current = RelativeRecallSession {
                day_rank: extract_explicit_date_rank(&line),
                header: Some(line),
                lines: Vec::new(),
            };
            continue;
        }
        current.lines.push(line);
    }
    if !current.lines.is_empty() || current.header.is_some() {
        sessions.push(current);
    }
    sessions
}

fn resolve_relative_temporal_recall_session(
    session: RelativeRecallSession,
    query: &RelativeTemporalRecallQuery,
) -> Option<(String, Vec<String>, usize)> {
    let session_day = session.day_rank?;
    let day_distance = relative_recall_day_distance(query, session_day)?;
    let session_text = session.lines.join(" ").to_ascii_lowercase();
    let session_overlap = relative_temporal_focus_overlap(&session_text, &query.focus_terms);
    for (line, line_score) in ranked_relative_recall_lines(&session.lines, query) {
        let answer = match query.answer_kind {
            RelativeTemporalRecallAnswerKind::BookTitle => extract_relative_book_answer(&line),
            RelativeTemporalRecallAnswerKind::SourcePerson => {
                extract_relative_source_person_answer(&line)
            },
            RelativeTemporalRecallAnswerKind::DirectObject => {
                extract_relative_direct_object_answer(&line, query)
            },
            RelativeTemporalRecallAnswerKind::EventClause => {
                extract_relative_event_clause_answer(&line, query)
            },
        };
        if let Some(answer) = answer {
            let mut evidence = Vec::new();
            if let Some(header) = session.header.clone() {
                evidence.push(header);
            }
            evidence.push(line);
            return Some((
                answer,
                dedupe_temporal_anchor_evidence(evidence),
                (session_overlap * 20 + line_score).saturating_sub(day_distance * 15),
            ));
        }
    }
    None
}

fn relative_recall_day_distance(
    query: &RelativeTemporalRecallQuery,
    session_day: i32,
) -> Option<usize> {
    let distance = (session_day - query.target_day).unsigned_abs() as usize;
    let lower = query.prompt_body.to_ascii_lowercase();
    let tolerance = if lower.contains(" week ago") || lower.contains(" weeks ago") {
        1
    } else {
        0
    };
    (distance <= tolerance).then_some(distance)
}

fn ranked_relative_recall_lines(
    lines: &[String],
    query: &RelativeTemporalRecallQuery,
) -> Vec<(String, usize)> {
    let mut scored = Vec::new();
    for (line_idx, line) in lines.iter().enumerate() {
        let lower = line.to_ascii_lowercase();
        if !lower.starts_with("user:") {
            continue;
        }
        let overlap = relative_temporal_focus_overlap(&lower, &query.focus_terms);
        let temporal_bonus = usize::from(
            lower.contains(" today")
                || lower.contains(" yesterday")
                || lower.contains(" this morning")
                || lower.contains(" this afternoon")
                || lower.contains(" last "),
        );
        let score = overlap * 10 + temporal_bonus * 5 + usize::from(line_idx == 0);
        scored.push((line.clone(), score, line_idx));
    }
    scored.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.2.cmp(&b.2)));
    scored
        .into_iter()
        .map(|(line, score, _)| (line, score))
        .collect()
}

fn relative_temporal_focus_overlap(text: &str, focus_terms: &[String]) -> usize {
    focus_terms
        .iter()
        .filter(|term| text.contains(term.as_str()))
        .count()
}

fn extract_relative_book_answer(line: &str) -> Option<String> {
    let body = normalize_session_answer_line_body(line);
    let title = extract_first_quoted_phrase(&body)?;
    let quoted_pattern = format!(r#""{}"\s+by\s+([^,.!?]+)"#, regex::escape(&title));
    if let Some(author) = compile_regex(&quoted_pattern)
        .captures(&body)
        .and_then(|captures| captures.get(1))
        .map(|matched| matched.as_str().trim())
    {
        return Some(format!("'{}' by {}", title, author));
    }
    Some(title)
}

fn extract_relative_source_person_answer(line: &str) -> Option<String> {
    let body = normalize_session_answer_line_body(line);
    let acquisition_regex =
        compile_regex(r"(?i)\b(got|received|acquired)\b[^.?!]*?\bfrom\s+([^,.!?]+)");
    let mut best: Option<(usize, String)> = None;
    for captures in acquisition_regex.captures_iter(&body) {
        let verb = captures
            .get(1)
            .map(|matched| matched.as_str().to_ascii_lowercase())
            .unwrap_or_default();
        let giver = captures
            .get(2)
            .map(|matched| clean_relative_giver_phrase(matched.as_str()))
            .filter(|value| !value.is_empty());
        let Some(giver) = giver else {
            continue;
        };
        let full_match = captures
            .get(0)
            .map(|matched| matched.as_str().to_ascii_lowercase())
            .unwrap_or_default();
        let score = match verb.as_str() {
            "received" => 30,
            "got" => 25,
            "acquired" => 5,
            _ => 0,
        } + usize::from(full_match.contains("today")) * 5;
        let should_replace = best
            .as_ref()
            .map(|(best_score, _)| score > *best_score)
            .unwrap_or(true);
        if should_replace {
            best = Some((score, giver));
        }
    }
    best.map(|(_, giver)| giver).or_else(|| {
        compile_regex(r"(?i)\bfrom\s+([^,.!?]+)")
            .captures(&body)
            .and_then(|captures| captures.get(1))
            .map(|matched| clean_relative_giver_phrase(matched.as_str()))
            .filter(|value| !value.is_empty())
    })
}

fn clean_relative_giver_phrase(raw: &str) -> String {
    compile_regex(
        r"(?i)\s+(today|yesterday|this morning|this afternoon|last (?:monday|tuesday|wednesday|thursday|friday|saturday|sunday))$",
    )
    .replace(raw.trim().split(" and ").next().unwrap_or_default().trim(), "")
    .trim()
    .to_string()
}

fn extract_relative_direct_object_answer(
    line: &str,
    query: &RelativeTemporalRecallQuery,
) -> Option<String> {
    let body = normalize_session_answer_line_body(line);
    let regex = compile_regex(
        r"(?i)\b(?:made|cooked|baked|prepared)\s+((?:a|an|the|some)\s+[^,.!?]+?)(?:[,!.?]|$)",
    );
    let mut best: Option<(usize, String)> = None;
    for captures in regex.captures_iter(&body) {
        let Some(object) = captures.get(1) else {
            continue;
        };
        let Some(full_match) = captures.get(0) else {
            continue;
        };
        let context_end = body.len().min(full_match.end() + 80);
        let context = &body[full_match.start()..context_end];
        let lower = context.to_ascii_lowercase();
        let score = relative_temporal_focus_overlap(&lower, &query.focus_terms) * 10
            + usize::from(lower.contains("friend")) * 10
            + usize::from(lower.contains("birthday")) * 5;
        let value = compile_regex(r"(?i)\s+for\b.*$|\s+that\b.*$")
            .replace(object.as_str().trim(), "")
            .trim()
            .to_string();
        let should_replace = best
            .as_ref()
            .map(|(best_score, _)| score > *best_score)
            .unwrap_or(true);
        if should_replace {
            best = Some((score, value));
        }
    }
    best.map(|(_, value)| value)
        .filter(|value| !value.is_empty())
}

fn extract_relative_event_clause_answer(
    line: &str,
    query: &RelativeTemporalRecallQuery,
) -> Option<String> {
    let body = normalize_session_answer_line_body(line);
    let mut best: Option<(usize, String)> = None;
    for sentence in compile_regex(r"[.!?]+")
        .split(&body)
        .map(str::trim)
        .filter(|sentence| !sentence.is_empty())
    {
        let lower = sentence.to_ascii_lowercase();
        let score = relative_temporal_focus_overlap(&lower, &query.focus_terms) * 10
            + usize::from(lower.contains(" today") || lower.contains(" yesterday")) * 20
            + usize::from(lower.starts_with("i "));
        let should_replace = best
            .as_ref()
            .map(|(best_score, _)| score > *best_score)
            .unwrap_or(true);
        if should_replace {
            best = Some((score, sentence.to_string()));
        }
    }
    let mut sentence = best?.1;
    for marker in [
        ", and i ",
        ", but i ",
        " and i ",
        " but i ",
        " do you ",
        " can you ",
    ] {
        if let Some(idx) = sentence.to_ascii_lowercase().find(marker) {
            sentence.truncate(idx);
        }
    }
    sentence = compile_regex(r"(?i)^i just\s+")
        .replace(&sentence, "I ")
        .into_owned();
    sentence = compile_regex(r"(?i)\bmy friend ([A-Z][A-Za-z'-]+)")
        .replace_all(&sentence, "$1")
        .into_owned();
    sentence = compile_regex(
        r"(?i)\s+(today|yesterday|this morning|this afternoon|last (?:monday|tuesday|wednesday|thursday|friday|saturday|sunday))(?:\s+(?:and|but)\b.*|,.*)?$",
    )
    .replace(&sentence, "")
    .trim()
    .trim_end_matches(',')
    .to_string();
    if sentence.is_empty() {
        return None;
    }
    if !sentence.ends_with('.') {
        sentence.push('.');
    }
    Some(sentence)
}
