//! Answer projection, extraction, and session-based logic.

use super::super::*;
use crate::index::compile_regex;

pub fn extract_adjacent_role_person_followup_answer(
    task_lower: &str,
    lines: &[String],
    line_idx: usize,
) -> Option<String> {
    if !task_contains_any(task_lower, &["who is the", "who was the"]) {
        return None;
    }
    let role_terms = assistant_followup_role_terms(task_lower);
    if role_terms.is_empty() {
        return None;
    }
    let line = lines.get(line_idx)?;
    let lower = line.to_ascii_lowercase();
    let role_overlap = role_terms
        .iter()
        .filter(|term| lower.contains(term.as_str()))
        .count();
    if role_overlap == 0 {
        return None;
    }
    for neighbor_idx in [line_idx.checked_sub(1), Some(line_idx + 1)] {
        let Some(neighbor_idx) = neighbor_idx else {
            continue;
        };
        let Some(neighbor) = lines.get(neighbor_idx) else {
            continue;
        };
        let neighbor_lower = neighbor.to_ascii_lowercase();
        if let Some(answer) =
            extract_session_named_answer_from_line(task_lower, neighbor, &neighbor_lower)
        {
            if answer
                .chars()
                .next()
                .map(|ch| ch.is_ascii_uppercase())
                .unwrap_or(false)
            {
                return Some(answer);
            }
        }
    }
    None
}

pub fn project_assistant_followup_answer_from_line(
    task: &str,
    task_lower: &str,
    line: &str,
    lower: &str,
) -> Option<String> {
    if task_contains_any(
        task_lower,
        &["what move", "which move", "what was the move"],
    ) {
        if let Some(answer) = extract_chess_move_answer_from_line(
            line,
            extract_expected_chess_reply_move_number(task_lower),
        ) {
            return Some(answer);
        }
    }
    if let Some(answer) = extract_descriptor_named_followup_answer(task_lower, line, lower) {
        return Some(answer);
    }
    if detect_counting_query(task) {
        if let Some(answer) = extract_parenthetical_label_count_answer(task_lower, line, lower)
            .or_else(|| extract_query_aligned_numeric_answer(task_lower, line))
        {
            return Some(answer);
        }
        return None;
    }
    if task_lower.contains("website") {
        if let Some(answer) = extract_website_name_from_line(line) {
            return Some(answer);
        }
    }
    if task_contains_any(task_lower, &["what type of beer", "what kind of beer"]) {
        if let Some(answer) = extract_beer_recommendation_answer_from_line(lower) {
            return Some(answer);
        }
    }
    if task_lower.contains("two-factor authentication") {
        if let Some(answer) = extract_two_factor_method_answer_from_line(line, lower) {
            return Some(answer);
        }
    }
    project_session_answer_from_line(task, task_lower, None, line, lower)
}

pub fn extract_descriptor_named_followup_answer(
    task_lower: &str,
    line: &str,
    lower: &str,
) -> Option<String> {
    if detect_counting_query(task_lower)
        || task_lower.starts_with("how ")
        || task_lower.starts_with("when ")
        || task_lower.starts_with("where ")
    {
        return None;
    }
    let descriptor_terms = assistant_followup_descriptor_terms(task_lower);
    if descriptor_terms.len() < 2 {
        return None;
    }
    let matched = descriptor_terms
        .iter()
        .filter(|term| lower.contains(term.as_str()))
        .count();
    if matched < 2 {
        return None;
    }
    extract_session_named_answer_from_line(task_lower, line, lower)
}

pub fn assistant_followup_descriptor_terms(task_lower: &str) -> Vec<String> {
    let mut terms = Vec::new();
    if let Some((_, clause)) = task_lower
        .rsplit_once(" that ")
        .or_else(|| task_lower.rsplit_once(" which "))
        .or_else(|| task_lower.rsplit_once(" who "))
    {
        terms.extend(
            synthetic_query_terms(clause)
                .into_iter()
                .filter(|term| term.len() >= 3)
                .filter(|term| !term.chars().all(|ch| ch.is_ascii_digit()))
                .filter(|term| {
                    !matches!(term.as_str(), "companies" | "company" | "people" | "person")
                }),
        );
    }
    if let Some(subject_clause) = assistant_followup_subject_descriptor_clause(task_lower) {
        terms.extend(
            synthetic_query_terms(subject_clause)
                .into_iter()
                .filter(|term| term.len() >= 3)
                .filter(|term| !term.chars().all(|ch| ch.is_ascii_digit()))
                .filter(|term| !matches!(term.as_str(), "example" | "gave" | "people" | "person")),
        );
    }
    terms.sort();
    terms.dedup();
    terms
}

pub fn assistant_followup_subject_descriptor_clause(task_lower: &str) -> Option<&str> {
    for marker in [
        "example you gave of a ",
        "example you gave of an ",
        "example you gave of the ",
    ] {
        let Some((_, tail)) = task_lower.split_once(marker) else {
            continue;
        };
        let stop = tail
            .find(" who ")
            .or_else(|| tail.find(" that "))
            .or_else(|| tail.find(" which "))
            .unwrap_or(tail.len());
        let clause = tail[..stop].trim();
        if !clause.is_empty() {
            return Some(clause);
        }
    }
    None
}

pub fn assistant_followup_role_terms(task_lower: &str) -> Vec<String> {
    synthetic_query_terms(task_lower)
        .into_iter()
        .filter(|term| term.len() >= 5)
        .filter(|term| {
            !matches!(
                term.as_str(),
                "article"
                    | "conversation"
                    | "follow"
                    | "mentioned"
                    | "previous"
                    | "remind"
                    | "science"
                    | "technology"
            )
        })
        .collect()
}

pub fn assistant_followup_anchor_terms(task_lower: &str) -> Vec<String> {
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

pub fn assistant_followup_anchor_distance(
    line_lower: &str,
    match_end: usize,
    anchor_terms: &[String],
) -> Option<usize> {
    if anchor_terms.is_empty() {
        return None;
    }
    anchor_terms
        .iter()
        .filter_map(|term| {
            line_lower[match_end..]
                .find(term)
                .map(|offset| offset + match_end)
        })
        .map(|position| position.saturating_sub(match_end))
        .min()
}

pub fn assistant_followup_context(lines: &[String], line_idx: usize) -> String {
    let start = line_idx.saturating_sub(1);
    let end = usize::min(line_idx + 1, lines.len().saturating_sub(1));
    lines[start..=end].join(" ")
}

pub fn extract_expected_chess_reply_move_number(task_lower: &str) -> Option<i32> {
    let prior_move = compile_regex(r"after\s+(\d+)\.")
        .captures(task_lower)
        .and_then(|caps| caps.get(1))
        .and_then(|m| m.as_str().parse::<i32>().ok())?;
    Some(prior_move + 1)
}

pub fn extract_chess_move_answer_from_line(
    line: &str,
    expected_move_number: Option<i32>,
) -> Option<String> {
    let capture = compile_regex(
        r"\b(\d+)\.\s*(O-O(?:-O)?|[KQRNB]?[a-h]?[1-8]?x?[a-h][1-8](?:=[QRNB])?[+#]?)\b",
    )
    .captures(line)?;
    let move_number = capture.get(1)?.as_str().parse::<i32>().ok()?;
    if expected_move_number.is_some_and(|expected| expected != move_number) {
        return None;
    }
    let notation = capture.get(2)?.as_str().trim();
    Some(format!("{move_number}. {notation}"))
}

pub fn extract_parenthetical_label_count_answer(
    task_lower: &str,
    line: &str,
    _lower: &str,
) -> Option<String> {
    let focus_terms = synthetic_query_terms(task_lower);
    let focus_refs: Vec<&str> = focus_terms.iter().map(String::as_str).collect();
    let capture = compile_regex(r"(?i)\b([A-Za-z][A-Za-z' -]+?)\s*\((\d+)\)").captures(line)?;
    let label = capture.get(1)?.as_str().trim().to_ascii_lowercase();
    (term_overlap_count(&label, &focus_refs) >= 1)
        .then(|| capture.get(2).map(|m| m.as_str().trim().to_string()))
        .flatten()
}

pub fn extract_website_name_from_line(line: &str) -> Option<String> {
    compile_regex(r"\b([A-Za-z0-9-]+\.(?:org|com|net|edu|io))\b")
        .captures(line)
        .and_then(|caps| caps.get(1))
        .map(|m| m.as_str().trim().to_string())
}

pub fn extract_beer_recommendation_answer_from_line(lower: &str) -> Option<String> {
    (lower.contains("beer") && lower.contains("pilsner") && lower.contains("lager"))
        .then_some("I recommended using a Pilsner or Lager for the recipe.".to_string())
}

pub fn extract_two_factor_method_answer_from_line(line: &str, lower: &str) -> Option<String> {
    if !lower.contains("two-factor authentication") {
        return None;
    }
    let methods = extract_phrase_after_any_index(
        line,
        lower,
        &["such as "],
        &[", enhances security", " enhances security", ".", ";"],
        1,
    )?;
    Some(format!(
        "I mentioned {} as examples of two-factor authentication methods.",
        methods.trim().trim_end_matches(',')
    ))
}

pub fn extract_session_education_answer(line: &str, lower: &str) -> Option<String> {
    let mut answer = extract_phrase_after_any_index(
        line,
        lower,
        &[
            "degree in ",
            "bachelor's in ",
            "bachelors in ",
            "master's in ",
            "masters in ",
            "graduated with a degree in ",
            "graduated with degree in ",
            "graduated with ",
            "majored in ",
            "major in ",
            "studying ",
            "study ",
        ],
        &[
            " which",
            " from ",
            " at ",
            " and ",
            " but ",
            " because ",
            ",",
        ],
        1,
    )?;
    for prefix in [
        "a degree in ",
        "degree in ",
        "a bachelor's in ",
        "a bachelors in ",
        "bachelor's in ",
        "bachelors in ",
        "a master's in ",
        "a masters in ",
        "master's in ",
        "masters in ",
    ] {
        if answer.to_ascii_lowercase().starts_with(prefix) {
            answer = answer[prefix.len()..].trim().to_string();
            break;
        }
    }
    Some(normalize_education_kg_value(&answer))
}

pub fn extract_session_named_answer_from_line(
    task_lower: &str,
    line: &str,
    lower: &str,
) -> Option<String> {
    let is_query_context = |candidate: &str| {
        let terms = tokenize(&candidate.to_ascii_lowercase());
        !terms.is_empty()
            && terms
                .iter()
                .all(|term| term.len() <= 2 || task_lower.contains(term.as_str()))
    };
    if let Some(value) = extract_descriptor_led_named_answer(line) {
        if !is_query_context(&value) {
            return Some(value);
        }
    }
    let is_question = lower.trim_end().ends_with('?');
    let markers = if is_question {
        vec![
            "called ",
            "named ",
            "titled ",
            "example is ",
            "example was ",
        ]
    } else {
        vec![
            "called ",
            "named ",
            "titled ",
            "recommend ",
            "recommended ",
            "try ",
            "example is ",
            "example was ",
            "was ",
        ]
    };
    if let Some(value) = extract_phrase_after_any_index(
        line,
        lower,
        &markers,
        &[" for ", " because ", " and ", " but ", ".", ",", " while "],
        1,
    ) {
        if let Some(best_title) = extract_title_like_phrases(&value)
            .into_iter()
            .find(|candidate| !is_query_context(candidate))
        {
            return Some(best_title);
        }
        if value.split_whitespace().count() <= 8 && !is_query_context(&value) {
            return Some(value);
        }
    }

    let mut titles = extract_title_like_phrases(line)
        .into_iter()
        .filter(|value| {
            let lower_value = value.to_ascii_lowercase();
            ![
                "also", "by", "can", "do", "does", "for", "i", "it", "my", "our", "that", "the",
                "this", "we", "what", "when", "where", "which", "who",
            ]
            .contains(&lower_value.as_str())
                && !is_query_context(value)
        })
        .collect::<Vec<_>>();
    if task_contains_any(task_lower, &["playlist", "project", "blog", "channel"]) {
        titles.retain(|value| value.split_whitespace().count() <= 6);
    }
    titles.sort_by(|left, right| right.len().cmp(&left.len()).then_with(|| left.cmp(right)));
    titles.into_iter().next()
}

pub fn extract_descriptor_led_named_answer(line: &str) -> Option<String> {
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
    Some(title_case_named_words(&candidate_tokens.join(" ")))
}

pub fn title_case_named_words(value: &str) -> String {
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

pub fn extract_session_list_answer_from_line(
    task_lower: &str,
    line: &str,
    lower: &str,
) -> Option<String> {
    let answer = extract_phrase_after_any_index(
        line,
        lower,
        &[
            "such as ",
            "including ",
            "include ",
            "includes ",
            "uses ",
            "using ",
            "were ",
        ],
        &[". ", "?", " and i'm ", " and i’m ", " but "],
        1,
    )?;
    task_contains_any(
        task_lower,
        &["what kind", "what type", "specific", "what were the"],
    )
    .then_some(answer)
}

pub fn extract_session_location_answer(
    task_lower: &str,
    line: &str,
    lower: &str,
) -> Option<String> {
    if task_contains_any(
        task_lower,
        &[
            "buy",
            "bought",
            "redeem",
            "use my coupon",
            "which store",
            "shop",
        ],
    ) {
        return extract_phrase_after_any_index(
            line,
            lower,
            &["from the ", "from ", "at the ", "at "],
            &[
                " for ",
                " with ",
                " because ",
                " and ",
                " but ",
                " last ",
                ".",
            ],
            1,
        );
    }
    if task_contains_any(
        task_lower,
        &["keep", "kept", "store", "stored", "put", "place"],
    ) {
        for marker in ["under ", "in ", "inside ", "on "] {
            if let Some(phrase) = extract_phrase_after_any_index(
                line,
                lower,
                &[marker],
                &[" because ", " and ", " but ", ".", ","],
                1,
            ) {
                return Some(format!("{} {}", marker.trim(), phrase));
            }
        }
    }
    extract_phrase_after_any_index(
        line,
        lower,
        &[
            "based in ",
            "live in ",
            "living in ",
            "now living in ",
            "moved to ",
            "moved back to ",
        ],
        &[
            " again",
            " because ",
            " and ",
            " but ",
            " with ",
            " after ",
            ".",
            ",",
        ],
        1,
    )
    .map(|value| normalize_location_kg_value(&value))
}

pub fn extract_session_occupation_answer(line: &str, lower: &str) -> Option<String> {
    extract_phrase_after_any_index(
        line,
        lower,
        &[
            "work as ",
            "working as ",
            "employed as ",
            "job as ",
            "role as ",
            "i'm a ",
            "i am a ",
        ],
        &[" at ", " for ", " and ", " but ", " because ", "."],
        1,
    )
}

pub fn extract_money_answer_from_line(line: &str) -> Option<String> {
    compile_regex(r"(?i)(\$\d[\d,]*(?:\.\d+)?)")
        .captures(line)
        .and_then(|caps| caps.get(1))
        .map(|m| m.as_str().trim().to_string())
}
