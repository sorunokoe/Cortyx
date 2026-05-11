use super::*;

pub(crate) fn candidate_weight(
    text: &str,
    task_terms: &[String],
    retrieval_score: f32,
    from_summary: bool,
) -> f32 {
    let lower = text.to_lowercase();
    let overlap = task_overlap_count(text, task_terms) as f32;
    let raw_token_count = text.split_whitespace().count();
    let token_count = raw_token_count.min(16) as f32;
    let has_number = lower.chars().any(|c| c.is_ascii_digit());
    let has_month = [
        "january",
        "february",
        "march",
        "april",
        "may",
        "june",
        "july",
        "august",
        "september",
        "october",
        "november",
        "december",
    ]
    .iter()
    .any(|month| lower.contains(month));
    let density_bonus = if has_number || has_month { 4.0 } else { 0.0 };
    let summary_bonus = if from_summary { 3.0 } else { 0.0 };
    let informational_bonus = token_count * 0.2;
    let concision_bonus = match raw_token_count {
        1..=3 => 0.5,
        4..=16 => 1.5,
        17..=28 => 0.0,
        _ => -2.0,
    };
    retrieval_score * 10.0
        + overlap * 15.0
        + density_bonus
        + summary_bonus
        + informational_bonus
        + concision_bonus
}

pub(crate) fn task_overlap_count(text: &str, task_terms: &[String]) -> usize {
    let lower = text.to_ascii_lowercase();
    let tokens = lower
        .split(|c: char| !c.is_ascii_alphanumeric())
        .filter(|token| !token.is_empty())
        .collect::<Vec<_>>();
    task_terms
        .iter()
        .filter(|term| {
            tokens
                .iter()
                .any(|token| query_term_matches_token(term, token))
        })
        .count()
}

pub(crate) fn max_task_overlap<'a>(
    texts: impl IntoIterator<Item = &'a str>,
    task_terms: &[String],
) -> usize {
    texts
        .into_iter()
        .map(|text| task_overlap_count(text, task_terms))
        .max()
        .unwrap_or(0)
}

pub(crate) fn candidate_has_required_anchor_support(task: &str, candidate: &CandidateLine) -> bool {
    if !looks_like_typed_open_qa_query(task)
        && !looks_like_relation_query(task)
        && parse_binary_choice(task).is_none()
        && parse_open_qa_choice_options(task).is_empty()
    {
        return true;
    }
    let task_terms = salient_query_terms(task);
    let subject_hints = extract_subject_hints(task);
    let anchor_terms = task_anchor_terms(task, &task_terms, &subject_hints);
    if institution_query_expected(task) {
        let specific_anchor_terms = institution_specific_anchor_terms(task);
        if !specific_anchor_terms.is_empty() {
            let min_overlap = if specific_anchor_terms.len() >= 2 {
                2
            } else {
                1
            };
            return candidate.specific_anchor_overlap >= min_overlap;
        }
    }
    anchor_terms.is_empty() || candidate.anchor_overlap > 0
}

pub(crate) fn validate_selected_answer(
    task: &str,
    answer: Option<String>,
    min_answer_confidence: Option<f32>,
) -> Option<String> {
    answer.filter(|answer| answer_meets_form_gate(task, answer, min_answer_confidence))
}

pub(crate) fn is_reading_progress_pages_left_query(task: &str) -> bool {
    task.to_ascii_lowercase().contains("pages do i have left")
}

pub(crate) fn answer_meets_form_gate(
    task: &str,
    text: &str,
    min_answer_confidence: Option<f32>,
) -> bool {
    let task_terms = salient_query_terms(task);
    let confidence = answer_form_confidence(task, text, &task_terms);
    confidence > 0.0
        && min_answer_confidence
            .map(|threshold| confidence >= threshold)
            .unwrap_or(true)
}

pub(crate) fn salient_query_terms(task: &str) -> Vec<String> {
    let mut terms: Vec<String> = task
        .split(|c: char| !c.is_alphanumeric() && c != '\'')
        .filter_map(|term| {
            let lower = term
                .trim_matches(|c: char| !c.is_ascii_alphanumeric() && c != '\'')
                .to_lowercase();
            if lower.len() < 3 || QUESTION_STOPWORDS.contains(&lower.as_str()) {
                None
            } else {
                Some(lower)
            }
        })
        .collect();
    terms.sort();
    terms.dedup();
    terms
}

pub(crate) fn is_enumerative_query(task: &str) -> bool {
    let lower = task.to_lowercase();
    if lower.contains(" or ")
        || lower.contains(" first")
        || lower.contains(" second")
        || lower.contains(" earlier")
        || lower.contains(" later")
        || lower.contains(" before ")
        || lower.contains(" after ")
    {
        return false;
    }

    lower.contains("list ")
        || lower.contains("what are")
        || lower.contains("who are")
        || lower.contains("which are")
        || lower.contains("which ones")
        || lower.contains("which people")
        || lower.contains("which items")
        || lower.contains("which activities")
        || lower.contains("which topics")
        || lower.contains("which books")
        || lower.contains("which movies")
        || lower.contains("which events were")
}

pub(crate) fn sanitize_answer_text(text: &str) -> String {
    let mut line = text.trim().trim_start_matches("- ").trim().to_string();
    if let Some((prefix, rest)) = line.split_once(": ") {
        let words = prefix.split_whitespace().count();
        let alpha_like = prefix
            .chars()
            .all(|c| c.is_alphabetic() || c == ' ' || c == '-' || c == '\'');
        if words <= 3 && alpha_like {
            line = rest.to_string();
        }
    }
    collapse_inline_whitespace(&line)
}

pub(crate) fn sanitize_inline(text: &str) -> String {
    collapse_inline_whitespace(text).chars().take(240).collect()
}

pub(crate) fn collapse_inline_whitespace(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

pub(crate) fn parse_binary_choice(task: &str) -> Option<(Vec<ChoiceOption>, TemporalDirection)> {
    let lower = task.to_ascii_lowercase();
    let direction = if lower.contains(" first")
        || lower.contains(" earlier")
        || lower.contains(" before ")
        || lower.contains(" oldest")
    {
        TemporalDirection::Earlier
    } else if lower.contains(" later")
        || lower.contains(" last")
        || lower.contains(" after ")
        || lower.contains(" newest")
        || lower.contains(" most recent")
    {
        TemporalDirection::Later
    } else {
        return None;
    };

    let tail = task
        .rsplit_once(',')
        .map(|(_, rest)| rest.trim())
        .unwrap_or(task.trim())
        .trim_end_matches('?')
        .trim();
    if !tail.to_ascii_lowercase().contains(" or ") {
        return None;
    }
    let mut parts = tail.splitn(2, " or ");
    let left = parts.next()?.trim();
    let right = parts.next()?.trim();
    let options = [left, right]
        .into_iter()
        .map(|raw| {
            let display = raw
                .trim()
                .trim_start_matches("the ")
                .trim_start_matches("a ")
                .trim_start_matches("an ")
                .trim_matches(|c: char| c == '?' || c == ',' || c == '.')
                .to_string();
            let tokens = display
                .split(|c: char| !c.is_alphanumeric())
                .filter_map(|token| {
                    let lower = token.to_ascii_lowercase();
                    if lower.len() < 2
                        || QUESTION_STOPWORDS.contains(&lower.as_str())
                        || parse_count_token(&lower).is_some()
                    {
                        None
                    } else {
                        Some(lower)
                    }
                })
                .collect::<Vec<_>>();
            ChoiceOption { display, tokens }
        })
        .filter(|option| !option.display.is_empty() && !option.tokens.is_empty())
        .collect::<Vec<_>>();
    if options.len() == 2 {
        Some((options, direction))
    } else {
        None
    }
}
