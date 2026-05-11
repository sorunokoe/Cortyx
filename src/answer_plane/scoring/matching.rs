use super::*;

pub(crate) fn update_best_answer(best: &mut Option<(f32, String)>, score: f32, answer: String) {
    if best
        .as_ref()
        .map(|(best_score, _)| score > *best_score)
        .unwrap_or(true)
    {
        *best = Some((score, answer));
    }
}

pub(crate) fn extract_subject_hints(task: &str) -> Vec<String> {
    let mut hints = Vec::new();
    for token in task.split(|c: char| !c.is_ascii_alphabetic() && c != '-') {
        let trimmed = token.trim();
        if trimmed.len() < 3 {
            continue;
        }
        let Some(first) = trimmed.chars().next() else {
            continue;
        };
        if !first.is_ascii_uppercase() {
            continue;
        }
        let lower = trimmed.to_ascii_lowercase();
        if ENTITY_STOPWORDS.contains(&lower.as_str()) {
            continue;
        }
        hints.push(lower);
    }
    hints.sort();
    hints.dedup();
    hints
}

pub(crate) fn dialogue_focus_terms(
    task: &str,
    task_terms: &[String],
    subject_hints: &[String],
) -> Vec<String> {
    let lower = task.to_ascii_lowercase();
    let mut focus = task_terms
        .iter()
        .filter(|term| !subject_hints.iter().any(|hint| hint == *term))
        .cloned()
        .collect::<Vec<_>>();
    focus.retain(|term| !matches!(term.as_str(), "likely" | "current" | "currently"));

    if is_education_field_query(&lower) {
        focus.extend(
            [
                "job",
                "jobs",
                "career",
                "work",
                "working",
                "study",
                "studying",
                "education",
                "school",
            ]
            .iter()
            .map(|term| (*term).to_string()),
        );
    }

    if lower.contains("research") || lower.contains("looking into") || lower.contains("look into") {
        focus.extend(
            [
                "research",
                "researching",
                "looking",
                "into",
                "check",
                "checking",
            ]
            .iter()
            .map(|term| (*term).to_string()),
        );
    }

    if lower.contains("support group") {
        focus.extend(["support", "group"].iter().map(|term| (*term).to_string()));
    }

    focus.sort();
    focus.dedup();
    focus
}

pub(crate) fn turn_matches_subject(turn: &DialogueTurn, subject_hints: &[String]) -> bool {
    if subject_hints.is_empty() {
        return true;
    }
    speaker_match_bonus(turn.speaker.as_deref(), subject_hints) > 0.0
        || task_overlap_count(&turn.text, subject_hints) > 0
}

pub(crate) fn normalize_match_term(term: &str) -> &str {
    term.strip_suffix("'s")
        .or_else(|| term.strip_suffix("s'"))
        .unwrap_or(term)
}

pub(crate) fn rough_match_term(term: &str) -> &str {
    term.strip_suffix("ing")
        .or_else(|| term.strip_suffix("ed"))
        .or_else(|| term.strip_suffix("es"))
        .or_else(|| term.strip_suffix('s'))
        .filter(|value| value.len() >= 4)
        .unwrap_or(term)
}

pub(crate) fn common_prefix_len(left: &str, right: &str) -> usize {
    left.chars()
        .zip(right.chars())
        .take_while(|(l, r)| l == r)
        .count()
}

pub(crate) fn within_edit_distance_one(left: &str, right: &str) -> bool {
    let left_chars = left.chars().collect::<Vec<_>>();
    let right_chars = right.chars().collect::<Vec<_>>();
    let left_len = left_chars.len();
    let right_len = right_chars.len();
    if left_len.abs_diff(right_len) > 1 {
        return false;
    }

    let mut left_idx = 0usize;
    let mut right_idx = 0usize;
    let mut seen_edit = false;
    while left_idx < left_len && right_idx < right_len {
        if left_chars[left_idx] == right_chars[right_idx] {
            left_idx += 1;
            right_idx += 1;
            continue;
        }
        if seen_edit {
            return false;
        }
        seen_edit = true;
        if left_len > right_len {
            left_idx += 1;
        } else if right_len > left_len {
            right_idx += 1;
        } else {
            left_idx += 1;
            right_idx += 1;
        }
    }
    true
}

pub(crate) fn query_term_matches_token(term: &str, token: &str) -> bool {
    let left = rough_match_term(normalize_match_term(term));
    let right = rough_match_term(normalize_match_term(token));
    if left.is_empty() || right.is_empty() {
        return false;
    }
    if left == right {
        return true;
    }
    if left.len() >= 4 && right.starts_with(left) {
        return true;
    }
    if right.len() >= 5 && left.starts_with(right) {
        return true;
    }
    left.len() >= 6
        && right.len() >= 6
        && common_prefix_len(left, right) >= 4
        && within_edit_distance_one(left, right)
}

pub(crate) fn term_list_overlap_count(left: &[String], right: &[String]) -> usize {
    left.iter()
        .filter(|term| {
            right
                .iter()
                .any(|candidate| query_term_matches_token(term, candidate))
        })
        .count()
}

pub(crate) fn speaker_match_bonus(speaker: Option<&str>, subject_hints: &[String]) -> f32 {
    let Some(speaker) = speaker else {
        return 0.0;
    };
    let lower = speaker.to_ascii_lowercase();
    if subject_hints.iter().any(|hint| hint == &lower) {
        14.0
    } else {
        0.0
    }
}

pub(crate) fn dialogue_match_score(text: &str, task_terms: &[String]) -> f32 {
    let overlap = task_overlap_count(text, task_terms) as f32;
    candidate_weight(text, task_terms, 0.0, false) + overlap * 6.0
}

pub(crate) fn extract_turn_answer(task: &str, text: &str, task_terms: &[String]) -> Option<String> {
    let clean = sanitize_answer_text(text);
    if clean.is_empty() {
        return None;
    }

    if is_reason_query(task) {
        if let Some(reason) = extract_reason_answer(&clean) {
            return Some(reason);
        }
    }

    if let Some(answer) = extract_relation_answer(task, &clean, task_terms) {
        return Some(answer);
    }

    if let Some(compact) = compact_answer(task, &clean, task_terms) {
        if is_informative_compact_answer(&compact) {
            return Some(compact);
        }
    }

    if task.to_ascii_lowercase().contains("research")
        && clean.to_ascii_lowercase().contains("research")
    {
        return None;
    }

    Some(summarize_turn_text(&clean, task_terms))
}
