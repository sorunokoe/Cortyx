use super::*;

pub(super) fn select_suggestion_list_item_answer(
    task: &str,
    evidence: &[EvidenceItem],
) -> Option<String> {
    if !is_suggestion_query(task) {
        return None;
    }

    let task_terms = salient_query_terms(task);
    let subject_hints = extract_subject_hints(task);
    let anchor_terms = task_anchor_terms(task, &task_terms, &subject_hints);
    let mut best: Option<(f32, String)> = None;

    for item in evidence {
        let Some(content) = read_context_text(&item.path, "suggestion list answer selection")
        else {
            continue;
        };
        let mut recent_lines: Vec<String> = Vec::new();
        for raw_line in content.lines() {
            let raw_trimmed = raw_line.trim();
            let Some(candidate) = extract_list_item_title(raw_line) else {
                if !raw_trimmed.is_empty() {
                    recent_lines.push(raw_trimmed.to_string());
                    if recent_lines.len() > 3 {
                        recent_lines.remove(0);
                    }
                }
                continue;
            };
            if !answer_meets_form_gate(task, &candidate, None) {
                if !raw_trimmed.is_empty() {
                    recent_lines.push(raw_trimmed.to_string());
                    if recent_lines.len() > 3 {
                        recent_lines.remove(0);
                    }
                }
                continue;
            }
            let context = recent_lines.join(" ");
            let anchor_overlap = if anchor_terms.is_empty() {
                0
            } else {
                max_task_overlap(
                    [context.as_str(), raw_line, candidate.as_str()],
                    &anchor_terms,
                )
            };
            if !anchor_terms.is_empty() && anchor_overlap == 0 {
                if !raw_trimmed.is_empty() {
                    recent_lines.push(raw_trimmed.to_string());
                    if recent_lines.len() > 3 {
                        recent_lines.remove(0);
                    }
                }
                continue;
            }
            let lower = candidate.to_ascii_lowercase();
            let food_bonus = if is_food_query(task)
                && FOOD_ITEM_HINTS.iter().any(|needle| lower.contains(needle))
            {
                6.0
            } else {
                0.0
            };
            let score = candidate_weight(raw_line, &task_terms, item.score, false)
                + anchor_overlap as f32 * 8.0
                + food_bonus;
            update_best_answer(&mut best, score, candidate);
            if !raw_trimmed.is_empty() {
                recent_lines.push(raw_trimmed.to_string());
                if recent_lines.len() > 3 {
                    recent_lines.remove(0);
                }
            }
        }
    }

    best.map(|(_, answer)| answer)
}

fn extract_list_item_title(line: &str) -> Option<String> {
    let trimmed = line.trim();
    if trimmed.is_empty() {
        return None;
    }

    let stripped = trimmed
        .trim_start_matches(|c: char| {
            c.is_ascii_digit() || matches!(c, '.' | ')' | '-' | '*' | ' ')
        })
        .trim();
    let head = stripped
        .split_once(':')
        .map(|(head, _)| head)
        .unwrap_or(stripped);
    let clean = normalized_validation_text(head)
        .trim_matches(|c: char| matches!(c, '"' | '\'' | '.' | ',' | ';' | ':' | '!' | '?'))
        .trim()
        .to_string();
    let word_count = clean.split_whitespace().count();
    if clean.is_empty() || !(1..=6).contains(&word_count) {
        return None;
    }
    let lower = clean.to_ascii_lowercase();
    if looks_like_heading_fragment(head, &clean)
        || ANSWER_REJECT_EXACT.contains(&lower.as_str())
        || looks_like_social_filler(&lower)
    {
        return None;
    }
    Some(clean)
}
