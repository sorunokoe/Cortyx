use super::*;

pub(in crate::index) fn best_temporal_rank_line(
    lines: &[String],
    phrase_lower: &str,
    terms: &[String],
) -> Option<(i32, usize, String)> {
    best_temporal_rank_line_with_min_overlap(lines, phrase_lower, terms, None)
}

pub(in crate::index) fn best_temporal_rank_line_with_min_overlap(
    lines: &[String],
    phrase_lower: &str,
    terms: &[String],
    min_overlap_override: Option<usize>,
) -> Option<(i32, usize, String)> {
    let keys = synthetic_answer_surface_term_key_set(terms);
    let min_overlap = min_overlap_override.unwrap_or_else(|| if keys.len() >= 3 { 2 } else { 1 });
    let mut best: Option<(i32, usize, usize, usize, String)> = None;
    for (line_idx, line) in lines.iter().enumerate() {
        let lower = line.to_ascii_lowercase();
        let line_keys = synthetic_answer_surface_term_key_set(&synthetic_query_terms(&lower));
        let overlap = synthetic_answer_surface_overlap_count(&line_keys, &keys);
        let exact = lower.contains(phrase_lower);
        if overlap < min_overlap && !exact {
            continue;
        }
        let Some(rank) = extract_temporal_rank_value(line) else {
            continue;
        };
        let exact_bonus = usize::from(exact);
        let score = overlap * 10 + exact_bonus * 5;
        let should_replace = best
            .as_ref()
            .map(|(_, best_score, best_exact, best_line_idx, _)| {
                score > *best_score
                    || (score == *best_score
                        && (exact_bonus > *best_exact
                            || (exact_bonus == *best_exact && line_idx > *best_line_idx)))
            })
            .unwrap_or(true);
        if should_replace {
            best = Some((rank, score, exact_bonus, line_idx, line.clone()));
        }
    }
    best.map(|(rank, score, _, _, line)| (rank, score, line))
}

pub(in crate::index) fn best_user_turn_line_with_min_overlap(
    lines: &[String],
    phrase_lower: &str,
    terms: &[String],
    min_overlap_override: Option<usize>,
) -> Option<(i32, usize, String)> {
    let keys = synthetic_answer_surface_term_key_set(terms);
    let min_overlap = min_overlap_override.unwrap_or_else(|| if keys.len() >= 3 { 2 } else { 1 });
    let mut best: Option<(i32, usize, usize, String)> = None;
    let mut user_turn = 0i32;
    for line in lines {
        let lower = line.to_ascii_lowercase();
        if !lower.starts_with("user:") {
            continue;
        }
        user_turn += 1;
        let line_keys = synthetic_answer_surface_term_key_set(&synthetic_query_terms(&lower));
        let overlap = synthetic_answer_surface_overlap_count(&line_keys, &keys);
        let exact = lower.contains(phrase_lower);
        if overlap < min_overlap && !exact {
            continue;
        }
        let exact_bonus = usize::from(exact);
        let score = overlap * 10 + exact_bonus * 5;
        let should_replace = best
            .as_ref()
            .map(|(best_turn, best_score, best_exact, _)| {
                score > *best_score
                    || (score == *best_score
                        && (exact_bonus > *best_exact
                            || (exact_bonus == *best_exact && user_turn > *best_turn)))
            })
            .unwrap_or(true);
        if should_replace {
            best = Some((user_turn, score, exact_bonus, line.clone()));
        }
    }
    best.map(|(turn, score, _, line)| (turn, score, line))
}

pub(in crate::index) fn best_temporal_duration_anchor_line(
    lines: &[String],
    phrase_lower: &str,
    terms: &[String],
) -> Option<(SyntheticDurationAnchor, usize, String)> {
    let keys = synthetic_answer_surface_term_key_set(terms);
    let min_overlap = if keys.len() >= 3 { 2 } else { 1 };
    let mut best: Option<(SyntheticDurationAnchor, usize, usize, usize, String)> = None;
    for (line_idx, line) in lines.iter().enumerate() {
        let lower = line.to_ascii_lowercase();
        let line_keys = synthetic_answer_surface_term_key_set(&synthetic_query_terms(&lower));
        let overlap = synthetic_answer_surface_overlap_count(&line_keys, &keys);
        let exact = lower.contains(phrase_lower);
        if overlap < min_overlap && !exact {
            continue;
        }
        let anchor = if let Some(days) = extract_current_duration_days(line) {
            SyntheticDurationAnchor::CurrentDays(days)
        } else if let Some(day) = extract_explicit_date_rank(line) {
            SyntheticDurationAnchor::AbsoluteDay(day)
        } else {
            continue;
        };
        let exact_bonus = usize::from(exact);
        let score = overlap * 10 + exact_bonus * 5;
        let should_replace = best
            .as_ref()
            .map(|(_, best_score, best_exact, best_line_idx, _)| {
                score > *best_score
                    || (score == *best_score
                        && (exact_bonus > *best_exact
                            || (exact_bonus == *best_exact && line_idx > *best_line_idx)))
            })
            .unwrap_or(true);
        if should_replace {
            best = Some((anchor, score, exact_bonus, line_idx, line.clone()));
        }
    }
    best.map(|(anchor, score, _, _, line)| (anchor, score, line))
}

pub(in crate::index) fn best_temporal_event_anchor_line(
    lines: &[String],
    phrase_lower: &str,
    terms: &[String],
) -> Option<(SyntheticEventAnchor, usize, String)> {
    let keys = synthetic_answer_surface_term_key_set(terms);
    let min_overlap = if keys.len() >= 3 { 2 } else { 1 };
    let required_action_key = terms
        .first()
        .map(|term| synthetic_answer_surface_term_key(term))
        .filter(|term| !term.is_empty());
    let mut best: Option<(SyntheticEventAnchor, usize, usize, usize, String)> = None;
    for (line_idx, line) in lines.iter().enumerate() {
        let lower = line.to_ascii_lowercase();
        let line_keys = synthetic_answer_surface_term_key_set(&synthetic_query_terms(&lower));
        if required_action_key
            .as_ref()
            .is_some_and(|term| !line_keys.contains(term))
        {
            continue;
        }
        let overlap = synthetic_answer_surface_overlap_count(&line_keys, &keys);
        let exact = lower.contains(phrase_lower);
        if overlap < min_overlap && !exact {
            continue;
        }
        let anchor = if let Some(days_ago) = extract_temporal_relative_days(line) {
            let adjusted = match extract_relative_reference_offset_days(line) {
                Some((SyntheticTemporalDirection::Earlier, offset)) => days_ago + offset,
                Some((SyntheticTemporalDirection::Later, offset)) => {
                    days_ago.saturating_sub(offset)
                },
                None => days_ago,
            };
            SyntheticEventAnchor::RelativeDaysAgo(adjusted)
        } else if let Some(day) = extract_explicit_date_rank(line) {
            SyntheticEventAnchor::AbsoluteDay(day)
        } else {
            continue;
        };
        let exact_bonus = usize::from(exact);
        let score = overlap * 10 + exact_bonus * 5;
        let should_replace = best
            .as_ref()
            .map(|(_, best_score, best_exact, best_line_idx, _)| {
                score > *best_score
                    || (score == *best_score
                        && (exact_bonus > *best_exact
                            || (exact_bonus == *best_exact && line_idx > *best_line_idx)))
            })
            .unwrap_or(true);
        if should_replace {
            best = Some((anchor, score, exact_bonus, line_idx, line.clone()));
        }
    }
    best.map(|(anchor, score, _, _, line)| (anchor, score, line))
}

pub(in crate::index) fn best_temporal_from_now_event_line(
    lines: &[String],
    phrase_lower: &str,
    terms: &[String],
) -> Option<(i32, usize, String)> {
    let focus_terms = temporal_from_now_focus_terms(terms);
    let min_overlap = if focus_terms.len() >= 3 { 2 } else { 1 };
    let mut best: Option<(i32, usize, usize, String)> = None;
    for (line_idx, line) in lines.iter().enumerate() {
        let lower = line.to_ascii_lowercase();
        let overlap = temporal_from_now_overlap_count(&lower, &focus_terms);
        let exact = lower.contains(phrase_lower);
        if overlap < min_overlap && !exact {
            continue;
        }
        let day = if let Some(base_day) = temporal_base_day_at_line(lines, line_idx) {
            if let Some(days_ago) = extract_temporal_relative_days(line) {
                let adjusted = match extract_relative_reference_offset_days(line) {
                    Some((SyntheticTemporalDirection::Earlier, offset)) => days_ago + offset,
                    Some((SyntheticTemporalDirection::Later, offset)) => {
                        days_ago.saturating_sub(offset)
                    },
                    None => days_ago,
                };
                base_day - adjusted
            } else if let Some(day) = extract_explicit_date_rank(line) {
                day
            } else {
                base_day
            }
        } else if let Some(days_ago) = extract_temporal_relative_days(line) {
            let adjusted = match extract_relative_reference_offset_days(line) {
                Some((SyntheticTemporalDirection::Earlier, offset)) => days_ago + offset,
                Some((SyntheticTemporalDirection::Later, offset)) => {
                    days_ago.saturating_sub(offset)
                },
                None => days_ago,
            };
            -adjusted
        } else if let Some(day) = extract_explicit_date_rank(line) {
            day
        } else {
            continue;
        };
        let score = overlap * 10 + usize::from(exact) * 5;
        let should_replace = best
            .as_ref()
            .map(|(best_day, best_score, best_line_idx, _)| {
                score > *best_score
                    || (score == *best_score
                        && (day > *best_day || (day == *best_day && line_idx > *best_line_idx)))
            })
            .unwrap_or(true);
        if should_replace {
            best = Some((day, score, line_idx, line.clone()));
        }
    }
    let (day, score, _, line) = best?;
    Some((day, score, line))
}
