//! Temporal state and dialogue answer selection.

use super::*;
use crate::kg;

pub(crate) fn select_temporal_state_answer(
    task: &str,
    evidence: &[EvidenceItem],
) -> Option<String> {
    let query = parse_temporal_state_query(task)?;
    let task_terms = salient_query_terms(task);
    if task_terms.is_empty() {
        return None;
    }

    let mut best: Option<(f32, String)> = None;
    for item in evidence {
        let Ok(entity) = kg::KgEntity::load(&item.path) else {
            continue;
        };
        if entity.facts.is_empty() {
            continue;
        }

        let mut predicates = entity
            .facts
            .iter()
            .map(|fact| fact.predicate.clone())
            .collect::<Vec<_>>();
        predicates.sort();
        predicates.dedup();

        for predicate in predicates {
            let score =
                temporal_state_candidate_score(&task_terms, item.score, &entity, &predicate);
            if score <= 0.0 {
                continue;
            }

            let Some(answer) = render_temporal_state_kg_answer(&query, &entity, &predicate) else {
                continue;
            };
            update_best_answer(&mut best, score, answer);
        }
    }

    best.map(|(_, answer)| answer)
}

pub(crate) fn select_dialogue_temporal_answer(
    task: &str,
    evidence: &[EvidenceItem],
) -> Option<String> {
    let lower = task.to_ascii_lowercase();
    let asks_when = lower.starts_with("when ")
        || lower.contains(" what date")
        || lower.contains(" which date")
        || lower.contains(" at what time");
    if !asks_when {
        return None;
    }

    let task_terms = salient_query_terms(task);
    if task_terms.is_empty() {
        return None;
    }
    let subject_hints = extract_subject_hints(task);
    let focus_terms = dialogue_focus_terms(task, &task_terms, &subject_hints);
    if focus_terms.is_empty() {
        return None;
    }
    let required_terms = required_tail_anchor_tokens(task);

    let mut candidates = Vec::new();
    for item in evidence {
        let Some(content) = read_context_text(&item.path, "dialogue temporal answer selection")
        else {
            continue;
        };
        for turn in parse_dialogue_turns(&content) {
            if !turn_matches_subject(&turn, &subject_hints) {
                continue;
            }
            let lower_turn = turn.text.to_ascii_lowercase();
            if !required_terms.is_empty()
                && !required_terms
                    .iter()
                    .all(|token| line_matches_event_token(&lower_turn, token))
            {
                continue;
            }
            let event_overlap = task_overlap_count(&turn.text, &focus_terms);
            if event_overlap == 0 {
                continue;
            }
            let Some(point) = extract_turn_temporal_answer(&turn) else {
                continue;
            };
            let subject_overlap = if subject_hints.is_empty() {
                0
            } else {
                task_overlap_count(&turn.text, &subject_hints)
            };
            let score = item.score * 10.0
                + dialogue_match_score(&turn.text, &task_terms)
                + event_overlap as f32 * 12.0
                + speaker_match_bonus(turn.speaker.as_deref(), &subject_hints)
                + subject_overlap as f32 * 4.0
                + temporal_point_specificity_bonus(point);
            candidates.push((score, render_temporal_answer(point)));
        }
    }

    candidates.sort_by(|left, right| {
        right
            .0
            .total_cmp(&left.0)
            .then_with(|| left.1.len().cmp(&right.1.len()))
            .then_with(|| left.1.cmp(&right.1))
    });
    let (top_score, top_answer) = candidates.first()?.clone();
    if candidates.iter().skip(1).any(|(score, answer)| {
        *score + 8.0 >= top_score && !answer_items_overlap(answer, &top_answer)
    }) {
        return None;
    }
    Some(top_answer)
}

fn extract_turn_temporal_answer(turn: &DialogueTurn) -> Option<TemporalAnswerPoint> {
    if let Some(date) = extract_explicit_date_match(&turn.text, turn.session_date) {
        return Some(match date {
            ExplicitDateMatch::Day { year, month, day } => {
                TemporalAnswerPoint::Day { year, month, day }
            },
            ExplicitDateMatch::Month { year, month } => TemporalAnswerPoint::Month { year, month },
        });
    }

    let base_date = turn.session_date?;
    let lower = turn.text.to_ascii_lowercase();

    if let Some((year, month, day)) = extract_relative_date(&turn.text, base_date) {
        return Some(TemporalAnswerPoint::Day { year, month, day });
    }
    if lower.contains("next month") {
        let (year, month) = shift_month(base_date.0, base_date.1, 1);
        return Some(TemporalAnswerPoint::Month { year, month });
    }
    if lower.contains("last month") {
        let (year, month) = shift_month(base_date.0, base_date.1, -1);
        return Some(TemporalAnswerPoint::Month { year, month });
    }
    if lower.contains("this month") {
        return Some(TemporalAnswerPoint::Month {
            year: base_date.0,
            month: base_date.1,
        });
    }
    if lower.contains("next year") {
        return Some(TemporalAnswerPoint::Year {
            year: base_date.0 + 1,
        });
    }
    if lower.contains("last year") {
        return Some(TemporalAnswerPoint::Year {
            year: base_date.0 - 1,
        });
    }
    if lower.contains("this year") {
        return Some(TemporalAnswerPoint::Year { year: base_date.0 });
    }
    None
}

fn render_temporal_answer(point: TemporalAnswerPoint) -> String {
    match point {
        TemporalAnswerPoint::Day { year, month, day } => {
            format!("{day} {} {year}", month_name(month))
        },
        TemporalAnswerPoint::Month { year, month } => format!("{} {year}", month_name(month)),
        TemporalAnswerPoint::Year { year } => year.to_string(),
    }
}

fn temporal_point_specificity_bonus(point: TemporalAnswerPoint) -> f32 {
    match point {
        TemporalAnswerPoint::Day { .. } => 12.0,
        TemporalAnswerPoint::Month { .. } => 8.0,
        TemporalAnswerPoint::Year { .. } => 6.0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extracts_month_year_turn_as_month_answer() {
        let turn = DialogueTurn {
            speaker: Some("User".to_string()),
            text: "The migration was scheduled for January 2023.".to_string(),
            session_date: None,
        };
        assert_eq!(
            extract_turn_temporal_answer(&turn),
            Some(TemporalAnswerPoint::Month {
                year: 2023,
                month: 1,
            })
        );
    }

    #[test]
    fn resolves_relative_turn_dates_only_with_anchor() {
        let anchored = DialogueTurn {
            speaker: Some("User".to_string()),
            text: "We met last Tuesday.".to_string(),
            session_date: Some((2024, 3, 15)),
        };
        assert_eq!(
            extract_turn_temporal_answer(&anchored),
            Some(TemporalAnswerPoint::Day {
                year: 2024,
                month: 3,
                day: 12,
            })
        );

        let unanchored = DialogueTurn {
            speaker: Some("User".to_string()),
            text: "We met last Tuesday.".to_string(),
            session_date: None,
        };
        assert_eq!(extract_turn_temporal_answer(&unanchored), None);
    }
}
