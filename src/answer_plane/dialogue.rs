use super::{
    aggregate_answer_candidates, answer_items_overlap, answer_meets_form_gate,
    candidate_has_required_anchor_support, candidate_weight, compact_answer, dialogue_focus_terms,
    dialogue_match_score, extract_explicit_date, extract_relation_answer, extract_subject_hints,
    extract_turn_answer, is_informative_compact_answer, is_reason_query, max_task_overlap,
    read_context_text, required_tail_anchor_tokens, salient_query_terms, sanitize_answer_text,
    speaker_match_bonus, task_anchor_terms, task_overlap_count, turn_matches_subject,
    update_best_answer, CandidateLine, DialogueTurn, EvidenceItem,
};
use crate::agent_memory::parse_structured_diary_entry;

pub(super) fn select_turn_pair_answer(
    task: &str,
    evidence: &[EvidenceItem],
    min_answer_confidence: Option<f32>,
) -> Option<String> {
    let task_terms = salient_query_terms(task);
    let subject_hints = extract_subject_hints(task);
    let focus_terms = dialogue_focus_terms(task, &task_terms, &subject_hints);
    let anchor_terms = task_anchor_terms(task, &task_terms, &subject_hints);
    let required_tail_terms = required_tail_anchor_tokens(task);
    let requires_reason = is_reason_query(task);
    let mut candidates = Vec::new();

    for item in evidence {
        let Some(content) = read_context_text(&item.path, "dialogue turn-pair selection") else {
            continue;
        };
        let turns = parse_dialogue_turns(&content);
        if turns.len() < 2 {
            continue;
        }

        for idx in 0..turns.len() - 1 {
            let question = &turns[idx];
            let answer = &turns[idx + 1];
            if !looks_like_question_turn(&question.text) {
                continue;
            }
            if question.speaker.is_some() && question.speaker == answer.speaker {
                continue;
            }

            let mut context = question.text.clone();
            if idx > 0 {
                context = format!("{} {}", turns[idx - 1].text, context);
            }
            let subject_overlap = if subject_hints.is_empty() {
                0
            } else {
                task_overlap_count(&context, &subject_hints)
                    .max(task_overlap_count(&answer.text, &subject_hints))
            };
            if !subject_hints.is_empty() && subject_overlap == 0 {
                continue;
            }
            let question_focus_overlap = if focus_terms.is_empty() {
                0
            } else {
                task_overlap_count(&context, &focus_terms)
            };
            let answer_focus_overlap = if focus_terms.is_empty() {
                0
            } else {
                task_overlap_count(&answer.text, &focus_terms)
            };
            if !focus_terms.is_empty() && question_focus_overlap == 0 && answer_focus_overlap == 0 {
                continue;
            }
            let question_score = dialogue_match_score(&context, &task_terms);
            let speaker_bonus = speaker_match_bonus(answer.speaker.as_deref(), &subject_hints);
            let total_score = question_score
                + speaker_bonus
                + question_focus_overlap as f32 * 8.0
                + answer_focus_overlap as f32 * 10.0;
            let threshold = if requires_reason {
                30.0
            } else if focus_terms.is_empty() {
                20.0
            } else {
                24.0
            };
            if total_score < threshold {
                continue;
            }

            let Some(candidate) = extract_turn_answer(task, &answer.text, &task_terms) else {
                continue;
            };
            let clean = sanitize_answer_text(&candidate);
            if clean.is_empty() {
                continue;
            }
            let candidate_focus_overlap = if focus_terms.is_empty() {
                0
            } else {
                task_overlap_count(&clean, &focus_terms)
            };
            if !focus_terms.is_empty()
                && candidate_focus_overlap == 0
                && answer_focus_overlap == 0
                && !is_reason_query(task)
            {
                continue;
            }
            let support_overlap = task_overlap_count(&context, &task_terms)
                .max(task_overlap_count(&answer.text, &task_terms))
                .max(task_overlap_count(&clean, &task_terms))
                .max(1);
            let anchor_overlap = if anchor_terms.is_empty() {
                0
            } else {
                max_task_overlap(
                    [context.as_str(), answer.text.as_str(), clean.as_str()],
                    &anchor_terms,
                )
            };
            if !anchor_terms.is_empty() && anchor_overlap == 0 {
                continue;
            }
            if !required_tail_terms.is_empty()
                && max_task_overlap(
                    [context.as_str(), answer.text.as_str(), clean.as_str()],
                    &required_tail_terms,
                ) < required_tail_terms.len()
            {
                continue;
            }
            candidates.push(CandidateLine {
                path: item.path.clone(),
                text: clean,
                weight: item.score * 10.0
                    + total_score
                    + candidate_focus_overlap as f32 * 8.0
                    + subject_overlap as f32 * 4.0
                    + 6.0,
                retrieval_score: item.score,
                support_overlap,
                anchor_overlap,
                specific_anchor_overlap: 0,
            });
        }
    }

    let candidates = aggregate_answer_candidates(candidates)
        .into_iter()
        .filter(|candidate| {
            candidate_has_required_anchor_support(task, candidate)
                && answer_meets_form_gate(task, &candidate.text, min_answer_confidence)
        })
        .collect::<Vec<_>>();
    let top = candidates.first()?.clone();
    if candidates.iter().skip(1).any(|candidate| {
        candidate.weight + 10.0 >= top.weight && !answer_items_overlap(&candidate.text, &top.text)
    }) {
        return None;
    }
    Some(top.text)
}

#[must_use]
pub fn mine_dialogue_question_pattern(question: &str) -> Option<String> {
    let clean = sanitize_answer_text(question);
    if clean.is_empty() || !looks_like_question_turn(&clean) {
        return None;
    }
    let terms = salient_query_terms(&clean);
    (terms.len() >= 2).then(|| terms.join(" "))
}

#[must_use]
pub fn mine_dialogue_answer_surface_span(question: &str, answer: &str) -> Option<String> {
    let clean_question = sanitize_answer_text(question);
    let task_terms = salient_query_terms(&clean_question);
    if clean_question.is_empty() || task_terms.is_empty() {
        return None;
    }
    let candidate = extract_turn_answer(&clean_question, answer, &task_terms)?;
    let clean = sanitize_answer_text(&candidate);
    if clean.is_empty() {
        return None;
    }
    if clean.split_whitespace().count() < 4 && !is_informative_compact_answer(&clean) {
        return None;
    }
    if !answer_meets_form_gate(&clean_question, &clean, None) {
        return None;
    }
    Some(clean)
}

pub(super) fn select_subject_turn_answer(
    task: &str,
    evidence: &[EvidenceItem],
    min_answer_confidence: Option<f32>,
) -> Option<String> {
    let task_terms = salient_query_terms(task);
    let subject_hints = extract_subject_hints(task);
    let focus_terms = dialogue_focus_terms(task, &task_terms, &subject_hints);
    let anchor_terms = task_anchor_terms(task, &task_terms, &subject_hints);
    let mut best: Option<(f32, String)> = None;

    for item in evidence {
        let Some(content) = read_context_text(&item.path, "relation answer selection") else {
            continue;
        };
        for turn in parse_dialogue_turns(&content) {
            if !subject_hints.is_empty() && !turn_matches_subject(&turn, &subject_hints) {
                continue;
            }
            let base_score = dialogue_match_score(&turn.text, &task_terms);
            let speaker_bonus = speaker_match_bonus(turn.speaker.as_deref(), &subject_hints);
            let Some(candidate) = extract_relation_answer(task, &turn.text, &task_terms) else {
                continue;
            };
            if !anchor_terms.is_empty()
                && max_task_overlap([turn.text.as_str(), candidate.as_str()], &anchor_terms) == 0
            {
                continue;
            }
            if !answer_meets_form_gate(task, &candidate, min_answer_confidence) {
                continue;
            }
            let focus_overlap = if focus_terms.is_empty() {
                0
            } else {
                task_overlap_count(&turn.text, &focus_terms)
            };
            if !focus_terms.is_empty() && focus_overlap == 0 {
                continue;
            }
            let relation_bonus = 10.0;
            let total_score =
                base_score + speaker_bonus + focus_overlap as f32 * 8.0 + relation_bonus;
            if total_score < 24.0 {
                continue;
            }

            let score = item.score * 10.0 + total_score;
            if best
                .as_ref()
                .map(|(best_score, _)| score > *best_score)
                .unwrap_or(true)
            {
                best = Some((score, candidate));
            }
        }
    }

    best.map(|(_, answer)| answer)
}

pub(super) fn select_structured_diary_answer(
    task: &str,
    evidence: &[EvidenceItem],
) -> Option<String> {
    let task_terms = salient_query_terms(task);
    let task_lower = task.to_ascii_lowercase();
    let wants_status = structured_diary_status_query(&task_lower);
    let wants_goal = structured_diary_goal_query(&task_lower);
    let wants_next_step = structured_diary_next_step_query(&task_lower);
    let wants_blocker = structured_diary_blocker_query(&task_lower);
    let wants_outcome = structured_diary_outcome_query(&task_lower);
    let wants_dependencies = structured_diary_dependencies_query(&task_lower);
    let wants_entities = structured_diary_entities_query(&task_lower);
    let wants_action = structured_diary_action_query(&task_lower);
    let wants_title = structured_diary_title_query(&task_lower);
    let mut best: Option<(f32, String)> = None;

    for item in evidence {
        let Some(content) = read_context_text(&item.path, "structured diary answer selection")
        else {
            continue;
        };
        let Some(entry) = parse_structured_diary_entry(&content) else {
            continue;
        };
        let entry_entities = entry.entities.join(" ");
        let entry_dependencies = entry.depends_on.join(" ");
        let entry_context = [
            entry.agent.as_deref(),
            entry.title.as_deref(),
            entry.status.as_deref(),
            entry.goal.as_deref(),
            entry.next_step.as_deref(),
            entry.blocker.as_deref(),
            entry.outcome.as_deref(),
            (!entry_entities.is_empty()).then_some(entry_entities.as_str()),
            (!entry_dependencies.is_empty()).then_some(entry_dependencies.as_str()),
            entry.action.as_deref(),
        ]
        .into_iter()
        .flatten()
        .collect::<Vec<_>>()
        .join(" ");
        let agent_bonus = entry
            .agent
            .as_ref()
            .map(|agent| agent.to_ascii_lowercase())
            .filter(|agent| task_lower.contains(agent))
            .map(|_| 8.0)
            .unwrap_or(0.0);

        if let Some(status) = entry.status.as_deref() {
            let score = candidate_weight(
                &format!("status progress blocked done {entry_context}"),
                &task_terms,
                item.score,
                false,
            ) + if wants_status && !wants_blocker {
                18.0
            } else if wants_blocker {
                3.0
            } else {
                2.0
            } + agent_bonus;
            update_best_answer(&mut best, score, status.to_string());
        }

        if let Some(title) = entry.title.as_deref() {
            let score = candidate_weight(
                &format!("task title working on doing focus {entry_context}"),
                &task_terms,
                item.score,
                false,
            ) + if wants_title || wants_action {
                14.0
            } else if wants_goal {
                6.0
            } else {
                4.0
            } + agent_bonus;
            update_best_answer(&mut best, score, title.to_string());
        }

        if let Some(goal) = entry.goal.as_deref() {
            let candidate =
                compact_answer(task, goal, &task_terms).unwrap_or_else(|| goal.to_string());
            let score = candidate_weight(
                &format!("goal objective target trying achieve mission aim {entry_context}"),
                &task_terms,
                item.score,
                false,
            ) + if wants_goal { 19.0 } else { 5.0 }
                + agent_bonus;
            update_best_answer(&mut best, score, candidate);
        }

        if let Some(next_step) = entry.next_step.as_deref() {
            let candidate = compact_answer(task, next_step, &task_terms)
                .unwrap_or_else(|| next_step.to_string());
            let score = candidate_weight(
                &format!(
                    "next step next action follow up follow-up unblock remaining {entry_context}"
                ),
                &task_terms,
                item.score,
                false,
            ) + if wants_next_step { 20.0 } else { 5.0 }
                + agent_bonus;
            update_best_answer(&mut best, score, candidate);
        }

        if let Some(blocker) = entry.blocker.as_deref() {
            let candidate =
                compact_answer(task, blocker, &task_terms).unwrap_or_else(|| blocker.to_string());
            let score = candidate_weight(
                &format!("blocker blocked blocking stuck waiting dependency {entry_context}"),
                &task_terms,
                item.score,
                false,
            ) + if wants_blocker { 21.0 } else { 4.0 }
                + agent_bonus;
            update_best_answer(&mut best, score, candidate);
        }

        if let Some(outcome) = entry.outcome.as_deref() {
            let candidate =
                compact_answer(task, outcome, &task_terms).unwrap_or_else(|| outcome.to_string());
            let score = candidate_weight(
                &format!("outcome result decision conclusion found changed fixed {entry_context}"),
                &task_terms,
                item.score,
                false,
            ) + if wants_outcome { 20.0 } else { 6.0 }
                + agent_bonus;
            update_best_answer(&mut best, score, candidate);
        }

        if !entry.entities.is_empty() {
            let answer = entry.entities.join(", ");
            let score = candidate_weight(
                &format!("entities files file modules module component area {entry_context}"),
                &task_terms,
                item.score,
                false,
            ) + if wants_entities { 16.0 } else { 4.0 }
                + agent_bonus;
            update_best_answer(&mut best, score, answer);
        }

        if !entry.depends_on.is_empty() {
            let answer = entry.depends_on.join(", ");
            let score = candidate_weight(
                &format!("depends dependency dependencies waiting on need handoff {entry_context}"),
                &task_terms,
                item.score,
                false,
            ) + if wants_dependencies {
                18.0
            } else if wants_blocker {
                8.0
            } else {
                4.0
            } + agent_bonus;
            update_best_answer(&mut best, score, answer);
        }

        if let Some(action) = entry.action.as_deref() {
            let candidate =
                compact_answer(task, action, &task_terms).unwrap_or_else(|| action.to_string());
            let score = candidate_weight(
                &format!("action doing working investigating reviewing changing {entry_context}"),
                &task_terms,
                item.score,
                false,
            ) + if wants_action || wants_title {
                12.0
            } else {
                5.0
            } + agent_bonus;
            update_best_answer(&mut best, score, candidate);
        }
    }

    best.map(|(_, answer)| answer)
}

pub(super) fn parse_dialogue_turns(content: &str) -> Vec<DialogueTurn> {
    let mut turns = Vec::new();
    let mut current: Option<DialogueTurn> = None;
    let mut session_date: Option<(i32, u32, u32)> = None;
    let mut in_generated_section = false;

    for raw_line in content.lines().map(str::trim) {
        if raw_line.starts_with("[Session") {
            session_date = extract_explicit_date(raw_line, None);
            continue;
        }
        if should_skip_generated_answer_line(raw_line, &mut in_generated_section) {
            continue;
        }

        if let Some((speaker, text)) = raw_line.split_once(": ") {
            if is_dialogue_speaker(speaker) {
                if let Some(turn) = current.take() {
                    if !turn.text.is_empty() {
                        turns.push(turn);
                    }
                }
                current = Some(DialogueTurn {
                    speaker: Some(speaker.trim().to_string()),
                    text: text.trim().to_string(),
                    session_date,
                });
                continue;
            }
        }

        if let Some(turn) = current.as_mut() {
            if !turn.text.is_empty() {
                turn.text.push(' ');
            }
            turn.text.push_str(raw_line);
        }
    }

    if let Some(turn) = current {
        if !turn.text.is_empty() {
            turns.push(turn);
        }
    }

    turns
}

fn is_dialogue_speaker(prefix: &str) -> bool {
    let trimmed = prefix.trim();
    if trimmed.is_empty() || trimmed.len() > 32 {
        return false;
    }
    let lower = trimmed.to_ascii_lowercase();
    if lower.starts_with("speaker ") {
        return false;
    }
    trimmed
        .chars()
        .all(|c| c.is_ascii_alphabetic() || c == ' ' || c == '-' || c == '\'')
}

pub(super) fn should_skip_generated_answer_line(
    raw_line: &str,
    in_generated_section: &mut bool,
) -> bool {
    let lower = raw_line.to_ascii_lowercase();
    if lower.contains("<!-- section:") {
        *in_generated_section = true;
        return true;
    }
    if *in_generated_section {
        if lower.contains("<!-- /section -->") {
            *in_generated_section = false;
        }
        return true;
    }

    raw_line.is_empty()
        || raw_line.starts_with("<!--")
        || raw_line.starts_with("##")
        || raw_line.starts_with('#')
        || raw_line.starts_with("===")
        || raw_line.starts_with("---")
        || raw_line.starts_with("- source:")
        || raw_line.starts_with('|')
}

pub(super) fn looks_like_question_turn(text: &str) -> bool {
    let lower = text.to_ascii_lowercase();
    text.contains('?')
        || lower.starts_with("what ")
        || lower.starts_with("who ")
        || lower.starts_with("when ")
        || lower.starts_with("where ")
        || lower.starts_with("why ")
        || lower.starts_with("how ")
        || lower.starts_with("which ")
}

pub(super) fn structured_diary_status_query(task_lower: &str) -> bool {
    task_lower.contains("status")
        || task_lower.contains("blocked")
        || task_lower.contains("in progress")
        || task_lower.contains("progress")
}

pub(super) fn structured_diary_goal_query(task_lower: &str) -> bool {
    task_lower.contains("goal")
        || task_lower.contains("objective")
        || task_lower.contains("trying to achieve")
        || task_lower.contains("target")
}

pub(super) fn structured_diary_next_step_query(task_lower: &str) -> bool {
    task_lower.contains("next step")
        || task_lower.contains("next action")
        || task_lower.contains("what next")
        || task_lower.contains("do next")
        || task_lower.contains("follow up")
        || task_lower.contains("follow-up")
}

pub(super) fn structured_diary_blocker_query(task_lower: &str) -> bool {
    task_lower.contains("blocker")
        || task_lower.contains("blocked")
        || task_lower.contains("blocking")
        || task_lower.contains("stuck")
        || task_lower.contains("waiting on")
}

pub(super) fn structured_diary_outcome_query(task_lower: &str) -> bool {
    task_lower.contains("decide")
        || task_lower.contains("decision")
        || task_lower.contains("conclude")
        || task_lower.contains("conclusion")
        || task_lower.contains("outcome")
        || task_lower.contains("result")
        || task_lower.contains("find")
        || task_lower.contains("found")
        || task_lower.contains("discover")
        || task_lower.contains("fixed")
        || task_lower.contains("changed")
}

pub(super) fn structured_diary_entities_query(task_lower: &str) -> bool {
    task_lower.contains("file")
        || task_lower.contains("files")
        || task_lower.contains("module")
        || task_lower.contains("modules")
        || task_lower.contains("component")
        || task_lower.contains("area")
        || task_lower.contains("entity")
        || task_lower.contains("entities")
}

pub(super) fn structured_diary_dependencies_query(task_lower: &str) -> bool {
    task_lower.contains("depends on")
        || task_lower.contains("dependency")
        || task_lower.contains("dependencies")
        || task_lower.contains("waiting on")
        || task_lower.contains("handoff")
}

pub(super) fn structured_diary_action_query(task_lower: &str) -> bool {
    task_lower.contains("working on")
        || task_lower.contains("doing now")
        || task_lower.contains("doing lately")
        || task_lower.contains("doing")
        || task_lower.contains("investigat")
        || task_lower.contains("review")
}

pub(super) fn structured_diary_title_query(task_lower: &str) -> bool {
    task_lower.contains("task")
        || task_lower.contains("title")
        || task_lower.contains("focus")
        || task_lower.contains("working on")
}
