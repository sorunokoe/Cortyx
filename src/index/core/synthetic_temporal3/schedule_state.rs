//! Artwork location, schedule slot, state transitions, latest purchased items.

use super::super::*;

impl NeuronIndex {
    pub fn synthetic_current_schedule_slot_answer(
        &self,
        task: &str,
        task_lower: &str,
    ) -> Option<PathBuf> {
        let answer_kind = if task_contains_any(
            task_lower,
            &["what day of the week do i ", "which day do i "],
        ) {
            "weekday"
        } else if task_lower.starts_with("what time do i ") {
            "time"
        } else {
            return None;
        };

        let focus_phrase = extract_schedule_slot_focus_phrase(task_lower)?;
        let mut focus_terms = synthetic_query_terms(&focus_phrase);
        focus_terms.retain(|term| {
            !matches!(
                term.as_str(),
                "usually"
                    | "normally"
                    | "typically"
                    | "take"
                    | "takes"
                    | "taking"
                    | "go"
                    | "goes"
                    | "going"
                    | "head"
                    | "heading"
                    | "do"
                    | "does"
            )
        });
        let task_terms = synthetic_query_terms(task_lower);
        let mut required_owned = task_terms
            .into_iter()
            .filter(|term| {
                !matches!(
                    term.as_str(),
                    "what" | "day" | "week" | "time" | "current" | "currently" | "previous"
                )
            })
            .collect::<Vec<_>>();
        if required_owned.is_empty() {
            required_owned = focus_terms.clone();
        }
        required_owned.sort();
        required_owned.dedup();
        if required_owned.is_empty() {
            return None;
        }
        let required_terms: Vec<&str> = required_owned.iter().map(String::as_str).collect();
        let focus_refs: Vec<&str> = focus_terms.iter().map(String::as_str).collect();

        let mut candidates = self
            .candidate_session_ids(task, &required_terms, 8)
            .into_iter()
            .enumerate()
            .map(|(idx, session_id)| (session_id, 8usize.saturating_sub(idx)))
            .collect::<Vec<_>>();
        if candidates.is_empty() {
            candidates = self.candidate_session_ids_by_line_overlap(&required_owned, 8);
        }

        #[allow(clippy::type_complexity)]
        let mut best: Option<(usize, usize, usize, String, Vec<String>)> = None;
        for (session_id, session_rank) in candidates {
            let lines = self.find_session_lines(&session_id, false, 128, |line, lower| {
                is_summary_or_user_line(line, lower)
                    && term_overlap_count(lower, &focus_refs) >= 1
                    && match answer_kind {
                        "weekday" => extract_weekday_surface_from_line(lower).is_some(),
                        "time" => {
                            extract_focus_aligned_time_answer_from_line(line, lower, &focus_terms)
                                .is_some()
                        },
                        _ => false,
                    }
            });
            for (line_idx, line) in lines.into_iter().enumerate() {
                let lower = line.to_ascii_lowercase();
                let Some(answer) = (match answer_kind {
                    "weekday" => extract_weekday_surface_from_line(&lower),
                    "time" => {
                        extract_focus_aligned_time_answer_from_line(&line, &lower, &focus_terms)
                    },
                    _ => None,
                }) else {
                    continue;
                };
                let focus_overlap = term_overlap_count(&lower, &focus_refs);
                let should_replace = best
                    .as_ref()
                    .map(|(best_rank, best_focus, best_line_idx, _, _)| {
                        session_rank > *best_rank
                            || (session_rank == *best_rank && focus_overlap > *best_focus)
                            || (session_rank == *best_rank
                                && focus_overlap == *best_focus
                                && line_idx > *best_line_idx)
                    })
                    .unwrap_or(true);
                if should_replace {
                    best = Some((
                        session_rank,
                        focus_overlap,
                        line_idx,
                        answer,
                        vec![line.clone()],
                    ));
                }
            }
        }

        let (_, _, _, answer, evidence) = best?;
        self.write_synthetic_answer("current-schedule-slot", task, &answer, &evidence)
    }

    pub fn synthetic_state_transition_answer(
        &self,
        task: &str,
        task_lower: &str,
    ) -> Option<PathBuf> {
        let state_kind = if task_lower.contains("score") {
            "score"
        } else if task_lower.contains("record") {
            "record"
        } else if task_lower.contains("status") {
            "status"
        } else if task_lower.contains("goal") {
            "goal"
        } else {
            return None;
        };
        let wants_previous = task_lower.contains("previous")
            && task_contains_any(
                task_lower,
                &["before i got", "before i updated", "before i changed"],
            );
        let wants_current = !wants_previous
            && task_contains_any(
                task_lower,
                &[
                    "current",
                    "currently",
                    "now",
                    "highest score",
                    "most recent",
                ],
            );
        if !wants_previous && !wants_current {
            return None;
        }

        let mut focus_terms = synthetic_query_terms(task_lower);
        focus_terms.retain(|term| {
            !matches!(
                term.as_str(),
                "what"
                    | "current"
                    | "currently"
                    | "previous"
                    | "before"
                    | "updated"
                    | "update"
                    | "got"
                    | "get"
                    | "goal"
                    | "score"
                    | "highest"
                    | "record"
                    | "status"
                    | "frequent"
                    | "flyer"
                    | "my"
            )
        });
        if focus_terms.is_empty() {
            return None;
        }
        let focus_refs: Vec<&str> = focus_terms.iter().map(String::as_str).collect();
        let mut required_owned = focus_terms.clone();
        required_owned.extend(match state_kind {
            "score" => vec!["points".to_string(), "score".to_string()],
            "record" => vec!["record".to_string(), "team".to_string()],
            "status" => vec!["status".to_string()],
            "goal" => vec!["level".to_string(), "goal".to_string()],
            _ => return None,
        });
        required_owned.sort();
        required_owned.dedup();
        let required_terms: Vec<&str> = required_owned.iter().map(String::as_str).collect();

        let mut candidates = self
            .candidate_session_ids(task, &required_terms, 8)
            .into_iter()
            .enumerate()
            .map(|(idx, session_id)| (session_id, 8usize.saturating_sub(idx)))
            .collect::<Vec<_>>();
        if candidates.is_empty() {
            candidates = self.candidate_session_ids_by_line_overlap(&required_owned, 8);
        }

        #[allow(clippy::type_complexity)]
        let mut best: Option<(usize, usize, usize, String, Vec<String>)> = None;
        for (session_id, session_rank) in candidates {
            let lines = self.find_session_lines(&session_id, false, 128, |line, lower| {
                is_summary_or_user_line(line, lower)
                    && term_overlap_count(lower, &focus_refs) >= 1
                    && extract_state_transition_surface_from_line(line, lower, state_kind).is_some()
            });
            let mut states: Vec<(usize, String, Vec<String>)> = Vec::new();
            for (line_idx, line) in lines.into_iter().enumerate() {
                let lower = line.to_ascii_lowercase();
                let Some(answer) =
                    extract_state_transition_surface_from_line(&line, &lower, state_kind)
                else {
                    continue;
                };
                if states
                    .last()
                    .is_some_and(|(_, previous, _)| previous.eq_ignore_ascii_case(&answer))
                {
                    continue;
                }
                states.push((line_idx, answer, vec![line.clone()]));
            }
            if states.is_empty() {
                continue;
            }
            let (line_idx, answer, evidence) = if wants_previous {
                if states.len() < 2 {
                    continue;
                }
                states[states.len() - 2].clone()
            } else {
                states.last().cloned()?
            };
            let state_count = states.len();
            let should_replace = best
                .as_ref()
                .map(|(best_rank, best_state_count, best_line_idx, _, _)| {
                    session_rank > *best_rank
                        || (session_rank == *best_rank && state_count > *best_state_count)
                        || (session_rank == *best_rank
                            && state_count == *best_state_count
                            && line_idx > *best_line_idx)
                })
                .unwrap_or(true);
            if should_replace {
                best = Some((session_rank, state_count, line_idx, answer, evidence));
            }
        }

        let (_, _, _, answer, evidence) = best?;
        self.write_synthetic_answer("state-transition", task, &answer, &evidence)
    }

    pub fn synthetic_previous_purchased_item_answer(
        &self,
        task: &str,
        task_lower: &str,
    ) -> Option<PathBuf> {
        if !task_contains_any(
            task_lower,
            &[
                "before getting",
                "before i got",
                "before buying",
                "before i bought",
                "before purchasing",
                "before i purchased",
            ],
        ) || !task_contains_any(task_lower, &["gadget", "appliance"])
        {
            return None;
        }

        let current_item = extract_relative_purchase_current_item(task_lower)?;
        let current_item_lower = current_item.to_ascii_lowercase();
        let mut required_owned = synthetic_query_terms(task_lower);
        required_owned.retain(|term| {
            !matches!(
                term.as_str(),
                "what"
                    | "new"
                    | "did"
                    | "before"
                    | "getting"
                    | "got"
                    | "buying"
                    | "bought"
                    | "purchasing"
                    | "purchased"
                    | "invest"
                    | "invested"
                    | "item"
                    | "current"
                    | "previous"
                    | "my"
            )
        });
        required_owned.extend(synthetic_query_terms(&current_item_lower));
        required_owned.push("gadget".to_string());
        required_owned.sort();
        required_owned.dedup();
        let required_terms: Vec<&str> = required_owned.iter().map(String::as_str).collect();

        let mut candidates = self
            .candidate_session_ids(task, &required_terms, 8)
            .into_iter()
            .enumerate()
            .map(|(idx, session_id)| (session_id, 8usize.saturating_sub(idx)))
            .collect::<Vec<_>>();
        if candidates.is_empty() {
            candidates = self.candidate_session_ids_by_line_overlap(&required_owned, 8);
        }

        let mut best: Option<(usize, usize, String, Vec<String>)> = None;
        for (session_id, session_rank) in candidates {
            let lines = self.find_session_lines(&session_id, false, 128, |line, lower| {
                is_summary_or_user_line(line, lower)
                    && (lower.contains(&current_item_lower)
                        || extract_purchase_family_item_from_line(line, lower, "gadget").is_some())
            });
            let mut items: Vec<(usize, String, String)> = Vec::new();
            for (line_idx, line) in lines.into_iter().enumerate() {
                let lower = line.to_ascii_lowercase();
                let candidate = if lower.contains(&current_item_lower) {
                    Some(current_item_lower.clone())
                } else {
                    extract_purchase_family_item_from_line(&line, &lower, "gadget")
                };
                let Some(item) = candidate else {
                    continue;
                };
                if items
                    .last()
                    .is_some_and(|(_, previous, _)| previous.eq_ignore_ascii_case(&item))
                {
                    continue;
                }
                items.push((line_idx, item, line.clone()));
            }

            let Some(current_pos) = items
                .iter()
                .rposition(|(_, item, _)| item.eq_ignore_ascii_case(&current_item_lower))
            else {
                continue;
            };
            if current_pos == 0 {
                continue;
            }

            let current_line_idx = items[current_pos].0;
            let previous_line = items[current_pos - 1].2.clone();
            let current_line = items[current_pos].2.clone();
            let mut evidence = vec![previous_line];
            if current_line != evidence[0] {
                evidence.push(current_line);
            }
            let answer = items[current_pos - 1].1.clone();
            let should_replace = best
                .as_ref()
                .map(|(best_rank, best_line_idx, _, _)| {
                    session_rank > *best_rank
                        || (session_rank == *best_rank && current_line_idx > *best_line_idx)
                })
                .unwrap_or(true);
            if should_replace {
                best = Some((session_rank, current_line_idx, answer, evidence));
            }
        }

        let (_, _, answer, evidence) = best?;
        self.write_synthetic_answer("previous-purchased-item", task, &answer, &evidence)
    }
}
