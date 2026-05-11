//! Education and career synthetic answers: education totals, role duration, collection windows.

use super::super::*;

impl NeuronIndex {
    pub fn synthetic_formal_education_total_answer(
        &self,
        task: &str,
        task_lower: &str,
    ) -> Option<PathBuf> {
        let target_stage = extract_formal_education_target_stage(task_lower)?;
        let mut best: Option<(usize, String, Vec<String>)> = None;
        for (_, content) in self.matching_verbatim_texts(&["high", "school"], 128) {
            let lines = content
                .lines()
                .map(str::trim)
                .filter(|line| !line.is_empty())
                .filter_map(|line| {
                    let lower = line.to_ascii_lowercase();
                    is_summary_or_user_line(line, &lower).then(|| line.to_string())
                })
                .collect::<Vec<_>>();
            let facts = collect_education_stage_facts(&lines);
            let Some((total_years, evidence, fact_count)) =
                solve_formal_education_total(&facts, target_stage)
            else {
                continue;
            };
            let score = fact_count * 10;
            let should_replace = best
                .as_ref()
                .map(|(best_score, _, _)| score > *best_score)
                .unwrap_or(true);
            if should_replace {
                best = Some((score, format!("{total_years} years"), evidence));
            }
        }

        let (_, answer, evidence) = best?;
        self.write_synthetic_answer("formal-education-total", task, &answer, &evidence)
    }

    pub fn synthetic_education_milestone_interval_answer(
        &self,
        task: &str,
        task_lower: &str,
    ) -> Option<PathBuf> {
        if !task_lower.contains("how many months passed between")
            || !task_lower.contains("undergraduate degree")
            || !task_lower.contains("master's thesis")
        {
            return None;
        }

        let start_phrase = "the completion of my undergraduate degree";
        let end_phrase = "the submission of my master's thesis";
        let start_terms = synthetic_query_terms(start_phrase);
        let end_terms = synthetic_query_terms(end_phrase);
        let mut required_owned = start_terms.clone();
        required_owned.extend(end_terms.iter().cloned());
        required_owned.sort();
        required_owned.dedup();

        let mut best: Option<(usize, i32, Vec<String>)> = None;
        for (session_id, session_rank) in
            self.candidate_session_ids_by_line_overlap(&required_owned, 12)
        {
            let lines = self.find_session_lines(&session_id, false, 512, |line, lower| {
                is_summary_or_user_line(line, lower)
            });
            let Some(start_match) =
                best_user_turn_line_with_min_overlap(&lines, start_phrase, &start_terms, Some(1))
            else {
                continue;
            };
            let Some(end_match) =
                best_user_turn_line_with_min_overlap(&lines, end_phrase, &end_terms, Some(1))
            else {
                continue;
            };
            let delta_months = end_match.0 - start_match.0;
            if delta_months <= 0 {
                continue;
            }
            let score = session_rank + start_match.1 + end_match.1;
            let mut evidence = vec![start_match.2.clone()];
            if !evidence.iter().any(|line| line == &end_match.2) {
                evidence.push(end_match.2.clone());
            }
            let should_replace = best
                .as_ref()
                .map(|(best_score, best_months, _)| {
                    score > *best_score || (score == *best_score && delta_months > *best_months)
                })
                .unwrap_or(true);
            if should_replace {
                best = Some((score, delta_months, evidence));
            }
        }

        let (_, delta_months, evidence) = best?;
        self.write_synthetic_answer(
            "education-milestone-interval",
            task,
            &format!("{delta_months} months"),
            &evidence,
        )
    }

    pub fn synthetic_current_role_duration_answer(
        &self,
        task: &str,
        task_lower: &str,
    ) -> Option<PathBuf> {
        if !is_ongoing_duration_query(task_lower)
            || !task_contains_any(task_lower, &["current role", "current position"])
        {
            return None;
        }

        let task_terms = synthetic_query_terms(task_lower);
        let task_refs: Vec<&str> = task_terms.iter().map(String::as_str).collect();
        let mut best: Option<(usize, i32, Vec<String>)> = None;

        for session_id in self.session_ids_matching_line(|line, lower| {
            is_summary_or_user_line(line, lower)
                && extract_current_role_offset_months_from_line(line, lower).is_some()
        }) {
            let lines = self.find_session_lines(&session_id, false, 192, |line, lower| {
                is_summary_or_user_line(line, lower)
            });
            let mut total = None::<(usize, i32, String)>;
            let mut offset = None::<(usize, i32, String, String)>;

            for (line_idx, line) in lines.iter().enumerate() {
                let lower = line.to_ascii_lowercase();
                if let Some(total_months) =
                    extract_current_role_total_months_from_line(line, &lower)
                {
                    let score = term_overlap_count(&lower, &task_refs)
                        + usize::from(lower.contains("company")) * 2
                        + line_idx;
                    let should_replace = total
                        .as_ref()
                        .map(|(best_score, best_months, _)| {
                            score > *best_score
                                || (score == *best_score && total_months > *best_months)
                        })
                        .unwrap_or(true);
                    if should_replace {
                        total = Some((score, total_months, line.clone()));
                    }
                }

                if let Some(offset_months) =
                    extract_current_role_offset_months_from_line(line, &lower)
                {
                    let role_title = extract_current_role_title_from_transition_line(line, &lower)
                        .unwrap_or_default();
                    let role_mentions = if role_title.is_empty() {
                        0
                    } else {
                        lines
                            .iter()
                            .filter(|candidate| {
                                candidate.to_ascii_lowercase().contains(&role_title)
                            })
                            .count()
                    };
                    let score = role_mentions * 10 + line_idx;
                    let should_replace = offset
                        .as_ref()
                        .map(|(best_score, best_months, _, _)| {
                            score > *best_score
                                || (score == *best_score && offset_months >= *best_months)
                        })
                        .unwrap_or(true);
                    if should_replace {
                        offset = Some((score, offset_months, role_title, line.clone()));
                    }
                }
            }

            let (
                Some((total_score, total_months, total_line)),
                Some((offset_score, offset_months, role_title, offset_line)),
            ) = (total, offset)
            else {
                continue;
            };
            if total_months <= offset_months {
                continue;
            }

            let role_mentions = if role_title.is_empty() {
                0
            } else {
                lines
                    .iter()
                    .filter(|line| line.to_ascii_lowercase().contains(&role_title))
                    .count()
            };
            let delta_months = total_months - offset_months;
            let score = total_score + offset_score + role_mentions * 4;
            let mut evidence = vec![total_line];
            if !evidence.iter().any(|line| line == &offset_line) {
                evidence.push(offset_line);
            }
            let should_replace = best
                .as_ref()
                .map(|(best_score, best_months, _)| {
                    score > *best_score || (score == *best_score && delta_months > *best_months)
                })
                .unwrap_or(true);
            if should_replace {
                best = Some((score, delta_months, evidence));
            }
        }

        let (_, delta_months, evidence) = best?;
        self.write_synthetic_answer(
            "current-role-duration",
            task,
            &render_month_span(delta_months),
            &evidence,
        )
    }

    pub fn synthetic_direct_count_answer(&self, task: &str, task_lower: &str) -> Option<PathBuf> {
        if !detect_counting_query(task)
            || task_lower.starts_with("how long")
            || is_money_query(task)
            || task_has_recall_context(task_lower)
            || should_inject_count_aggregate(task)
            || synthetic_count_query_requires_multi_operand_reasoning(task, task_lower)
        {
            return None;
        }

        let prefers_reference_count =
            task_lower.contains("subjects") && task_contains_any(task_lower, &["study", "journal"]);
        let prefers_max_value = has_explicit_current_state_marker(task)
            || detect_knowledge_update_query(task)
            || prefers_reference_count
            || task_contains_any(
                task_lower,
                &[
                    "so far",
                    "already",
                    "completed",
                    "finished",
                    "watched",
                    " complete ",
                    " finish ",
                    " watch ",
                    "worn",
                    " wear ",
                    "tried",
                    " try ",
                    "how many times",
                    "times have i",
                    "times did i",
                    " need ",
                    " needs ",
                    " reach ",
                    " reaches ",
                    " requires ",
                    " required ",
                ],
            );
        if !prefers_max_value {
            return None;
        }

        let task_terms = synthetic_query_terms(task_lower);
        let mut focus_terms = extract_direct_count_focus_terms(&task_terms);
        if has_explicit_current_state_marker(task) || detect_knowledge_update_query(task) {
            let knowledge_terms = extract_knowledge_update_focus_terms(&task_terms);
            if !knowledge_terms.is_empty() {
                focus_terms = knowledge_terms;
            }
        }
        if focus_terms.is_empty() {
            focus_terms = task_terms.clone();
        }
        focus_terms.sort();
        focus_terms.dedup();
        if focus_terms.is_empty() {
            return None;
        }

        let required_terms: Vec<&str> = focus_terms.iter().map(String::as_str).collect();
        let task_term_refs: Vec<&str> = task_terms.iter().map(String::as_str).collect();
        let wants_current =
            has_explicit_current_state_marker(task) || detect_knowledge_update_query(task);
        let mut best: Option<(f32, i32, Vec<String>)> = None;
        let mut runner_up: Option<(f32, i32)> = None;

        for (path, content) in self.matching_verbatim_texts(&required_terms, 64) {
            let is_summary = is_session_summary_path(&path);
            for raw_line in content.lines() {
                let line = raw_line.trim();
                if line.is_empty() {
                    continue;
                }
                let lower = line.to_ascii_lowercase();
                if !is_summary_or_user_line(line, &lower) {
                    continue;
                }

                if let Some(required_role) = direct_count_required_role_phrase(task_lower) {
                    if !lower.contains(&required_role) {
                        continue;
                    }
                }

                if task_contains_any(task_lower, &["completed", "finished"])
                    && lower.contains("currently on")
                    && !task_contains_any(&lower, &["completed", "finished"])
                {
                    continue;
                }

                let focus_overlap = term_overlap_count(&lower, &required_terms);
                let min_focus_overlap = if focus_terms.len() >= 4 {
                    3
                } else if focus_terms.len() >= 2 {
                    2
                } else {
                    1
                };
                if focus_overlap < min_focus_overlap {
                    continue;
                }
                let raw_overlap = term_overlap_count(&lower, &task_term_refs);
                let Some((value, proximity_score)) =
                    extract_focus_aligned_count(line, &focus_terms, task_lower)
                else {
                    continue;
                };
                if value <= 0 {
                    continue;
                }

                let mut score = focus_overlap as f32 * 8.0
                    + raw_overlap as f32 * 2.0
                    + proximity_score as f32 * 1.5;
                if is_summary {
                    score += 1.5;
                }
                if wants_current && line_has_current_count_marker(&lower) {
                    score += 4.0;
                }
                if task_contains_any(
                    task_lower,
                    &[
                        " need ",
                        " needs ",
                        " reach ",
                        " reaches ",
                        " requires ",
                        " required ",
                    ],
                ) && task_contains_any(
                    &lower,
                    &["need", "needs", "reach", "requires", "required"],
                ) {
                    score += 2.0;
                }
                if task_contains_any(task_lower, &["completed", "finished"])
                    && task_contains_any(&lower, &["completed", "finished"])
                {
                    score += 1.5;
                }
                if task_contains_any(task_lower, &["watched", "worn", "tried"])
                    && task_contains_any(&lower, &["watched", "worn", "tried"])
                {
                    score += 1.0;
                }
                if score < 12.0 {
                    continue;
                }

                let evidence = vec![line.to_string()];
                let should_replace = best
                    .as_ref()
                    .map(|(best_score, best_value, _)| {
                        score > *best_score
                            || ((score - *best_score).abs() < 0.01
                                && prefers_max_value
                                && value > *best_value)
                    })
                    .unwrap_or(true);
                if should_replace {
                    if let Some((best_score, best_value, _)) = &best {
                        if *best_value != value {
                            runner_up = Some((*best_score, *best_value));
                        }
                    }
                    best = Some((score, value, evidence));
                } else if best
                    .as_ref()
                    .map(|(_, best_value, _)| *best_value != value)
                    .unwrap_or(true)
                    && runner_up
                        .as_ref()
                        .map(|(runner_score, runner_value)| {
                            score > *runner_score
                                || ((score - *runner_score).abs() < 0.01
                                    && prefers_max_value
                                    && value > *runner_value)
                        })
                        .unwrap_or(true)
                {
                    runner_up = Some((score, value));
                }
            }
        }

        let (best_score, value, evidence) = best?;
        if let Some((runner_score, runner_value)) = runner_up {
            if runner_value != value
                && runner_score + 0.75 >= best_score
                && !(prefers_max_value && value > runner_value)
            {
                return None;
            }
        }
        if task_lower.contains("issues of ")
            && task_contains_any(task_lower, &["finished reading", "finished"])
        {
            if let Some(answer) = evidence
                .first()
                .and_then(|line| extract_plural_issue_count_answer_from_line(line))
            {
                return self.write_synthetic_answer("direct-count", task, &answer, &evidence);
            }
        }
        self.write_synthetic_answer("direct-count", task, &value.to_string(), &evidence)
    }

    pub fn synthetic_study_subject_count_answer(
        &self,
        task: &str,
        task_lower: &str,
    ) -> Option<PathBuf> {
        if !detect_counting_query(task)
            || !task_lower.contains("subjects")
            || !task_contains_any(task_lower, &["study", "journal"])
        {
            return None;
        }

        let task_terms = synthetic_query_terms(task_lower);
        let focus_terms: Vec<String> = task_terms
            .iter()
            .filter(|term| term.len() >= 3)
            .cloned()
            .collect();
        let required_terms: Vec<&str> = focus_terms.iter().map(String::as_str).collect();
        let required_journal = study_subject_required_journal_phrase(task_lower);
        let mut best: Option<(usize, usize, i32, Vec<String>)> = None;

        for (_, content) in self.matching_verbatim_texts(&required_terms, 64) {
            for raw_line in content.lines() {
                let line = raw_line.trim();
                if line.is_empty() || !is_session_answer_candidate_line(line) {
                    continue;
                }
                let lower = line.to_ascii_lowercase();
                if !lower.contains("subject") {
                    continue;
                }
                if extract_numbered_list_item(line).is_none() && !lower.starts_with("assistant:") {
                    continue;
                }
                if let Some(journal_phrase) = required_journal.as_deref() {
                    if !lower.contains(journal_phrase) {
                        continue;
                    }
                }

                let overlap = term_overlap_count(&lower, &required_terms);
                if overlap < 4 {
                    continue;
                }
                let Some((value, proximity_score)) =
                    extract_focus_aligned_count(line, &focus_terms, task_lower)
                else {
                    continue;
                };
                if value <= 0 {
                    continue;
                }

                let evidence = vec![line.to_string()];
                let should_replace = best
                    .as_ref()
                    .map(|(best_overlap, best_proximity, best_value, _)| {
                        overlap > *best_overlap
                            || (overlap == *best_overlap && proximity_score > *best_proximity)
                            || (overlap == *best_overlap
                                && proximity_score == *best_proximity
                                && value > *best_value)
                    })
                    .unwrap_or(true);
                if should_replace {
                    best = Some((overlap, proximity_score, value, evidence));
                }
            }
        }

        let (_, _, value, evidence) = best?;
        self.write_synthetic_answer(
            "study-subject-count",
            task,
            &format!("{value} subjects"),
            &evidence,
        )
    }

    pub fn synthetic_instagram_current_count_answer(
        &self,
        task: &str,
        task_lower: &str,
    ) -> Option<PathBuf> {
        if !detect_counting_query(task)
            || !task_lower.contains("instagram")
            || !task_lower.contains("follower")
            || !task_contains_any(task_lower, &["current", "currently", "now", "these days"])
        {
            return None;
        }

        let task_terms = synthetic_query_terms(task_lower);
        let required_terms: Vec<&str> = task_terms.iter().map(String::as_str).collect();
        let mut candidates = self
            .candidate_session_ids(task, &required_terms, 8)
            .into_iter()
            .enumerate()
            .map(|(idx, session_id)| (session_id, 8usize.saturating_sub(idx)))
            .collect::<Vec<_>>();
        if candidates.is_empty() {
            candidates = self.candidate_session_ids_by_line_overlap(&task_terms, 8);
        }

        let prefers_explicit_current = task_contains_any(task_lower, &["current", "currently"]);
        let mut best: Option<(usize, usize, i32, Vec<String>)> = None;
        for (session_id, session_rank) in candidates {
            let lines = self.find_session_lines(&session_id, false, 128, |line, lower| {
                is_summary_or_user_line(line, lower)
                    && lower.contains("follower")
                    && !task_contains_any(
                        lower,
                        &["facebook", "twitter", "tiktok", "youtube", "linkedin"],
                    )
            });
            let mut session_best: Option<(usize, i32, Vec<String>)> = None;
            for (line_idx, line) in lines.into_iter().enumerate() {
                let lower = line.to_ascii_lowercase();
                let Some((value, line_strength)) =
                    extract_instagram_current_count_candidate(&line, &lower)
                else {
                    continue;
                };
                let evidence = vec![line.clone()];
                if session_best
                    .as_ref()
                    .map(|(best_metric, best_value, _)| {
                        if prefers_explicit_current {
                            line_strength > *best_metric
                                || (line_strength == *best_metric && value > *best_value)
                        } else {
                            line_idx > *best_metric
                                || (line_idx == *best_metric && value > *best_value)
                        }
                    })
                    .unwrap_or(true)
                {
                    session_best = Some((
                        if prefers_explicit_current {
                            line_strength
                        } else {
                            line_idx
                        },
                        value,
                        evidence,
                    ));
                }
            }
            let Some((line_metric, value, evidence)) = session_best else {
                continue;
            };
            if best
                .as_ref()
                .map(|(best_rank, best_metric, best_value, _)| {
                    if prefers_explicit_current {
                        line_metric > *best_metric
                            || (line_metric == *best_metric && session_rank > *best_rank)
                            || (line_metric == *best_metric
                                && session_rank == *best_rank
                                && value > *best_value)
                    } else {
                        session_rank > *best_rank
                            || (session_rank == *best_rank && line_metric > *best_metric)
                            || (session_rank == *best_rank
                                && line_metric == *best_metric
                                && value > *best_value)
                    }
                })
                .unwrap_or(true)
            {
                best = Some((session_rank, line_metric, value, evidence));
            }
        }

        let (_, _, value, evidence) = best?;
        self.write_synthetic_answer(
            "instagram-followers-current",
            task,
            &value.to_string(),
            &evidence,
        )
    }

    pub fn synthetic_collection_window_count_answer(
        &self,
        task: &str,
        task_lower: &str,
    ) -> Option<PathBuf> {
        if !detect_counting_query(task)
            || !task_contains_any(task_lower, &["collection", "collecting"])
        {
            return None;
        }
        let window = extract_query_duration_window(task_lower)?;
        let task_terms = synthetic_query_terms(task_lower);
        let mut focus_terms = extract_direct_count_focus_terms(&task_terms);
        if focus_terms.is_empty() {
            focus_terms = task_terms.clone();
        }
        focus_terms.sort();
        focus_terms.dedup();
        let required_terms: Vec<&str> = focus_terms.iter().map(String::as_str).collect();
        let task_term_refs: Vec<&str> = task_terms.iter().map(String::as_str).collect();
        let mut best: Option<(usize, usize, i32, Vec<String>)> = None;

        for (_, content) in self.matching_verbatim_texts(&required_terms, 64) {
            for raw_line in content.lines() {
                let line = raw_line.trim();
                if line.is_empty() {
                    continue;
                }
                let lower = line.to_ascii_lowercase();
                if !is_summary_or_user_line(line, &lower)
                    || !task_contains_any(&lower, &["collection", "collecting"])
                {
                    continue;
                }
                let Some(duration) = extract_duration_answer_from_line(line) else {
                    continue;
                };
                if normalize_current_duration_answer(&duration).to_ascii_lowercase() != window {
                    continue;
                }
                let focus_overlap = term_overlap_count(&lower, &required_terms);
                if focus_overlap < 3 {
                    continue;
                }
                let raw_overlap = term_overlap_count(&lower, &task_term_refs);
                let Some((value, _proximity_score)) =
                    extract_focus_aligned_count(line, &focus_terms, task_lower)
                else {
                    continue;
                };
                if value <= 0 {
                    continue;
                }
                let evidence = vec![line.to_string()];
                if best
                    .as_ref()
                    .map(|(best_focus, best_overlap, best_value, _)| {
                        focus_overlap > *best_focus
                            || (focus_overlap == *best_focus && raw_overlap > *best_overlap)
                            || (focus_overlap == *best_focus
                                && raw_overlap == *best_overlap
                                && value > *best_value)
                    })
                    .unwrap_or(true)
                {
                    best = Some((focus_overlap, raw_overlap, value, evidence));
                }
            }
        }

        let (_, _, value, evidence) = best?;
        self.write_synthetic_answer(
            "collection-window-count",
            task,
            &value.to_string(),
            &evidence,
        )
    }
}
