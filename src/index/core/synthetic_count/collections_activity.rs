//! Collection and activity synthetic answers: publications, weight loss, collection counts.

use super::super::*;

impl NeuronIndex {
    pub fn synthetic_daily_time_commitment_answer(
        &self,
        task: &str,
        task_lower: &str,
    ) -> Option<PathBuf> {
        if !detect_counting_query(task)
            || is_money_query(task)
            || !task_contains_any(
                task_lower,
                &[
                    "how much time do i dedicate to ",
                    "how much time do i spend on ",
                    "how much time do i spend ",
                ],
            )
            || !task_contains_any(task_lower, &["each day", "every day", "daily"])
        {
            return None;
        }

        let focus_phrase = extract_daily_duration_commitment_phrase(task_lower)?;
        let phrase_terms = synthetic_query_terms(&focus_phrase);
        let mut focus_terms = extract_direct_count_focus_terms(&phrase_terms);
        focus_terms.retain(|term| {
            !matches!(
                term.as_str(),
                "dedicate"
                    | "dedicating"
                    | "spend"
                    | "spending"
                    | "practice"
                    | "practicing"
                    | "practise"
                    | "practising"
            )
        });
        if focus_terms.is_empty() {
            focus_terms = phrase_terms;
        }
        let focus_keys = synthetic_answer_surface_term_key_set(&focus_terms);
        let min_focus_overlap = if focus_keys.len() >= 3 { 2 } else { 1 };
        let mut required_owned = focus_terms.clone();
        required_owned.extend([
            "daily".to_string(),
            "day".to_string(),
            "each".to_string(),
            "every".to_string(),
        ]);
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

        let mut best: Option<(usize, usize, f32, String, Vec<String>)> = None;
        for (session_id, session_rank) in candidates {
            let lines = self.find_session_lines(&session_id, false, 128, |line, lower| {
                is_summary_or_user_line(line, lower)
                    && line_has_daily_duration_marker(lower)
                    && extract_duration_answer_from_line(line).is_some()
            });
            for (line_idx, line) in lines.into_iter().enumerate() {
                let lower = line.to_ascii_lowercase();
                let line_keys =
                    synthetic_answer_surface_term_key_set(&synthetic_query_terms(&lower));
                if synthetic_answer_surface_overlap_count(&line_keys, &focus_keys)
                    < min_focus_overlap
                {
                    continue;
                }
                let Some(answer) = extract_duration_answer_from_line(&line) else {
                    continue;
                };
                let Some(magnitude) =
                    duration_answer_magnitude(&normalize_current_duration_answer(&answer))
                else {
                    continue;
                };
                let rendered = answer.to_ascii_lowercase();
                let should_replace = best
                    .as_ref()
                    .map(|(best_rank, best_line_idx, best_magnitude, _, _)| {
                        session_rank > *best_rank
                            || (session_rank == *best_rank && line_idx > *best_line_idx)
                            || (session_rank == *best_rank
                                && line_idx == *best_line_idx
                                && magnitude > *best_magnitude)
                    })
                    .unwrap_or(true);
                if should_replace {
                    best = Some((
                        session_rank,
                        line_idx,
                        magnitude,
                        rendered,
                        vec![line.clone()],
                    ));
                }
            }
        }

        if let Some((_, _, _, answer, evidence)) = best {
            return self.write_synthetic_answer("daily-time-commitment", task, &answer, &evidence);
        }

        None
    }

    pub fn synthetic_time_spent_range_answer(
        &self,
        task: &str,
        task_lower: &str,
    ) -> Option<PathBuf> {
        if !detect_counting_query(task)
            || is_money_query(task)
            || !task_contains_any(task_lower, &["how many hours", "hours have i spent"])
            || !task_contains_any(task_lower, &["spent on", "spent on my"])
        {
            return None;
        }

        let task_terms = synthetic_query_terms(task_lower);
        let mut focus_terms = extract_direct_count_focus_terms(&task_terms);
        if focus_terms.is_empty() {
            focus_terms = task_terms.clone();
        }
        focus_terms.sort();
        focus_terms.dedup();
        let required_terms: Vec<&str> = focus_terms.iter().map(String::as_str).collect();
        let mut candidates = self
            .candidate_session_ids(task, &required_terms, 8)
            .into_iter()
            .enumerate()
            .map(|(idx, session_id)| (session_id, 8usize.saturating_sub(idx)))
            .collect::<Vec<_>>();
        if candidates.is_empty() {
            candidates = self.candidate_session_ids_by_line_overlap(&task_terms, 8);
        }

        let mut best: Option<(usize, usize, f32, String, Vec<String>)> = None;
        for (session_id, session_rank) in candidates {
            let lines = self.find_session_lines(&session_id, false, 128, |line, lower| {
                is_summary_or_user_line(line, lower)
                    && task_contains_any(lower, &["spent", "put in", "working on"])
                    && extract_duration_answer_from_line(line).is_some()
            });
            for (line_idx, line) in lines.into_iter().enumerate() {
                let lower = line.to_ascii_lowercase();
                let focus_overlap = term_overlap_count(&lower, &required_terms);
                if focus_overlap < 3 {
                    continue;
                }
                let Some(answer) = extract_duration_answer_from_line(&line) else {
                    continue;
                };
                let normalized = normalize_current_duration_answer(&answer);
                if !normalized.to_ascii_lowercase().contains("hour") {
                    continue;
                }
                let magnitude = duration_answer_magnitude(&normalized).unwrap_or(0.0);
                let evidence = vec![line.clone()];
                if best
                    .as_ref()
                    .map(|(best_rank, best_line_idx, best_magnitude, _, _)| {
                        session_rank > *best_rank
                            || (session_rank == *best_rank && line_idx > *best_line_idx)
                            || (session_rank == *best_rank
                                && line_idx == *best_line_idx
                                && magnitude > *best_magnitude)
                    })
                    .unwrap_or(true)
                {
                    best = Some((session_rank, line_idx, magnitude, normalized, evidence));
                }
            }
        }

        if let Some((_, _, _, answer, evidence)) = best {
            return self.write_synthetic_answer("time-spent-range", task, &answer, &evidence);
        }

        for (_, content) in self.matching_verbatim_texts(&required_terms, 32) {
            for raw_line in content.lines() {
                let line = raw_line.trim();
                if line.is_empty() {
                    continue;
                }
                let lower = line.to_ascii_lowercase();
                if !is_summary_or_user_line(line, &lower)
                    || term_overlap_count(&lower, &required_terms) < 3
                {
                    continue;
                }
                let Some(answer) = extract_duration_answer_from_line(line) else {
                    continue;
                };
                let normalized = normalize_current_duration_answer(&answer);
                if normalized.to_ascii_lowercase().contains("hour") {
                    return self.write_synthetic_answer(
                        "time-spent-range",
                        task,
                        &normalized,
                        &[line.to_string()],
                    );
                }
            }
        }

        None
    }

    pub fn synthetic_publication_issue_count_answer(
        &self,
        task: &str,
        task_lower: &str,
    ) -> Option<PathBuf> {
        if !detect_counting_query(task)
            || !task_lower.contains("issues of ")
            || !task_contains_any(task_lower, &["finished reading", "finished"])
        {
            return None;
        }

        let publication_phrase = extract_issue_publication_phrase(task_lower)?;
        let publication_terms = synthetic_query_terms(&publication_phrase);
        if publication_terms.is_empty() {
            return None;
        }
        let mut required_owned = publication_terms.clone();
        required_owned.push("issues".to_string());
        required_owned.push("finished".to_string());
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
                    && lower.contains(&publication_phrase)
                    && lower.contains("issue")
                    && lower.contains("finished")
            });
            for (line_idx, line) in lines.into_iter().enumerate() {
                let lower = line.to_ascii_lowercase();
                if term_overlap_count(&lower, &required_terms) < 3 {
                    continue;
                }
                let Some(answer) = extract_plural_issue_count_answer_from_line(&line) else {
                    continue;
                };
                let evidence = vec![line.clone()];
                if best
                    .as_ref()
                    .map(|(best_rank, best_line_idx, _, _)| {
                        session_rank > *best_rank
                            || (session_rank == *best_rank && line_idx > *best_line_idx)
                    })
                    .unwrap_or(true)
                {
                    best = Some((session_rank, line_idx, answer, evidence));
                }
            }
        }

        if let Some((_, _, answer, evidence)) = best {
            return self.write_synthetic_answer(
                "publication-issue-count",
                task,
                &answer,
                &evidence,
            );
        }

        for (_, content) in self.matching_verbatim_texts(&required_terms, 32) {
            for raw_line in content.lines() {
                let line = raw_line.trim();
                if line.is_empty() {
                    continue;
                }
                let lower = line.to_ascii_lowercase();
                if !is_summary_or_user_line(line, &lower)
                    || !lower.contains(&publication_phrase)
                    || term_overlap_count(&lower, &required_terms) < 3
                {
                    continue;
                }
                if let Some(answer) = extract_plural_issue_count_answer_from_line(line) {
                    return self.write_synthetic_answer(
                        "publication-issue-count",
                        task,
                        &answer,
                        &[line.to_string()],
                    );
                }
            }
        }

        None
    }

    pub fn synthetic_collection_restart_count_answer(
        &self,
        task: &str,
        task_lower: &str,
    ) -> Option<PathBuf> {
        if !detect_counting_query(task)
            || !task_lower.contains("collecting again")
            || !task_contains_any(task_lower, &["collection", "collecting"])
        {
            return None;
        }

        let task_terms = synthetic_query_terms(task_lower);
        let mut focus_terms = extract_direct_count_focus_terms(&task_terms);
        if focus_terms.is_empty() {
            focus_terms = task_terms.clone();
        }
        focus_terms.sort();
        focus_terms.dedup();
        let required_terms: Vec<&str> = focus_terms.iter().map(String::as_str).collect();
        let mut candidates = self
            .candidate_session_ids(task, &required_terms, 8)
            .into_iter()
            .enumerate()
            .map(|(idx, session_id)| (session_id, 8usize.saturating_sub(idx)))
            .collect::<Vec<_>>();
        if candidates.is_empty() {
            candidates = self.candidate_session_ids_by_line_overlap(&focus_terms, 8);
        }

        let mut best: Option<(usize, usize, i32, Vec<String>)> = None;
        for (session_id, session_rank) in candidates {
            let lines = self.find_session_lines(&session_id, false, 128, |line, lower| {
                is_summary_or_user_line(line, lower)
                    && lower.contains("collecting again")
                    && task_contains_any(lower, &["collection", "collecting"])
            });
            for (line_idx, line) in lines.into_iter().enumerate() {
                let lower = line.to_ascii_lowercase();
                let focus_overlap = term_overlap_count(&lower, &required_terms);
                if focus_overlap < 3 {
                    continue;
                }
                let Some((value, _)) = extract_focus_aligned_count(&line, &focus_terms, task_lower)
                else {
                    continue;
                };
                if value <= 0 {
                    continue;
                }
                let evidence = vec![line.clone()];
                if best
                    .as_ref()
                    .map(|(best_rank, best_line_idx, best_value, _)| {
                        session_rank > *best_rank
                            || (session_rank == *best_rank && line_idx > *best_line_idx)
                            || (session_rank == *best_rank
                                && line_idx == *best_line_idx
                                && value > *best_value)
                    })
                    .unwrap_or(true)
                {
                    best = Some((session_rank, line_idx, value, evidence));
                }
            }
        }

        if let Some((_, _, value, evidence)) = best {
            return self.write_synthetic_answer(
                "collection-restart-count",
                task,
                &value.to_string(),
                &evidence,
            );
        }

        for (_, content) in self.matching_verbatim_texts(&required_terms, 32) {
            for raw_line in content.lines() {
                let line = raw_line.trim();
                if line.is_empty() {
                    continue;
                }
                let lower = line.to_ascii_lowercase();
                if !is_summary_or_user_line(line, &lower)
                    || !lower.contains("collecting again")
                    || term_overlap_count(&lower, &required_terms) < 3
                {
                    continue;
                }
                let Some((value, _)) = extract_focus_aligned_count(line, &focus_terms, task_lower)
                else {
                    continue;
                };
                if value > 0 {
                    return self.write_synthetic_answer(
                        "collection-restart-count",
                        task,
                        &value.to_string(),
                        &[line.to_string()],
                    );
                }
            }
        }

        None
    }

    pub fn synthetic_weight_loss_since_start_answer(
        &self,
        task: &str,
        task_lower: &str,
    ) -> Option<PathBuf> {
        if !detect_counting_query(task)
            || !task_contains_any(task_lower, &["how much weight", "weight have i lost"])
            || (!task_lower.contains("since starting ") && !task_lower.contains("since i started "))
        {
            return None;
        }

        let anchor_phrase = extract_since_start_anchor_phrase(task_lower)?;
        let anchor_terms = synthetic_query_terms(&anchor_phrase);
        if anchor_terms.is_empty() {
            return None;
        }
        let anchor_keys = synthetic_answer_surface_term_key_set(&anchor_terms);
        let min_anchor_overlap = if anchor_keys.len() >= 3 { 2 } else { 1 };
        let focus_terms = vec![
            "lost".to_string(),
            "weight".to_string(),
            "pounds".to_string(),
        ];
        let focus_keys = synthetic_answer_surface_term_key_set(&focus_terms);
        let mut required_owned = focus_terms.clone();
        required_owned.extend(anchor_terms.iter().cloned());
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

        let mut best: Option<(usize, usize, i32, String, Vec<String>)> = None;
        for (session_id, session_rank) in candidates {
            let lines = self.find_session_lines(&session_id, false, 128, |line, lower| {
                is_summary_or_user_line(line, lower)
                    && lower.contains("lost")
                    && lower.contains("pound")
            });
            for (line_idx, line) in lines.into_iter().enumerate() {
                let lower = line.to_ascii_lowercase();
                let line_keys =
                    synthetic_answer_surface_term_key_set(&synthetic_query_terms(&lower));
                if synthetic_answer_surface_overlap_count(&line_keys, &focus_keys) == 0
                    || synthetic_answer_surface_overlap_count(&line_keys, &anchor_keys)
                        < min_anchor_overlap
                {
                    continue;
                }
                let Some((value, answer)) = extract_weight_loss_answer_from_line(&line, &lower)
                else {
                    continue;
                };
                let should_replace = best
                    .as_ref()
                    .map(|(best_rank, best_line_idx, best_value, _, _)| {
                        value > *best_value
                            || (value == *best_value
                                && (session_rank > *best_rank
                                    || (session_rank == *best_rank && line_idx > *best_line_idx)))
                    })
                    .unwrap_or(true);
                if should_replace {
                    best = Some((session_rank, line_idx, value, answer, vec![line.clone()]));
                }
            }
        }

        if let Some((_, _, _, answer, evidence)) = best {
            return self.write_synthetic_answer(
                "weight-loss-since-start",
                task,
                &answer,
                &evidence,
            );
        }

        let mut best_fallback: Option<(i32, String, Vec<String>)> = None;
        for (_, content) in self.matching_verbatim_texts(&required_terms, 32) {
            for raw_line in content.lines() {
                let line = raw_line.trim();
                if line.is_empty() {
                    continue;
                }
                let lower = line.to_ascii_lowercase();
                let line_keys =
                    synthetic_answer_surface_term_key_set(&synthetic_query_terms(&lower));
                if !is_summary_or_user_line(line, &lower)
                    || synthetic_answer_surface_overlap_count(&line_keys, &anchor_keys)
                        < min_anchor_overlap
                    || synthetic_answer_surface_overlap_count(&line_keys, &focus_keys) == 0
                {
                    continue;
                }
                let Some((value, answer)) = extract_weight_loss_answer_from_line(line, &lower)
                else {
                    continue;
                };
                let should_replace = best_fallback
                    .as_ref()
                    .map(|(best_value, _, _)| value > *best_value)
                    .unwrap_or(true);
                if should_replace {
                    best_fallback = Some((value, answer, vec![line.to_string()]));
                }
            }
        }

        let (_, answer, evidence) = best_fallback?;
        self.write_synthetic_answer("weight-loss-since-start", task, &answer, &evidence)
    }

    pub fn synthetic_since_start_count_answer(
        &self,
        task: &str,
        task_lower: &str,
    ) -> Option<PathBuf> {
        if !detect_counting_query(task)
            || (!task_lower.contains("since starting ") && !task_lower.contains("since i started "))
            || task_lower.contains("collecting again")
            || task_contains_any(
                task_lower,
                &["how many times", "times have i", "times did i"],
            )
        {
            return None;
        }

        let anchor_phrase = extract_since_start_anchor_phrase(task_lower)?;
        let anchor_terms = synthetic_query_terms(&anchor_phrase);
        if anchor_terms.is_empty() {
            return None;
        }

        let task_terms = synthetic_query_terms(task_lower);
        let mut focus_terms = extract_direct_count_focus_terms(&task_terms);
        if focus_terms.is_empty() {
            focus_terms = task_terms.clone();
        }
        focus_terms
            .retain(|term| !matches!(term.as_str(), "since" | "start" | "starting" | "started"));
        focus_terms.sort();
        focus_terms.dedup();

        let mut required_owned = focus_terms.clone();
        required_owned.extend(anchor_terms.iter().cloned());
        required_owned.sort();
        required_owned.dedup();
        let required_terms: Vec<&str> = required_owned.iter().map(String::as_str).collect();
        let focus_keys = synthetic_answer_surface_term_key_set(&focus_terms);
        let anchor_keys = synthetic_answer_surface_term_key_set(&anchor_terms);
        let required_keys = synthetic_answer_surface_term_key_set(&required_owned);

        let mut candidates = self
            .candidate_session_ids(task, &required_terms, 8)
            .into_iter()
            .enumerate()
            .map(|(idx, session_id)| (session_id, 8usize.saturating_sub(idx)))
            .collect::<Vec<_>>();
        if candidates.is_empty() {
            candidates = self.candidate_session_ids_by_line_overlap(&required_owned, 8);
        }

        let mut best: Option<(String, usize, i32, usize, Vec<String>)> = None;
        for (session_id, session_rank) in candidates {
            let lines = self.find_session_lines(&session_id, false, 128, |line, lower| {
                let line_keys =
                    synthetic_answer_surface_term_key_set(&synthetic_query_terms(lower));
                is_summary_or_user_line(line, lower)
                    && synthetic_answer_surface_overlap_count(&line_keys, &required_keys) >= 3
                    && synthetic_answer_surface_overlap_count(&line_keys, &anchor_keys) >= 1
                    && line_has_progress_count_marker(lower)
                    && !line_has_future_goal_marker(lower)
            });
            for (line_idx, line) in lines.into_iter().enumerate() {
                let lower = line.to_ascii_lowercase();
                let line_keys =
                    synthetic_answer_surface_term_key_set(&synthetic_query_terms(&lower));
                if synthetic_answer_surface_overlap_count(&line_keys, &focus_keys) == 0
                    || synthetic_answer_surface_overlap_count(&line_keys, &anchor_keys) == 0
                {
                    continue;
                }
                let Some((value, _)) = extract_focus_aligned_count(&line, &focus_terms, task_lower)
                else {
                    continue;
                };
                if value <= 0 {
                    continue;
                }
                let should_replace = best
                    .as_ref()
                    .map(|(_, best_rank, best_value, best_line_idx, _)| {
                        value > *best_value
                            || (value == *best_value
                                && (session_rank > *best_rank
                                    || (session_rank == *best_rank && line_idx > *best_line_idx)))
                    })
                    .unwrap_or(true);
                if should_replace {
                    best = Some((
                        session_id.clone(),
                        session_rank,
                        value,
                        line_idx,
                        vec![line.clone()],
                    ));
                }
            }
        }

        if let Some((session_id, _, value, _, evidence)) = best {
            let session_lines =
                self.find_session_lines(&session_id, false, 192, |line, _| !line.trim().is_empty());
            let answer = supporting_word_count_surface(&session_lines, value, &focus_terms)
                .unwrap_or_else(|| value.to_string());
            return self.write_synthetic_answer("since-start-count", task, &answer, &evidence);
        }

        let mut best_fallback: Option<(i32, Vec<String>, Vec<String>)> = None;
        for (_, content) in self.matching_verbatim_texts(&required_terms, 32) {
            let content_lines: Vec<String> = content
                .lines()
                .map(str::trim)
                .map(ToString::to_string)
                .collect();
            for raw_line in content.lines() {
                let line = raw_line.trim();
                if line.is_empty() {
                    continue;
                }
                let lower = line.to_ascii_lowercase();
                let line_keys =
                    synthetic_answer_surface_term_key_set(&synthetic_query_terms(&lower));
                if !is_summary_or_user_line(line, &lower)
                    || synthetic_answer_surface_overlap_count(&line_keys, &required_keys) < 3
                    || synthetic_answer_surface_overlap_count(&line_keys, &anchor_keys) == 0
                    || !line_has_progress_count_marker(&lower)
                    || line_has_future_goal_marker(&lower)
                {
                    continue;
                }
                let Some((value, _)) = extract_focus_aligned_count(line, &focus_terms, task_lower)
                else {
                    continue;
                };
                if value <= 0 {
                    continue;
                }
                let should_replace = best_fallback
                    .as_ref()
                    .map(|(best_value, _, _)| value > *best_value)
                    .unwrap_or(true);
                if should_replace {
                    best_fallback = Some((value, vec![line.to_string()], content_lines.clone()));
                }
            }
        }

        let (value, evidence, content_lines) = best_fallback?;
        let answer = supporting_word_count_surface(&content_lines, value, &focus_terms)
            .unwrap_or_else(|| value.to_string());
        self.write_synthetic_answer("since-start-count", task, &answer, &evidence)
    }
}
