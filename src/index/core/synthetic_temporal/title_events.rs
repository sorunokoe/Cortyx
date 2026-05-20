//! Title duration, event intervals, and usage frequency.

use super::super::*;

impl NeuronIndex {
    pub fn synthetic_title_duration_answer(&self, task: &str, task_lower: &str) -> Option<PathBuf> {
        if !task_contains_any(task_lower, &["how long", "how many days"]) {
            return None;
        }

        let titles = extract_quoted_titles(task);
        if titles.is_empty() || !task_lower.contains("finish") {
            return None;
        }

        let combined = titles.len() >= 2
            && task_contains_any(
                task_lower,
                &[" combined", " altogether", " together", " total"],
            );
        let wants_days = task_lower.contains("how many days");

        let mut parsed = Vec::new();
        let mut evidence = Vec::new();
        for title in &titles {
            let title_lower = title.to_ascii_lowercase();
            let mut required_owned = synthetic_query_terms(&title_lower);
            required_owned.extend(["took".to_string(), "finish".to_string()]);
            required_owned.sort();
            required_owned.dedup();
            let required_terms: Vec<&str> = required_owned.iter().map(String::as_str).collect();

            let lines = self.find_matching_lines(&required_terms, 48, false, 12, |line, lower| {
                is_summary_or_user_line(line, lower)
                    && lower.contains(&title_lower)
                    && extract_title_duration_value(line, &title_lower).is_some()
            });

            let mut best: Option<(usize, SyntheticDurationValue, String)> = None;
            for (line_idx, line) in lines.into_iter().enumerate() {
                let Some(duration) = extract_title_duration_value(&line, &title_lower) else {
                    continue;
                };
                let overlap = synthetic_answer_surface_overlap_count(
                    &synthetic_answer_surface_term_key_set(&synthetic_query_terms(
                        &line.to_ascii_lowercase(),
                    )),
                    &synthetic_answer_surface_term_key_set(&synthetic_query_terms(&title_lower)),
                );
                let score = overlap * 10 + line_idx;
                let should_replace = best
                    .as_ref()
                    .map(|(best_score, _, _)| score > *best_score)
                    .unwrap_or(true);
                if should_replace {
                    best = Some((score, duration, line.clone()));
                }
            }

            let (_, duration, line) = best?;
            if !evidence.iter().any(|existing| existing == &line) {
                evidence.push(line);
            }
            parsed.push(duration);
        }

        let answer = if wants_days && parsed.len() == 1 {
            let days = parsed[0].days.round() as i32;
            format!("{days} days")
        } else if combined {
            let first_unit = parsed.first()?.unit;
            if parsed.iter().all(|value| value.unit == first_unit) {
                let total = parsed.iter().map(|value| value.amount).sum::<f32>();
                format!(
                    "{} {}",
                    compact_decimal_string(total),
                    render_duration_unit(first_unit, total)
                )
            } else {
                let total_days = parsed.iter().map(|value| value.days).sum::<f32>().round() as i32;
                render_elapsed_duration_answer(total_days)
            }
        } else {
            let duration = parsed.first()?;
            format!(
                "{} {}",
                compact_decimal_string(duration.amount),
                render_duration_unit(duration.unit, duration.amount)
            )
        };

        self.write_synthetic_answer("title-duration", task, &answer, &evidence)
    }

    pub fn synthetic_temporal_interval_between_events_answer(
        &self,
        task: &str,
        task_lower: &str,
    ) -> Option<PathBuf> {
        if !extract_quoted_titles(task).is_empty() {
            return None;
        }
        let (end_phrase, start_phrase) = extract_temporal_interval_phrases(task_lower)?;
        let end_terms = synthetic_query_terms(&end_phrase);
        let start_terms = synthetic_query_terms(&start_phrase);
        if end_terms.is_empty() || start_terms.is_empty() {
            return None;
        }

        let end_lower = end_phrase.to_ascii_lowercase();
        let start_lower = start_phrase.to_ascii_lowercase();
        let mut required_owned = end_terms.clone();
        required_owned.extend(start_terms.iter().cloned());
        required_owned.sort();
        required_owned.dedup();

        let candidates = self.candidate_session_ids_by_line_overlap(&required_owned, 12);
        let mut best: Option<(usize, i32, Vec<String>)> = None;
        for (session_id, session_rank) in candidates {
            let lines = self.find_session_lines(&session_id, false, 192, |line, lower| {
                is_summary_or_user_line(line, lower)
            });
            let Some(start_match) = best_temporal_rank_line_with_min_overlap(
                &lines,
                &start_lower,
                &start_terms,
                Some(1),
            ) else {
                continue;
            };
            let Some(end_match) =
                best_temporal_rank_line_with_min_overlap(&lines, &end_lower, &end_terms, Some(1))
            else {
                continue;
            };
            let delta_days = end_match.0 - start_match.0;
            if delta_days <= 0 {
                continue;
            }
            let combined_score = session_rank + start_match.1 + end_match.1;
            let mut evidence = vec![start_match.2.clone()];
            if !evidence.iter().any(|line| line == &end_match.2) {
                evidence.push(end_match.2.clone());
            }
            let should_replace = best
                .as_ref()
                .map(|(best_score, best_days, _)| {
                    combined_score > *best_score
                        || (combined_score == *best_score && delta_days > *best_days)
                })
                .unwrap_or(true);
            if should_replace {
                best = Some((combined_score, delta_days, evidence));
            }
        }

        let (_, delta_days, evidence) = best?;
        self.write_synthetic_answer(
            "temporal-interval-between-events",
            task,
            &format!("{delta_days} days"),
            &evidence,
        )
    }

    pub fn synthetic_item_usage_frequency_answer(
        &self,
        task: &str,
        task_lower: &str,
    ) -> Option<PathBuf> {
        if !detect_counting_query(task) {
            return None;
        }

        let (usage_kind, item_phrase) = extract_item_usage_phrase(task_lower)?;
        let item_terms = synthetic_query_terms(&item_phrase);
        if item_terms.is_empty() {
            return None;
        }

        let mut required_owned = item_terms.clone();
        match usage_kind.as_str() {
            "wear" => {
                required_owned.extend(["times".to_string(), "worn".to_string(), "wore".to_string()])
            },
            "trip" => required_owned.extend([
                "trip".to_string(),
                "trips".to_string(),
                "adventure".to_string(),
                "adventures".to_string(),
            ]),
            _ => return None,
        }
        required_owned.sort();
        required_owned.dedup();
        let required_terms: Vec<&str> = required_owned.iter().map(String::as_str).collect();
        let item_keys = synthetic_answer_surface_term_key_set(&item_terms);
        let required_keys = synthetic_answer_surface_term_key_set(&required_owned);
        let min_item_overlap = if item_keys.len() >= 2 { 2 } else { 1 };

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
        let mut best: Option<(String, usize, i32, String, usize, Vec<String>)> = None;
        for (session_id, session_rank) in candidates {
            let lines = self.find_session_lines(&session_id, false, 128, |line, lower| {
                let line_keys =
                    synthetic_answer_surface_term_key_set(&synthetic_query_terms(lower));
                is_summary_or_user_line(line, lower)
                    && synthetic_answer_surface_overlap_count(&line_keys, &required_keys) >= 2
                    && !line_has_future_goal_marker(lower)
            });
            for (line_idx, line) in lines.into_iter().enumerate() {
                let lower = line.to_ascii_lowercase();
                let line_keys =
                    synthetic_answer_surface_term_key_set(&synthetic_query_terms(&lower));
                if synthetic_answer_surface_overlap_count(&line_keys, &item_keys) < min_item_overlap
                {
                    continue;
                }
                let Some(value) = extract_item_usage_count_from_line(&line, &lower, &usage_kind)
                else {
                    continue;
                };
                if value <= 0 {
                    continue;
                }
                let answer = extract_item_usage_count_surface_from_line(&line, &lower, &usage_kind)
                    .unwrap_or_else(|| value.to_string());
                let should_replace = best
                    .as_ref()
                    .map(|(_, best_rank, best_value, _, best_line_idx, _)| {
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
                        answer,
                        line_idx,
                        vec![line.clone()],
                    ));
                }
            }
        }

        if let Some((session_id, _, value, answer, _, evidence)) = best {
            let session_lines =
                self.find_session_lines(&session_id, false, 192, |line, _| !line.trim().is_empty());
            let rendered = if answer.chars().all(|ch| ch.is_ascii_digit()) {
                supporting_word_count_surface(&session_lines, value, &item_terms).unwrap_or(answer)
            } else {
                answer
            };
            return self.write_synthetic_answer("item-usage-count", task, &rendered, &evidence);
        }

        #[allow(clippy::type_complexity)]
        let mut best_fallback: Option<(i32, String, Vec<String>, Vec<String>)> = None;
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
                    || synthetic_answer_surface_overlap_count(&line_keys, &required_keys) < 2
                    || synthetic_answer_surface_overlap_count(&line_keys, &item_keys)
                        < min_item_overlap
                    || line_has_future_goal_marker(&lower)
                {
                    continue;
                }
                let Some(value) = extract_item_usage_count_from_line(line, &lower, &usage_kind)
                else {
                    continue;
                };
                let answer = extract_item_usage_count_surface_from_line(line, &lower, &usage_kind)
                    .unwrap_or_else(|| value.to_string());
                let should_replace = best_fallback
                    .as_ref()
                    .map(|(best_value, _, _, _)| value > *best_value)
                    .unwrap_or(true);
                if should_replace {
                    best_fallback =
                        Some((value, answer, vec![line.to_string()], content_lines.clone()));
                }
            }
        }

        let (value, answer, evidence, content_lines) = best_fallback?;
        let rendered = if answer.chars().all(|ch| ch.is_ascii_digit()) {
            supporting_word_count_surface(&content_lines, value, &item_terms).unwrap_or(answer)
        } else {
            answer
        };
        self.write_synthetic_answer("item-usage-count", task, &rendered, &evidence)
    }
}
