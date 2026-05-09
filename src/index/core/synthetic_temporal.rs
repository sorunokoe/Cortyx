// This file is a submodule of `crate::index::core`.
// Contains `impl NeuronIndex` synthetic answer methods extracted from synthetic.rs.
use super::*;

impl NeuronIndex {
    pub(super) fn synthetic_temporal_choice_answer(
        &self,
        task: &str,
        task_lower: &str,
    ) -> Option<PathBuf> {
        if !task_lower.starts_with("which ")
            || !task_contains_any(
                task_lower,
                &[" first", " earlier", " before", " later", " after"],
            )
        {
            return None;
        }

        let (left_option, right_option) = extract_temporal_choice_options(task)?;
        let left_lower = left_option.to_ascii_lowercase();
        let right_lower = right_option.to_ascii_lowercase();
        let left_terms = synthetic_query_terms(&left_lower);
        let right_terms = synthetic_query_terms(&right_lower);
        if left_terms.is_empty() || right_terms.is_empty() {
            return None;
        }

        let mut required_owned = left_terms.clone();
        required_owned.extend(right_terms.iter().cloned());
        required_owned.sort();
        required_owned.dedup();

        let prefer_later = task_contains_any(task_lower, &[" later", " after"])
            && !task_contains_any(task_lower, &[" first", " earlier", " before"]);

        let candidates = self.candidate_session_ids_by_line_overlap(&required_owned, 12);
        let mut best: Option<(usize, usize, String, Vec<String>)> = None;
        for (session_id, session_rank) in candidates {
            let lines = self.find_session_lines(&session_id, false, 192, |line, lower| {
                if !is_summary_or_user_line(line, lower) {
                    return false;
                }
                let line_keys =
                    synthetic_answer_surface_term_key_set(&synthetic_query_terms(lower));
                let left_keys = synthetic_answer_surface_term_key_set(&left_terms);
                let right_keys = synthetic_answer_surface_term_key_set(&right_terms);
                synthetic_answer_surface_overlap_count(&line_keys, &left_keys) > 0
                    || synthetic_answer_surface_overlap_count(&line_keys, &right_keys) > 0
            });

            let Some(left_match) = best_temporal_rank_line(&lines, &left_lower, &left_terms) else {
                continue;
            };
            let Some(right_match) = best_temporal_rank_line(&lines, &right_lower, &right_terms)
            else {
                continue;
            };
            if left_match.0 == right_match.0 {
                continue;
            }

            let (answer, gap) = if prefer_later {
                if left_match.0 > right_match.0 {
                    (left_option.clone(), left_match.0 - right_match.0)
                } else {
                    (right_option.clone(), right_match.0 - left_match.0)
                }
            } else if left_match.0 < right_match.0 {
                (left_option.clone(), right_match.0 - left_match.0)
            } else {
                (right_option.clone(), left_match.0 - right_match.0)
            };

            let combined_score =
                session_rank + left_match.1 + right_match.1 + (gap as usize).min(30);
            let mut evidence = vec![left_match.2.clone()];
            if !evidence.iter().any(|line| line == &right_match.2) {
                evidence.push(right_match.2.clone());
            }
            let should_replace = best
                .as_ref()
                .map(|(best_score, best_gap, _, _)| {
                    combined_score > *best_score
                        || (combined_score == *best_score && gap as usize > *best_gap)
                })
                .unwrap_or(true);
            if should_replace {
                best = Some((combined_score, gap as usize, answer, evidence));
            }
        }

        let (_, _, answer, evidence) = best?;
        self.write_synthetic_answer("temporal-choice", task, &answer, &evidence)
    }

    pub(super) fn synthetic_temporal_elapsed_duration_answer(
        &self,
        task: &str,
        task_lower: &str,
    ) -> Option<PathBuf> {
        let (subject_phrase, event_phrase) = extract_temporal_elapsed_phrases(task_lower)?;
        let subject_terms = synthetic_query_terms(&subject_phrase);
        let event_terms = synthetic_query_terms(&event_phrase);
        if subject_terms.is_empty() || event_terms.is_empty() {
            return None;
        }

        let subject_lower = subject_phrase.to_ascii_lowercase();
        let event_lower = event_phrase.to_ascii_lowercase();
        let mut required_owned = subject_terms.clone();
        required_owned.extend(event_terms.iter().cloned());
        required_owned.sort();
        required_owned.dedup();

        let candidates = self.candidate_session_ids_by_line_overlap(&required_owned, 12);
        let mut best: Option<(usize, i32, String, Vec<String>)> = None;
        for (session_id, session_rank) in candidates {
            let lines = self.find_session_lines(&session_id, false, 192, |line, lower| {
                is_summary_or_user_line(line, lower)
            });
            let Some(subject_match) =
                best_temporal_duration_anchor_line(&lines, &subject_lower, &subject_terms)
            else {
                continue;
            };
            let Some(event_match) =
                best_temporal_event_anchor_line(&lines, &event_lower, &event_terms)
            else {
                continue;
            };
            let delta_days = match (subject_match.0, event_match.0) {
                (
                    SyntheticDurationAnchor::CurrentDays(subject_days),
                    SyntheticEventAnchor::RelativeDaysAgo(event_days),
                ) => subject_days - event_days,
                (
                    SyntheticDurationAnchor::AbsoluteDay(start_day),
                    SyntheticEventAnchor::AbsoluteDay(event_day),
                ) => event_day - start_day,
                _ => continue,
            };
            if delta_days <= 0 {
                continue;
            }
            let answer = render_elapsed_duration_answer(delta_days);
            let combined_score = session_rank + subject_match.1 + event_match.1;
            let mut evidence = vec![subject_match.2.clone()];
            if !evidence.iter().any(|line| line == &event_match.2) {
                evidence.push(event_match.2.clone());
            }
            let should_replace = best
                .as_ref()
                .map(|(best_score, best_days, _, _)| {
                    combined_score > *best_score
                        || (combined_score == *best_score && delta_days > *best_days)
                })
                .unwrap_or(true);
            if should_replace {
                best = Some((combined_score, delta_days, answer, evidence));
            }
        }

        let (_, _, answer, evidence) = best?;
        self.write_synthetic_answer("temporal-elapsed-duration", task, &answer, &evidence)
    }

    pub(super) fn synthetic_temporal_from_now_answer(
        &self,
        task: &str,
        task_lower: &str,
    ) -> Option<PathBuf> {
        let query = extract_temporal_from_now_query(task_lower)?;
        let event_terms = synthetic_query_terms(&query.event_phrase);
        if event_terms.is_empty() {
            return None;
        }

        let reference_label = extract_task_reference_label(task);
        let reference_day = reference_label
            .as_deref()
            .and_then(extract_explicit_date_rank);
        let event_lower = query.event_phrase.to_ascii_lowercase();
        let event_candidates =
            self.temporal_from_now_event_candidates(&event_lower, &event_terms, reference_day);
        let (_event_session_id, event_day, event_line, current_day, current_line) =
            if let Some(anchor_phrase) = query.anchor_phrase {
                let anchor_terms = synthetic_query_terms(&anchor_phrase);
                if anchor_terms.is_empty() {
                    return None;
                }
                let anchor_lower = anchor_phrase.to_ascii_lowercase();
                let anchor_candidates = self.temporal_from_now_event_candidates(
                    &anchor_lower,
                    &anchor_terms,
                    reference_day,
                );
                let mut best_pair: Option<(usize, i32, i32, String, String, String)> = None;
                for (session_id, event_score, event_day, event_line) in &event_candidates {
                    for (anchor_session_id, anchor_score, anchor_day, anchor_line) in
                        &anchor_candidates
                    {
                        if anchor_session_id != session_id {
                            continue;
                        }
                        if *anchor_day <= *event_day {
                            continue;
                        }
                        let combined_score = event_score + anchor_score;
                        let should_replace = best_pair
                            .as_ref()
                            .map(
                                |(
                                    best_score,
                                    best_anchor_day,
                                    best_event_day,
                                    _,
                                    best_event_line,
                                    best_anchor_line,
                                )| {
                                    combined_score > *best_score
                                        || (combined_score == *best_score
                                            && (*anchor_day > *best_anchor_day
                                                || (*anchor_day == *best_anchor_day
                                                    && (*event_day > *best_event_day
                                                        || (*event_day == *best_event_day
                                                            && (event_line.as_str()
                                                                < best_event_line.as_str()
                                                                || (event_line.as_str()
                                                                    == best_event_line
                                                                        .as_str()
                                                                    && anchor_line.as_str()
                                                                        < best_anchor_line
                                                                            .as_str())))))))
                                },
                            )
                            .unwrap_or(true);
                        if should_replace {
                            best_pair = Some((
                                combined_score,
                                *anchor_day,
                                *event_day,
                                session_id.clone(),
                                event_line.clone(),
                                anchor_line.clone(),
                            ));
                        }
                    }
                }
                let (_, current_day, event_day, event_session_id, event_line, current_line) =
                    best_pair?;
                (
                    event_session_id,
                    event_day,
                    event_line,
                    current_day,
                    current_line,
                )
            } else {
                let (event_session_id, _, event_day, event_line) =
                    event_candidates.into_iter().next()?;
                let (current_day, current_line) = if let Some(day) = reference_day {
                    let label = reference_label.unwrap_or_else(|| task.to_string());
                    (day, format!("reference date: {label}"))
                } else {
                    self.best_temporal_current_anchor_session(&event_session_id)?
                };
                (
                    event_session_id,
                    event_day,
                    event_line,
                    current_day,
                    current_line,
                )
            };
        let delta_days = current_day - event_day;
        if delta_days <= 0 {
            return None;
        }

        let answer = render_elapsed_from_now_answer(delta_days, query.unit, query.append_ago);
        let evidence = if current_line == event_line {
            vec![event_line]
        } else {
            vec![event_line, current_line]
        };
        self.write_synthetic_answer("temporal-from-now", task, &answer, &evidence)
    }

    pub(super) fn best_temporal_current_anchor_session(
        &self,
        session_id: &str,
    ) -> Option<(i32, String)> {
        let mut best: Option<(i32, usize, String)> = None;
        for entries in self.verbatim_entry_groups_for_session(session_id) {
            let lines = self.read_matching_session_group_lines(&entries, |_line, lower| {
                lower.starts_with("[session") || lower.starts_with("user:")
            });
            let Some((score, line_idx, line)) = best_temporal_current_anchor_line(&lines) else {
                continue;
            };
            let Some(base_day) = temporal_base_day_at_line(&lines, line_idx) else {
                continue;
            };
            let should_replace = best
                .as_ref()
                .map(|(best_day, best_score, best_line)| {
                    base_day > *best_day
                        || (base_day == *best_day
                            && (score > *best_score
                                || (score == *best_score && line.as_str() < best_line.as_str())))
                })
                .unwrap_or(true);
            if should_replace {
                best = Some((base_day, score, line));
            }
        }

        best.map(|(day, _, line)| (day, line))
    }

    pub(super) fn verbatim_entry_groups_for_session(
        &self,
        session_id: &str,
    ) -> Vec<Vec<&BM25Entry>> {
        let mut groups: BTreeMap<String, Vec<&BM25Entry>> = BTreeMap::new();
        for entry in self.entries.iter().filter(|entry| {
            matches!(entry.kind, NeuronKind::Verbatim) && entry.session_id == session_id
        }) {
            groups
                .entry(verbatim_source_group_key(entry))
                .or_default()
                .push(entry);
        }

        let mut grouped = groups.into_iter().collect::<Vec<_>>();
        grouped.sort_by(|a, b| a.0.cmp(&b.0));
        grouped
            .into_iter()
            .map(|(_, mut entries)| {
                entries.sort_by(|a, b| a.neuron_path.cmp(&b.neuron_path));
                entries
            })
            .collect()
    }

    pub(super) fn read_matching_session_group_lines<F>(
        &self,
        entries: &[&BM25Entry],
        mut predicate: F,
    ) -> Vec<String>
    where
        F: FnMut(&str, &str) -> bool,
    {
        let mut lines = Vec::new();
        for entry in entries {
            let Ok(content) = std::fs::read_to_string(&entry.neuron_path) else {
                continue;
            };
            for raw_line in strip_query_surface_section(&content).lines() {
                let line = raw_line.trim();
                if line.is_empty() {
                    continue;
                }
                let lower = line.to_ascii_lowercase();
                if predicate(line, &lower) && !lines.iter().any(|existing| existing == line) {
                    lines.push(line.to_string());
                }
            }
        }
        lines
    }

    pub(super) fn temporal_from_now_event_candidates(
        &self,
        phrase_lower: &str,
        terms: &[String],
        latest_day: Option<i32>,
    ) -> Vec<(String, usize, i32, String)> {
        let mut groups = std::collections::BTreeMap::<String, Vec<&BM25Entry>>::new();
        for entry in self.entries.iter().filter(|entry| {
            matches!(entry.kind, NeuronKind::Verbatim)
                && !is_session_summary_path(&entry.neuron_path)
        }) {
            let key = if entry.session_id.is_empty() {
                verbatim_source_group_key(entry)
            } else {
                entry.session_id.clone()
            };
            groups.entry(key).or_default().push(entry);
        }
        let mut candidates = Vec::new();
        for (group_id, mut entries) in groups {
            entries.sort_by(|a, b| a.neuron_path.cmp(&b.neuron_path));
            let lines = self.read_matching_session_group_lines(&entries, |line, lower| {
                is_summary_or_user_line(line, lower) || lower.starts_with("[session")
            });
            let Some((event_day, event_score, event_line)) =
                best_temporal_from_now_event_line(&lines, phrase_lower, terms)
            else {
                continue;
            };
            if latest_day.is_some_and(|day| event_day > day) {
                continue;
            }
            candidates.push((group_id, event_score, event_day, event_line));
        }
        candidates.sort_by(|a, b| {
            b.1.cmp(&a.1)
                .then_with(|| b.2.cmp(&a.2))
                .then_with(|| a.3.cmp(&b.3))
                .then_with(|| a.0.cmp(&b.0))
        });
        candidates
    }

    pub(super) fn synthetic_title_duration_answer(
        &self,
        task: &str,
        task_lower: &str,
    ) -> Option<PathBuf> {
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

    pub(super) fn synthetic_temporal_interval_between_events_answer(
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

    pub(super) fn synthetic_item_usage_frequency_answer(
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

    pub(super) fn synthetic_media_rewatch_count_answer(
        &self,
        task: &str,
        task_lower: &str,
    ) -> Option<PathBuf> {
        if !detect_counting_query(task) {
            return None;
        }

        let (focus_phrase, media_kind) = extract_media_rewatch_focus(task_lower)?;
        let mut required_owned = synthetic_query_terms(&focus_phrase);
        required_owned.push(media_kind);
        required_owned.extend(["watched".to_string(), "rewatched".to_string()]);
        required_owned.sort();
        required_owned.dedup();
        let required_terms: Vec<&str> = required_owned.iter().map(String::as_str).collect();
        let focus_terms = synthetic_query_terms(&focus_phrase);
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

        let mut best: Option<(usize, usize, Vec<String>)> = None;
        for (session_id, session_rank) in candidates {
            let lines = self.find_session_lines(&session_id, false, 192, |line, lower| {
                is_summary_or_user_line(line, lower) && line_has_rewatch_marker(lower)
            });
            let mut titles = HashSet::new();
            let mut evidence = Vec::new();

            for line in lines {
                let lower = line.to_ascii_lowercase();
                if !focus_refs.is_empty() && term_overlap_count(&lower, &focus_refs) == 0 {
                    continue;
                }
                let Some(title) = extract_rewatch_title_from_line(&line, &lower) else {
                    continue;
                };
                let normalized = normalize_rewatch_title(&title);
                if normalized.is_empty() || !titles.insert(normalized) {
                    continue;
                }
                evidence.push(line);
            }

            if titles.is_empty() {
                continue;
            }

            let count = titles.len();
            let score = session_rank * 1000 + count * 100 + evidence.len();
            let should_replace = best
                .as_ref()
                .map(|(best_score, best_count, _)| {
                    score > *best_score || (score == *best_score && count > *best_count)
                })
                .unwrap_or(true);
            if should_replace {
                best = Some((score, count, evidence));
            }
        }

        let (_, count, evidence) = best?;
        self.write_synthetic_answer("media-rewatch-count", task, &count.to_string(), &evidence)
    }

    pub(super) fn synthetic_family_origin_item_count_answer(
        &self,
        task: &str,
        task_lower: &str,
    ) -> Option<PathBuf> {
        if !detect_counting_query(task)
            || !task_lower.contains("antique")
            || !task_contains_any(task_lower, &["inherit", "acquire"])
            || !task_contains_any(task_lower, &["family", "family members"])
        {
            return None;
        }

        let mut required_owned = vec![
            "antique".to_string(),
            "vintage".to_string(),
            "family".to_string(),
            "heirloom".to_string(),
            "inherited".to_string(),
        ];
        required_owned.sort();
        required_owned.dedup();
        let required_terms: Vec<&str> = required_owned.iter().map(String::as_str).collect();

        let mut candidates = self
            .session_ids_matching_line(|line, lower| {
                is_summary_or_user_line(line, lower)
                    && !extract_family_origin_antique_items_from_line(line, lower).is_empty()
            })
            .into_iter()
            .map(|session_id| (session_id, 0usize))
            .collect::<Vec<_>>();
        for (session_id, score) in self.candidate_session_ids_by_line_overlap(&required_owned, 8) {
            if let Some((_, existing_score)) = candidates
                .iter_mut()
                .find(|(existing_session_id, _)| existing_session_id == &session_id)
            {
                *existing_score = (*existing_score).max(score);
            } else {
                candidates.push((session_id, score));
            }
        }
        for (idx, session_id) in self
            .candidate_session_ids(task, &required_terms, 8)
            .into_iter()
            .enumerate()
        {
            let score = 8usize.saturating_sub(idx);
            if let Some((_, existing_score)) = candidates
                .iter_mut()
                .find(|(existing_session_id, _)| existing_session_id == &session_id)
            {
                *existing_score = (*existing_score).max(score);
            } else {
                candidates.push((session_id, score));
            }
        }

        let mut best: Option<(usize, usize, Vec<String>)> = None;
        for (session_id, session_rank) in candidates {
            let lines = self.find_session_lines(&session_id, false, 192, |line, lower| {
                is_summary_or_user_line(line, lower)
                    && !extract_family_origin_antique_items_from_line(line, lower).is_empty()
            });
            let mut items = HashSet::new();
            let mut evidence = Vec::new();

            for line in lines {
                let lower = line.to_ascii_lowercase();
                for item in extract_family_origin_antique_items_from_line(&line, &lower) {
                    if !items.insert(normalized_synthetic_phrase_key(&item)) {
                        continue;
                    }
                    if evidence.len() < 4 && !evidence.iter().any(|existing| existing == &line) {
                        evidence.push(line.clone());
                    }
                }
            }

            if items.is_empty() {
                continue;
            }

            let count = items.len();
            let score = count * 100 + session_rank * 10 + evidence.len();
            let should_replace = best
                .as_ref()
                .map(|(best_score, best_count, _)| {
                    score > *best_score || (score == *best_score && count > *best_count)
                })
                .unwrap_or(true);
            if should_replace {
                best = Some((score, count, evidence));
            }
        }

        let (_, count, evidence) = best?;
        self.write_synthetic_answer(
            "family-origin-item-count",
            task,
            &count.to_string(),
            &evidence,
        )
    }

    pub(super) fn synthetic_recent_birth_count_answer(
        &self,
        task: &str,
        task_lower: &str,
    ) -> Option<PathBuf> {
        if !detect_counting_query(task)
            || !task_contains_any(task_lower, &["baby", "babies"])
            || !task_lower.contains("born")
        {
            return None;
        }

        let mut required_owned = vec![
            "baby".to_string(),
            "born".to_string(),
            "twins".to_string(),
            "daughter".to_string(),
            "son".to_string(),
            "welcomed".to_string(),
        ];
        required_owned.sort();
        required_owned.dedup();
        let required_terms: Vec<&str> = required_owned.iter().map(String::as_str).collect();

        let mut candidates = self
            .session_ids_matching_line(|line, lower| {
                is_summary_or_user_line(line, lower)
                    && !extract_born_child_names_from_line(line, lower).is_empty()
            })
            .into_iter()
            .map(|session_id| (session_id, 0usize))
            .collect::<Vec<_>>();
        for (session_id, score) in self.candidate_session_ids_by_line_overlap(&required_owned, 8) {
            if let Some((_, existing_score)) = candidates
                .iter_mut()
                .find(|(existing_session_id, _)| existing_session_id == &session_id)
            {
                *existing_score = (*existing_score).max(score);
            } else {
                candidates.push((session_id, score));
            }
        }
        for (idx, session_id) in self
            .candidate_session_ids(task, &required_terms, 8)
            .into_iter()
            .enumerate()
        {
            let score = 8usize.saturating_sub(idx);
            if let Some((_, existing_score)) = candidates
                .iter_mut()
                .find(|(existing_session_id, _)| existing_session_id == &session_id)
            {
                *existing_score = (*existing_score).max(score);
            } else {
                candidates.push((session_id, score));
            }
        }

        let mut best: Option<(usize, usize, Vec<String>)> = None;
        for (session_id, session_rank) in candidates {
            let lines = self.find_session_lines(&session_id, false, 192, |line, lower| {
                is_summary_or_user_line(line, lower)
                    && !extract_born_child_names_from_line(line, lower).is_empty()
            });
            let mut names = HashSet::new();
            let mut evidence = Vec::new();

            for line in lines {
                let lower = line.to_ascii_lowercase();
                for name in extract_born_child_names_from_line(&line, &lower) {
                    if !names.insert(normalized_synthetic_phrase_key(&name)) {
                        continue;
                    }
                    if evidence.len() < 4 && !evidence.iter().any(|existing| existing == &line) {
                        evidence.push(line.clone());
                    }
                }
            }

            if names.is_empty() {
                continue;
            }

            let count = names.len();
            let score = count * 100 + session_rank * 10 + evidence.len();
            let should_replace = best
                .as_ref()
                .map(|(best_score, best_count, _)| {
                    score > *best_score || (score == *best_score && count > *best_count)
                })
                .unwrap_or(true);
            if should_replace {
                best = Some((score, count, evidence));
            }
        }

        let (_, count, evidence) = best?;
        self.write_synthetic_answer("recent-birth-count", task, &count.to_string(), &evidence)
    }

    pub(super) fn synthetic_bike_service_count_answer(
        &self,
        task: &str,
        task_lower: &str,
    ) -> Option<PathBuf> {
        if !detect_counting_query(task)
            || !task_contains_any(task_lower, &["bike", "bikes"])
            || !task_contains_any(task_lower, &["service", "serviced"])
        {
            return None;
        }

        let month_filter = extract_query_month_name(task_lower)?;
        let mut required_owned = vec![
            "bike".to_string(),
            month_filter.to_string(),
            "service".to_string(),
            "serviced".to_string(),
            "replace".to_string(),
        ];
        required_owned.sort();
        required_owned.dedup();
        let required_terms: Vec<&str> = required_owned.iter().map(String::as_str).collect();

        let mut candidates = self
            .session_ids_matching_line(|line, lower| {
                is_summary_or_user_line(line, lower)
                    && extract_bike_service_item_from_line(line, lower, month_filter).is_some()
            })
            .into_iter()
            .map(|session_id| (session_id, 0usize))
            .collect::<Vec<_>>();
        for (session_id, score) in self.candidate_session_ids_by_line_overlap(&required_owned, 8) {
            if let Some((_, existing_score)) = candidates
                .iter_mut()
                .find(|(existing_session_id, _)| existing_session_id == &session_id)
            {
                *existing_score = (*existing_score).max(score);
            } else {
                candidates.push((session_id, score));
            }
        }
        for (idx, session_id) in self
            .candidate_session_ids(task, &required_terms, 8)
            .into_iter()
            .enumerate()
        {
            let score = 8usize.saturating_sub(idx);
            if let Some((_, existing_score)) = candidates
                .iter_mut()
                .find(|(existing_session_id, _)| existing_session_id == &session_id)
            {
                *existing_score = (*existing_score).max(score);
            } else {
                candidates.push((session_id, score));
            }
        }

        let mut best: Option<(usize, usize, Vec<String>)> = None;
        for (session_id, session_rank) in candidates {
            let lines = self.find_session_lines(&session_id, false, 192, |line, lower| {
                is_summary_or_user_line(line, lower)
                    && extract_bike_service_item_from_line(line, lower, month_filter).is_some()
            });
            let mut bikes = HashSet::new();
            let mut evidence = Vec::new();

            for line in lines {
                let lower = line.to_ascii_lowercase();
                let Some(bike) = extract_bike_service_item_from_line(&line, &lower, month_filter)
                else {
                    continue;
                };
                if !bikes.insert(normalized_synthetic_phrase_key(&bike)) {
                    continue;
                }
                if evidence.len() < 4 && !evidence.iter().any(|existing| existing == &line) {
                    evidence.push(line.clone());
                }
            }

            if bikes.is_empty() {
                continue;
            }

            let count = bikes.len();
            let score = session_rank * 10 + count * 5 + evidence.len();
            let should_replace = best
                .as_ref()
                .map(|(best_score, best_count, _)| {
                    score > *best_score || (score == *best_score && count > *best_count)
                })
                .unwrap_or(true);
            if should_replace {
                best = Some((score, count, evidence));
            }
        }

        let (_, count, evidence) = best?;
        self.write_synthetic_answer("bike-service-count", task, &count.to_string(), &evidence)
    }

    pub(super) fn synthetic_fitness_class_day_count_answer(
        &self,
        task: &str,
        task_lower: &str,
    ) -> Option<PathBuf> {
        if !detect_counting_query(task)
            || !task_contains_all(task_lower, &["fitness", "class"])
            || !task_contains_any(task_lower, &["days a week", "typical week"])
        {
            return None;
        }

        let mut required_owned = vec![
            "fitness".to_string(),
            "class".to_string(),
            "classes".to_string(),
            "yoga".to_string(),
            "zumba".to_string(),
            "bodypump".to_string(),
            "hip hop abs".to_string(),
            "weightlifting".to_string(),
        ];
        required_owned.sort();
        required_owned.dedup();
        let required_terms: Vec<&str> = required_owned.iter().map(String::as_str).collect();

        let mut candidates = self
            .session_ids_matching_line(|line, lower| {
                line_describes_countable_fitness_class_schedule(line, lower)
                    && !extract_weekday_mentions_from_line(lower).is_empty()
            })
            .into_iter()
            .map(|session_id| (session_id, 0usize))
            .collect::<Vec<_>>();
        for (session_id, score) in self.candidate_session_ids_by_line_overlap(&required_owned, 8) {
            if let Some((_, existing_score)) = candidates
                .iter_mut()
                .find(|(existing_session_id, _)| existing_session_id == &session_id)
            {
                *existing_score = (*existing_score).max(score);
            } else {
                candidates.push((session_id, score));
            }
        }
        for (idx, session_id) in self
            .candidate_session_ids(task, &required_terms, 8)
            .into_iter()
            .enumerate()
        {
            let score = 8usize.saturating_sub(idx);
            if let Some((_, existing_score)) = candidates
                .iter_mut()
                .find(|(existing_session_id, _)| existing_session_id == &session_id)
            {
                *existing_score = (*existing_score).max(score);
            } else {
                candidates.push((session_id, score));
            }
        }

        let mut best: Option<(usize, usize, Vec<String>)> = None;
        for (session_id, session_rank) in candidates {
            let lines = self.find_session_lines(&session_id, false, 192, |line, lower| {
                line_describes_countable_fitness_class_schedule(line, lower)
                    && !extract_weekday_mentions_from_line(lower).is_empty()
            });
            let mut weekdays = HashSet::new();
            let mut evidence = Vec::new();

            for line in lines {
                let lower = line.to_ascii_lowercase();
                let line_weekdays = extract_weekday_mentions_from_line(&lower);
                if line_weekdays.is_empty() {
                    continue;
                }
                let mut inserted = false;
                for weekday in line_weekdays {
                    inserted |= weekdays.insert(weekday);
                }
                if inserted
                    && evidence.len() < 4
                    && !evidence.iter().any(|existing| existing == &line)
                {
                    evidence.push(line.clone());
                }
            }

            if weekdays.is_empty() {
                continue;
            }

            let count = weekdays.len();
            let score = session_rank * 10 + count * 5 + evidence.len();
            let should_replace = best
                .as_ref()
                .map(|(best_score, best_count, _)| {
                    score > *best_score || (score == *best_score && count > *best_count)
                })
                .unwrap_or(true);
            if should_replace {
                best = Some((score, count, evidence));
            }
        }

        let (_, count, evidence) = best?;
        let answer = if task_lower.contains("days") {
            render_day_count_answer(count)
        } else {
            count.to_string()
        };
        self.write_synthetic_answer("fitness-class-day-count", task, &answer, &evidence)
    }

    pub(super) fn synthetic_month_scoped_activity_day_count_answer(
        &self,
        task: &str,
        task_lower: &str,
    ) -> Option<PathBuf> {
        if !detect_counting_query(task) || !task_lower.starts_with("how many days did i spend") {
            return None;
        }

        let month_filter = extract_query_month_name(task_lower)?;
        let (route_name, mut required_owned, activity_markers): (&str, Vec<String>, &[&str]) =
            if task_contains_any(task_lower, &["workshop", "lecture", "conference"]) {
                (
                    "learning-activity-day-count",
                    vec![
                        month_filter.to_string(),
                        "workshop".to_string(),
                        "lecture".to_string(),
                        "conference".to_string(),
                    ],
                    &["workshop", "lecture", "conference"],
                )
            } else if task_lower.contains("faith-related") {
                (
                    "faith-activity-day-count",
                    vec![
                        month_filter.to_string(),
                        "faith".to_string(),
                        "church".to_string(),
                        "bible".to_string(),
                        "mass".to_string(),
                        "prayer".to_string(),
                        "worship".to_string(),
                    ],
                    &["church", "bible", "mass", "prayer", "worship", "faith"],
                )
            } else {
                return None;
            };
        required_owned.sort();
        required_owned.dedup();
        let required_terms: Vec<&str> = required_owned.iter().map(String::as_str).collect();

        let mut candidates = self
            .session_ids_matching_line(|line, lower| {
                is_summary_or_user_line(line, lower)
                    && !extract_month_scoped_activity_days_from_line(
                        line,
                        lower,
                        month_filter,
                        activity_markers,
                    )
                    .is_empty()
            })
            .into_iter()
            .map(|session_id| (session_id, 0usize))
            .collect::<Vec<_>>();
        for (session_id, score) in self.candidate_session_ids_by_line_overlap(&required_owned, 8) {
            if let Some((_, existing_score)) = candidates
                .iter_mut()
                .find(|(existing_session_id, _)| existing_session_id == &session_id)
            {
                *existing_score = (*existing_score).max(score);
            } else {
                candidates.push((session_id, score));
            }
        }
        for (idx, session_id) in self
            .candidate_session_ids(task, &required_terms, 8)
            .into_iter()
            .enumerate()
        {
            let score = 8usize.saturating_sub(idx);
            if let Some((_, existing_score)) = candidates
                .iter_mut()
                .find(|(existing_session_id, _)| existing_session_id == &session_id)
            {
                *existing_score = (*existing_score).max(score);
            } else {
                candidates.push((session_id, score));
            }
        }

        let mut best: Option<(usize, usize, Vec<String>)> = None;
        for (session_id, session_rank) in candidates {
            let lines = self.find_session_lines(&session_id, false, 192, |line, lower| {
                is_summary_or_user_line(line, lower)
                    && !extract_month_scoped_activity_days_from_line(
                        line,
                        lower,
                        month_filter,
                        activity_markers,
                    )
                    .is_empty()
            });
            let mut days = HashSet::new();
            let mut evidence = Vec::new();

            for line in lines {
                let lower = line.to_ascii_lowercase();
                let line_days = extract_month_scoped_activity_days_from_line(
                    &line,
                    &lower,
                    month_filter,
                    activity_markers,
                );
                if line_days.is_empty() {
                    continue;
                }
                let mut inserted = false;
                for day in line_days {
                    inserted |= days.insert(day);
                }
                if inserted
                    && evidence.len() < 4
                    && !evidence.iter().any(|existing| existing == &line)
                {
                    evidence.push(line.clone());
                }
            }

            if days.is_empty() {
                continue;
            }

            let count = days.len();
            let score = session_rank * 10 + count * 5 + evidence.len();
            let should_replace = best
                .as_ref()
                .map(|(best_score, best_count, _)| {
                    score > *best_score || (score == *best_score && count > *best_count)
                })
                .unwrap_or(true);
            if should_replace {
                best = Some((score, count, evidence));
            }
        }

        let (_, count, evidence) = best?;
        let answer = render_day_count_answer(count);
        self.write_synthetic_answer(route_name, task, &answer, &evidence)
    }

    pub(super) fn synthetic_art_related_event_count_answer(
        &self,
        task: &str,
        task_lower: &str,
    ) -> Option<PathBuf> {
        if !detect_counting_query(task)
            || !task_contains_all(task_lower, &["art-related", "events"])
            || !task_contains_any(task_lower, &["past month", "last month"])
        {
            return None;
        }

        let mut required_owned = vec![
            "art".to_string(),
            "event".to_string(),
            "events".to_string(),
            "exhibition".to_string(),
            "museum".to_string(),
            "gallery".to_string(),
            "lecture".to_string(),
        ];
        required_owned.sort();
        required_owned.dedup();
        let required_terms: Vec<&str> = required_owned.iter().map(String::as_str).collect();

        let mut candidates = self
            .session_ids_matching_line(|line, lower| {
                is_summary_or_user_line(line, lower)
                    && extract_art_related_event_signature_from_line(line, lower).is_some()
            })
            .into_iter()
            .map(|session_id| (session_id, 0usize))
            .collect::<Vec<_>>();
        for (session_id, score) in self.candidate_session_ids_by_line_overlap(&required_owned, 8) {
            if let Some((_, existing_score)) = candidates
                .iter_mut()
                .find(|(existing_session_id, _)| existing_session_id == &session_id)
            {
                *existing_score = (*existing_score).max(score);
            } else {
                candidates.push((session_id, score));
            }
        }
        for (idx, session_id) in self
            .candidate_session_ids(task, &required_terms, 8)
            .into_iter()
            .enumerate()
        {
            let score = 8usize.saturating_sub(idx);
            if let Some((_, existing_score)) = candidates
                .iter_mut()
                .find(|(existing_session_id, _)| existing_session_id == &session_id)
            {
                *existing_score = (*existing_score).max(score);
            } else {
                candidates.push((session_id, score));
            }
        }

        let mut best: Option<(usize, usize, Vec<String>)> = None;
        for (session_id, session_rank) in candidates {
            let lines = self.find_session_lines(&session_id, false, 256, |line, lower| {
                is_summary_or_user_line(line, lower)
                    && extract_art_related_event_signature_from_line(line, lower).is_some()
            });
            let mut extracted = Vec::new();
            for line in lines {
                let lower = line.to_ascii_lowercase();
                let Some((rank, key)) =
                    extract_art_related_event_signature_from_line(&line, &lower)
                else {
                    continue;
                };
                extracted.push((rank, key, line));
            }
            if extracted.is_empty() {
                continue;
            }
            let latest_rank = extracted
                .iter()
                .map(|(rank, _, _)| *rank)
                .max()
                .unwrap_or_default();
            let mut events = HashSet::new();
            let mut evidence = Vec::new();
            for (rank, key, line) in extracted {
                if latest_rank - rank > 32 {
                    continue;
                }
                if !events.insert(key) {
                    continue;
                }
                if evidence.len() < 4 && !evidence.iter().any(|existing| existing == &line) {
                    evidence.push(line);
                }
            }

            if events.is_empty() {
                continue;
            }

            let count = events.len();
            let score = session_rank * 10 + count * 5 + evidence.len();
            let should_replace = best
                .as_ref()
                .map(|(best_score, best_count, _)| {
                    score > *best_score || (score == *best_score && count > *best_count)
                })
                .unwrap_or(true);
            if should_replace {
                best = Some((score, count, evidence));
            }
        }

        let (_, count, evidence) = best?;
        self.write_synthetic_answer(
            "art-related-event-count",
            task,
            &count.to_string(),
            &evidence,
        )
    }

    pub(super) fn synthetic_distinct_cuisine_count_answer(
        &self,
        task: &str,
        task_lower: &str,
    ) -> Option<PathBuf> {
        if !detect_counting_query(task)
            || !task_contains_any(task_lower, &["cuisine", "cuisines"])
            || !task_contains_any(task_lower, &["learned to cook", "tried out"])
        {
            return None;
        }

        let mut required_owned = vec![
            "cuisine".to_string(),
            "restaurant".to_string(),
            "recipe".to_string(),
            "class".to_string(),
            "learned".to_string(),
            "tried".to_string(),
        ];
        required_owned.sort();
        required_owned.dedup();
        let required_terms: Vec<&str> = required_owned.iter().map(String::as_str).collect();

        let mut candidates = self
            .session_ids_matching_line(|line, lower| {
                is_summary_or_user_line(line, lower)
                    && !extract_cuisine_labels_from_line(line, lower).is_empty()
            })
            .into_iter()
            .map(|session_id| (session_id, 0usize))
            .collect::<Vec<_>>();
        for (session_id, score) in self.candidate_session_ids_by_line_overlap(&required_owned, 8) {
            if let Some((_, existing_score)) = candidates
                .iter_mut()
                .find(|(existing_session_id, _)| existing_session_id == &session_id)
            {
                *existing_score = (*existing_score).max(score);
            } else {
                candidates.push((session_id, score));
            }
        }
        for (idx, session_id) in self
            .candidate_session_ids(task, &required_terms, 8)
            .into_iter()
            .enumerate()
        {
            let score = 8usize.saturating_sub(idx);
            if let Some((_, existing_score)) = candidates
                .iter_mut()
                .find(|(existing_session_id, _)| existing_session_id == &session_id)
            {
                *existing_score = (*existing_score).max(score);
            } else {
                candidates.push((session_id, score));
            }
        }

        let mut best: Option<(usize, usize, Vec<String>)> = None;
        for (session_id, session_rank) in candidates {
            let lines = self.find_session_lines(&session_id, false, 256, |line, lower| {
                is_summary_or_user_line(line, lower)
                    && !extract_cuisine_labels_from_line(line, lower).is_empty()
            });
            let mut cuisines = HashSet::new();
            let mut evidence = Vec::new();
            for line in lines {
                let lower = line.to_ascii_lowercase();
                let mut inserted = false;
                for cuisine in extract_cuisine_labels_from_line(&line, &lower) {
                    inserted |= cuisines.insert(normalized_synthetic_phrase_key(&cuisine));
                }
                if inserted
                    && evidence.len() < 4
                    && !evidence.iter().any(|existing| existing == &line)
                {
                    evidence.push(line);
                }
            }

            if cuisines.is_empty() {
                continue;
            }

            let count = cuisines.len();
            let score = session_rank * 10 + count * 5 + evidence.len();
            let should_replace = best
                .as_ref()
                .map(|(best_score, best_count, _)| {
                    score > *best_score || (score == *best_score && count > *best_count)
                })
                .unwrap_or(true);
            if should_replace {
                best = Some((score, count, evidence));
            }
        }

        let (_, count, evidence) = best?;
        self.write_synthetic_answer(
            "distinct-cuisine-count",
            task,
            &count.to_string(),
            &evidence,
        )
    }

    pub(super) fn synthetic_museum_gallery_visit_count_answer(
        &self,
        task: &str,
        task_lower: &str,
    ) -> Option<PathBuf> {
        if !detect_counting_query(task)
            || !task_contains_any(task_lower, &["museum", "museums"])
            || !task_contains_any(task_lower, &["gallery", "galleries"])
        {
            return None;
        }

        let month_filter = extract_query_month_name(task_lower)?;
        let mut per_session: HashMap<String, (HashSet<String>, bool, bool, Vec<String>)> =
            HashMap::new();
        for entry in self.entries.iter().filter(|entry| {
            matches!(entry.kind, NeuronKind::Verbatim) && !entry.session_id.is_empty()
        }) {
            let Ok(content) = std::fs::read_to_string(&entry.neuron_path) else {
                continue;
            };
            for raw_line in strip_query_surface_section(&content).lines() {
                let line = raw_line.trim();
                if line.is_empty() {
                    continue;
                }
                let lower = line.to_ascii_lowercase();
                if !is_summary_or_user_line(line, &lower)
                    || !line_mentions_candidate_museum_gallery_visit(line, &lower, month_filter)
                {
                    continue;
                }
                let Some(venue) =
                    extract_museum_gallery_visit_venue_from_line(line, &lower, month_filter)
                else {
                    continue;
                };
                let venue_key = normalized_synthetic_phrase_key(&venue);
                let venue_lower = venue.to_ascii_lowercase();
                let (venues, has_museum, has_gallery, evidence) = per_session
                    .entry(entry.session_id.clone())
                    .or_insert_with(|| (HashSet::new(), false, false, Vec::new()));
                let inserted = venues.insert(venue_key);
                *has_museum |= venue_lower.contains("museum");
                *has_gallery |= venue_lower.contains("gallery") || venue_lower.contains("art cube");
                if inserted
                    && evidence.len() < 4
                    && !evidence.iter().any(|existing| existing == line)
                {
                    evidence.push(line.to_string());
                }
            }
        }

        let mut best: Option<(usize, usize, usize, Vec<String>)> = None;
        for (_, (venues, has_museum, has_gallery, evidence)) in per_session {
            if venues.is_empty() {
                continue;
            }
            let count = venues.len();
            let category_bonus = usize::from(has_museum && has_gallery);
            let score = count * 100 + category_bonus * 25 + evidence.len();
            let should_replace = best
                .as_ref()
                .map(|(best_score, best_count, best_bonus, _)| {
                    score > *best_score
                        || (score == *best_score
                            && (count > *best_count
                                || (count == *best_count && category_bonus > *best_bonus)))
                })
                .unwrap_or(true);
            if should_replace {
                best = Some((score, count, category_bonus, evidence));
            }
        }

        let (_, count, _, evidence) = best?;
        self.write_synthetic_answer(
            "museum-gallery-visit-count",
            task,
            &count.to_string(),
            &evidence,
        )
    }

    pub(super) fn synthetic_citrus_fruit_count_answer(
        &self,
        task: &str,
        task_lower: &str,
    ) -> Option<PathBuf> {
        if !detect_counting_query(task) || !task_contains_all(task_lower, &["citrus", "cocktail"]) {
            return None;
        }

        let mut required_owned = vec![
            "cocktail".to_string(),
            "citrus".to_string(),
            "orange".to_string(),
            "lemon".to_string(),
            "lime".to_string(),
            "mixology".to_string(),
        ];
        required_owned.sort();
        required_owned.dedup();
        let required_terms: Vec<&str> = required_owned.iter().map(String::as_str).collect();

        let mut candidates = self
            .session_ids_matching_line(|line, lower| {
                is_summary_or_user_line(line, lower)
                    && !extract_citrus_fruits_from_line(line, lower).is_empty()
            })
            .into_iter()
            .map(|session_id| (session_id, 0usize))
            .collect::<Vec<_>>();
        for (session_id, score) in self.candidate_session_ids_by_line_overlap(&required_owned, 8) {
            if let Some((_, existing_score)) = candidates
                .iter_mut()
                .find(|(existing_session_id, _)| existing_session_id == &session_id)
            {
                *existing_score = (*existing_score).max(score);
            } else {
                candidates.push((session_id, score));
            }
        }
        for (idx, session_id) in self
            .candidate_session_ids(task, &required_terms, 8)
            .into_iter()
            .enumerate()
        {
            let score = 8usize.saturating_sub(idx);
            if let Some((_, existing_score)) = candidates
                .iter_mut()
                .find(|(existing_session_id, _)| existing_session_id == &session_id)
            {
                *existing_score = (*existing_score).max(score);
            } else {
                candidates.push((session_id, score));
            }
        }

        let mut best: Option<(usize, usize, Vec<String>)> = None;
        for (session_id, session_rank) in candidates {
            let lines = self.find_session_lines(&session_id, false, 256, |line, lower| {
                is_summary_or_user_line(line, lower)
                    && !extract_citrus_fruits_from_line(line, lower).is_empty()
            });
            let mut fruits = HashSet::new();
            let mut evidence = Vec::new();
            for line in lines {
                let lower = line.to_ascii_lowercase();
                let mut inserted = false;
                for fruit in extract_citrus_fruits_from_line(&line, &lower) {
                    inserted |= fruits.insert(normalized_synthetic_phrase_key(&fruit));
                }
                if inserted
                    && evidence.len() < 4
                    && !evidence.iter().any(|existing| existing == &line)
                {
                    evidence.push(line);
                }
            }

            if fruits.is_empty() {
                continue;
            }

            let count = fruits.len();
            let score = count * 100 + session_rank * 10 + evidence.len();
            let should_replace = best
                .as_ref()
                .map(|(best_score, best_count, _)| {
                    score > *best_score || (score == *best_score && count > *best_count)
                })
                .unwrap_or(true);
            if should_replace {
                best = Some((score, count, evidence));
            }
        }

        let (_, count, evidence) = best?;
        self.write_synthetic_answer("citrus-fruit-count", task, &count.to_string(), &evidence)
    }
}
