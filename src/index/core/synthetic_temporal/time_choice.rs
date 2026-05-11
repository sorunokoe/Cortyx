//! Temporal choice, elapsed duration, and from-now estimation.

use super::super::*;

impl NeuronIndex {
    pub fn synthetic_temporal_choice_answer(
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

    pub fn synthetic_temporal_elapsed_duration_answer(
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

    pub fn synthetic_temporal_from_now_answer(
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

    pub fn best_temporal_current_anchor_session(&self, session_id: &str) -> Option<(i32, String)> {
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

    pub fn verbatim_entry_groups_for_session(&self, session_id: &str) -> Vec<Vec<&BM25Entry>> {
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

    pub fn read_matching_session_group_lines<F>(
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

    pub fn temporal_from_now_event_candidates(
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
}
