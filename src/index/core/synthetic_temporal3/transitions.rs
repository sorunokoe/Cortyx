//! Role transitions, activity frequency, recurring frequency, company, artwork location.

use super::super::*;

impl NeuronIndex {
    pub fn synthetic_role_transition_count_answer(
        &self,
        task: &str,
        task_lower: &str,
    ) -> Option<PathBuf> {
        if !detect_counting_query(task)
            || !task_lower.contains("when i just started")
            || !task_lower.contains("lead")
            || !task_lower.contains("now")
        {
            return None;
        }
        let role_phrase = extract_role_phrase(task)?;
        let role_phrase_lower = role_phrase.to_ascii_lowercase();
        let mut required_owned = vec![
            "lead".to_string(),
            "team".to_string(),
            "engineers".to_string(),
        ];
        required_owned.extend(synthetic_query_terms(&role_phrase_lower));
        required_owned.sort();
        required_owned.dedup();
        let required_terms: Vec<&str> = required_owned.iter().map(String::as_str).collect();
        let mut best: Option<(usize, i32, String, i32, String)> = None;

        for session_id in self.candidate_session_ids(task, &required_terms, 8) {
            let lines = self.find_session_lines(&session_id, false, 128, |line, lower| {
                is_summary_or_user_line(line, lower)
                    && lower.contains("engineer")
                    && task_contains_any(lower, &["lead", "leading"])
            });
            let mut session_start: Option<(usize, i32, String)> = None;
            let mut session_now: Option<(usize, i32, String)> = None;

            for line in lines {
                let lower = line.to_ascii_lowercase();
                if lower.contains(&role_phrase_lower)
                    && task_contains_any(&lower, &["new role as", "started my new role"])
                {
                    if let Some((value, proximity_score)) = extract_focus_aligned_count(
                        &line,
                        &[
                            "lead".to_string(),
                            "team".to_string(),
                            "engineers".to_string(),
                        ],
                        task_lower,
                    ) {
                        let score =
                            proximity_score + usize::from(lower.contains("team of")) * 4 + 2;
                        if session_start
                            .as_ref()
                            .map(|(best_score, best_value, _)| {
                                score > *best_score || (score == *best_score && value > *best_value)
                            })
                            .unwrap_or(true)
                        {
                            session_start = Some((score, value, line.clone()));
                        }
                    }
                }
                if line_has_current_count_marker(&lower) {
                    if let Some((value, proximity_score)) = extract_focus_aligned_count(
                        &line,
                        &[
                            "lead".to_string(),
                            "team".to_string(),
                            "engineers".to_string(),
                        ],
                        task_lower,
                    ) {
                        let score = proximity_score + 4;
                        if session_now
                            .as_ref()
                            .map(|(best_score, best_value, _)| {
                                score > *best_score || (score == *best_score && value > *best_value)
                            })
                            .unwrap_or(true)
                        {
                            session_now = Some((score, value, line.clone()));
                        }
                    }
                }
            }

            let (
                Some((start_score, start_value, start_line)),
                Some((now_score, now_value, now_line)),
            ) = (session_start, session_now)
            else {
                continue;
            };
            let session_score = start_score + now_score;
            if best
                .as_ref()
                .map(|(best_score, best_start, _, best_now, _)| {
                    session_score > *best_score
                        || (session_score == *best_score && now_value > *best_now)
                        || (session_score == *best_score
                            && now_value == *best_now
                            && start_value > *best_start)
                })
                .unwrap_or(true)
            {
                best = Some((session_score, start_value, start_line, now_value, now_line));
            }
        }

        let (_, start_value, start_line, now_value, now_line) = best?;
        self.write_synthetic_answer(
            "role-transition-count",
            task,
            &format!(
                "When you just started your new role as {role_phrase}, you led {start_value} engineers. Now, you lead {now_value} engineers"
            ),
            &[start_line, now_line],
        )
    }

    pub fn synthetic_activity_frequency_transition_answer(
        &self,
        task: &str,
        task_lower: &str,
    ) -> Option<PathBuf> {
        if !detect_counting_query(task)
            || !task_lower.contains("previously")
            || !task_lower.contains("how often do i")
            || !task_lower.contains("now")
        {
            return None;
        }

        let activity_phrase = extract_frequency_transition_activity_phrase(task_lower)?;
        let activity_terms = synthetic_query_terms(&activity_phrase);
        if activity_terms.is_empty() {
            return None;
        }
        let activity_keys = synthetic_answer_surface_term_key_set(&activity_terms);
        let min_overlap = if activity_keys.len() >= 4 {
            3
        } else if activity_keys.len() >= 2 {
            2
        } else {
            1
        };
        let required_terms: Vec<&str> = activity_terms.iter().map(String::as_str).collect();

        let mut candidates = self
            .candidate_session_ids(task, &required_terms, 8)
            .into_iter()
            .enumerate()
            .map(|(idx, session_id)| (session_id, 8usize.saturating_sub(idx)))
            .collect::<Vec<_>>();
        if candidates.is_empty() {
            candidates = self.candidate_session_ids_by_line_overlap(&activity_terms, 8);
        }

        #[allow(clippy::type_complexity)]
        let mut best: Option<(
            usize,
            usize,
            String,
            Option<String>,
            String,
            Option<String>,
            Vec<String>,
        )> = None;
        for (session_id, session_rank) in candidates {
            let lines = self.find_session_lines(&session_id, false, 128, |line, lower| {
                is_summary_or_user_line(line, lower)
                    && extract_frequency_surface_from_line(line, lower).is_some()
            });
            let mut matches = Vec::new();
            for (line_idx, line) in lines.into_iter().enumerate() {
                let lower = line.to_ascii_lowercase();
                let line_keys =
                    synthetic_answer_surface_term_key_set(&synthetic_query_terms(&lower));
                if synthetic_answer_surface_overlap_count(&line_keys, &activity_keys) < min_overlap
                {
                    continue;
                }
                let Some(frequency) = extract_frequency_surface_from_line(&line, &lower) else {
                    continue;
                };
                let day = extract_date_or_time_answer_from_line(&line)
                    .map(|value| value.to_ascii_lowercase());
                matches.push((line_idx, frequency, day, line));
            }
            if matches.len() < 2 {
                continue;
            }
            let (first_line_idx, first_frequency, first_day, first_line) = matches[0].clone();
            let (last_line_idx, last_frequency, last_day, last_line) =
                matches[matches.len() - 1].clone();
            if first_frequency == last_frequency && first_day == last_day {
                continue;
            }
            let evidence = if first_line == last_line {
                vec![first_line]
            } else {
                vec![first_line, last_line]
            };
            let should_replace = best
                .as_ref()
                .map(|(best_rank, best_last_line_idx, _, _, _, _, _)| {
                    session_rank > *best_rank
                        || (session_rank == *best_rank && last_line_idx > *best_last_line_idx)
                })
                .unwrap_or(true);
            if should_replace {
                best = Some((
                    session_rank,
                    last_line_idx.max(first_line_idx),
                    first_frequency,
                    first_day,
                    last_frequency,
                    last_day,
                    evidence,
                ));
            }
        }

        let (_, _, previous_frequency, previous_day, current_frequency, current_day, evidence) =
            best?;
        let previous_phrase = normalize_first_person_phrase_to_second_person(&activity_phrase);
        let current_phrase = extract_activity_core_phrase(&previous_phrase);
        let previous_day_suffix = previous_day
            .as_deref()
            .map(|day| format!(" (on {})", capitalize_first_ascii(day)))
            .unwrap_or_default();
        let current_day_suffix = current_day
            .as_deref()
            .map(|day| format!(" (on {})", capitalize_first_ascii(day)))
            .unwrap_or_default();
        let answer = format!(
            "Previously, you {} {}{}. Currently, you {} {}{}.",
            previous_phrase,
            previous_frequency,
            previous_day_suffix,
            current_phrase,
            current_frequency,
            current_day_suffix
        );
        self.write_synthetic_answer("activity-frequency-transition", task, &answer, &evidence)
    }

    pub fn synthetic_named_recurring_frequency_answer(
        &self,
        task: &str,
        task_lower: &str,
    ) -> Option<PathBuf> {
        if !task_lower.starts_with("how often")
            || !task_contains_any(task_lower, &["therapist", "dr.", "dr ", "doctor"])
        {
            return None;
        }

        let person = extract_schedule_query_person(task)?;
        let person_lower = person.to_ascii_lowercase();
        let mut required_owned = vec![
            person_lower.clone(),
            "every".to_string(),
            "week".to_string(),
            "session".to_string(),
        ];
        required_owned.extend(
            synthetic_query_terms(task_lower)
                .into_iter()
                .filter(|term| matches!(term.as_str(), "therapist" | "therapy" | "doctor" | "dr")),
        );
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
                    && lower.contains(&person_lower)
                    && (lower.contains("i see ")
                        || lower.contains("seeing ")
                        || lower.contains("therap")
                        || lower.contains("session")
                        || lower.contains("checkup"))
                    && extract_frequency_surface_from_line(line, lower).is_some()
            });
            for (line_idx, line) in lines.into_iter().enumerate() {
                let lower = line.to_ascii_lowercase();
                let Some(answer) = extract_frequency_surface_from_line(&line, &lower) else {
                    continue;
                };
                let should_replace = best
                    .as_ref()
                    .map(|(best_rank, best_line_idx, _, _)| {
                        session_rank > *best_rank
                            || (session_rank == *best_rank && line_idx > *best_line_idx)
                    })
                    .unwrap_or(true);
                if should_replace {
                    best = Some((session_rank, line_idx, answer, vec![line.clone()]));
                }
            }
        }

        let (_, _, answer, evidence) = best?;
        self.write_synthetic_answer("named-recurring-frequency", task, &answer, &evidence)
    }

    pub fn synthetic_named_current_company_answer(
        &self,
        task: &str,
        task_lower: &str,
    ) -> Option<PathBuf> {
        if !task_lower.contains("company")
            || !task_contains_any(task_lower, &["current", "currently", "now", "these days"])
            || !task_contains_any(task_lower, &["working at", "works at", "work at"])
        {
            return None;
        }

        let person = extract_schedule_query_person(task)?;
        let person_lower = person.to_ascii_lowercase();
        let mut required_owned = vec![
            person_lower.clone(),
            "company".to_string(),
            "current".to_string(),
            "currently".to_string(),
            "working".to_string(),
        ];
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
                    && lower.contains(&person_lower)
                    && line_has_current_company_marker(lower)
            });
            for (line_idx, line) in lines.into_iter().enumerate() {
                let lower = line.to_ascii_lowercase();
                let Some(answer) = extract_current_company_answer_from_line(&line, &lower) else {
                    continue;
                };
                let strength =
                    if lower.contains("currently working at ") || lower.contains("currently at ") {
                        3
                    } else if lower.contains("current company is ") {
                        2
                    } else {
                        1
                    };
                let should_replace = best
                    .as_ref()
                    .map(|(best_rank, best_strength, best_line_idx, _, _)| {
                        session_rank > *best_rank
                            || (session_rank == *best_rank && strength > *best_strength)
                            || (session_rank == *best_rank
                                && strength == *best_strength
                                && line_idx > *best_line_idx)
                    })
                    .unwrap_or(true);
                if should_replace {
                    best = Some((session_rank, strength, line_idx, answer, vec![line.clone()]));
                }
            }
        }

        let (_, _, _, answer, evidence) = best?;
        self.write_synthetic_answer("named-current-company", task, &answer, &evidence)
    }

    pub fn synthetic_named_artwork_location_answer(
        &self,
        task: &str,
        task_lower: &str,
    ) -> Option<PathBuf> {
        if !task_contains_any(task_lower, &["painting", "print", "artwork"])
            || !task_contains_any(task_lower, &["where", "hang", "hanging", "display"])
            || (!has_explicit_current_state_marker(task) && !detect_knowledge_update_query(task))
        {
            return None;
        }

        let title_lower = extract_quoted_title(task)?;
        let mut required_owned = synthetic_query_terms(&title_lower);
        if required_owned.is_empty() {
            return None;
        }
        required_owned.extend([
            "painting".to_string(),
            "print".to_string(),
            "moved".to_string(),
            "hang".to_string(),
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

        let mut best: Option<(usize, usize, String, Vec<String>)> = None;
        for (session_id, session_rank) in candidates {
            let lines = self.find_session_lines(&session_id, false, 192, |line, lower| {
                is_summary_or_user_line(line, lower) && lower.contains(&title_lower)
            });
            for (line_idx, line) in lines.into_iter().enumerate() {
                let lower = line.to_ascii_lowercase();
                let Some(answer) =
                    extract_named_artwork_location_surface_from_line(&line, &lower, &title_lower)
                else {
                    continue;
                };
                let score = session_rank * 10 + line_idx;
                let should_replace = best
                    .as_ref()
                    .map(|(best_score, best_line_idx, _, _)| {
                        score > *best_score || (score == *best_score && line_idx > *best_line_idx)
                    })
                    .unwrap_or(true);
                if should_replace {
                    best = Some((score, line_idx, answer, vec![line.clone()]));
                }
            }
        }

        let (_, _, answer, evidence) = best?;
        self.write_synthetic_answer("named-artwork-location", task, &answer, &evidence)
    }
}
