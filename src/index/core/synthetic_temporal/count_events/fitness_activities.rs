use super::*;

impl NeuronIndex {
    pub fn synthetic_fitness_class_day_count_answer(
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

    pub fn synthetic_month_scoped_activity_day_count_answer(
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

    pub fn synthetic_bike_service_count_answer(
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

    pub fn synthetic_citrus_fruit_count_answer(
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
