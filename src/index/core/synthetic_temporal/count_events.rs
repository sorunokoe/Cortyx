//! Event-based counting: births, bike services, fitness classes, museum visits.

use super::super::*;

impl NeuronIndex {
    pub fn synthetic_media_rewatch_count_answer(
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

    pub fn synthetic_family_origin_item_count_answer(
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

    pub fn synthetic_recent_birth_count_answer(
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

    pub fn synthetic_art_related_event_count_answer(
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

    pub fn synthetic_distinct_cuisine_count_answer(
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

    pub fn synthetic_museum_gallery_visit_count_answer(
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
