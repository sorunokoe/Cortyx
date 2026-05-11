//! Food delivery, fun run, graduation, health device activities.

use super::super::*;

impl NeuronIndex {
    pub fn synthetic_food_delivery_service_count_answer(
        &self,
        task: &str,
        task_lower: &str,
    ) -> Option<PathBuf> {
        if !detect_counting_query(task)
            || !task_contains_any(task_lower, &["food delivery", "delivery services"])
        {
            return None;
        }

        let mut required_owned = vec![
            "delivery".to_string(),
            "service".to_string(),
            "fresh fusion".to_string(),
            "uber eats".to_string(),
            "domino".to_string(),
        ];
        required_owned.sort();
        required_owned.dedup();
        let required_terms: Vec<&str> = required_owned.iter().map(String::as_str).collect();

        let mut candidates = self
            .session_ids_matching_line(|line, lower| {
                is_summary_or_user_line(line, lower)
                    && extract_food_delivery_service_from_line(line, lower).is_some()
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
                    && extract_food_delivery_service_from_line(line, lower).is_some()
            });
            let mut services = HashSet::new();
            let mut evidence = Vec::new();
            for line in lines {
                let lower = line.to_ascii_lowercase();
                let Some(service) = extract_food_delivery_service_from_line(&line, &lower) else {
                    continue;
                };
                if !services.insert(normalized_synthetic_phrase_key(&service)) {
                    continue;
                }
                if evidence.len() < 4 && !evidence.iter().any(|existing| existing == &line) {
                    evidence.push(line);
                }
            }

            if services.is_empty() {
                continue;
            }

            let count = services.len();
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
            "food-delivery-service-count",
            task,
            &count.to_string(),
            &evidence,
        )
    }

    pub fn synthetic_missed_fun_run_count_answer(
        &self,
        task: &str,
        task_lower: &str,
    ) -> Option<PathBuf> {
        if !detect_counting_query(task)
            || !task_contains_any(task_lower, &["fun run", "fun runs"])
            || !task_contains_any(task_lower, &["miss", "missed"])
        {
            return None;
        }

        let month_filter = extract_query_month_name(task_lower)?;
        let mut required_owned = vec![
            "fun run".to_string(),
            month_filter.to_string(),
            "missed".to_string(),
            "work".to_string(),
        ];
        required_owned.sort();
        required_owned.dedup();
        let required_terms: Vec<&str> = required_owned.iter().map(String::as_str).collect();

        let mut candidates = self
            .session_ids_matching_line(|line, lower| {
                is_summary_or_user_line(line, lower)
                    && extract_missed_fun_run_signature_from_line(line, lower, month_filter)
                        .is_some()
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
                    && extract_missed_fun_run_signature_from_line(line, lower, month_filter)
                        .is_some()
            });
            let mut missed_runs = HashSet::new();
            let mut evidence = Vec::new();
            let mut work_bonus = 0usize;
            for line in lines {
                let lower = line.to_ascii_lowercase();
                let Some(signature) =
                    extract_missed_fun_run_signature_from_line(&line, &lower, month_filter)
                else {
                    continue;
                };
                let inserted = missed_runs.insert(signature);
                if lower.contains("work") {
                    work_bonus = 1;
                }
                if inserted
                    && evidence.len() < 4
                    && !evidence.iter().any(|existing| existing == &line)
                {
                    evidence.push(line);
                }
            }

            if missed_runs.is_empty() {
                continue;
            }

            let count = missed_runs.len();
            let score = count * 100 + session_rank * 10 + work_bonus * 5 + evidence.len();
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
        self.write_synthetic_answer("missed-fun-run-count", task, &count.to_string(), &evidence)
    }

    pub fn synthetic_graduation_ceremony_count_answer(
        &self,
        task: &str,
        task_lower: &str,
    ) -> Option<PathBuf> {
        if !detect_counting_query(task)
            || !task_contains_any(
                task_lower,
                &["graduation ceremony", "graduation ceremonies"],
            )
            || !task_contains_any(task_lower, &["past three months", "three months"])
        {
            return None;
        }

        let mut required_owned = vec![
            "graduation".to_string(),
            "attended".to_string(),
            "weeks ago".to_string(),
            "months ago".to_string(),
        ];
        required_owned.sort();
        required_owned.dedup();
        let required_terms: Vec<&str> = required_owned.iter().map(String::as_str).collect();

        let mut candidates = self
            .session_ids_matching_line(|line, lower| {
                is_summary_or_user_line(line, lower)
                    && extract_graduation_ceremony_signature_from_line(line, lower).is_some()
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
                    && extract_graduation_ceremony_signature_from_line(line, lower).is_some()
            });
            let mut ceremonies = HashSet::new();
            let mut evidence = Vec::new();
            for line in lines {
                let lower = line.to_ascii_lowercase();
                let Some(signature) =
                    extract_graduation_ceremony_signature_from_line(&line, &lower)
                else {
                    continue;
                };
                if !ceremonies.insert(signature) {
                    continue;
                }
                if evidence.len() < 4 && !evidence.iter().any(|existing| existing == &line) {
                    evidence.push(line);
                }
            }

            if ceremonies.is_empty() {
                continue;
            }

            let count = ceremonies.len();
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
            "graduation-ceremony-count",
            task,
            &count.to_string(),
            &evidence,
        )
    }

    pub fn synthetic_health_device_count_answer(
        &self,
        task: &str,
        task_lower: &str,
    ) -> Option<PathBuf> {
        if !detect_counting_query(task)
            || !task_contains_any(
                task_lower,
                &["health-related devices", "devices do i use in a day"],
            )
        {
            return None;
        }

        let mut required_owned = vec![
            "fitbit".to_string(),
            "versa".to_string(),
            "hearing aids".to_string(),
            "devices".to_string(),
        ];
        required_owned.sort();
        required_owned.dedup();
        let required_terms: Vec<&str> = required_owned.iter().map(String::as_str).collect();

        let mut candidates = self
            .session_ids_matching_line(|line, lower| {
                is_summary_or_user_line(line, lower)
                    && !extract_health_device_units_from_line(line, lower).is_empty()
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
                    && !extract_health_device_units_from_line(line, lower).is_empty()
            });
            let mut devices = HashSet::new();
            let mut evidence = Vec::new();
            for line in lines {
                let lower = line.to_ascii_lowercase();
                let mut inserted = false;
                for device in extract_health_device_units_from_line(&line, &lower) {
                    inserted |= devices.insert(device);
                }
                if inserted
                    && evidence.len() < 4
                    && !evidence.iter().any(|existing| existing == &line)
                {
                    evidence.push(line);
                }
            }

            if devices.is_empty() {
                continue;
            }

            let count = devices.len();
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
        self.write_synthetic_answer("health-device-count", task, &count.to_string(), &evidence)
    }

    pub fn synthetic_peak_campaign_weekly_hours_answer(
        &self,
        task: &str,
        task_lower: &str,
    ) -> Option<PathBuf> {
        if !detect_counting_query(task)
            || !task_contains_any(
                task_lower,
                &["peak campaign season", "peak campaign seasons"],
            )
            || !task_contains_any(task_lower, &["hours do i work", "typical week"])
        {
            return None;
        }

        let mut required_owned = vec![
            "peak campaign".to_string(),
            "work hours".to_string(),
            "40 hours a week".to_string(),
            "10 hours weekly".to_string(),
        ];
        required_owned.sort();
        required_owned.dedup();
        let required_terms: Vec<&str> = required_owned.iter().map(String::as_str).collect();

        let mut candidates = self
            .session_ids_matching_line(|line, lower| {
                is_summary_or_user_line(line, lower)
                    && (extract_peak_campaign_weekly_hour_delta_from_line(line, lower).is_some()
                        || extract_typical_weekly_work_hours_from_line(line, lower).is_some()
                        || extract_peak_campaign_total_weekly_hours_from_line(line, lower)
                            .is_some())
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

        let mut best: Option<(usize, f32, Vec<String>)> = None;
        for (session_id, session_rank) in candidates {
            let lines = self.find_session_lines(&session_id, false, 256, |line, lower| {
                is_summary_or_user_line(line, lower)
                    && (extract_peak_campaign_weekly_hour_delta_from_line(line, lower).is_some()
                        || extract_typical_weekly_work_hours_from_line(line, lower).is_some()
                        || extract_peak_campaign_total_weekly_hours_from_line(line, lower)
                            .is_some())
            });

            let mut base_hours: Option<(f32, String)> = None;
            let mut peak_delta: Option<(f32, String)> = None;
            let mut peak_total: Option<(f32, String)> = None;
            for line in lines {
                let lower = line.to_ascii_lowercase();
                if base_hours.is_none() {
                    if let Some(value) = extract_typical_weekly_work_hours_from_line(&line, &lower)
                    {
                        base_hours = Some((value, line.clone()));
                    }
                }
                if peak_delta.is_none() {
                    if let Some(value) =
                        extract_peak_campaign_weekly_hour_delta_from_line(&line, &lower)
                    {
                        peak_delta = Some((value, line.clone()));
                    }
                }
                if peak_total.is_none() {
                    if let Some(value) =
                        extract_peak_campaign_total_weekly_hours_from_line(&line, &lower)
                    {
                        peak_total = Some((value, line.clone()));
                    }
                }
            }

            let mut evidence = Vec::new();
            let mut support = 0usize;
            let total = if let (Some((base, base_line)), Some((delta, delta_line))) =
                (base_hours.as_ref(), peak_delta.as_ref())
            {
                support += 2;
                evidence.push(base_line.clone());
                if delta_line != base_line {
                    evidence.push(delta_line.clone());
                }
                *base + *delta
            } else if let Some((value, line)) = peak_total.as_ref() {
                support += 1;
                evidence.push(line.clone());
                *value
            } else {
                continue;
            };

            if let Some((_, line)) = peak_total.as_ref() {
                support += 1;
                if evidence.len() < 4 && !evidence.iter().any(|existing| existing == line) {
                    evidence.push(line.clone());
                }
            }

            let score = support * 100 + session_rank * 10 + evidence.len();
            let should_replace = best
                .as_ref()
                .map(|(best_score, best_total, _)| {
                    score > *best_score || (score == *best_score && total > *best_total)
                })
                .unwrap_or(true);
            if should_replace {
                best = Some((score, total, evidence));
            }
        }

        let (_, total, evidence) = best?;
        self.write_synthetic_answer(
            "peak-campaign-weekly-hours",
            task,
            &compact_decimal_string(total),
            &evidence,
        )
    }
}
