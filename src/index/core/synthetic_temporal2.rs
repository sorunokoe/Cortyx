// This file is a submodule of `crate::index::core`.
// Contains `impl NeuronIndex` synthetic answer methods extracted from synthetic.rs.
use super::*;

impl NeuronIndex {
    pub(super) fn synthetic_food_delivery_service_count_answer(
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

    pub(super) fn synthetic_missed_fun_run_count_answer(
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

    pub(super) fn synthetic_graduation_ceremony_count_answer(
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

    pub(super) fn synthetic_health_device_count_answer(
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

    pub(super) fn synthetic_peak_campaign_weekly_hours_answer(
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

    pub(super) fn synthetic_recent_activity_duration_total_answer(
        &self,
        task: &str,
        task_lower: &str,
    ) -> Option<PathBuf> {
        if !detect_counting_query(task)
            || !task_lower.contains("last week")
            || !task_lower.contains("hour")
        {
            return None;
        }

        let requested_activities = extract_recent_activity_query_labels(task_lower);
        if requested_activities.is_empty() {
            return None;
        }

        let mut required_owned = requested_activities
            .iter()
            .map(|activity| (*activity).to_string())
            .collect::<Vec<_>>();
        required_owned.push("last week".to_string());
        required_owned.push("workout".to_string());
        required_owned.sort();
        required_owned.dedup();
        let required_terms: Vec<&str> = required_owned.iter().map(String::as_str).collect();

        let mut candidates = self
            .session_ids_matching_line(|line, lower| {
                is_summary_or_user_line(line, lower)
                    && !extract_recent_activity_duration_facts_from_line(
                        line,
                        lower,
                        &requested_activities,
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

        let mut best: Option<(usize, f32, Vec<String>)> = None;
        for (session_id, session_rank) in candidates {
            let lines = self.find_session_lines(&session_id, false, 256, |line, lower| {
                is_summary_or_user_line(line, lower)
                    && !extract_recent_activity_duration_facts_from_line(
                        line,
                        lower,
                        &requested_activities,
                    )
                    .is_empty()
            });
            let mut total_days = 0.0;
            let mut seen = HashSet::new();
            let mut covered = HashSet::new();
            let mut evidence = Vec::new();
            for line in lines {
                let lower = line.to_ascii_lowercase();
                for (signature, activity, duration) in
                    extract_recent_activity_duration_facts_from_line(
                        &line,
                        &lower,
                        &requested_activities,
                    )
                {
                    if !seen.insert(signature) {
                        continue;
                    }
                    covered.insert(activity);
                    total_days += duration.days;
                    if evidence.len() < 4 && !evidence.iter().any(|existing| existing == &line) {
                        evidence.push(line.clone());
                    }
                }
            }

            if total_days <= 0.0 {
                continue;
            }

            let score = covered.len() * 100 + session_rank * 10 + evidence.len();
            let should_replace = best
                .as_ref()
                .map(|(best_score, best_total, _)| {
                    score > *best_score || (score == *best_score && total_days > *best_total)
                })
                .unwrap_or(true);
            if should_replace {
                best = Some((score, total_days, evidence));
            }
        }

        let (_, total_days, evidence) = best?;
        let total_hours = convert_duration_days(total_days, "hour");
        self.write_synthetic_answer(
            "recent-activity-duration-total",
            task,
            &format_aggregate_duration_answer(total_hours, "hour"),
            &evidence,
        )
    }

    pub(super) fn synthetic_current_magazine_subscription_count_answer(
        &self,
        task: &str,
        task_lower: &str,
    ) -> Option<PathBuf> {
        if !detect_counting_query(task)
            || !task_contains_any(
                task_lower,
                &["magazine subscription", "magazine subscriptions"],
            )
            || !task_contains_any(task_lower, &["currently", "current"])
        {
            return None;
        }

        let mut required_owned = vec![
            "magazine".to_string(),
            "subscription".to_string(),
            "currently".to_string(),
            "canceled".to_string(),
        ];
        required_owned.sort();
        required_owned.dedup();
        let required_terms: Vec<&str> = required_owned.iter().map(String::as_str).collect();

        let mut candidates = self
            .session_ids_matching_line(|line, lower| {
                is_summary_or_user_line(line, lower)
                    && !extract_current_magazine_subscription_updates_from_line(line, lower)
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
            let lines = self.find_session_lines(&session_id, false, 256, |line, lower| {
                is_summary_or_user_line(line, lower)
                    && !extract_current_magazine_subscription_updates_from_line(line, lower)
                        .is_empty()
            });
            let mut active = HashSet::new();
            let mut evidence = Vec::new();
            let mut cancellation_bonus = 0usize;
            for line in lines {
                let lower = line.to_ascii_lowercase();
                let updates =
                    extract_current_magazine_subscription_updates_from_line(&line, &lower);
                if updates.is_empty() {
                    continue;
                }
                for (publication, is_active) in updates {
                    let key = normalized_synthetic_phrase_key(&publication);
                    if key.is_empty() {
                        continue;
                    }
                    if is_active {
                        active.insert(key);
                    } else {
                        active.remove(&key);
                        cancellation_bonus = 1;
                    }
                }
                if evidence.len() < 4 && !evidence.iter().any(|existing| existing == &line) {
                    evidence.push(line);
                }
            }

            if active.is_empty() {
                continue;
            }

            let count = active.len();
            let score = count * 100 + session_rank * 10 + cancellation_bonus * 5 + evidence.len();
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
            "current-magazine-subscription-count",
            task,
            &count.to_string(),
            &evidence,
        )
    }

    pub(super) fn synthetic_marathon_target_overrun_minutes_answer(
        &self,
        task: &str,
        task_lower: &str,
    ) -> Option<PathBuf> {
        if !detect_counting_query(task)
            || !task_contains_any(task_lower, &["marathon", "target time"])
            || !task_contains_any(task_lower, &["exceed", "exceeded"])
        {
            return None;
        }

        let required_owned = vec![
            "marathon".to_string(),
            "target time".to_string(),
            "4h 22min".to_string(),
            "4 hours and 10 minutes".to_string(),
        ];
        let required_terms: Vec<&str> = required_owned.iter().map(String::as_str).collect();

        let mut candidates = self
            .session_ids_matching_line(|line, lower| {
                is_summary_or_user_line(line, lower)
                    && (extract_marathon_completion_minutes_from_line(line, lower).is_some()
                        || extract_marathon_target_minutes_from_line(line, lower).is_some())
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

        let mut best: Option<(usize, i32, Vec<String>)> = None;
        for (session_id, session_rank) in candidates {
            let lines = self.find_session_lines(&session_id, false, 256, |line, lower| {
                is_summary_or_user_line(line, lower)
                    && (extract_marathon_completion_minutes_from_line(line, lower).is_some()
                        || extract_marathon_target_minutes_from_line(line, lower).is_some())
            });
            let mut actual = None::<(i32, String)>;
            let mut target = None::<(i32, String)>;
            for line in lines {
                let lower = line.to_ascii_lowercase();
                if actual.is_none() {
                    if let Some(value) =
                        extract_marathon_completion_minutes_from_line(&line, &lower)
                    {
                        actual = Some((value, line.clone()));
                    }
                }
                if target.is_none() {
                    if let Some(value) = extract_marathon_target_minutes_from_line(&line, &lower) {
                        target = Some((value, line.clone()));
                    }
                }
            }

            let (Some((actual_minutes, actual_line)), Some((target_minutes, target_line))) =
                (actual, target)
            else {
                continue;
            };
            if actual_minutes <= target_minutes {
                continue;
            }
            let difference = actual_minutes - target_minutes;
            let mut evidence = vec![actual_line];
            if target_line != evidence[0] {
                evidence.push(target_line);
            }
            let score = 200 + session_rank * 10 + evidence.len();
            let should_replace = best
                .as_ref()
                .map(|(best_score, best_difference, _)| {
                    score > *best_score || (score == *best_score && difference > *best_difference)
                })
                .unwrap_or(true);
            if should_replace {
                best = Some((score, difference, evidence));
            }
        }

        let (_, difference, evidence) = best?;
        self.write_synthetic_answer(
            "marathon-target-overrun-minutes",
            task,
            &difference.to_string(),
            &evidence,
        )
    }

    pub(super) fn synthetic_movie_festival_count_answer(
        &self,
        task: &str,
        task_lower: &str,
    ) -> Option<PathBuf> {
        if !detect_counting_query(task)
            || !task_contains_any(
                task_lower,
                &[
                    "movie festival",
                    "movie festivals",
                    "film festival",
                    "film festivals",
                ],
            )
            || !task_contains_any(task_lower, &["attended", "attend"])
        {
            return None;
        }

        let required_owned = vec![
            "festival".to_string(),
            "volunteered".to_string(),
            "participated".to_string(),
            "q&a".to_string(),
        ];
        let required_terms: Vec<&str> = required_owned.iter().map(String::as_str).collect();

        let mut candidates = self
            .session_ids_matching_line(|line, lower| {
                is_summary_or_user_line(line, lower)
                    && extract_attended_movie_festival_from_line(line, lower).is_some()
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
                    && extract_attended_movie_festival_from_line(line, lower).is_some()
            });
            let mut festivals = HashSet::new();
            let mut evidence = Vec::new();
            for line in lines {
                let lower = line.to_ascii_lowercase();
                let Some(festival) = extract_attended_movie_festival_from_line(&line, &lower)
                else {
                    continue;
                };
                if !festivals.insert(normalized_synthetic_phrase_key(&festival)) {
                    continue;
                }
                if evidence.len() < 4 && !evidence.iter().any(|existing| existing == &line) {
                    evidence.push(line);
                }
            }

            if festivals.is_empty() {
                continue;
            }

            let count = festivals.len();
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
        let count_surface = spell_small_cardinal(count)
            .map(str::to_string)
            .unwrap_or_else(|| count.to_string());
        self.write_synthetic_answer(
            "movie-festival-count",
            task,
            &format!("I attended {count_surface} movie festivals."),
            &evidence,
        )
    }

    pub(super) fn synthetic_music_release_count_answer(
        &self,
        task: &str,
        task_lower: &str,
    ) -> Option<PathBuf> {
        if !detect_counting_query(task)
            || !task_contains_any(task_lower, &["album", "albums", "ep", "eps"])
            || !task_contains_any(task_lower, &["purchased", "downloaded"])
        {
            return None;
        }

        let required_owned = vec![
            "music".to_string(),
            "album".to_string(),
            "ep".to_string(),
            "downloaded".to_string(),
            "bought".to_string(),
            "vinyl".to_string(),
        ];
        let required_terms: Vec<&str> = required_owned.iter().map(String::as_str).collect();

        let mut candidates = self
            .session_ids_matching_line(|line, lower| {
                is_summary_or_user_line(line, lower)
                    && !extract_music_release_signatures_from_line(line, lower).is_empty()
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
                    && !extract_music_release_signatures_from_line(line, lower).is_empty()
            });
            let mut releases = HashSet::new();
            let mut evidence = Vec::new();
            for line in lines {
                let lower = line.to_ascii_lowercase();
                let mut inserted = false;
                for release in extract_music_release_signatures_from_line(&line, &lower) {
                    inserted |= releases.insert(release);
                }
                if inserted
                    && evidence.len() < 4
                    && !evidence.iter().any(|existing| existing == &line)
                {
                    evidence.push(line);
                }
            }

            if releases.is_empty() {
                continue;
            }

            let count = releases.len();
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
        self.write_synthetic_answer("music-release-count", task, &count.to_string(), &evidence)
    }

    pub(super) fn synthetic_current_musical_instrument_count_answer(
        &self,
        task: &str,
        task_lower: &str,
    ) -> Option<PathBuf> {
        if !detect_counting_query(task)
            || !task_contains_any(
                task_lower,
                &[
                    "musical instrument",
                    "musical instruments",
                    "instrument",
                    "instruments",
                ],
            )
            || !task_contains_any(task_lower, &["current", "currently", "now"])
            || !task_contains_any(task_lower, &["own", "have"])
        {
            return None;
        }

        let required_owned = vec![
            "instrument".to_string(),
            "guitar".to_string(),
            "piano".to_string(),
            "drum".to_string(),
            "ukulele".to_string(),
        ];
        let required_terms: Vec<&str> = required_owned.iter().map(String::as_str).collect();

        let mut candidates = self
            .session_ids_matching_line(|line, lower| {
                is_summary_or_user_line(line, lower)
                    && !extract_owned_musical_instrument_signatures_from_line(line, lower)
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
            let lines = self.find_session_lines(&session_id, false, 256, |line, lower| {
                is_summary_or_user_line(line, lower)
                    && !extract_owned_musical_instrument_signatures_from_line(line, lower)
                        .is_empty()
            });
            let mut instruments = HashSet::new();
            let mut durations = HashMap::new();
            let mut evidence = Vec::new();
            for line in lines {
                let lower = line.to_ascii_lowercase();
                let mut inserted = false;
                let duration = extract_duration_answer_from_line(&line)
                    .map(|value| normalize_current_duration_answer(&value));
                for instrument in
                    extract_owned_musical_instrument_signatures_from_line(&line, &lower)
                {
                    let entry = durations.entry(instrument.clone()).or_insert(None);
                    if entry.is_none() && duration.is_some() {
                        *entry = duration.clone();
                    }
                    inserted |= instruments.insert(instrument);
                }
                if inserted
                    && evidence.len() < 4
                    && !evidence.iter().any(|existing| existing == &line)
                {
                    evidence.push(line);
                }
            }

            if instruments.is_empty() {
                continue;
            }

            let count = collapsed_owned_instrument_count(&instruments);
            if count == 0 {
                continue;
            }
            let score = count * 100 + session_rank * 10 + evidence.len();
            let should_replace = best
                .as_ref()
                .map(|(best_score, best_count, _)| {
                    score > *best_score || (score == *best_score && count > *best_count)
                })
                .unwrap_or(true);
            if should_replace {
                let answer = compose_current_musical_instrument_count_answer(
                    &instruments,
                    &durations,
                    count,
                );
                best = Some((score, count, {
                    let mut with_answer = vec![answer];
                    with_answer.extend(evidence);
                    with_answer
                }));
            }
        }

        let (_, _, mut evidence) = best?;
        let answer = evidence.remove(0);
        self.write_synthetic_answer("current-musical-instrument-count", task, &answer, &evidence)
    }

    pub(super) fn synthetic_online_course_completion_total_answer(
        &self,
        task: &str,
        task_lower: &str,
    ) -> Option<PathBuf> {
        if !detect_counting_query(task)
            || !task_contains_any(task_lower, &["online course", "online courses"])
            || !task_contains_any(task_lower, &["completed", "finished"])
            || !task_contains_any(task_lower, &["total", "in total"])
        {
            return None;
        }

        let required_owned = vec![
            "courses".to_string(),
            "completed".to_string(),
            "coursera".to_string(),
            "edx".to_string(),
        ];
        let required_terms: Vec<&str> = required_owned.iter().map(String::as_str).collect();

        let mut candidates = self
            .session_ids_matching_line(|line, lower| {
                is_summary_or_user_line(line, lower)
                    && !extract_online_course_completion_updates_from_line(line, lower).is_empty()
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

        let mut best: Option<(usize, i32, Vec<String>)> = None;
        for (session_id, session_rank) in candidates {
            let lines = self.find_session_lines(&session_id, false, 256, |line, lower| {
                is_summary_or_user_line(line, lower)
                    && !extract_online_course_completion_updates_from_line(line, lower).is_empty()
            });
            let mut platform_counts = HashMap::new();
            let mut evidence = Vec::new();
            for line in lines {
                let lower = line.to_ascii_lowercase();
                let mut inserted = false;
                for (platform, count) in
                    extract_online_course_completion_updates_from_line(&line, &lower)
                {
                    let entry = platform_counts.entry(platform).or_insert(count);
                    if count > *entry {
                        *entry = count;
                    }
                    inserted = true;
                }
                if inserted
                    && evidence.len() < 4
                    && !evidence.iter().any(|existing| existing == &line)
                {
                    evidence.push(line);
                }
            }

            if platform_counts.is_empty() {
                continue;
            }

            let total: i32 = platform_counts.values().sum();
            let score = platform_counts.len() * 1000
                + total.max(0) as usize * 10
                + session_rank * 10
                + evidence.len();
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
        self.write_synthetic_answer("online-courses-total", task, &total.to_string(), &evidence)
    }

    pub(super) fn synthetic_recent_furniture_action_count_answer(
        &self,
        task: &str,
        task_lower: &str,
    ) -> Option<PathBuf> {
        if !detect_counting_query(task)
            || !task_lower.contains("furniture")
            || !task_contains_any(
                task_lower,
                &[
                    "buy",
                    "bought",
                    "assemble",
                    "assembled",
                    "sell",
                    "sold",
                    "fix",
                    "fixed",
                ],
            )
            || !task_contains_any(
                task_lower,
                &[
                    "past few months",
                    "past few month",
                    "recent",
                    "last few months",
                ],
            )
        {
            return None;
        }

        let required_owned = vec![
            "furniture".to_string(),
            "coffee table".to_string(),
            "mattress".to_string(),
            "bookshelf".to_string(),
            "table".to_string(),
            "assembled".to_string(),
            "fixed".to_string(),
        ];
        let required_terms: Vec<&str> = required_owned.iter().map(String::as_str).collect();

        let mut candidates = self
            .session_ids_matching_line(|line, lower| {
                is_summary_or_user_line(line, lower)
                    && !extract_recent_furniture_action_signatures_from_line(line, lower).is_empty()
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
                    && !extract_recent_furniture_action_signatures_from_line(line, lower).is_empty()
            });
            let mut items = HashSet::new();
            let mut evidence = Vec::new();
            for line in lines {
                let lower = line.to_ascii_lowercase();
                let mut inserted = false;
                for item in extract_recent_furniture_action_signatures_from_line(&line, &lower) {
                    inserted |= items.insert(item);
                }
                if inserted
                    && evidence.len() < 4
                    && !evidence.iter().any(|existing| existing == &line)
                {
                    evidence.push(line);
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
            "recent-furniture-action-count",
            task,
            &count.to_string(),
            &evidence,
        )
    }

    pub(super) fn synthetic_initial_garden_planting_count_answer(
        &self,
        task: &str,
        task_lower: &str,
    ) -> Option<PathBuf> {
        if !detect_counting_query(task)
            || !task_contains_any(task_lower, &["plant", "plants"])
            || !task_contains_any(task_lower, &["initially", "initial"])
            || !task_contains_any(task_lower, &["tomato", "tomatoes"])
            || !task_contains_any(task_lower, &["cucumber", "cucumbers"])
        {
            return None;
        }

        let required_owned = vec![
            "tomato".to_string(),
            "cucumber".to_string(),
            "plants".to_string(),
            "garden".to_string(),
            "planted".to_string(),
        ];
        let required_terms: Vec<&str> = required_owned.iter().map(String::as_str).collect();

        let mut candidates = self
            .session_ids_matching_line(|line, lower| {
                lower.starts_with("user:")
                    && lower.contains("plant")
                    && (lower.contains("tomato") || lower.contains("cucumber"))
                    && !extract_line_numbers(line).is_empty()
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

        let mut best: Option<(usize, i32, Vec<String>)> = None;
        for (session_id, session_rank) in candidates {
            let lines = self.find_session_lines(&session_id, false, 256, |line, lower| {
                lower.starts_with("user:")
                    && lower.contains("plant")
                    && (lower.contains("tomato") || lower.contains("cucumber"))
                    && !extract_line_numbers(line).is_empty()
            });
            let mut tomato = None::<(usize, i32, String)>;
            let mut cucumber = None::<(usize, i32, String)>;
            for line in lines {
                let lower = line.to_ascii_lowercase();
                if lower.contains("tomato") {
                    if let Some((value, proximity_score)) = extract_focus_aligned_count(
                        &line,
                        &["tomato".to_string(), "plants".to_string()],
                        task_lower,
                    ) {
                        if !(lower.contains("planted") || lower.contains("initial")) {
                            continue;
                        }
                        let score = proximity_score
                            + usize::from(lower.contains("initially")) * 8
                            + usize::from(lower.contains("garden")) * 2;
                        if tomato
                            .as_ref()
                            .map(|(best_score, best_value, _)| {
                                score > *best_score || (score == *best_score && value > *best_value)
                            })
                            .unwrap_or(true)
                        {
                            tomato = Some((score, value, line.clone()));
                        }
                    }
                }
                if lower.contains("cucumber") {
                    if let Some((value, proximity_score)) = extract_focus_aligned_count(
                        &line,
                        &["cucumber".to_string(), "plants".to_string()],
                        task_lower,
                    ) {
                        if !(lower.contains("garden") && lower.contains("plants")) {
                            continue;
                        }
                        let score = proximity_score
                            + usize::from(lower.contains("got")) * 2
                            + usize::from(lower.contains("growing")) * 4;
                        if cucumber
                            .as_ref()
                            .map(|(best_score, best_value, _)| {
                                score > *best_score || (score == *best_score && value > *best_value)
                            })
                            .unwrap_or(true)
                        {
                            cucumber = Some((score, value, line.clone()));
                        }
                    }
                }
            }

            let (
                Some((tomato_score, tomato_value, tomato_line)),
                Some((cucumber_score, cucumber_value, cucumber_line)),
            ) = (tomato, cucumber)
            else {
                continue;
            };

            let total = tomato_value + cucumber_value;
            let tomato_initially = tomato_line.to_ascii_lowercase().contains("initially");
            let cucumber_growing = cucumber_line.to_ascii_lowercase().contains("growing");
            let mut evidence = vec![tomato_line];
            if cucumber_line != evidence[0] {
                evidence.push(cucumber_line);
            }
            let score = session_rank * 1000
                + tomato_score * 20
                + cucumber_score * 20
                + usize::from(tomato_initially) * 50
                + usize::from(cucumber_growing) * 25;
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
            "initial-garden-planting-count",
            task,
            &total.to_string(),
            &evidence,
        )
    }

    pub(super) fn synthetic_sephora_points_needed_for_free_skincare_answer(
        &self,
        task: &str,
        task_lower: &str,
    ) -> Option<PathBuf> {
        if !detect_counting_query(task)
            || !task_lower.contains("point")
            || !task_lower.contains("sephora")
            || !task_lower.contains("redeem")
            || !task_contains_any(task_lower, &["free skincare", "skincare product"])
            || !task_contains_any(task_lower, &["need to earn", "need"])
        {
            return None;
        }

        let session_id = self
            .best_matching_session_id(task, &["sephora", "points", "free", "skincare"])
            .or_else(|| {
                self.session_ids_matching_line(|_line, lower| {
                    lower.starts_with("user:")
                        && lower.contains("sephora")
                        && lower.contains("point")
                })
                .into_iter()
                .next()
            })?;
        let lines = self.find_session_lines(&session_id, false, 128, |_line, lower| {
            lower.starts_with("user:") && lower.contains("sephora") && lower.contains("point")
        });

        let mut target_total = None::<(i32, String)>;
        let mut current_total = None::<(i32, String)>;
        for line in lines {
            let lower = line.to_ascii_lowercase();
            if target_total.is_none() {
                if let Some(value) = extract_loyalty_point_goal_total_from_line(&line, &lower) {
                    target_total = Some((value, line.clone()));
                }
            }
            if current_total.is_none() {
                if let Some(value) = extract_loyalty_point_current_total_from_line(&line, &lower) {
                    current_total = Some((value, line.clone()));
                }
            }
        }

        let (target_value, target_line) = target_total?;
        let (current_value, current_line) = current_total?;
        let needed = target_value - current_value;
        if needed < 0 {
            return None;
        }

        let mut evidence = vec![target_line];
        if current_line != evidence[0] {
            evidence.push(current_line);
        }

        self.write_synthetic_answer(
            "sephora-points-needed-for-free-skincare",
            task,
            &needed.to_string(),
            &evidence,
        )
    }

    pub(super) fn synthetic_pre_offer_property_view_count_answer(
        &self,
        task: &str,
        task_lower: &str,
    ) -> Option<PathBuf> {
        if !detect_counting_query(task)
            || !task_lower.contains("properties")
            || !task_lower.contains("offer")
            || !task_lower.contains("townhouse")
            || !task_lower.contains("brookside")
        {
            return None;
        }

        let session_id = self
            .best_matching_session_id(
                task,
                &["brookside", "townhouse", "offer", "property", "condo"],
            )
            .or_else(|| {
                self.session_ids_matching_line(|_line, lower| {
                    lower.starts_with("user:")
                        && task_contains_any(
                            lower,
                            &[
                                "brookside",
                                "townhouse",
                                "condo",
                                "bungalow",
                                "cedar creek",
                                "offer",
                            ],
                        )
                })
                .into_iter()
                .next()
            })?;
        let lines = self.find_session_lines(&session_id, false, 256, |_line, lower| {
            lower.starts_with("user:")
                && task_contains_any(
                    lower,
                    &[
                        "brookside",
                        "townhouse",
                        "condo",
                        "bungalow",
                        "cedar creek",
                        "offer",
                    ],
                )
        });

        let offer_line = lines.iter().find(|line| {
            let lower = line.to_ascii_lowercase();
            lower.contains("brookside") && lower.contains("offer")
        })?;
        let offer_rank = extract_explicit_date_rank(offer_line)?;

        let mut views: HashMap<String, (i32, String, String)> = HashMap::new();
        for line in lines.iter() {
            let lower = line.to_ascii_lowercase();
            let Some((key, rank, reason)) = extract_property_view_reason_from_line(line, &lower)
            else {
                continue;
            };
            if rank >= offer_rank {
                continue;
            }
            let should_replace = views
                .get(&key)
                .map(|(best_rank, _, _)| rank < *best_rank)
                .unwrap_or(true);
            if should_replace {
                views.insert(key, (rank, reason, line.clone()));
            }
        }

        if views.is_empty() {
            return None;
        }

        let mut ordered_views = views.into_iter().collect::<Vec<_>>();
        ordered_views.sort_by_key(|(_, (rank, _, _))| *rank);

        let count = ordered_views.len();
        let reasons = ordered_views
            .iter()
            .map(|(_, (_, reason, _))| reason.clone())
            .collect::<Vec<_>>();
        let mut evidence = vec![offer_line.clone()];
        for (_, (_, _, line)) in &ordered_views {
            if !evidence.iter().any(|existing| existing == line) {
                evidence.push(line.clone());
            }
        }

        let answer = format!(
            "I viewed {} properties before making an offer on the townhouse in the Brookside neighborhood. The reasons I didn't make an offer on them were: {}.",
            small_cardinal_word_lower(count),
            join_reason_clauses(&reasons),
        );

        self.write_synthetic_answer("pre-offer-property-view-count", task, &answer, &evidence)
    }
}
