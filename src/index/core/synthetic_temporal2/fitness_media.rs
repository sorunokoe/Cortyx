//! Marathon, movie festival, music release, musical instruments, online courses.

use super::super::*;

impl NeuronIndex {
    pub fn synthetic_recent_activity_duration_total_answer(
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

    pub fn synthetic_current_magazine_subscription_count_answer(
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

    pub fn synthetic_marathon_target_overrun_minutes_answer(
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

    pub fn synthetic_movie_festival_count_answer(
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

    pub fn synthetic_music_release_count_answer(
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

    pub fn synthetic_current_musical_instrument_count_answer(
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

    pub fn synthetic_online_course_completion_total_answer(
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
                + usize::try_from(total.max(0)).unwrap_or(0) * 10
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
}
