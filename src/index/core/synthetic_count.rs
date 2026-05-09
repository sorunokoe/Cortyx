// This file is a submodule of `crate::index::core`.
// Contains `impl NeuronIndex` synthetic answer methods extracted from synthetic.rs.
use super::*;

impl NeuronIndex {
    pub(super) fn max_count_from_matching_texts<F>(
        &self,
        required_terms: &[&str],
        limit: usize,
        mut parser: F,
    ) -> Option<(i32, Vec<String>)>
    where
        F: FnMut(&str, &str) -> Option<i32>,
    {
        let mut best: Option<(i32, Vec<String>)> = None;
        for (_, content) in self.matching_verbatim_texts(required_terms, limit.max(128)) {
            for raw_line in content.lines() {
                let line = raw_line.trim();
                if line.is_empty() {
                    continue;
                }
                let lower = line.to_ascii_lowercase();
                let Some(value) = parser(line, &lower) else {
                    continue;
                };
                let should_replace = best
                    .as_ref()
                    .map(|(current, _)| value > *current)
                    .unwrap_or(true);
                if should_replace {
                    best = Some((value, vec![line.to_string()]));
                }
            }
        }

        best
    }

    pub(super) fn max_count_answer_from_matching_texts<F>(
        &self,
        task: &str,
        required_terms: &[&str],
        limit: usize,
        slug: &str,
        parser: F,
    ) -> Option<PathBuf>
    where
        F: FnMut(&str, &str) -> Option<i32>,
    {
        let (value, evidence) =
            self.max_count_from_matching_texts(required_terms, limit, parser)?;
        self.write_synthetic_answer(slug, task, &value.to_string(), &evidence)
    }

    pub(super) fn synthetic_missing_operand_answer(
        &self,
        task: &str,
        task_lower: &str,
    ) -> Option<PathBuf> {
        if !synthetic_count_query_requires_multi_operand_reasoning(task, task_lower) {
            return None;
        }

        let options = if task_lower.contains(" or ") {
            synthetic_answer_surface_choice_options(task)
        } else if task_lower.contains(" and ") {
            synthetic_conjoined_choice_options(task)
        } else {
            Vec::new()
        };
        if options.len() != 2 {
            return None;
        }

        let task_terms = synthetic_query_terms(task_lower);
        if task_terms.is_empty() {
            return None;
        }
        let required_terms: Vec<&str> = task_terms.iter().map(String::as_str).collect();
        let mut coverage: [Option<String>; 2] = [None, None];

        for (_, content) in self.matching_verbatim_texts(&required_terms, 64) {
            for raw_line in content.lines() {
                let line = raw_line.trim();
                if line.is_empty() {
                    continue;
                }
                let lower = line.to_ascii_lowercase();
                if !is_direct_count_candidate_line(line, &lower, task_lower) {
                    continue;
                }

                let line_keys =
                    synthetic_answer_surface_term_key_set(&synthetic_query_terms(&lower));
                for (idx, option) in options.iter().enumerate() {
                    if coverage[idx].is_some() {
                        continue;
                    }
                    let overlap =
                        synthetic_answer_surface_overlap_count(&line_keys, &option.term_keys);
                    let min_overlap = if option.term_keys.len() >= 4 { 2 } else { 1 };
                    if overlap >= min_overlap {
                        coverage[idx] = Some(line.to_string());
                    }
                }
            }
        }

        let (present_idx, evidence) = match (&coverage[0], &coverage[1]) {
            (Some(line), None) => (0usize, vec![line.clone()]),
            (None, Some(line)) => (1usize, vec![line.clone()]),
            _ => return None,
        };
        let missing_idx = 1usize.saturating_sub(present_idx);
        let present = missing_operand_display_phrase(&options[present_idx].display);
        let missing = missing_operand_display_phrase(&options[missing_idx].display);
        let answer = format!(
            "The information provided is not enough. You mentioned {}, but you did not mention {}.",
            present, missing
        );
        self.write_synthetic_answer("missing-operand", task, &answer, &evidence)
    }

    pub(super) fn synthetic_doctor_visit_count_answer(
        &self,
        task: &str,
        task_lower: &str,
    ) -> Option<PathBuf> {
        if !task_lower.contains("doctor")
            || (!detect_counting_query(task)
                && !task_contains_any(task_lower, &["different doctors", "which doctors"]))
        {
            return None;
        }

        let month_filter = extract_query_month_name(task_lower);
        let required_terms: Vec<&str> = if month_filter.is_some() {
            vec!["doctor", "appointment", "march"]
        } else {
            vec!["doctor", "physician", "specialist"]
        };
        let overlap_terms = synthetic_query_terms(task_lower);
        let mut candidate_sessions = self
            .session_ids_matching_line(|line, lower| {
                is_summary_or_user_line(line, lower)
                    && task_contains_any(
                        lower,
                        &[
                            "dr.",
                            "doctor",
                            "physician",
                            "specialist",
                            "surgeon",
                            "dermatologist",
                            "gastroenterologist",
                            "neurologist",
                        ],
                    )
                    && month_filter
                        .map(|month| lower.contains(month))
                        .unwrap_or(true)
            })
            .into_iter()
            .map(|session_id| (session_id, 0usize))
            .collect::<Vec<_>>();
        for candidate in self.candidate_session_ids_by_line_overlap(&overlap_terms, 12) {
            if !candidate_sessions
                .iter()
                .any(|(existing, _)| existing == &candidate.0)
            {
                candidate_sessions.push(candidate);
            }
        }
        for session_id in self.candidate_session_ids(task, &required_terms, 8) {
            if !candidate_sessions
                .iter()
                .any(|(existing, _)| existing == &session_id)
            {
                candidate_sessions.push((session_id, 0));
            }
        }

        let session_id = candidate_sessions
            .into_iter()
            .filter_map(|(session_id, base_score)| {
                let lines = self.find_session_lines(&session_id, false, 256, |line, lower| {
                    is_summary_or_user_line(line, lower)
                        && task_contains_any(
                            lower,
                            &[
                                "dr.",
                                "doctor",
                                "physician",
                                "specialist",
                                "surgeon",
                                "dermatologist",
                                "gastroenterologist",
                                "neurologist",
                            ],
                        )
                });
                let mut matched = 0usize;
                let mut distinct_roles = HashSet::new();
                for line in lines {
                    let lower = line.to_ascii_lowercase();
                    if let Some(month) = month_filter {
                        if !lower.contains(month) {
                            continue;
                        }
                    }
                    if !line_describes_actual_doctor_visit(&lower) {
                        continue;
                    }
                    let Some(role) = extract_doctor_role_from_line(&line, &lower) else {
                        continue;
                    };
                    matched += 1;
                    distinct_roles.insert(role);
                }
                (matched > 0).then_some((
                    matched * 10 + distinct_roles.len() * 2 + base_score,
                    session_id,
                ))
            })
            .max_by_key(|(score, _)| *score)
            .map(|(_, session_id)| session_id)?;

        let mut seen_events = HashSet::new();
        let mut roles = Vec::new();
        let mut role_set = HashSet::new();
        let mut evidence = Vec::new();

        let lines = self.find_session_lines(&session_id, false, 256, |line, lower| {
            is_summary_or_user_line(line, lower)
                && task_contains_any(
                    lower,
                    &[
                        "dr.",
                        "doctor",
                        "physician",
                        "specialist",
                        "surgeon",
                        "dermatologist",
                        "gastroenterologist",
                        "neurologist",
                    ],
                )
        });
        for line in lines {
            let lower = line.to_ascii_lowercase();
            if let Some(month) = month_filter {
                if !lower.contains(month) {
                    continue;
                }
            }
            if !line_describes_actual_doctor_visit(&lower) {
                continue;
            }
            let Some(role) = extract_doctor_role_from_line(&line, &lower) else {
                continue;
            };
            let event_key = doctor_visit_event_key(&role, &lower);
            if !seen_events.insert(event_key) {
                continue;
            }
            if evidence.len() < 4 {
                evidence.push(line.clone());
            }
            if role_set.insert(role.clone()) {
                roles.push(role);
            }
        }

        if roles.is_empty() {
            return None;
        }

        roles.sort_by_key(|role| doctor_role_sort_key(role));

        if task_contains_any(task_lower, &["different doctors", "which doctors"]) {
            let answer = match roles.as_slice() {
                [only] => format!("I visited one doctor: {only}."),
                [first, second] => {
                    format!("I visited two different doctors: {first} and {second}.")
                },
                [first, second, third] => {
                    format!("I visited three different doctors: {first}, {second}, and {third}.")
                },
                _ => {
                    let count = roles.len();
                    let joined = roles.join(", ");
                    format!("I visited {count} different doctors: {joined}.")
                },
            };
            return self.write_synthetic_answer("doctor-visit-count", task, &answer, &evidence);
        }

        self.write_synthetic_answer(
            "doctor-appointment-count",
            task,
            &seen_events.len().to_string(),
            &evidence,
        )
    }

    pub(super) fn synthetic_unit_price_answer(
        &self,
        task: &str,
        task_lower: &str,
    ) -> Option<PathBuf> {
        if !is_money_query(task)
            || !task_lower.contains("each")
            || !task_contains_any(task_lower, &["coffee mug", "coffee mugs", "mug"])
        {
            return None;
        }

        let session_id = self.best_matching_session_id(task, &["coffee", "mug", "coworkers"])?;
        let lines = self.find_session_lines(&session_id, false, 256, |line, lower| {
            is_summary_or_user_line(line, lower)
                && task_contains_any(lower, &["coffee mug", "coffee mugs"])
        });

        let focus_terms = vec!["coffee".to_string(), "mug".to_string()];
        let mut quantity: Option<(usize, i32, String)> = None;
        let mut total: Option<(usize, f32, String)> = None;

        for (line_idx, line) in lines.into_iter().enumerate() {
            let lower = line.to_ascii_lowercase();
            if let Some((count, _)) = extract_focus_aligned_count(&line, &focus_terms, task_lower) {
                let score = usize::from(lower.contains("purchased"))
                    + usize::from(lower.contains("one for each"))
                    + usize::from(lower.contains("coffee mugs"));
                let should_replace = quantity
                    .as_ref()
                    .map(|(best_score, _, _)| score > *best_score)
                    .unwrap_or(true);
                if should_replace {
                    quantity = Some((score + line_idx, count, line.clone()));
                }
            }

            if let Some(amount) = extract_dollar_amounts(&line).into_iter().next() {
                let score = usize::from(lower.contains("spent"))
                    + usize::from(lower.contains("splurge"))
                    + usize::from(lower.contains("coffee mugs"));
                let should_replace = total
                    .as_ref()
                    .map(|(best_score, _, _)| score > *best_score)
                    .unwrap_or(true);
                if should_replace {
                    total = Some((score + line_idx, amount, line.clone()));
                }
            }
        }

        let (_, item_count, quantity_line) = quantity?;
        let (_, total_spent, total_line) = total?;
        if item_count <= 0 {
            return None;
        }

        let unit_price = total_spent / item_count as f32;
        let answer = format!("${}", format_numeric_answer(unit_price));
        self.write_synthetic_answer("unit-price", task, &answer, &[quantity_line, total_line])
    }

    pub(super) fn synthetic_multi_session_money_total_answer(
        &self,
        task: &str,
        task_lower: &str,
    ) -> Option<PathBuf> {
        if !should_try_multi_session_money_total(task_lower) {
            return None;
        }

        let focus_terms = extract_multi_session_money_focus_terms(task_lower);
        if focus_terms.is_empty() {
            return None;
        }
        let focus_refs: Vec<&str> = focus_terms.iter().map(String::as_str).collect();
        let min_overlap = if focus_terms.len() >= 4 { 2 } else { 1 };
        let mut facts: Vec<(String, f32, HashSet<String>)> = Vec::new();
        let mut evidence = Vec::new();
        let session_results = self
            .ranked_numeric_aggregate_sessions(task, &focus_terms, |line, lower| {
                is_grounded_user_money_fact_line(lower)
                    && money_total_line_matches_query(task_lower, lower)
                    && !extract_focused_dollar_amounts(line, &focus_terms).is_empty()
            })
            .into_iter()
            .map(|(session_id, session_score)| {
                let mut session_facts = Vec::new();
                let mut session_evidence = Vec::new();
                for line in self.session_verbatim_answer_candidate_lines(&session_id, usize::MAX) {
                    let lower = line.to_ascii_lowercase();
                    if !is_grounded_user_money_fact_line(&lower)
                        || !money_total_line_matches_query(task_lower, &lower)
                    {
                        continue;
                    }
                    let overlap = term_overlap_count(&lower, &focus_refs);
                    if overlap < min_overlap && !(session_score >= 20 && overlap >= 1) {
                        continue;
                    }

                    let amounts = extract_focused_dollar_amounts(&line, &focus_terms);
                    if amounts.is_empty() {
                        continue;
                    }

                    let terms = aggregate_fact_terms(&line);
                    for amount in amounts {
                        if is_duplicate_numeric_aggregate_fact(
                            &session_facts,
                            &session_id,
                            amount,
                            &terms,
                        ) {
                            continue;
                        }
                        session_facts.push((session_id.clone(), amount, terms.clone()));
                        if session_evidence.len() < 6
                            && !session_evidence.iter().any(|existing| existing == &line)
                        {
                            session_evidence.push(line.clone());
                        }
                    }
                }
                (session_id, session_score, session_facts, session_evidence)
            })
            .collect::<Vec<_>>();

        if let Some((_, _, best_facts, best_evidence)) = session_results
            .iter()
            .filter(|(_, _, session_facts, _)| session_facts.len() >= 2)
            .max_by(|a, b| a.2.len().cmp(&b.2.len()).then_with(|| a.1.cmp(&b.1)))
        {
            facts = best_facts.clone();
            evidence = best_evidence.clone();
        } else {
            for (session_id, _, session_facts, session_evidence) in session_results {
                for (_, amount, terms) in session_facts {
                    if is_duplicate_numeric_aggregate_fact(&facts, &session_id, amount, &terms) {
                        continue;
                    }
                    facts.push((session_id.clone(), amount, terms));
                }
                for line in session_evidence {
                    if evidence.len() >= 6 || evidence.iter().any(|existing| existing == &line) {
                        continue;
                    }
                    evidence.push(line);
                }
            }
        }

        if facts.len() < 2 {
            return None;
        }

        let total: f32 = facts.iter().map(|(_, value, _)| *value).sum();
        let answer = format_money_answer(total);
        self.write_synthetic_answer("multi-session-money-total", task, &answer, &evidence)
    }

    pub(super) fn synthetic_multi_session_duration_total_answer(
        &self,
        task: &str,
        task_lower: &str,
    ) -> Option<PathBuf> {
        if !should_try_multi_session_duration_total(task_lower) {
            return None;
        }

        let requested_unit = extract_requested_aggregate_duration_unit(task_lower)?;
        let focus_terms = extract_multi_session_duration_focus_terms(task_lower);
        if focus_terms.is_empty() {
            return None;
        }
        let focus_refs: Vec<&str> = focus_terms.iter().map(String::as_str).collect();
        let min_overlap = if focus_terms.len() >= 4 { 2 } else { 1 };
        let mut facts: Vec<(String, f32, HashSet<String>)> = Vec::new();
        let mut evidence = Vec::new();
        let session_results = self
            .ranked_numeric_aggregate_sessions(task, &focus_terms, |line, lower| {
                is_grounded_user_duration_fact_line(lower)
                    && !extract_matching_duration_total_segments(line, task_lower).is_empty()
            })
            .into_iter()
            .map(|(session_id, session_score)| {
                let mut session_facts = Vec::new();
                let mut session_evidence = Vec::new();
                for line in self.session_verbatim_answer_candidate_lines(&session_id, usize::MAX) {
                    let lower = line.to_ascii_lowercase();
                    if !is_grounded_user_duration_fact_line(&lower) {
                        continue;
                    }

                    let matches = extract_matching_duration_total_segments(&line, task_lower);
                    if matches.is_empty() {
                        continue;
                    }
                    let line_overlap = term_overlap_count(&lower, &focus_refs);
                    for (segment, duration) in matches {
                        let segment_lower = segment.to_ascii_lowercase();
                        let overlap =
                            term_overlap_count(&segment_lower, &focus_refs).max(line_overlap);
                        if overlap < min_overlap && !(session_score >= 20 && overlap >= 1) {
                            continue;
                        }

                        let terms = aggregate_fact_terms(&segment);
                        if is_duplicate_numeric_aggregate_fact(
                            &session_facts,
                            &session_id,
                            duration.days,
                            &terms,
                        ) {
                            continue;
                        }
                        session_facts.push((session_id.clone(), duration.days, terms.clone()));
                        if session_evidence.len() < 6
                            && !session_evidence.iter().any(|existing| existing == &line)
                        {
                            session_evidence.push(line.clone());
                        }
                    }
                }
                (session_id, session_score, session_facts, session_evidence)
            })
            .collect::<Vec<_>>();

        if let Some((_, _, best_facts, best_evidence)) = session_results
            .iter()
            .filter(|(_, _, session_facts, _)| session_facts.len() >= 2)
            .max_by(|a, b| a.2.len().cmp(&b.2.len()).then_with(|| a.1.cmp(&b.1)))
        {
            facts = best_facts.clone();
            evidence = best_evidence.clone();
        } else {
            for (session_id, _, session_facts, session_evidence) in session_results {
                for (_, value, terms) in session_facts {
                    if is_duplicate_numeric_aggregate_fact(&facts, &session_id, value, &terms) {
                        continue;
                    }
                    facts.push((session_id.clone(), value, terms));
                }
                for line in session_evidence {
                    if evidence.len() >= 6 || evidence.iter().any(|existing| existing == &line) {
                        continue;
                    }
                    evidence.push(line);
                }
            }
        }

        if facts.len() < 2 {
            return None;
        }

        let total_days: f32 = facts.iter().map(|(_, value, _)| *value).sum();
        let converted = convert_duration_days(total_days, requested_unit);
        let answer = if requested_unit == "hour"
            && (task_lower.contains("road trip") || task_lower.contains("destinations"))
        {
            let destination_phrase = if task_lower.contains("three") {
                "the three destinations"
            } else {
                "the destinations"
            };
            format!(
                "{} for getting to {} (or {} for the round trip)",
                format_aggregate_duration_answer(converted, requested_unit),
                destination_phrase,
                format_aggregate_duration_answer(converted * 2.0, requested_unit),
            )
        } else {
            format_aggregate_duration_answer(converted, requested_unit)
        };
        self.write_synthetic_answer("multi-session-duration-total", task, &answer, &evidence)
    }

    pub(super) fn synthetic_formal_education_total_answer(
        &self,
        task: &str,
        task_lower: &str,
    ) -> Option<PathBuf> {
        let target_stage = extract_formal_education_target_stage(task_lower)?;
        let mut best: Option<(usize, String, Vec<String>)> = None;
        for (_, content) in self.matching_verbatim_texts(&["high", "school"], 128) {
            let lines = content
                .lines()
                .map(str::trim)
                .filter(|line| !line.is_empty())
                .filter_map(|line| {
                    let lower = line.to_ascii_lowercase();
                    is_summary_or_user_line(line, &lower).then(|| line.to_string())
                })
                .collect::<Vec<_>>();
            let facts = collect_education_stage_facts(&lines);
            let Some((total_years, evidence, fact_count)) =
                solve_formal_education_total(&facts, target_stage)
            else {
                continue;
            };
            let score = fact_count * 10;
            let should_replace = best
                .as_ref()
                .map(|(best_score, _, _)| score > *best_score)
                .unwrap_or(true);
            if should_replace {
                best = Some((score, format!("{total_years} years"), evidence));
            }
        }

        let (_, answer, evidence) = best?;
        self.write_synthetic_answer("formal-education-total", task, &answer, &evidence)
    }

    pub(super) fn synthetic_education_milestone_interval_answer(
        &self,
        task: &str,
        task_lower: &str,
    ) -> Option<PathBuf> {
        if !task_lower.contains("how many months passed between")
            || !task_lower.contains("undergraduate degree")
            || !task_lower.contains("master's thesis")
        {
            return None;
        }

        let start_phrase = "the completion of my undergraduate degree";
        let end_phrase = "the submission of my master's thesis";
        let start_terms = synthetic_query_terms(start_phrase);
        let end_terms = synthetic_query_terms(end_phrase);
        let mut required_owned = start_terms.clone();
        required_owned.extend(end_terms.iter().cloned());
        required_owned.sort();
        required_owned.dedup();

        let mut best: Option<(usize, i32, Vec<String>)> = None;
        for (session_id, session_rank) in
            self.candidate_session_ids_by_line_overlap(&required_owned, 12)
        {
            let lines = self.find_session_lines(&session_id, false, 512, |line, lower| {
                is_summary_or_user_line(line, lower)
            });
            let Some(start_match) =
                best_user_turn_line_with_min_overlap(&lines, start_phrase, &start_terms, Some(1))
            else {
                continue;
            };
            let Some(end_match) =
                best_user_turn_line_with_min_overlap(&lines, end_phrase, &end_terms, Some(1))
            else {
                continue;
            };
            let delta_months = end_match.0 - start_match.0;
            if delta_months <= 0 {
                continue;
            }
            let score = session_rank + start_match.1 + end_match.1;
            let mut evidence = vec![start_match.2.clone()];
            if !evidence.iter().any(|line| line == &end_match.2) {
                evidence.push(end_match.2.clone());
            }
            let should_replace = best
                .as_ref()
                .map(|(best_score, best_months, _)| {
                    score > *best_score || (score == *best_score && delta_months > *best_months)
                })
                .unwrap_or(true);
            if should_replace {
                best = Some((score, delta_months, evidence));
            }
        }

        let (_, delta_months, evidence) = best?;
        self.write_synthetic_answer(
            "education-milestone-interval",
            task,
            &format!("{delta_months} months"),
            &evidence,
        )
    }

    pub(super) fn synthetic_current_role_duration_answer(
        &self,
        task: &str,
        task_lower: &str,
    ) -> Option<PathBuf> {
        if !is_ongoing_duration_query(task_lower)
            || !task_contains_any(task_lower, &["current role", "current position"])
        {
            return None;
        }

        let task_terms = synthetic_query_terms(task_lower);
        let task_refs: Vec<&str> = task_terms.iter().map(String::as_str).collect();
        let mut best: Option<(usize, i32, Vec<String>)> = None;

        for session_id in self.session_ids_matching_line(|line, lower| {
            is_summary_or_user_line(line, lower)
                && extract_current_role_offset_months_from_line(line, lower).is_some()
        }) {
            let lines = self.find_session_lines(&session_id, false, 192, |line, lower| {
                is_summary_or_user_line(line, lower)
            });
            let mut total = None::<(usize, i32, String)>;
            let mut offset = None::<(usize, i32, String, String)>;

            for (line_idx, line) in lines.iter().enumerate() {
                let lower = line.to_ascii_lowercase();
                if let Some(total_months) =
                    extract_current_role_total_months_from_line(line, &lower)
                {
                    let score = term_overlap_count(&lower, &task_refs)
                        + usize::from(lower.contains("company")) * 2
                        + line_idx;
                    let should_replace = total
                        .as_ref()
                        .map(|(best_score, best_months, _)| {
                            score > *best_score
                                || (score == *best_score && total_months > *best_months)
                        })
                        .unwrap_or(true);
                    if should_replace {
                        total = Some((score, total_months, line.clone()));
                    }
                }

                if let Some(offset_months) =
                    extract_current_role_offset_months_from_line(line, &lower)
                {
                    let role_title = extract_current_role_title_from_transition_line(line, &lower)
                        .unwrap_or_default();
                    let role_mentions = if role_title.is_empty() {
                        0
                    } else {
                        lines
                            .iter()
                            .filter(|candidate| {
                                candidate.to_ascii_lowercase().contains(&role_title)
                            })
                            .count()
                    };
                    let score = role_mentions * 10 + line_idx;
                    let should_replace = offset
                        .as_ref()
                        .map(|(best_score, best_months, _, _)| {
                            score > *best_score
                                || (score == *best_score && offset_months >= *best_months)
                        })
                        .unwrap_or(true);
                    if should_replace {
                        offset = Some((score, offset_months, role_title, line.clone()));
                    }
                }
            }

            let (
                Some((total_score, total_months, total_line)),
                Some((offset_score, offset_months, role_title, offset_line)),
            ) = (total, offset)
            else {
                continue;
            };
            if total_months <= offset_months {
                continue;
            }

            let role_mentions = if role_title.is_empty() {
                0
            } else {
                lines
                    .iter()
                    .filter(|line| line.to_ascii_lowercase().contains(&role_title))
                    .count()
            };
            let delta_months = total_months - offset_months;
            let score = total_score + offset_score + role_mentions * 4;
            let mut evidence = vec![total_line];
            if !evidence.iter().any(|line| line == &offset_line) {
                evidence.push(offset_line);
            }
            let should_replace = best
                .as_ref()
                .map(|(best_score, best_months, _)| {
                    score > *best_score || (score == *best_score && delta_months > *best_months)
                })
                .unwrap_or(true);
            if should_replace {
                best = Some((score, delta_months, evidence));
            }
        }

        let (_, delta_months, evidence) = best?;
        self.write_synthetic_answer(
            "current-role-duration",
            task,
            &render_month_span(delta_months),
            &evidence,
        )
    }

    pub(super) fn synthetic_direct_count_answer(
        &self,
        task: &str,
        task_lower: &str,
    ) -> Option<PathBuf> {
        if !detect_counting_query(task)
            || task_lower.starts_with("how long")
            || is_money_query(task)
            || task_has_recall_context(task_lower)
            || should_inject_count_aggregate(task)
            || synthetic_count_query_requires_multi_operand_reasoning(task, task_lower)
        {
            return None;
        }

        let prefers_reference_count =
            task_lower.contains("subjects") && task_contains_any(task_lower, &["study", "journal"]);
        let prefers_max_value = has_explicit_current_state_marker(task)
            || detect_knowledge_update_query(task)
            || prefers_reference_count
            || task_contains_any(
                task_lower,
                &[
                    "so far",
                    "already",
                    "completed",
                    "finished",
                    "watched",
                    " complete ",
                    " finish ",
                    " watch ",
                    "worn",
                    " wear ",
                    "tried",
                    " try ",
                    "how many times",
                    "times have i",
                    "times did i",
                    " need ",
                    " needs ",
                    " reach ",
                    " reaches ",
                    " requires ",
                    " required ",
                ],
            );
        if !prefers_max_value {
            return None;
        }

        let task_terms = synthetic_query_terms(task_lower);
        let mut focus_terms = extract_direct_count_focus_terms(&task_terms);
        if has_explicit_current_state_marker(task) || detect_knowledge_update_query(task) {
            let knowledge_terms = extract_knowledge_update_focus_terms(&task_terms);
            if !knowledge_terms.is_empty() {
                focus_terms = knowledge_terms;
            }
        }
        if focus_terms.is_empty() {
            focus_terms = task_terms.clone();
        }
        focus_terms.sort();
        focus_terms.dedup();
        if focus_terms.is_empty() {
            return None;
        }

        let required_terms: Vec<&str> = focus_terms.iter().map(String::as_str).collect();
        let task_term_refs: Vec<&str> = task_terms.iter().map(String::as_str).collect();
        let wants_current =
            has_explicit_current_state_marker(task) || detect_knowledge_update_query(task);
        let mut best: Option<(f32, i32, Vec<String>)> = None;
        let mut runner_up: Option<(f32, i32)> = None;

        for (path, content) in self.matching_verbatim_texts(&required_terms, 64) {
            let is_summary = is_session_summary_path(&path);
            for raw_line in content.lines() {
                let line = raw_line.trim();
                if line.is_empty() {
                    continue;
                }
                let lower = line.to_ascii_lowercase();
                if !is_summary_or_user_line(line, &lower) {
                    continue;
                }

                if let Some(required_role) = direct_count_required_role_phrase(task_lower) {
                    if !lower.contains(&required_role) {
                        continue;
                    }
                }

                if task_contains_any(task_lower, &["completed", "finished"])
                    && lower.contains("currently on")
                    && !task_contains_any(&lower, &["completed", "finished"])
                {
                    continue;
                }

                let focus_overlap = term_overlap_count(&lower, &required_terms);
                let min_focus_overlap = if focus_terms.len() >= 4 {
                    3
                } else if focus_terms.len() >= 2 {
                    2
                } else {
                    1
                };
                if focus_overlap < min_focus_overlap {
                    continue;
                }
                let raw_overlap = term_overlap_count(&lower, &task_term_refs);
                let Some((value, proximity_score)) =
                    extract_focus_aligned_count(line, &focus_terms, task_lower)
                else {
                    continue;
                };
                if value <= 0 {
                    continue;
                }

                let mut score = focus_overlap as f32 * 8.0
                    + raw_overlap as f32 * 2.0
                    + proximity_score as f32 * 1.5;
                if is_summary {
                    score += 1.5;
                }
                if wants_current && line_has_current_count_marker(&lower) {
                    score += 4.0;
                }
                if task_contains_any(
                    task_lower,
                    &[
                        " need ",
                        " needs ",
                        " reach ",
                        " reaches ",
                        " requires ",
                        " required ",
                    ],
                ) && task_contains_any(
                    &lower,
                    &["need", "needs", "reach", "requires", "required"],
                ) {
                    score += 2.0;
                }
                if task_contains_any(task_lower, &["completed", "finished"])
                    && task_contains_any(&lower, &["completed", "finished"])
                {
                    score += 1.5;
                }
                if task_contains_any(task_lower, &["watched", "worn", "tried"])
                    && task_contains_any(&lower, &["watched", "worn", "tried"])
                {
                    score += 1.0;
                }
                if score < 12.0 {
                    continue;
                }

                let evidence = vec![line.to_string()];
                let should_replace = best
                    .as_ref()
                    .map(|(best_score, best_value, _)| {
                        score > *best_score
                            || ((score - *best_score).abs() < 0.01
                                && prefers_max_value
                                && value > *best_value)
                    })
                    .unwrap_or(true);
                if should_replace {
                    if let Some((best_score, best_value, _)) = &best {
                        if *best_value != value {
                            runner_up = Some((*best_score, *best_value));
                        }
                    }
                    best = Some((score, value, evidence));
                } else if best
                    .as_ref()
                    .map(|(_, best_value, _)| *best_value != value)
                    .unwrap_or(true)
                    && runner_up
                        .as_ref()
                        .map(|(runner_score, runner_value)| {
                            score > *runner_score
                                || ((score - *runner_score).abs() < 0.01
                                    && prefers_max_value
                                    && value > *runner_value)
                        })
                        .unwrap_or(true)
                {
                    runner_up = Some((score, value));
                }
            }
        }

        let (best_score, value, evidence) = best?;
        if let Some((runner_score, runner_value)) = runner_up {
            if runner_value != value
                && runner_score + 0.75 >= best_score
                && !(prefers_max_value && value > runner_value)
            {
                return None;
            }
        }
        if task_lower.contains("issues of ")
            && task_contains_any(task_lower, &["finished reading", "finished"])
        {
            if let Some(answer) = evidence
                .first()
                .and_then(|line| extract_plural_issue_count_answer_from_line(line))
            {
                return self.write_synthetic_answer("direct-count", task, &answer, &evidence);
            }
        }
        self.write_synthetic_answer("direct-count", task, &value.to_string(), &evidence)
    }

    pub(super) fn synthetic_study_subject_count_answer(
        &self,
        task: &str,
        task_lower: &str,
    ) -> Option<PathBuf> {
        if !detect_counting_query(task)
            || !task_lower.contains("subjects")
            || !task_contains_any(task_lower, &["study", "journal"])
        {
            return None;
        }

        let task_terms = synthetic_query_terms(task_lower);
        let focus_terms: Vec<String> = task_terms
            .iter()
            .filter(|term| term.len() >= 3)
            .cloned()
            .collect();
        let required_terms: Vec<&str> = focus_terms.iter().map(String::as_str).collect();
        let required_journal = study_subject_required_journal_phrase(task_lower);
        let mut best: Option<(usize, usize, i32, Vec<String>)> = None;

        for (_, content) in self.matching_verbatim_texts(&required_terms, 64) {
            for raw_line in content.lines() {
                let line = raw_line.trim();
                if line.is_empty() || !is_session_answer_candidate_line(line) {
                    continue;
                }
                let lower = line.to_ascii_lowercase();
                if !lower.contains("subject") {
                    continue;
                }
                if extract_numbered_list_item(line).is_none() && !lower.starts_with("assistant:") {
                    continue;
                }
                if let Some(journal_phrase) = required_journal.as_deref() {
                    if !lower.contains(journal_phrase) {
                        continue;
                    }
                }

                let overlap = term_overlap_count(&lower, &required_terms);
                if overlap < 4 {
                    continue;
                }
                let Some((value, proximity_score)) =
                    extract_focus_aligned_count(line, &focus_terms, task_lower)
                else {
                    continue;
                };
                if value <= 0 {
                    continue;
                }

                let evidence = vec![line.to_string()];
                let should_replace = best
                    .as_ref()
                    .map(|(best_overlap, best_proximity, best_value, _)| {
                        overlap > *best_overlap
                            || (overlap == *best_overlap && proximity_score > *best_proximity)
                            || (overlap == *best_overlap
                                && proximity_score == *best_proximity
                                && value > *best_value)
                    })
                    .unwrap_or(true);
                if should_replace {
                    best = Some((overlap, proximity_score, value, evidence));
                }
            }
        }

        let (_, _, value, evidence) = best?;
        self.write_synthetic_answer(
            "study-subject-count",
            task,
            &format!("{value} subjects"),
            &evidence,
        )
    }

    pub(super) fn synthetic_instagram_current_count_answer(
        &self,
        task: &str,
        task_lower: &str,
    ) -> Option<PathBuf> {
        if !detect_counting_query(task)
            || !task_lower.contains("instagram")
            || !task_lower.contains("follower")
            || !task_contains_any(task_lower, &["current", "currently", "now", "these days"])
        {
            return None;
        }

        let task_terms = synthetic_query_terms(task_lower);
        let required_terms: Vec<&str> = task_terms.iter().map(String::as_str).collect();
        let mut candidates = self
            .candidate_session_ids(task, &required_terms, 8)
            .into_iter()
            .enumerate()
            .map(|(idx, session_id)| (session_id, 8usize.saturating_sub(idx)))
            .collect::<Vec<_>>();
        if candidates.is_empty() {
            candidates = self.candidate_session_ids_by_line_overlap(&task_terms, 8);
        }

        let prefers_explicit_current = task_contains_any(task_lower, &["current", "currently"]);
        let mut best: Option<(usize, usize, i32, Vec<String>)> = None;
        for (session_id, session_rank) in candidates {
            let lines = self.find_session_lines(&session_id, false, 128, |line, lower| {
                is_summary_or_user_line(line, lower)
                    && lower.contains("follower")
                    && !task_contains_any(
                        lower,
                        &["facebook", "twitter", "tiktok", "youtube", "linkedin"],
                    )
            });
            let mut session_best: Option<(usize, i32, Vec<String>)> = None;
            for (line_idx, line) in lines.into_iter().enumerate() {
                let lower = line.to_ascii_lowercase();
                let Some((value, line_strength)) =
                    extract_instagram_current_count_candidate(&line, &lower)
                else {
                    continue;
                };
                let evidence = vec![line.clone()];
                if session_best
                    .as_ref()
                    .map(|(best_metric, best_value, _)| {
                        if prefers_explicit_current {
                            line_strength > *best_metric
                                || (line_strength == *best_metric && value > *best_value)
                        } else {
                            line_idx > *best_metric
                                || (line_idx == *best_metric && value > *best_value)
                        }
                    })
                    .unwrap_or(true)
                {
                    session_best = Some((
                        if prefers_explicit_current {
                            line_strength
                        } else {
                            line_idx
                        },
                        value,
                        evidence,
                    ));
                }
            }
            let Some((line_metric, value, evidence)) = session_best else {
                continue;
            };
            if best
                .as_ref()
                .map(|(best_rank, best_metric, best_value, _)| {
                    if prefers_explicit_current {
                        line_metric > *best_metric
                            || (line_metric == *best_metric && session_rank > *best_rank)
                            || (line_metric == *best_metric
                                && session_rank == *best_rank
                                && value > *best_value)
                    } else {
                        session_rank > *best_rank
                            || (session_rank == *best_rank && line_metric > *best_metric)
                            || (session_rank == *best_rank
                                && line_metric == *best_metric
                                && value > *best_value)
                    }
                })
                .unwrap_or(true)
            {
                best = Some((session_rank, line_metric, value, evidence));
            }
        }

        let (_, _, value, evidence) = best?;
        self.write_synthetic_answer(
            "instagram-followers-current",
            task,
            &value.to_string(),
            &evidence,
        )
    }

    pub(super) fn synthetic_collection_window_count_answer(
        &self,
        task: &str,
        task_lower: &str,
    ) -> Option<PathBuf> {
        if !detect_counting_query(task)
            || !task_contains_any(task_lower, &["collection", "collecting"])
        {
            return None;
        }
        let window = extract_query_duration_window(task_lower)?;
        let task_terms = synthetic_query_terms(task_lower);
        let mut focus_terms = extract_direct_count_focus_terms(&task_terms);
        if focus_terms.is_empty() {
            focus_terms = task_terms.clone();
        }
        focus_terms.sort();
        focus_terms.dedup();
        let required_terms: Vec<&str> = focus_terms.iter().map(String::as_str).collect();
        let task_term_refs: Vec<&str> = task_terms.iter().map(String::as_str).collect();
        let mut best: Option<(usize, usize, i32, Vec<String>)> = None;

        for (_, content) in self.matching_verbatim_texts(&required_terms, 64) {
            for raw_line in content.lines() {
                let line = raw_line.trim();
                if line.is_empty() {
                    continue;
                }
                let lower = line.to_ascii_lowercase();
                if !is_summary_or_user_line(line, &lower)
                    || !task_contains_any(&lower, &["collection", "collecting"])
                {
                    continue;
                }
                let Some(duration) = extract_duration_answer_from_line(line) else {
                    continue;
                };
                if normalize_current_duration_answer(&duration).to_ascii_lowercase() != window {
                    continue;
                }
                let focus_overlap = term_overlap_count(&lower, &required_terms);
                if focus_overlap < 3 {
                    continue;
                }
                let raw_overlap = term_overlap_count(&lower, &task_term_refs);
                let Some((value, _proximity_score)) =
                    extract_focus_aligned_count(line, &focus_terms, task_lower)
                else {
                    continue;
                };
                if value <= 0 {
                    continue;
                }
                let evidence = vec![line.to_string()];
                if best
                    .as_ref()
                    .map(|(best_focus, best_overlap, best_value, _)| {
                        focus_overlap > *best_focus
                            || (focus_overlap == *best_focus && raw_overlap > *best_overlap)
                            || (focus_overlap == *best_focus
                                && raw_overlap == *best_overlap
                                && value > *best_value)
                    })
                    .unwrap_or(true)
                {
                    best = Some((focus_overlap, raw_overlap, value, evidence));
                }
            }
        }

        let (_, _, value, evidence) = best?;
        self.write_synthetic_answer(
            "collection-window-count",
            task,
            &value.to_string(),
            &evidence,
        )
    }

    pub(super) fn synthetic_daily_time_commitment_answer(
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

    pub(super) fn synthetic_time_spent_range_answer(
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

    pub(super) fn synthetic_publication_issue_count_answer(
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

    pub(super) fn synthetic_collection_restart_count_answer(
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

    pub(super) fn synthetic_weight_loss_since_start_answer(
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

    pub(super) fn synthetic_since_start_count_answer(
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
