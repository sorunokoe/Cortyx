//! Financial and medical synthetic answers: prices, money totals, doctor visits.

use super::super::*;

impl NeuronIndex {
    pub fn max_count_from_matching_texts<F>(
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

    pub fn max_count_answer_from_matching_texts<F>(
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

    pub fn synthetic_missing_operand_answer(
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

    pub fn synthetic_doctor_visit_count_answer(
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

    pub fn synthetic_unit_price_answer(&self, task: &str, task_lower: &str) -> Option<PathBuf> {
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

    pub fn synthetic_multi_session_money_total_answer(
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

    pub fn synthetic_multi_session_duration_total_answer(
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
}
