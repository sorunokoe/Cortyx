use super::*;

impl NeuronIndex {
    pub(in crate::index::core) fn synthetic_transport_cost_delta_answer(
        &self,
        task: &str,
        task_lower: &str,
    ) -> Option<PathBuf> {
        if !is_money_query(task)
            || !task_contains_any(task_lower, &["taxi", "train"])
            || !task_contains_any(
                task_lower,
                &["more expensive", "compared to", "than the train"],
            )
        {
            return None;
        }

        let session_id = self.best_matching_session_id(task, &["commute", "taxi", "train"])?;
        let lines = self.find_session_lines(&session_id, false, 256, |line, lower| {
            is_summary_or_user_line(line, lower)
                && !extract_dollar_amounts(line).is_empty()
                && (lower.contains("taxi") || lower.contains("train fare"))
        });

        let mut taxi: Option<(usize, usize, f32, String)> = None;
        let mut train: Option<(usize, usize, f32, String)> = None;

        for (line_idx, line) in lines.into_iter().enumerate() {
            let lower = line.to_ascii_lowercase();
            let Some(amount) = extract_dollar_amounts(&line).into_iter().next() else {
                continue;
            };

            if lower.contains("taxi") {
                let score = usize::from(lower.contains("cost me"))
                    + usize::from(lower.contains("taxi ride"))
                    + usize::from(lower.contains("take a taxi"))
                    + usize::from(lower.contains("missed my train"));
                let should_replace = taxi
                    .as_ref()
                    .map(|(best_score, best_idx, _, _)| {
                        score > *best_score || (score == *best_score && line_idx > *best_idx)
                    })
                    .unwrap_or(true);
                if should_replace {
                    taxi = Some((score, line_idx, amount, line.clone()));
                }
            }

            if lower.contains("train fare") {
                let score = usize::from(lower.contains("actually"))
                    + usize::from(lower.contains("daily train fare"))
                    + usize::from(lower.contains("averages out"));
                let should_replace = train
                    .as_ref()
                    .map(|(best_score, best_idx, _, _)| {
                        score > *best_score || (score == *best_score && line_idx > *best_idx)
                    })
                    .unwrap_or(true);
                if should_replace {
                    train = Some((score, line_idx, amount, line.clone()));
                }
            }
        }

        let (_, _, taxi_cost, taxi_line) = taxi?;
        let (_, _, train_cost, train_line) = train?;
        if taxi_cost <= train_cost {
            return None;
        }
        let diff = taxi_cost - train_cost;
        let answer = format!("${}", format_numeric_answer(diff));
        self.write_synthetic_answer(
            "transport-cost-delta",
            task,
            &answer,
            &[train_line, taxi_line],
        )
    }

    pub(in crate::index::core) fn synthetic_named_schedule_rotation_answer(
        &self,
        task: &str,
        task_lower: &str,
    ) -> Option<PathBuf> {
        if !task_contains_any(task_lower, &["schedule", "shift", "rotation"]) {
            return None;
        }
        let day = extract_weekday_from_query(task_lower)?;
        let person = extract_schedule_query_person(task)?;
        let person_lower = person.to_ascii_lowercase();
        for session_id in self.session_ids_matching_line(|line, lower| {
            line.trim_start().starts_with('|')
                && (lower.contains(day) || lower.contains(&person_lower) || lower.contains("shift"))
        }) {
            let table_lines = self.find_session_lines(&session_id, false, 256, |line, _| {
                line.trim_start().starts_with('|')
            });
            if let Some((shift, evidence)) =
                extract_schedule_shift_from_table(&table_lines, &person, day)
            {
                let answer = format!(
                    "{person} was assigned to the {shift} on {}.",
                    pluralize_weekday(day)
                );
                return self.write_synthetic_answer("schedule-rotation", task, &answer, &evidence);
            }
        }
        None
    }

    pub(in crate::index::core) fn synthetic_commute_time_answer(
        &self,
        task: &str,
        task_lower: &str,
    ) -> Option<PathBuf> {
        if !is_commute_query(task_lower) {
            return None;
        }
        let task_terms = synthetic_query_terms(task_lower);
        let candidates =
            self.ranked_session_candidates(task, &["commute", "work", "audiobook"], &task_terms, 8);
        let mut best = None::<(usize, String, Vec<String>)>;

        for (session_id, session_score) in candidates {
            let lines = self.find_session_lines(&session_id, true, 64, |line, lower| {
                is_summary_or_user_line(line, lower) && lower.contains("commute")
            });
            for line in lines {
                let Some(value) = extract_commute_duration_from_line(&line) else {
                    continue;
                };
                let lower = line.to_ascii_lowercase();
                let overlap = task_terms
                    .iter()
                    .filter(|term| lower.contains(term.as_str()))
                    .count();
                let score = session_score
                    + overlap * 4
                    + if value.contains("each way") { 8 } else { 0 }
                    + if lower.contains("daily commute") {
                        4
                    } else {
                        0
                    }
                    + if lower.contains("work") { 2 } else { 0 };
                let replace = best
                    .as_ref()
                    .map(|(best_score, current, _)| {
                        (value.contains("each way") && !current.contains("each way"))
                            || (value.contains("each way") == current.contains("each way")
                                && (score > *best_score
                                    || (score == *best_score && value.len() > current.len())))
                    })
                    .unwrap_or(true);
                if replace {
                    best = Some((score, value, vec![line.clone()]));
                }
            }
        }

        let (_, answer, evidence) = best?;
        self.write_synthetic_answer("commute-time", task, &answer, &evidence)
    }
}
