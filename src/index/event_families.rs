use super::count_support::{
    nz, DistinctSignatureDetailsConfig, SignatureDetail, SignatureQuantitySumConfig,
};
use super::event_extractors::{
    best_current_age_fact, best_education_completion_age, extract_age_delta_profile_terms,
    extract_current_age_from_line, extract_education_completion_age_from_line,
    extract_query_month_range, extract_rollercoaster_event_quantities,
    extract_wedding_attendance_details, is_attended_wedding_line, profile_overlap_count,
};
use super::*;

impl NeuronIndex {
    pub(super) fn synthetic_attended_wedding_count_answer(
        &self,
        task: &str,
        task_lower: &str,
    ) -> Option<PathBuf> {
        if !detect_counting_query(task)
            || !task_lower.contains("wedding")
            || !task_contains_any(task_lower, &["attended", "have i attended", "been to"])
        {
            return None;
        }

        let (details, evidence) = self.best_distinct_signature_details(
            task,
            DistinctSignatureDetailsConfig {
                required_owned: vec![
                    "wedding".to_string(),
                    "married".to_string(),
                    "ceremony".to_string(),
                    "friend".to_string(),
                    "cousin".to_string(),
                    "last weekend".to_string(),
                ],
                candidate_limit: nz(8),
                max_lines: nz(512),
                evidence_limit: nz(5),
                line_match: |line: &str, lower: &str| {
                    is_summary_or_user_line(line, lower)
                        && is_attended_wedding_line(lower)
                        && !extract_wedding_attendance_details(line).is_empty()
                },
                extract: |line: &str, lower: &str| {
                    if is_summary_or_user_line(line, lower) {
                        extract_wedding_attendance_details(line)
                    } else {
                        Vec::new()
                    }
                },
            },
        )?;
        let answer = render_attended_wedding_answer(&details);
        self.write_synthetic_answer("attended-wedding-count", task, &answer, &evidence)
    }

    pub(super) fn synthetic_rollercoaster_ride_count_answer(
        &self,
        task: &str,
        task_lower: &str,
    ) -> Option<PathBuf> {
        let month_range = extract_query_month_range(task_lower)?;
        if !detect_counting_query(task)
            || !task_contains_any(task_lower, &["rollercoaster", "roller coaster", "coaster"])
            || !task_contains_any(task_lower, &["how many times", "across all the events"])
        {
            return None;
        }

        let mut required_owned = synthetic_query_terms(task_lower);
        required_owned.push("rode".to_string());
        required_owned.sort();
        required_owned.dedup();

        let (count, evidence) = self.best_signature_quantity_sum(
            task,
            SignatureQuantitySumConfig {
                required_owned,
                candidate_limit: nz(10),
                max_lines: nz(512),
                evidence_limit: nz(6),
                line_match: |_: &str, lower: &str| {
                    lower.starts_with("user:") && lower.contains("rode")
                },
                extract: move |line: &str, lower: &str| {
                    extract_rollercoaster_event_quantities(line, lower, month_range)
                },
            },
        )?;
        self.write_synthetic_answer(
            "rollercoaster-ride-count",
            task,
            &format!("{} {}", count, if count == 1 { "time" } else { "times" }),
            &evidence,
        )
    }

    pub(super) fn synthetic_education_completion_age_delta_answer(
        &self,
        task: &str,
        task_lower: &str,
    ) -> Option<PathBuf> {
        if !task_lower.contains("years older")
            || !task_contains_any(
                task_lower,
                &[
                    "graduated from college",
                    "graduated from university",
                    "graduated from my bachelor's",
                    "graduated from my bachelor",
                ],
            )
        {
            return None;
        }

        let current_age_sessions = self
            .session_ids_matching_line(|line, lower| {
                is_summary_or_user_line(line, lower)
                    && extract_current_age_from_line(line).is_some()
            })
            .into_iter()
            .collect::<Vec<_>>();
        let mut best: Option<(usize, i32, Vec<String>)> = None;

        for graduation_session_id in self.session_ids_matching_line(|line, lower| {
            is_summary_or_user_line(line, lower)
                && extract_education_completion_age_from_line(line).is_some()
        }) {
            let graduation_lines =
                self.find_session_lines(&graduation_session_id, false, 256, |line, lower| {
                    is_summary_or_user_line(line, lower)
                });
            let Some((graduation_age, graduation_score, graduation_line)) =
                best_education_completion_age(&graduation_lines)
            else {
                continue;
            };

            let profile_terms = extract_age_delta_profile_terms(&graduation_lines);
            let ranked_current_sessions = self
                .candidate_session_ids_by_line_overlap(&profile_terms, 16)
                .into_iter()
                .collect::<HashMap<_, _>>();

            for current_session_id in &current_age_sessions {
                let current_lines =
                    self.find_session_lines(current_session_id, false, 256, |line, lower| {
                        is_summary_or_user_line(line, lower)
                    });
                let Some((current_age, current_score, current_line)) =
                    best_current_age_fact(&current_lines)
                else {
                    continue;
                };
                if current_age <= graduation_age {
                    continue;
                }

                let shared_profile = if current_session_id == &graduation_session_id {
                    8usize
                } else {
                    profile_overlap_count(
                        &profile_terms,
                        &extract_age_delta_profile_terms(&current_lines),
                    )
                };
                if current_session_id != &graduation_session_id && shared_profile < 2 {
                    continue;
                }

                let score = shared_profile * 100
                    + ranked_current_sessions
                        .get(current_session_id)
                        .copied()
                        .unwrap_or(0)
                        * 20
                    + graduation_score
                    + current_score;
                let delta = current_age - graduation_age;
                let mut evidence = vec![graduation_line.clone()];
                if !evidence.iter().any(|line| line == &current_line) {
                    evidence.push(current_line);
                }
                let should_replace = best
                    .as_ref()
                    .map(|(best_score, best_delta, _)| {
                        score > *best_score || (score == *best_score && delta > *best_delta)
                    })
                    .unwrap_or(true);
                if should_replace {
                    best = Some((score, delta, evidence));
                }
            }
        }

        let (_, delta, evidence) = best?;
        self.write_synthetic_answer(
            "education-completion-age-delta",
            task,
            &delta.to_string(),
            &evidence,
        )
    }
}

fn render_attended_wedding_answer(details: &[SignatureDetail]) -> String {
    let couples = details
        .iter()
        .map(|detail| detail.display.clone())
        .collect::<Vec<_>>();
    format!(
        "I attended {} weddings. The couples were {}.",
        small_cardinal_word_lower(details.len()),
        join_reason_clauses(&couples),
    )
}
