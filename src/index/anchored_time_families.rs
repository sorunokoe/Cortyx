use super::anchored_time_extractors::{
    add_minutes_to_clock_time, extract_bedtime_fact_from_line,
    extract_clinic_travel_duration_fact_from_line, extract_departure_home_fact_from_line,
    extract_doctor_appointment_fact_from_line, parse_anchored_time_query, previous_weekday,
    AnchoredTimeQuery, BedtimeBeforeAppointmentQuery, ClinicArrivalQuery, TravelDurationFact,
    WeekdayTimeFact,
};
use super::conversation_scan_support::session_score;
use super::*;

impl NeuronIndex {
    pub(super) fn synthetic_anchored_time_answer(
        &self,
        task: &str,
        task_lower: &str,
    ) -> Option<PathBuf> {
        match parse_anchored_time_query(task_lower)? {
            AnchoredTimeQuery::BedtimeBeforeAppointment(query) => {
                self.synthetic_bedtime_before_appointment_answer(task, &query)
            },
            AnchoredTimeQuery::ClinicArrival(query) => {
                self.synthetic_clinic_arrival_answer(task, &query)
            },
        }
    }

    fn synthetic_bedtime_before_appointment_answer(
        &self,
        task: &str,
        query: &BedtimeBeforeAppointmentQuery,
    ) -> Option<PathBuf> {
        let candidates =
            self.ranked_numeric_aggregate_sessions(task, &query.required_terms, |line, lower| {
                extract_bedtime_fact_from_line(line, lower).is_some()
                    || extract_doctor_appointment_fact_from_line(line, lower).is_some()
            });
        let (answer, evidence) = best_bedtime_answer(self, &candidates)
            .or_else(|| best_bedtime_answer_all_entries(self))?;
        self.write_synthetic_answer("bedtime-before-appointment", task, &answer, &evidence)
    }

    fn synthetic_clinic_arrival_answer(
        &self,
        task: &str,
        query: &ClinicArrivalQuery,
    ) -> Option<PathBuf> {
        let candidates =
            self.ranked_numeric_aggregate_sessions(task, &query.required_terms, |line, lower| {
                extract_departure_home_fact_from_line(line, lower, &query.weekday).is_some()
                    || extract_clinic_travel_duration_fact_from_line(line, lower).is_some()
            });
        let (answer, evidence) = best_clinic_arrival_answer(self, &candidates, query)
            .or_else(|| best_clinic_arrival_answer_all_entries(self, query))?;
        self.write_synthetic_answer("clinic-arrival-time", task, &answer, &evidence)
    }
}

fn best_bedtime_answer(
    idx: &NeuronIndex,
    candidates: &[(String, usize)],
) -> Option<(String, Vec<String>)> {
    candidates
        .iter()
        .filter_map(|(session_id, session_rank)| {
            resolve_bedtime_answer(idx.session_answer_candidate_lines(session_id, usize::MAX)).map(
                |(answer, evidence, score)| (session_score(*session_rank, score), answer, evidence),
            )
        })
        .max_by_key(|(score, _, _)| *score)
        .map(|(_, answer, evidence)| (answer, evidence))
}

fn best_clinic_arrival_answer(
    idx: &NeuronIndex,
    candidates: &[(String, usize)],
    query: &ClinicArrivalQuery,
) -> Option<(String, Vec<String>)> {
    candidates
        .iter()
        .filter_map(|(session_id, session_rank)| {
            resolve_clinic_arrival_answer(
                idx.session_answer_candidate_lines(session_id, usize::MAX),
                &query.weekday,
            )
            .map(|(answer, evidence, score)| {
                (session_score(*session_rank, score), answer, evidence)
            })
        })
        .max_by_key(|(score, _, _)| *score)
        .map(|(_, answer, evidence)| (answer, evidence))
}

fn best_bedtime_answer_all_entries(idx: &NeuronIndex) -> Option<(String, Vec<String>)> {
    let lines = all_verbatim_candidate_lines(idx);
    resolve_bedtime_answer(lines).map(|(answer, evidence, _)| (answer, evidence))
}

fn best_clinic_arrival_answer_all_entries(
    idx: &NeuronIndex,
    query: &ClinicArrivalQuery,
) -> Option<(String, Vec<String>)> {
    let lines = all_verbatim_candidate_lines(idx);
    resolve_clinic_arrival_answer(lines, &query.weekday)
        .map(|(answer, evidence, _)| (answer, evidence))
}

fn resolve_bedtime_answer(lines: Vec<(String, bool)>) -> Option<(String, Vec<String>, usize)> {
    let mut bedtimes = Vec::new();
    let mut appointments = Vec::new();
    for (line, is_summary) in lines {
        let lower = line.to_ascii_lowercase();
        if let Some(mut fact) = extract_bedtime_fact_from_line(&line, &lower) {
            if is_summary {
                fact.score += 3;
            }
            bedtimes.push(fact);
        }
        if let Some(mut fact) = extract_doctor_appointment_fact_from_line(&line, &lower) {
            if is_summary {
                fact.score += 3;
            }
            appointments.push(fact);
        }
    }
    appointments.into_iter().find_map(|appointment| {
        let prior_day = previous_weekday(&appointment.weekday)?;
        let bedtime = bedtimes
            .iter()
            .filter(|fact| fact.weekday == prior_day)
            .max_by_key(|fact| fact.score)?;
        Some((
            bedtime.time.clone(),
            dedupe_time_evidence([bedtime.evidence.clone(), appointment.evidence.clone()]),
            bedtime.score + appointment.score,
        ))
    })
}

fn resolve_clinic_arrival_answer(
    lines: Vec<(String, bool)>,
    weekday: &str,
) -> Option<(String, Vec<String>, usize)> {
    let mut departure: Option<WeekdayTimeFact> = None;
    let mut duration: Option<TravelDurationFact> = None;
    for (line, is_summary) in lines {
        let lower = line.to_ascii_lowercase();
        if let Some(mut fact) = extract_departure_home_fact_from_line(&line, &lower, weekday) {
            if is_summary {
                fact.score += 3;
            }
            if departure
                .as_ref()
                .map(|best| fact.score > best.score)
                .unwrap_or(true)
            {
                departure = Some(fact);
            }
        }
        if let Some(mut fact) = extract_clinic_travel_duration_fact_from_line(&line, &lower) {
            if is_summary {
                fact.score += 3;
            }
            if duration
                .as_ref()
                .map(|best| fact.score > best.score)
                .unwrap_or(true)
            {
                duration = Some(fact);
            }
        }
    }
    let departure = departure?;
    let duration = duration?;
    Some((
        add_minutes_to_clock_time(&departure.time, duration.minutes)?,
        dedupe_time_evidence([departure.evidence.clone(), duration.evidence.clone()]),
        departure.score + duration.score,
    ))
}

fn dedupe_time_evidence<I>(evidence: I) -> Vec<String>
where
    I: IntoIterator<Item = String>,
{
    let mut deduped = Vec::new();
    for line in evidence {
        if !deduped.iter().any(|existing| existing == &line) {
            deduped.push(line);
        }
    }
    deduped
}

fn all_verbatim_candidate_lines(idx: &NeuronIndex) -> Vec<(String, bool)> {
    let mut lines = Vec::new();
    for entry in idx
        .entries
        .iter()
        .filter(|entry| matches!(entry.kind, NeuronKind::Verbatim))
    {
        let Ok(content) = std::fs::read_to_string(&entry.neuron_path) else {
            continue;
        };
        let is_summary = is_session_summary_path(&entry.neuron_path);
        for raw_line in strip_query_surface_section(&content).lines() {
            let line = raw_line.trim();
            if !line.is_empty() && is_session_answer_candidate_line(line) {
                lines.push((line.to_string(), is_summary));
            }
        }
    }
    lines
}
