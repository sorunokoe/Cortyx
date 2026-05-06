use super::average_value_extractors::{
    extract_academic_gpa_facts_from_line, extract_family_age_facts_from_line, format_average_value,
    parse_average_value_query, AcademicGpaFact, AcademicGpaQuery, AcademicStageKind,
    AverageValueQuery, FamilyAgeFact, FamilyAgeGroup, FamilyAgeQuery,
};
use super::conversation_scan_support::{scanned_conversation_lines, session_score};
use super::*;

impl NeuronIndex {
    pub(super) fn synthetic_average_value_answer(
        &self,
        task: &str,
        task_lower: &str,
    ) -> Option<PathBuf> {
        match parse_average_value_query(task_lower)? {
            AverageValueQuery::AcademicGpa(query) => {
                self.synthetic_academic_gpa_average_answer(task, &query)
            },
            AverageValueQuery::FamilyAge(query) => {
                self.synthetic_family_age_average_answer(task, &query)
            },
        }
    }

    fn synthetic_academic_gpa_average_answer(
        &self,
        task: &str,
        query: &AcademicGpaQuery,
    ) -> Option<PathBuf> {
        let candidates =
            self.collect_signature_candidates(task, &query.required_terms, 12, |line, lower| {
                !extract_academic_gpa_facts_from_line(line, lower).is_empty()
            });
        let facts = best_entry_scanned_academic_gpa_facts(self)
            .or_else(|| best_same_session_academic_gpa_facts(self, &candidates))?;
        let undergraduate = best_academic_gpa_fact(&facts, AcademicStageKind::Undergraduate)?;
        let graduate = best_academic_gpa_fact(&facts, AcademicStageKind::Graduate)?;
        let average = (undergraduate.value + graduate.value) / 2.0;
        self.write_synthetic_answer(
            "average-academic-gpa",
            task,
            &format_average_value(average),
            &[undergraduate.evidence, graduate.evidence],
        )
    }

    fn synthetic_family_age_average_answer(
        &self,
        task: &str,
        query: &FamilyAgeQuery,
    ) -> Option<PathBuf> {
        let candidates =
            self.collect_signature_candidates(task, &query.required_terms, 12, |line, lower| {
                !extract_family_age_facts_from_line(line, lower).is_empty()
            });
        let facts = best_entry_scanned_family_age_facts(self)
            .or_else(|| best_same_session_family_age_facts(self, &candidates))?;
        has_required_family_age_groups(&facts).then_some(())?;
        let values = facts.values().map(|fact| fact.value).collect::<Vec<_>>();
        let average = values.iter().sum::<f64>() / values.len() as f64;
        let evidence = facts
            .values()
            .map(|fact| fact.evidence.clone())
            .collect::<Vec<_>>();
        self.write_synthetic_answer(
            "average-family-age",
            task,
            &format_average_value(average),
            &evidence,
        )
    }
}

fn best_same_session_academic_gpa_facts(
    idx: &NeuronIndex,
    candidates: &[(String, usize)],
) -> Option<Vec<AcademicGpaFact>> {
    candidates
        .iter()
        .filter_map(|(session_id, session_rank)| {
            let lines = idx.find_session_lines(session_id, false, 256, |line, lower| {
                lower.starts_with("user:")
                    && !extract_academic_gpa_facts_from_line(line, lower).is_empty()
            });
            let facts = lines
                .iter()
                .flat_map(|line| {
                    let lower = line.to_ascii_lowercase();
                    extract_academic_gpa_facts_from_line(line, &lower)
                })
                .collect::<Vec<_>>();
            let undergraduate = best_academic_gpa_fact(&facts, AcademicStageKind::Undergraduate)?;
            let graduate = best_academic_gpa_fact(&facts, AcademicStageKind::Graduate)?;
            Some((
                session_score(*session_rank, undergraduate.score + graduate.score),
                vec![undergraduate, graduate],
            ))
        })
        .max_by_key(|(score, _)| *score)
        .map(|(_, facts)| facts)
}

fn best_same_session_family_age_facts(
    idx: &NeuronIndex,
    candidates: &[(String, usize)],
) -> Option<HashMap<String, FamilyAgeFact>> {
    candidates
        .iter()
        .filter_map(|(session_id, session_rank)| {
            let lines = idx.find_session_lines(session_id, false, 256, |line, lower| {
                lower.starts_with("user:")
                    && !extract_family_age_facts_from_line(line, lower).is_empty()
            });
            let facts = collect_family_age_facts(lines);
            has_required_family_age_groups(&facts).then_some((
                session_score(*session_rank, facts.values().map(|fact| fact.score).sum()),
                facts,
            ))
        })
        .max_by_key(|(score, facts)| (*score, facts.len()))
        .map(|(_, facts)| facts)
}

fn best_entry_scanned_academic_gpa_facts(idx: &NeuronIndex) -> Option<Vec<AcademicGpaFact>> {
    scanned_conversation_lines(idx)
        .filter_map(|lines| {
            let facts = lines
                .iter()
                .flat_map(|line| {
                    let lower = line.to_ascii_lowercase();
                    extract_academic_gpa_facts_from_line(line, &lower)
                })
                .collect::<Vec<_>>();
            let undergraduate = best_academic_gpa_fact(&facts, AcademicStageKind::Undergraduate)?;
            let graduate = best_academic_gpa_fact(&facts, AcademicStageKind::Graduate)?;
            Some((
                undergraduate.score + graduate.score,
                vec![undergraduate, graduate],
            ))
        })
        .max_by_key(|(score, _)| *score)
        .map(|(_, facts)| facts)
}

fn best_entry_scanned_family_age_facts(
    idx: &NeuronIndex,
) -> Option<HashMap<String, FamilyAgeFact>> {
    scanned_conversation_lines(idx)
        .filter_map(|lines| {
            let facts = collect_family_age_facts(lines);
            has_required_family_age_groups(&facts)
                .then_some((facts.values().map(|fact| fact.score).sum::<usize>(), facts))
        })
        .max_by_key(|(score, facts)| (*score, facts.len()))
        .map(|(_, facts)| facts)
}

fn best_academic_gpa_fact(
    facts: &[AcademicGpaFact],
    stage: AcademicStageKind,
) -> Option<AcademicGpaFact> {
    facts
        .iter()
        .filter(|fact| fact.stage == stage)
        .cloned()
        .max_by_key(|fact| fact.score)
}

fn collect_family_age_facts(lines: Vec<String>) -> HashMap<String, FamilyAgeFact> {
    let mut best = HashMap::new();
    for line in lines {
        let lower = line.to_ascii_lowercase();
        for fact in extract_family_age_facts_from_line(&line, &lower) {
            let should_replace = best
                .get(&fact.key)
                .map(|existing: &FamilyAgeFact| fact.score > existing.score)
                .unwrap_or(true);
            if should_replace {
                best.insert(fact.key.clone(), fact);
            }
        }
    }
    best
}

fn has_required_family_age_groups(facts: &HashMap<String, FamilyAgeFact>) -> bool {
    let self_count = facts
        .values()
        .filter(|fact| fact.group == FamilyAgeGroup::SelfPerson)
        .count();
    let parent_count = facts
        .values()
        .filter(|fact| fact.group == FamilyAgeGroup::Parent)
        .count();
    let grandparent_count = facts
        .values()
        .filter(|fact| fact.group == FamilyAgeGroup::Grandparent)
        .count();
    self_count >= 1 && parent_count >= 2 && grandparent_count >= 2
}
