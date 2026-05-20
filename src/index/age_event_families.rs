use super::age_event_extractors::{
    extract_current_user_age, extract_first_person_marriage_year_offset,
    extract_named_marriage_year_offset, extract_named_person_age,
    line_mentions_first_person_marriage, parse_age_event_query, AgeEventQuery, AgeSubjectQuery,
};
use super::*;

impl NeuronIndex {
    pub(super) fn synthetic_age_event_answer(
        &self,
        task: &str,
        task_lower: &str,
    ) -> Option<PathBuf> {
        let query = parse_age_event_query(task, task_lower)?;
        match query {
            AgeEventQuery::OlderThanMe(subject) => {
                self.synthetic_older_than_me_age_answer(task, &subject)
            },
            AgeEventQuery::MyAgeWhenNamedPersonWasBorn(subject) => {
                self.synthetic_age_when_named_person_was_born_answer(task, &subject)
            },
            AgeEventQuery::MyAgeWhenNamedPersonGetsMarried(subject) => {
                self.synthetic_age_when_named_person_gets_married_answer(task, &subject)
            },
            AgeEventQuery::NamedPersonAgeWhenIGetMarried(subject) => {
                self.synthetic_named_person_age_when_i_get_married_answer(task, &subject)
            },
        }
    }

    fn synthetic_older_than_me_age_answer(
        &self,
        task: &str,
        subject: &AgeSubjectQuery,
    ) -> Option<PathBuf> {
        let facts = select_age_event_facts(
            self,
            subject,
            &AgeEventNeed::UserAndSubjectAge,
            &["birthday", "turned"],
        );
        let (user_age, subject_age) = (facts.user_age?, facts.subject_age?);
        (subject_age > user_age).then_some(())?;
        self.write_synthetic_answer(
            "age-gap-older-than-me",
            task,
            &(subject_age - user_age).to_string(),
            &facts.evidence,
        )
    }

    fn synthetic_age_when_named_person_was_born_answer(
        &self,
        task: &str,
        subject: &AgeSubjectQuery,
    ) -> Option<PathBuf> {
        let facts = select_age_event_facts(self, subject, &AgeEventNeed::UserAndSubjectAge, &[]);
        let (user_age, subject_age) = (facts.user_age?, facts.subject_age?);
        (user_age >= subject_age).then_some(())?;
        self.write_synthetic_answer(
            "age-when-named-person-was-born",
            task,
            &(user_age - subject_age).to_string(),
            &facts.evidence,
        )
    }

    fn synthetic_age_when_named_person_gets_married_answer(
        &self,
        task: &str,
        subject: &AgeSubjectQuery,
    ) -> Option<PathBuf> {
        let facts = select_age_event_facts(
            self,
            subject,
            &AgeEventNeed::UserAgeAndNamedMarriage,
            &["married", "wedding"],
        );
        let (user_age, year_offset) = (facts.user_age?, facts.named_marriage_offset?);
        self.write_synthetic_answer(
            "age-when-named-person-gets-married",
            task,
            &(user_age + year_offset).to_string(),
            &facts.evidence,
        )
    }

    fn synthetic_named_person_age_when_i_get_married_answer(
        &self,
        task: &str,
        subject: &AgeSubjectQuery,
    ) -> Option<PathBuf> {
        let facts = select_age_event_facts(
            self,
            subject,
            &AgeEventNeed::SubjectAgeAndUserMarriage,
            &["married", "wedding"],
        );
        if let (Some(subject_age), Some(year_offset)) =
            (facts.subject_age, facts.user_marriage_offset)
        {
            return self.write_synthetic_answer(
                "named-person-age-when-i-get-married",
                task,
                &(subject_age + year_offset).to_string(),
                &facts.evidence,
            );
        }

        (!facts.evidence.is_empty()).then_some(())?;
        self.write_synthetic_answer(
            "missing-named-person-age-at-my-marriage",
            task,
            &format!(
                "The information provided is not enough. You did not mention how old {} is right now, nor when will you get married.",
                subject.display_name
            ),
            &facts.evidence,
        )
    }
}

#[derive(Clone, Copy)]
enum AgeEventNeed {
    UserAndSubjectAge,
    UserAgeAndNamedMarriage,
    SubjectAgeAndUserMarriage,
}

#[derive(Default, Clone)]
struct AgeEventFacts {
    user_age: Option<i32>,
    subject_age: Option<i32>,
    named_marriage_offset: Option<i32>,
    user_marriage_offset: Option<i32>,
    evidence: Vec<String>,
}

fn select_age_event_facts(
    idx: &NeuronIndex,
    subject: &AgeSubjectQuery,
    need: &AgeEventNeed,
    extra_terms: &[&str],
) -> AgeEventFacts {
    let search_terms = build_age_event_search_terms(subject, extra_terms);
    let borrowed_terms: Vec<&str> = search_terms.iter().map(String::as_str).collect();

    let mut merged = AgeEventFacts::default();
    let mut best_complete: Option<AgeEventFacts> = None;

    for (_, content) in idx.matching_verbatim_texts(&borrowed_terms, idx.retrieval.entries.len()) {
        let facts = collect_age_event_facts(&content, subject);
        merge_age_event_facts(&mut merged, &facts);
        if age_event_completion_score(&facts, need) == required_age_event_score(need)
            && best_complete.as_ref().is_none_or(|best| {
                age_event_candidate_rank(&facts, need) > age_event_candidate_rank(best, need)
            })
        {
            best_complete = Some(facts);
        }
    }

    best_complete.unwrap_or(merged)
}

fn build_age_event_search_terms(subject: &AgeSubjectQuery, extra_terms: &[&str]) -> Vec<String> {
    let mut terms = subject.subject_terms.clone();
    for extra in extra_terms {
        if !terms.iter().any(|term| term == extra) {
            terms.push((*extra).to_string());
        }
    }
    terms
}

fn collect_age_event_facts(content: &str, subject: &AgeSubjectQuery) -> AgeEventFacts {
    let mut facts = AgeEventFacts::default();
    for raw_line in content.lines() {
        let line = raw_line.trim();
        if line.is_empty() {
            continue;
        }
        let lower = line.to_ascii_lowercase();

        if subject
            .subject_terms
            .iter()
            .any(|term| !term.is_empty() && lower.contains(term))
            || line_mentions_first_person_marriage(&lower)
        {
            push_unique(&mut facts.evidence, line);
        }

        if let Some(value) = extract_current_user_age(line) {
            facts.user_age = Some(facts.user_age.unwrap_or(value));
            push_unique(&mut facts.evidence, line);
        }
        if let Some(value) = extract_named_person_age(line, &subject.subject_terms) {
            facts.subject_age = Some(facts.subject_age.unwrap_or(value));
            push_unique(&mut facts.evidence, line);
        }
        if let Some(value) = extract_named_marriage_year_offset(line, &subject.subject_terms) {
            facts.named_marriage_offset = Some(facts.named_marriage_offset.unwrap_or(value));
            push_unique(&mut facts.evidence, line);
        }
        if let Some(value) = extract_first_person_marriage_year_offset(line) {
            facts.user_marriage_offset = Some(facts.user_marriage_offset.unwrap_or(value));
            push_unique(&mut facts.evidence, line);
        }
    }
    facts
}

fn merge_age_event_facts(target: &mut AgeEventFacts, source: &AgeEventFacts) {
    if target.user_age.is_none() {
        target.user_age = source.user_age;
    }
    if target.subject_age.is_none() {
        target.subject_age = source.subject_age;
    }
    if target.named_marriage_offset.is_none() {
        target.named_marriage_offset = source.named_marriage_offset;
    }
    if target.user_marriage_offset.is_none() {
        target.user_marriage_offset = source.user_marriage_offset;
    }
    for line in &source.evidence {
        push_unique(&mut target.evidence, line);
    }
}

fn age_event_completion_score(facts: &AgeEventFacts, need: &AgeEventNeed) -> usize {
    match need {
        AgeEventNeed::UserAndSubjectAge => {
            usize::from(facts.user_age.is_some()) + usize::from(facts.subject_age.is_some())
        },
        AgeEventNeed::UserAgeAndNamedMarriage => {
            usize::from(facts.user_age.is_some())
                + usize::from(facts.named_marriage_offset.is_some())
        },
        AgeEventNeed::SubjectAgeAndUserMarriage => {
            usize::from(facts.subject_age.is_some())
                + usize::from(facts.user_marriage_offset.is_some())
        },
    }
}

fn required_age_event_score(need: &AgeEventNeed) -> usize {
    match need {
        AgeEventNeed::UserAndSubjectAge
        | AgeEventNeed::UserAgeAndNamedMarriage
        | AgeEventNeed::SubjectAgeAndUserMarriage => 2,
    }
}

fn age_event_candidate_rank(facts: &AgeEventFacts, need: &AgeEventNeed) -> (usize, usize) {
    (
        age_event_completion_score(facts, need),
        facts.evidence.len(),
    )
}

fn push_unique(lines: &mut Vec<String>, line: &str) {
    if !lines.iter().any(|existing| existing == line) {
        lines.push(line.to_string());
    }
}
