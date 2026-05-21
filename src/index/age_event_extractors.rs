use super::*;

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) enum AgeEventQuery {
    OlderThanMe(AgeSubjectQuery),
    MyAgeWhenNamedPersonWasBorn(AgeSubjectQuery),
    MyAgeWhenNamedPersonGetsMarried(AgeSubjectQuery),
    NamedPersonAgeWhenIGetMarried(AgeSubjectQuery),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct AgeSubjectQuery {
    pub display_name: String,
    pub subject_terms: Vec<String>,
}

pub(super) fn parse_age_event_query(task: &str, task_lower: &str) -> Option<AgeEventQuery> {
    if let Some(subject) =
        extract_subject_between(task, task_lower, &["how many years older is "], " than me")
    {
        return Some(AgeEventQuery::OlderThanMe(build_age_subject_query(
            &subject,
        )?));
    }

    if let Some(subject) =
        extract_subject_between(task, task_lower, &["how old was i when "], " was born")
    {
        return Some(AgeEventQuery::MyAgeWhenNamedPersonWasBorn(
            build_age_subject_query(&subject)?,
        ));
    }

    if let Some(subject) = extract_subject_between(
        task,
        task_lower,
        &["how many years will i be when ", "how old will i be when "],
        " gets married",
    ) {
        return Some(AgeEventQuery::MyAgeWhenNamedPersonGetsMarried(
            build_age_subject_query(&subject)?,
        ));
    }

    if let Some(subject) = extract_subject_between(
        task,
        task_lower,
        &["how old will ", "what age will "],
        " be when i get married",
    ) {
        return Some(AgeEventQuery::NamedPersonAgeWhenIGetMarried(
            build_age_subject_query(&subject)?,
        ));
    }

    None
}

pub(super) fn extract_current_user_age(line: &str) -> Option<i32> {
    extract_age_with_patterns(
        &line.to_ascii_lowercase(),
        &[
            r"\bi just turned\s+(\d{1,3})\b",
            r"\bi turned\s+(\d{1,3})\b",
            r"\bi'm\s+(\d{1,3})\b",
            r"\bi am\s+(\d{1,3})\b",
            r"\byou(?:'re| are)\s+(?:currently\s+)?(\d{1,3})\b",
            r"\bcurrent age[:\s]+(\d{1,3})\b",
        ],
    )
}

pub(super) fn extract_named_person_age(line: &str, subject_terms: &[String]) -> Option<i32> {
    let lower = line.to_ascii_lowercase();
    subject_line_matches(&lower, subject_terms).then_some(())?;
    extract_age_with_patterns(
        &lower,
        &[
            r"\bturned\s+(\d{1,3})\b",
            r"\bis\s+(?:just\s+)?(\d{1,3})\b",
            r"\b(?:he's|he is|she's|she is|who's|who is|they're|they are)\s+(?:just\s+)?(\d{1,3})\b",
            r"\b(\d{1,3})-year-old\b",
            r"\b(\d{1,3})\s+years old\b",
        ],
    )
}

pub(super) fn extract_named_marriage_year_offset(
    line: &str,
    subject_terms: &[String],
) -> Option<i32> {
    let lower = line.to_ascii_lowercase();
    if !subject_line_matches(&lower, subject_terms) || !line_mentions_marriage(&lower) {
        return None;
    }
    extract_year_offset(&lower)
}

pub(super) fn extract_first_person_marriage_year_offset(line: &str) -> Option<i32> {
    let lower = line.to_ascii_lowercase();
    if !line_mentions_first_person_marriage(&lower) {
        return None;
    }
    extract_year_offset(&lower)
}

fn build_age_subject_query(subject: &str) -> Option<AgeSubjectQuery> {
    let cleaned = subject
        .trim()
        .trim_matches(|c: char| !c.is_alphanumeric() && c != '\'' && c != ' ')
        .trim();
    if cleaned.is_empty() {
        return None;
    }

    let stop_words = ["my", "friend", "the", "a", "an"];
    let words: Vec<&str> = cleaned.split_whitespace().collect();
    let meaningful_words: Vec<&str> = words
        .iter()
        .copied()
        .skip_while(|word| stop_words.contains(&word.to_ascii_lowercase().as_str()))
        .collect();

    let display_name = if meaningful_words.is_empty() {
        cleaned.to_string()
    } else {
        meaningful_words.join(" ")
    };

    let subject_terms: Vec<String> = display_name
        .split_whitespace()
        .map(|term| {
            term.trim_matches(|c: char| !c.is_alphanumeric() && c != '\'')
                .to_ascii_lowercase()
        })
        .filter(|term| term.len() >= 3 && !stop_words.contains(&term.as_str()))
        .collect();

    (!subject_terms.is_empty()).then_some(AgeSubjectQuery {
        display_name,
        subject_terms,
    })
}

fn extract_subject_between(
    task: &str,
    task_lower: &str,
    prefixes: &[&str],
    suffix: &str,
) -> Option<String> {
    for prefix in prefixes {
        let Some(start) = task_lower.find(prefix) else {
            continue;
        };
        let start = start + prefix.len();
        let tail_lower = &task_lower[start..];
        let Some(end) = tail_lower.find(suffix) else {
            continue;
        };
        let subject = task[start..start + end].trim();
        if !subject.is_empty() {
            return Some(subject.to_string());
        }
    }
    None
}

fn extract_age_with_patterns(lower: &str, patterns: &[&str]) -> Option<i32> {
    patterns.iter().find_map(|pattern| {
        compile_regex_static(pattern)
            .captures(lower)
            .and_then(|caps| caps.get(1))
            .and_then(|m| m.as_str().parse::<i32>().ok())
            .filter(|value| (1..=120).contains(value))
    })
}

fn extract_year_offset(lower: &str) -> Option<i32> {
    if lower.contains("next year") {
        return Some(1);
    }
    if lower.contains("this year") || lower.contains("later this year") {
        return Some(0);
    }

    compile_regex_static(r"\bin\s+(\d{1,2})\s+years?\b")
        .captures(lower)
        .and_then(|caps| caps.get(1))
        .and_then(|m| m.as_str().parse::<i32>().ok())
        .or_else(|| {
            compile_regex_static(r"\b(\d{1,2})\s+years?\s+from now\b")
                .captures(lower)
                .and_then(|caps| caps.get(1))
                .and_then(|m| m.as_str().parse::<i32>().ok())
        })
}

fn line_mentions_marriage(lower: &str) -> bool {
    task_contains_any(lower, &["married", "marriage", "wedding"])
}

pub(super) fn line_mentions_first_person_marriage(lower: &str) -> bool {
    line_mentions_marriage(lower)
        && task_contains_any(
            lower,
            &[
                "i'm getting married",
                "i am getting married",
                "my wedding",
                "our wedding",
            ],
        )
}

fn subject_line_matches(lower: &str, subject_terms: &[String]) -> bool {
    subject_terms
        .iter()
        .any(|term| !term.is_empty() && lower.contains(term))
}
