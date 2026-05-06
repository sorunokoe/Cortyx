use super::numeric_delta_extractors::format_numeric_delta;
use super::*;

#[derive(Clone, Debug, PartialEq)]
pub(super) enum AverageValueQuery {
    AcademicGpa(AcademicGpaQuery),
    FamilyAge(FamilyAgeQuery),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct AcademicGpaQuery {
    pub(super) required_terms: Vec<String>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct FamilyAgeQuery {
    pub(super) required_terms: Vec<String>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum AcademicStageKind {
    Undergraduate,
    Graduate,
}

#[derive(Clone, Debug, PartialEq)]
pub(super) struct AcademicGpaFact {
    pub(super) value: f64,
    pub(super) stage: AcademicStageKind,
    pub(super) score: usize,
    pub(super) evidence: String,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum FamilyAgeGroup {
    SelfPerson,
    Parent,
    Grandparent,
}

#[derive(Clone, Debug, PartialEq)]
pub(super) struct FamilyAgeFact {
    pub(super) key: String,
    pub(super) value: f64,
    pub(super) group: FamilyAgeGroup,
    pub(super) score: usize,
    pub(super) evidence: String,
}

pub(super) fn parse_average_value_query(task_lower: &str) -> Option<AverageValueQuery> {
    parse_academic_gpa_average_query(task_lower)
        .map(AverageValueQuery::AcademicGpa)
        .or_else(|| parse_family_age_average_query(task_lower).map(AverageValueQuery::FamilyAge))
}

pub(super) fn extract_academic_gpa_facts_from_line(
    line: &str,
    lower: &str,
) -> Vec<AcademicGpaFact> {
    if !lower.starts_with("user:") || !lower.contains("gpa") {
        return Vec::new();
    }
    let Some(value) = extract_gpa_from_line(line) else {
        return Vec::new();
    };
    let stage = academic_stage_for_line(lower);
    vec![AcademicGpaFact {
        value,
        stage,
        score: usize::from(lower.contains("gpa")) * 10
            + usize::from(lower.contains("master")) * 8
            + usize::from(has_standalone_graduate_keyword(lower)) * 6
            + usize::from(lower.contains("bachelor")) * 6
            + usize::from(lower.contains("graduated")) * 4
            + 8,
        evidence: line.trim().to_string(),
    }]
}

pub(super) fn extract_family_age_facts_from_line(line: &str, lower: &str) -> Vec<FamilyAgeFact> {
    if !lower.starts_with("user:") {
        return Vec::new();
    }
    let mut facts = Vec::new();
    if let Some(value) = extract_self_age_from_line(line) {
        facts.push(FamilyAgeFact {
            key: "self".to_string(),
            value,
            group: FamilyAgeGroup::SelfPerson,
            score: usize::from(lower.contains("turned")) * 10
                + usize::from(lower.contains("years old")) * 8
                + 10,
            evidence: line.trim().to_string(),
        });
    }
    extend_family_age_facts(
        &mut facts,
        line,
        lower,
        FamilyAgeGroup::Parent,
        &[
            ("mom", r"(?i)\bmy mom is (\d{1,2})\b"),
            ("dad", r"(?i)\bmy dad is (\d{1,2})\b"),
        ],
    );
    extend_family_age_facts(
        &mut facts,
        line,
        lower,
        FamilyAgeGroup::Grandparent,
        &[
            ("grandma", r"(?i)\bmy grandma is (\d{1,2})\b"),
            ("grandpa", r"(?i)\bmy grandpa is (\d{1,2})\b"),
        ],
    );
    facts
}

pub(super) fn format_average_value(value: f64) -> String {
    format_numeric_delta(((value * 100.0).round() / 100.0).max(0.0))
}

fn parse_academic_gpa_average_query(task_lower: &str) -> Option<AcademicGpaQuery> {
    if !task_lower.contains("average gpa")
        || !task_lower.contains("undergraduate")
        || !task_lower.contains("graduate")
    {
        return None;
    }
    Some(AcademicGpaQuery {
        required_terms: vec![
            "gpa".to_string(),
            "graduate".to_string(),
            "undergraduate".to_string(),
        ],
    })
}

fn parse_family_age_average_query(task_lower: &str) -> Option<FamilyAgeQuery> {
    task_lower
        .contains("average age of me, my parents, and my grandparents")
        .then_some(FamilyAgeQuery {
            required_terms: vec![
                "age".to_string(),
                "parents".to_string(),
                "grandparents".to_string(),
            ],
        })
}

fn academic_stage_for_line(lower: &str) -> AcademicStageKind {
    if lower.contains("master") || has_standalone_graduate_keyword(lower) {
        AcademicStageKind::Graduate
    } else {
        AcademicStageKind::Undergraduate
    }
}

fn has_standalone_graduate_keyword(lower: &str) -> bool {
    compile_regex(r"(?i)\bgraduate\b").is_match(lower)
}

fn extract_gpa_from_line(line: &str) -> Option<f64> {
    compile_regex(r"(?i)\bGPA of (\d+(?:\.\d+)?)\s+out of 4\.0\b")
        .captures(line)
        .and_then(|captures| captures.get(1))
        .and_then(|value| value.as_str().parse::<f64>().ok())
}

fn extract_self_age_from_line(line: &str) -> Option<f64> {
    compile_regex(
        r"(?i)\b(?:i(?:'m| am) currently|i just turned|i'm currently)\s+(\d{1,2})(?:\s+years old)?\b",
    )
    .captures(line)
    .and_then(|captures| captures.get(1))
    .and_then(|value| value.as_str().parse::<f64>().ok())
}

fn extend_family_age_facts(
    facts: &mut Vec<FamilyAgeFact>,
    line: &str,
    lower: &str,
    group: FamilyAgeGroup,
    patterns: &[(&str, &str)],
) {
    for (key, pattern) in patterns {
        let Some(value) = compile_regex(pattern)
            .captures(line)
            .and_then(|captures| captures.get(1))
            .and_then(|raw| raw.as_str().parse::<f64>().ok())
        else {
            continue;
        };
        facts.push(FamilyAgeFact {
            key: (*key).to_string(),
            value,
            group,
            score: usize::from(lower.contains(*key)) * 10 + 8,
            evidence: line.trim().to_string(),
        });
    }
}
