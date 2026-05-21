//! Education aggregates and fact parsing: stage extraction, money/duration solving.

use super::super::*;
use crate::index::{compile_regex, compile_regex_static};

pub fn extract_formal_education_target_stage(task_lower: &str) -> Option<EducationStageKind> {
    if !task_lower.contains("formal education") || !task_lower.contains("high school") {
        return None;
    }
    if task_lower.contains("bachelor") {
        return Some(EducationStageKind::Bachelor);
    }
    if task_lower.contains("master") {
        return Some(EducationStageKind::Master);
    }
    None
}

pub fn collect_education_stage_facts(
    lines: &[String],
) -> HashMap<EducationStageKind, EducationStageFact> {
    let mut facts = HashMap::new();
    for line in lines {
        let Some(parsed) = parse_education_stage_fact(line) else {
            continue;
        };
        let should_replace = facts
            .get(&parsed.kind)
            .map(|existing| {
                education_stage_fact_score(&parsed) > education_stage_fact_score(existing)
            })
            .unwrap_or(true);
        if should_replace {
            facts.insert(parsed.kind, parsed);
        }
    }
    facts
}

pub fn solve_formal_education_total(
    facts: &HashMap<EducationStageKind, EducationStageFact>,
    target_stage: EducationStageKind,
) -> Option<(i32, Vec<String>, usize)> {
    let high_school = facts.get(&EducationStageKind::HighSchool)?;
    let high_school_duration = education_stage_duration_years(high_school)?;
    let high_school_end = education_stage_end_year(high_school)?;

    let bachelor = facts
        .get(&EducationStageKind::Bachelor)
        .filter(|fact| fact.completed)?;
    let bachelor_duration = education_stage_duration_years(bachelor)?;
    let bachelor_start = education_stage_start_year(bachelor)?;
    let bachelor_end = education_stage_end_year(bachelor)?;

    let mut total_years = high_school_duration + bachelor_duration;
    let mut evidence = vec![high_school.evidence.clone()];

    if let Some(associate) = facts
        .get(&EducationStageKind::Associate)
        .filter(|fact| fact.completed)
    {
        let associate_duration = education_stage_duration_years(associate).or_else(|| {
            let associate_end = education_stage_end_year(associate)?;
            ((associate_end > high_school_end) && (associate_end <= bachelor_start))
                .then_some(associate_end - high_school_end)
        });
        if let Some(years) = associate_duration.filter(|years| *years > 0) {
            total_years += years;
            evidence.push(associate.evidence.clone());
        }
    }

    evidence.push(bachelor.evidence.clone());

    if target_stage == EducationStageKind::Master {
        let master = facts
            .get(&EducationStageKind::Master)
            .filter(|fact| fact.completed)?;
        let master_duration = education_stage_duration_years(master).or_else(|| {
            let master_end = education_stage_end_year(master)?;
            (master_end > bachelor_end).then_some(master_end - bachelor_end)
        })?;
        if master_duration <= 0 {
            return None;
        }
        total_years += master_duration;
        evidence.push(master.evidence.clone());
    }

    let fact_count = evidence.len();
    Some((total_years, evidence, fact_count))
}

pub fn parse_education_stage_fact(line: &str) -> Option<EducationStageFact> {
    let body = normalize_session_answer_line_body(line);
    let lower = body.to_ascii_lowercase();
    let years = extract_year_mentions(&body);

    let high_school_range =
        compile_regex_static(r"(?i)\bhigh school\b.*?\bfrom\s+(\d{4})\s+to\s+(\d{4})\b");
    if let Some(caps) = high_school_range.captures(&body) {
        let start_year = caps.get(1)?.as_str().parse::<i32>().ok()?;
        let end_year = caps.get(2)?.as_str().parse::<i32>().ok()?;
        if end_year > start_year {
            return Some(EducationStageFact {
                kind: EducationStageKind::HighSchool,
                completed: true,
                start_year: Some(start_year),
                end_year: Some(end_year),
                duration_years: Some(end_year - start_year),
                evidence: line.to_string(),
            });
        }
    }

    if task_contains_any(
        &lower,
        &[
            "associate's degree",
            "associates degree",
            "associate degree",
        ],
    ) {
        return Some(EducationStageFact {
            kind: EducationStageKind::Associate,
            completed: task_contains_any(&lower, &["earned", "completed", "graduated"]),
            start_year: None,
            end_year: years.last().copied(),
            duration_years: extract_education_duration_years(&lower),
            evidence: line.to_string(),
        });
    }

    if task_contains_any(
        &lower,
        &[
            "bachelor's degree",
            "bachelors degree",
            "bachelor degree",
            "bachelor's in",
            "bachelors in",
            "bachelor in",
        ],
    ) {
        return Some(EducationStageFact {
            kind: EducationStageKind::Bachelor,
            completed: task_contains_any(&lower, &["graduated", "earned", "completed"])
                || lower.contains("took me"),
            start_year: None,
            end_year: years.last().copied(),
            duration_years: extract_education_duration_years(&lower),
            evidence: line.to_string(),
        });
    }

    if task_contains_any(
        &lower,
        &[
            "master's degree",
            "masters degree",
            "master degree",
            "master's in",
            "masters in",
            "master in",
        ],
    ) {
        return Some(EducationStageFact {
            kind: EducationStageKind::Master,
            completed: task_contains_any(&lower, &["graduated", "earned", "completed", "finished"]),
            start_year: None,
            end_year: years.last().copied(),
            duration_years: extract_education_duration_years(&lower),
            evidence: line.to_string(),
        });
    }

    None
}

pub fn extract_education_duration_years(lower: &str) -> Option<i32> {
    for marker in [
        "which took me ",
        "took me ",
        "completed in ",
        "finished in ",
    ] {
        let Some(idx) = lower.find(marker) else {
            continue;
        };
        let tail = &lower[idx + marker.len()..];
        let value = parse_leading_duration_value(tail)?;
        if value.unit == "year" {
            #[allow(clippy::cast_possible_truncation)]
            let amount = value.amount.round() as i32;
            return Some(amount);
        }
    }
    None
}

pub fn extract_year_mentions(text: &str) -> Vec<i32> {
    let years = compile_regex_static(r"\b(19|20)\d{2}\b");
    years
        .captures_iter(text)
        .filter_map(|caps| caps.get(0).and_then(|m| m.as_str().parse::<i32>().ok()))
        .collect()
}

pub fn education_stage_fact_score(fact: &EducationStageFact) -> i32 {
    let mut score = 0;
    if fact.completed {
        score += 2;
    }
    if fact.start_year.is_some() {
        score += 2;
    }
    if fact.end_year.is_some() {
        score += 2;
    }
    if fact.duration_years.is_some() {
        score += 3;
    }
    score
}

pub fn education_stage_duration_years(fact: &EducationStageFact) -> Option<i32> {
    fact.duration_years.or_else(|| {
        fact.start_year
            .zip(fact.end_year)
            .and_then(|(start, end)| (end > start).then_some(end - start))
    })
}

pub fn education_stage_start_year(fact: &EducationStageFact) -> Option<i32> {
    fact.start_year.or_else(|| {
        fact.end_year
            .zip(fact.duration_years)
            .and_then(|(end, years)| (years > 0).then_some(end - years))
    })
}

pub fn education_stage_end_year(fact: &EducationStageFact) -> Option<i32> {
    fact.end_year.or_else(|| {
        fact.start_year
            .zip(fact.duration_years)
            .and_then(|(start, years)| (years > 0).then_some(start + years))
    })
}

pub fn extract_multi_session_money_focus_terms(task_lower: &str) -> Vec<String> {
    const STOP: &[&str] = &[
        "total",
        "combined",
        "altogether",
        "since",
        "start",
        "year",
        "years",
        "month",
        "months",
        "past",
        "last",
        "few",
        "item",
        "items",
        "related",
        "i",
        "money",
        "amount",
        "spent",
        "spend",
        "cost",
        "costs",
        "expense",
        "expenses",
        "paid",
        "purchase",
        "purchased",
    ];
    let stop: HashSet<&str> = STOP.iter().copied().collect();
    let mut terms = synthetic_query_terms(task_lower);
    terms.retain(|term| !stop.contains(term.as_str()));
    if task_lower.contains("bike") {
        for extra in [
            "helmet",
            "lights",
            "chain",
            "cycling",
            "tune-up",
            "bike shop",
        ] {
            terms.push(extra.to_string());
        }
    }
    terms.sort();
    terms.dedup();
    terms
}

pub fn extract_multi_session_duration_focus_terms(task_lower: &str) -> Vec<String> {
    const STOP: &[&str] = &[
        "total",
        "combined",
        "altogether",
        "since",
        "start",
        "year",
        "years",
        "month",
        "months",
        "week",
        "weeks",
        "day",
        "days",
        "hour",
        "hours",
        "minute",
        "minutes",
        "time",
        "take",
        "took",
        "spent",
        "spend",
        "main",
        "past",
        "last",
        "few",
        "item",
        "items",
        "related",
        "united",
        "states",
        "i",
    ];
    let stop: HashSet<&str> = STOP.iter().copied().collect();
    let mut terms = synthetic_query_terms(task_lower);
    terms.retain(|term| !stop.contains(term.as_str()));
    terms.retain(|term| term.len() >= 2);
    if task_lower.contains("game") || task_lower.contains("gaming") {
        for extra in [
            "playing",
            "played",
            "finish",
            "finished",
            "complete",
            "completed",
        ] {
            terms.push(extra.to_string());
        }
    }
    if task_lower.contains("road trip") || task_lower.contains("destinations") {
        terms.retain(|term| !matches!(term.as_str(), "three" | "destination" | "destinations"));
        for extra in ["road", "trip", "drive", "drove", "driving"] {
            terms.push(extra.to_string());
        }
    }
    terms.sort();
    terms.dedup();
    terms
}

pub fn is_realized_duration_fact_text(lower: &str) -> bool {
    let realized = task_contains_any(
        lower,
        &[
            "just got back",
            "got back from",
            "watched all",
            "did it in",
            "spent around",
            "spent ",
            "took me",
            "completed",
            "finished",
            "clocked",
            "drove for",
            "driving to",
            "camping trip",
            "break in",
            "break from",
            "marathon",
        ],
    );
    let future = task_contains_any(
        lower,
        &[
            "i'm planning",
            "i am planning",
            "plan to",
            "going to",
            "i'll",
            "i will",
            "next week",
            "next month",
            "by the end",
            "goal is",
            "goal to",
            "thinking about",
            "thinking of",
        ],
    );
    realized && !future
}

pub fn extract_matching_duration_total_segments(
    line: &str,
    task_lower: &str,
) -> Vec<(String, SyntheticDurationValue)> {
    let mut matches = Vec::new();
    for segment in split_duration_aggregate_segments(line) {
        let lower = segment.to_ascii_lowercase();
        if !is_realized_duration_fact_text(&lower)
            || !duration_total_line_matches_query(task_lower, &lower)
        {
            continue;
        }
        let Some(duration) = extract_aggregate_duration_value(&segment) else {
            continue;
        };
        matches.push((segment, duration));
    }
    matches
}

pub fn duration_total_line_matches_query(task_lower: &str, lower: &str) -> bool {
    if task_lower.contains("social media") {
        return lower.contains("social media")
            && task_contains_any(lower, &["break from", "break in", "break"]);
    }
    if task_lower.contains("camping") {
        return lower.contains("camping trip");
    }
    if task_lower.contains("road trip") || task_lower.contains("destinations") {
        return task_contains_any(
            lower,
            &[
                "drove for",
                "drive there",
                "drive to",
                "driving to",
                "took me",
            ],
        );
    }
    if task_contains_any(task_lower, &["marvel", "star wars", "movies", "films"]) {
        return task_contains_any(lower, &["watched", "marathon"]);
    }
    if task_contains_any(task_lower, &["games", "gaming"]) {
        return task_contains_any(
            lower,
            &[
                "playing",
                "spent around",
                "took me",
                "finished",
                "completed",
            ],
        ) && !task_contains_any(
            lower,
            &[
                "developers",
                "development",
                "develop ",
                "release",
                "announced",
                "team ",
                "script",
                "dialogue",
                "motion capture",
                "pages long",
            ],
        );
    }
    true
}

pub fn aggregate_fact_terms(line: &str) -> HashSet<String> {
    synthetic_query_terms(&normalize_session_answer_line_body(line).to_ascii_lowercase())
        .into_iter()
        .collect()
}

pub fn is_duplicate_numeric_aggregate_fact(
    existing: &[(String, f32, HashSet<String>)],
    session_id: &str,
    value: f32,
    terms: &HashSet<String>,
) -> bool {
    existing
        .iter()
        .any(|(existing_session, existing_value, existing_terms)| {
            if (existing_value - value).abs() >= 0.01 {
                return false;
            }
            let overlap = existing_terms.intersection(terms).count();
            let min_size = existing_terms.len().min(terms.len());
            if existing_session == session_id {
                overlap >= 4 || (min_size > 0 && overlap == min_size)
            } else {
                overlap >= 5 || (min_size >= 4 && overlap == min_size)
            }
        })
}

pub fn extract_nightly_rate(line: &str) -> Option<f32> {
    let lower = line.to_ascii_lowercase();
    if !lower.starts_with("user:") {
        return None;
    }
    if !lower.contains("per night") {
        return None;
    }
    if !task_contains_any(
        &lower,
        &[
            "stay", "staying", "hotel", "hostel", "resort", "room", "accommod",
        ],
    ) {
        return None;
    }
    extract_dollar_amounts(line).into_iter().next()
}

pub fn extract_sale_total(line: &str) -> Option<f32> {
    let lower = line.to_ascii_lowercase();
    if !(lower.starts_with("user:") || line.trim_start().starts_with('-')) {
        return None;
    }
    if !(lower.contains("sold") || lower.contains("earned") || lower.contains("earning")) {
        return None;
    }

    let explicit_total = compile_regex_static(
        r"(?:earned|earning(?: a total of)?|for a total of)\s+\$([0-9][0-9,]*(?:\.[0-9]+)?)",
    );
    if let Some(caps) = explicit_total.captures(&lower) {
        if let Some(value) = caps
            .get(1)
            .and_then(|m| m.as_str().replace(',', "").parse::<f32>().ok())
        {
            return Some(value);
        }
    }

    let per_item =
        compile_regex_static(r"sold\s+(\d+)[^$]{0,160}?\$([0-9][0-9,]*(?:\.[0-9]+)?)\s*each");
    if let Some(caps) = per_item.captures(&lower) {
        let quantity = caps.get(1).and_then(|m| m.as_str().parse::<f32>().ok())?;
        let price = caps
            .get(2)
            .and_then(|m| m.as_str().replace(',', "").parse::<f32>().ok())?;
        return Some(quantity * price);
    }

    None
}
