use super::count_support::SignatureDetail;
use super::*;

const MONTH_NAMES: [(&str, u8); 12] = [
    ("january", 1),
    ("february", 2),
    ("march", 3),
    ("april", 4),
    ("may", 5),
    ("june", 6),
    ("july", 7),
    ("august", 8),
    ("september", 9),
    ("october", 10),
    ("november", 11),
    ("december", 12),
];

const AGE_PROFILE_STOP: &[&str] = &[
    "about",
    "background",
    "college",
    "completed",
    "considering",
    "course",
    "courses",
    "current",
    "currently",
    "degree",
    "experience",
    "future",
    "going",
    "help",
    "interested",
    "looking",
    "master",
    "masters",
    "program",
    "programs",
    "school",
    "suitable",
    "university",
    "updated",
    "while",
    "wondering",
    "work",
    "working",
    "year",
    "years",
];

pub(super) fn extract_wedding_attendance_details(line: &str) -> Vec<SignatureDetail> {
    let mut details = Vec::new();
    let mut seen = HashSet::new();
    let mut push = |key: &str, display: &str| {
        let normalized_key = normalized_synthetic_phrase_key(key);
        if !normalized_key.is_empty() && seen.insert(normalized_key.clone()) {
            details.push(SignatureDetail::new(normalized_key, display.trim()));
        }
    };

    if let Some((name, partner)) = compile_regex(
        r"\b(?:the|The)\s+bride,\s*([A-Z][a-z]+),.*?\b(?:husband|wife|partner),\s*([A-Z][a-z]+)\b",
    )
    .captures(line)
    .and_then(|captures| Some((captures.get(1)?.as_str(), captures.get(2)?.as_str())))
    {
        push(name, &format!("{name} and {partner}"));
    }
    if let Some((name, partner)) = compile_regex(
        r"\b([A-Z][a-z]+)\s+(?:finally\s+)?(?:got to\s+)?tie the knot with\s+(?:her|his|their)\s+(?:partner|husband|wife)\s+([A-Z][a-z]+)\b",
    )
    .captures(line)
    .and_then(|captures| Some((captures.get(1)?.as_str(), captures.get(2)?.as_str())))
    {
        push(name, &format!("{name} and {partner}"));
    }
    if let Some(name) = compile_regex(
        r"\b(?:my|My)\s+(?:friend|cousin|roommate|college roommate)\s+([A-Z][a-z]+)\s+got married\b",
    )
    .captures(line)
    .and_then(|captures| captures.get(1).map(|m| m.as_str()))
    {
        push(name, name);
    }
    if let Some(name) = compile_regex(r"\b([A-Z][a-z]+)'s wedding\b")
        .captures(line)
        .and_then(|captures| captures.get(1).map(|m| m.as_str()))
    {
        push(name, name);
    }

    details
}

pub(super) fn is_attended_wedding_line(lower: &str) -> bool {
    task_contains_any(lower, &["wedding", "married", "tie the knot"])
        && task_contains_any(
            lower,
            &[
                "'s wedding",
                "got back from",
                "been to",
                "bridesmaid",
                "tie the knot",
            ],
        )
}

pub(super) fn extract_rollercoaster_event_quantities(
    line: &str,
    lower: &str,
    month_range: (u8, u8),
) -> Vec<(String, usize)> {
    if !lower.starts_with("user:") || !lower.contains("rode") {
        return Vec::new();
    }
    let Some(month) =
        extract_first_month_number(lower).filter(|month| month_in_range(*month, month_range))
    else {
        return Vec::new();
    };

    let key = format!("{month}:{}", normalized_synthetic_phrase_key(line));
    if let Some(count) = count_listed_rollercoasters(line) {
        return vec![(key, count)];
    }
    if let Some(count) = extract_explicit_ride_count(line) {
        return vec![(key, count)];
    }
    if task_contains_any(lower, &["rollercoaster", "roller coaster", "coaster"]) {
        return vec![(key, 1)];
    }
    Vec::new()
}

pub(super) fn extract_query_month_range(task_lower: &str) -> Option<(u8, u8)> {
    let months = MONTH_NAMES
        .iter()
        .filter_map(|(name, value)| task_lower.contains(name).then_some(*value))
        .collect::<Vec<_>>();
    if months.len() >= 2 {
        Some((months[0], months[1]))
    } else {
        None
    }
}

pub(super) fn extract_education_completion_age_from_line(line: &str) -> Option<i32> {
    let lower = line.to_ascii_lowercase();
    if !task_contains_any(
        &lower,
        &[
            "degree",
            "college",
            "university",
            "graduated",
            "completed",
            "bachelor",
        ],
    ) {
        return None;
    }
    compile_regex(
        r"(?i)\b(?:completed|graduated|finished|earned)[^.]{0,80}?\bat (?:the )?age of (\d{1,2})\b",
    )
    .captures(line)
    .and_then(|captures| captures.get(1))
    .and_then(|value| value.as_str().parse::<i32>().ok())
}

pub(super) fn extract_current_age_from_line(line: &str) -> Option<i32> {
    compile_regex(r"(?i)\b(?:i am|i'm|im)\s+(?:currently\s+)?(\d{1,2})\s+years old\b")
        .captures(line)
        .and_then(|captures| captures.get(1))
        .and_then(|value| value.as_str().parse::<i32>().ok())
}

pub(super) fn best_education_completion_age(lines: &[String]) -> Option<(i32, usize, String)> {
    best_explicit_age_fact(lines, extract_education_completion_age_from_line, |lower| {
        usize::from(task_contains_any(
            lower,
            &["bachelor", "degree", "university"],
        )) * 10
            + usize::from(lower.contains("completed")) * 4
            + usize::from(lower.contains("age")) * 2
    })
}

pub(super) fn best_current_age_fact(lines: &[String]) -> Option<(i32, usize, String)> {
    best_explicit_age_fact(lines, extract_current_age_from_line, |lower| {
        usize::from(lower.contains("currently")) * 10
            + usize::from(lower.contains("years old")) * 6
            + usize::from(lower.starts_with("user:")) * 2
    })
}

pub(super) fn extract_age_delta_profile_terms(lines: &[String]) -> Vec<String> {
    let relevant = lines
        .iter()
        .filter_map(|line| {
            let lower = line.to_ascii_lowercase();
            task_contains_any(
                &lower,
                &[
                    "marketing",
                    "digital",
                    "content",
                    "leadership",
                    "career",
                    "industry",
                    "specialist",
                ],
            )
            .then_some(lower)
        })
        .collect::<Vec<_>>()
        .join(" ");
    let mut terms = synthetic_query_terms(&relevant);
    terms.retain(|term| term.len() >= 4 && !AGE_PROFILE_STOP.contains(&term.as_str()));
    terms.sort();
    terms.dedup();
    terms
}

pub(super) fn profile_overlap_count(left: &[String], right: &[String]) -> usize {
    let right_terms = right.iter().map(String::as_str).collect::<HashSet<_>>();
    left.iter()
        .filter(|term| right_terms.contains(term.as_str()))
        .count()
}

fn count_listed_rollercoasters(line: &str) -> Option<usize> {
    let captures =
        compile_regex(r"(?i)\brode\s+(?:the\s+)?(.+?)\s+rollercoasters?\b").captures(line)?;
    let count = captures
        .get(1)?
        .as_str()
        .replace(", and ", ",")
        .replace(" and ", ",")
        .split(',')
        .map(|item| {
            item.trim()
                .trim_start_matches("the ")
                .trim_matches(|c: char| {
                    !c.is_ascii_alphanumeric() && c != ':' && c != '\'' && c != '-'
                })
        })
        .filter(|item| !item.is_empty())
        .count();
    (count > 1).then_some(count)
}

fn extract_explicit_ride_count(line: &str) -> Option<usize> {
    compile_regex(
        r"(?i)\brode\b.+?\b(once|twice|thrice|one|two|three|four|five|six|seven|eight|nine|ten|\d+)\s+times?\b",
    )
    .captures(line)
    .and_then(|captures| captures.get(1))
    .and_then(|value| parse_count_token_value(value.as_str()))
    .and_then(|value| usize::try_from(value).ok())
}

fn extract_first_month_number(lower: &str) -> Option<u8> {
    MONTH_NAMES
        .iter()
        .find_map(|(name, value)| lower.contains(name).then_some(*value))
}

fn month_in_range(month: u8, range: (u8, u8)) -> bool {
    if range.0 <= range.1 {
        (range.0..=range.1).contains(&month)
    } else {
        month >= range.0 || month <= range.1
    }
}

fn best_explicit_age_fact<FExtract, FScore>(
    lines: &[String],
    extract: FExtract,
    score_line: FScore,
) -> Option<(i32, usize, String)>
where
    FExtract: Fn(&str) -> Option<i32>,
    FScore: Fn(&str) -> usize,
{
    lines.iter().fold(None, |best, line| {
        let lower = line.to_ascii_lowercase();
        let Some(value) = extract(line) else {
            return best;
        };
        let score = score_line(&lower);
        let candidate = (value, score, line.clone());
        let should_replace = best
            .as_ref()
            .map(|(_, best_score, _)| score > *best_score)
            .unwrap_or(true);
        should_replace.then_some(candidate).or(best)
    })
}
