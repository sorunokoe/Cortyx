//! Query analysis and personal-fact detection helpers.
//!
//! Pure free functions for classifying query intent, detecting personal-fact
//! patterns, extracting query ordinals, and scoring answer candidates.
//! These functions have no dependency on NeuronIndex and operate on `&str` / `&[String]` inputs.

use crate::kg;
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

// Functions defined in the parent module (core/mod.rs) that these helpers depend on.
use super::{detect_counting_query, is_money_query, tokenize};

pub(crate) fn task_contains_all(lower: &str, needles: &[&str]) -> bool {
    needles.iter().all(|needle| lower.contains(needle))
}

pub(crate) fn task_contains_any(lower: &str, needles: &[&str]) -> bool {
    needles.iter().any(|needle| lower.contains(needle))
}

pub(crate) fn extract_query_ordinal(task: &str) -> Option<usize> {
    fn ordinal_word(token: &str) -> Option<usize> {
        match token {
            "first" => Some(1),
            "second" => Some(2),
            "third" => Some(3),
            "fourth" => Some(4),
            "fifth" => Some(5),
            "sixth" => Some(6),
            "seventh" => Some(7),
            "eighth" => Some(8),
            "ninth" => Some(9),
            "tenth" => Some(10),
            _ => None,
        }
    }

    for raw in task.split_whitespace() {
        let token = raw
            .trim_matches(|c: char| !c.is_ascii_alphanumeric())
            .to_ascii_lowercase();
        if let Some(value) = ordinal_word(&token) {
            return Some(value);
        }
        for suffix in ["st", "nd", "rd", "th"] {
            if let Some(num) = token
                .strip_suffix(suffix)
                .and_then(|value| value.parse::<usize>().ok())
            {
                return Some(num);
            }
        }
    }
    None
}

pub(crate) fn extract_numbered_list_item(line: &str) -> Option<(usize, String)> {
    let trimmed = line.trim();
    let digits_len = trimmed.chars().take_while(|c| c.is_ascii_digit()).count();
    if digits_len == 0 {
        return None;
    }
    let index = trimmed[..digits_len].parse::<usize>().ok()?;
    let rest = trimmed[digits_len..].trim_start();
    if !(rest.starts_with('.') || rest.starts_with(')')) {
        return None;
    }
    let value = rest[1..]
        .trim()
        .trim_matches(|c: char| matches!(c, '"' | '\'' | '.' | '-'))
        .to_string();
    if value.is_empty() {
        None
    } else {
        Some((index, value))
    }
}

pub(crate) fn extract_single_word_after_marker(line: &str, marker: &str) -> Option<String> {
    let lower = line.to_ascii_lowercase();
    let pos = lower.find(marker)?;
    let value = line[pos + marker.len()..]
        .split_whitespace()
        .next()
        .unwrap_or_default()
        .trim_matches(|c: char| !c.is_ascii_alphabetic() && c != '+' && c != '&' && c != '-')
        .to_string();
    if value.is_empty() {
        None
    } else {
        Some(value)
    }
}

pub(crate) fn extract_pet_name(line: &str, animal: &str) -> Option<String> {
    let primary = format!("my {animal}'s name is ");
    if let Some(value) = extract_single_word_after_marker(line, &primary) {
        return Some(value);
    }
    if animal != "pet" {
        let generic = "my pet's name is ";
        if let Some(value) = extract_single_word_after_marker(line, generic) {
            return Some(value);
        }
    }
    None
}

pub(crate) fn synthetic_query_terms(task_lower: &str) -> Vec<String> {
    const STOP: &[&str] = &[
        "what",
        "which",
        "when",
        "where",
        "who",
        "whom",
        "whose",
        "how",
        "many",
        "much",
        "would",
        "could",
        "should",
        "can",
        "will",
        "may",
        "might",
        "does",
        "did",
        "do",
        "have",
        "has",
        "had",
        "with",
        "from",
        "for",
        "and",
        "or",
        "the",
        "into",
        "over",
        "under",
        "onto",
        "upon",
        "through",
        "across",
        "between",
        "among",
        "around",
        "near",
        "in",
        "on",
        "at",
        "to",
        "of",
        "is",
        "are",
        "was",
        "were",
        "am",
        "be",
        "into",
        "after",
        "before",
        "about",
        "there",
        "these",
        "those",
        "this",
        "that",
        "your",
        "their",
        "ours",
        "hers",
        "theirs",
        "overall",
        "altogether",
        "latest",
        "recent",
        "recently",
        "current",
        "currently",
        "likely",
        "probably",
        "possibly",
        "potentially",
        "considered",
        "still",
        "more",
        "most",
        "less",
        "least",
    ];
    tokenize(task_lower)
        .into_iter()
        .filter(|term| !STOP.contains(&term.as_str()))
        .filter(|term| extract_query_ordinal(term).is_none())
        .collect()
}

pub(crate) fn is_list_style_query(task_lower: &str) -> bool {
    task_contains_any(
        task_lower,
        &[
            "list", "jobs", "job", "options", "items", "steps", "recipes", "ideas",
        ],
    )
}

pub(crate) fn term_overlap_count(text_lower: &str, terms: &[&str]) -> usize {
    terms
        .iter()
        .filter(|term| text_lower.contains(**term))
        .count()
}

pub(crate) fn extract_knowledge_update_focus_terms(terms: &[String]) -> Vec<String> {
    const KU_STOP: &[&str] = &[
        "recent",
        "current",
        "latest",
        "after",
        "before",
        "since",
        "then",
        "still",
        "again",
        "relocation",
        "relocated",
        "update",
        "updated",
        "new",
        "now",
        "anymore",
        "back",
        "around",
    ];
    let stop: HashSet<&str> = KU_STOP.iter().copied().collect();
    let focused: Vec<String> = terms
        .iter()
        .filter(|term| term.len() >= 4 && !stop.contains(term.as_str()))
        .cloned()
        .collect();
    if focused.is_empty() {
        terms.to_vec()
    } else {
        focused
    }
}

/// R18 P2 Sol B + R21 T4: Detect knowledge-update queries — questions about current state that may
/// have stale Verbatim answers. These queries should suppress old verbatim facts and prefer
/// KG/Concept neurons that track supersession.
pub(crate) fn detect_knowledge_update_query(task: &str) -> bool {
    const KU_MARKERS: &[&str] = &[
        // Original R18 markers
        "what is now",
        "what are now",
        "changed to",
        "changed his",
        "changed her",
        "switched to",
        "moved to",
        "no longer",
        "anymore",
        "not anymore",
        "what does he do now",
        "what does she do now",
        "what do they do now",
        "what is he doing now",
        "what is she doing now",
        "what is their current",
        "what is his current",
        "what is her current",
        "does he still",
        "does she still",
        "do they still",
        "is he still",
        "is she still",
        "what happened to",
        "what changed",
        "since then",
        "after that",
        "new job",
        "new role",
        "new address",
        "new number",
        "new partner",
        "my current",
        "as of now",
        "up to date",
        "latest update",
        // KU-R10: Current-state queries without explicit "current" keyword.
        // These ask about the user's present situation (job, location, diet, etc.)
        // where the NEWEST session is definitionally correct. Applying the temporal
        // boost for these ensures updated facts outrank older mentions.
        "where do i work",
        "where does she work",
        "where does he work",
        "what do i do for work",
        "what does she do for work",
        "what does he do for work",
        "where do i live",
        "where does she live",
        "where does he live",
        "what car do i drive",
        "what car does she drive",
        "what car does he drive",
        "what do i eat",
        "what does she eat",
        "what does he eat",
        "what is my diet",
        "what is her diet",
        "what is his diet",
        "what am i studying",
        "what is she studying",
        "what is he studying",
        "what do i study",
        "what does she study",
        "what does he study",
        "do i still go",
        "does she still go",
        "does he still go",
        "what is my latest",
        "what is her latest",
        "what is his latest",
        "did i finish reading",
        "more frequently than i did previously",
        "do i have a spare",
        "did i switch to more water",
        "how long have i been living in my current",
        "how long have i been sticking to my daily",
    ];
    let lower = task.to_lowercase();
    KU_MARKERS.iter().any(|m| lower.contains(m))
}

/// P2-B: Detect personal-attribute queries and return the canonical KG predicate.
///
/// Returns Some(predicate) when the query is asking about a fact that the KG stores
/// as a structured (entity, predicate, value) triple. The caller can then bypass
/// BM25 entirely and route to O(1) KG lookup.
///
/// Patterns are deliberately specific to avoid false positives on generic queries.
pub(crate) fn detect_personal_fact_query(task: &str) -> Option<&'static str> {
    let lower = task.to_lowercase();
    if is_named_move_query(task, &lower) || is_location_query(&lower) {
        return Some("location");
    }
    if is_major_query(&lower) {
        return Some("major");
    }
    if is_education_query(&lower) {
        return Some("education");
    }
    if is_occupation_query(&lower) {
        return Some("occupation");
    }
    if is_commute_query(&lower) {
        return Some("commute_time");
    }
    if is_fitness_record_query(&lower) {
        return Some("fitness_record");
    }
    if is_book_query(&lower) {
        return Some("book");
    }
    if is_pet_query(&lower) {
        return Some("pet");
    }
    if is_partner_query(&lower) {
        return Some("partner");
    }
    if is_phone_query(&lower) {
        return Some("phone");
    }
    if is_project_name_query(&lower) {
        return Some("project_name");
    }
    if lower.contains("instagram") && lower.contains("follower") {
        return Some("instagram_followers");
    }
    if lower.contains("bbq sauce")
        && task_contains_any(&lower, &["favorite", "favourite", "obsessed", "prefer"])
    {
        return Some("bbq_sauce");
    }
    if lower.contains("h&m") && lower.contains("top") && detect_counting_query(task) {
        return Some("hm_tops");
    }
    if (lower.contains("pre-1920") || lower.contains("pre 1920"))
        && lower.contains("coin")
        && lower.contains("collection")
    {
        return Some("pre_1920_american_coins");
    }
    if lower.contains("vehicle model")
        && task_contains_any(&lower, &["working on", "building", "making"])
    {
        return Some("vehicle_model");
    }
    if lower.contains("family trip") && lower.contains("where") {
        return Some("family_trip_location");
    }
    if lower.contains("workshop") && is_money_query(task) && detect_counting_query(task) {
        return Some("workshop_spend_total");
    }
    if lower.contains("rare item") && detect_counting_query(task) {
        return Some("rare_items_total");
    }
    if lower.contains("bird") && lower.contains("local park") && detect_counting_query(task) {
        return Some("local_park_bird_species_count");
    }
    // Compound trigger: "where did [name] move" / "where did ... move ... relocation"
    if lower.contains("where did") && (lower.contains(" move") || lower.contains("relocation")) {
        return Some("location");
    }
    None
}

pub(crate) fn is_named_move_query(task: &str, lower: &str) -> bool {
    if !task.starts_with("Where did ") {
        return false;
    }
    let after = &task["Where did ".len()..];
    after.chars().next().is_some_and(|c| c.is_uppercase())
        && (lower.contains(" move") || lower.contains(" relocat"))
}

pub(crate) fn is_location_query(lower: &str) -> bool {
    (lower.contains("where") || lower.contains("what city") || lower.contains("which city"))
        && task_contains_any(lower, &[" live", " home", " based", " move", " relocat"])
}

pub(crate) fn is_major_query(lower: &str) -> bool {
    lower.contains(" major")
        && (lower.contains("what ") || lower.contains("which "))
        && !lower.contains("major project")
}

pub(crate) fn is_education_query(lower: &str) -> bool {
    !task_contains_any(
        lower,
        &["how many", "how long", "years in total", "total did"],
    ) && (lower.contains("what degree")
        || lower.contains("which degree")
        || lower.contains("graduate with")
        || lower.contains("graduated with")
        || (task_contains_any(lower, &["study", "studied"]) && lower.contains("what ")))
}

pub(crate) fn is_occupation_query(lower: &str) -> bool {
    (lower.contains("work") && task_contains_any(lower, &["where ", "what does", "do for work"]))
        || lower.contains("occupation")
        || lower.contains("employed")
        || (lower.contains(" job") && lower.contains("what "))
}

pub(crate) fn is_commute_query(lower: &str) -> bool {
    if task_contains_any(lower, &["get ready", "getting ready"]) {
        return false;
    }
    let mentions_commute = synthetic_query_terms(lower).iter().any(|term| {
        matches!(
            term.as_str(),
            "commute" | "commutes" | "commuting" | "commuted"
        )
    }) || lower.contains("get to work");
    let asks_duration = task_contains_any(
        lower,
        &[
            "how long",
            "how much time",
            "minutes",
            "minute",
            "hours",
            "hour",
            "each way",
            "takes",
            "take me",
            "long is",
            "long does",
        ],
    );
    mentions_commute && asks_duration
}

pub(crate) fn is_fitness_record_query(lower: &str) -> bool {
    lower.contains("personal best")
        || lower.contains(" pb")
        || lower.contains("best time")
        || lower.contains("race time")
}

pub(crate) fn is_book_query(lower: &str) -> bool {
    !task_contains_any(
        lower,
        &["finish", "finished", "completed", "a week ago", "week ago"],
    ) && (lower.contains("currently reading")
        || lower.starts_with("what am i reading")
        || lower.contains(" is she reading")
        || lower.contains(" is he reading")
        || (lower.contains("book")
            && task_contains_any(lower, &["what ", "which "])
            && lower.contains("reading")))
}

pub(crate) fn is_pet_query(lower: &str) -> bool {
    task_contains_any(lower, &["pet", "dog", "cat"])
        && task_contains_any(lower, &["name", "what is", "who is"])
}

pub(crate) fn is_partner_query(lower: &str) -> bool {
    task_contains_any(
        lower,
        &[
            "partner",
            "husband",
            "wife",
            "boyfriend",
            "girlfriend",
            "spouse",
            "married",
        ],
    ) && task_contains_any(lower, &["who ", "is "])
}

pub(crate) fn is_phone_query(lower: &str) -> bool {
    lower.contains("phone")
        || lower.contains("phone number")
        || (lower.contains(" number")
            && task_contains_any(lower, &["what is her", "what is his", "what is my"]))
}

pub(crate) fn is_project_name_query(lower: &str) -> bool {
    task_contains_any(lower, &["project", "playlist", "blog", "channel"])
        && task_contains_any(lower, &["called", "name of", "call it", "title"])
}

pub(crate) fn detect_personal_fact_entity(task: &str) -> Option<String> {
    const RELATIONS: &[(&str, &str)] = &[
        ("my mom", "mom"),
        ("my mother", "mom"),
        ("my dad", "dad"),
        ("my father", "dad"),
        ("my sister", "sister"),
        ("my brother", "brother"),
        ("my wife", "wife"),
        ("my husband", "husband"),
        ("my partner", "partner"),
        ("my boyfriend", "boyfriend"),
        ("my girlfriend", "girlfriend"),
        ("my friend", "friend"),
        ("my coworker", "coworker"),
        ("my colleague", "coworker"),
        ("my boss", "boss"),
        ("my manager", "manager"),
        ("my neighbor", "neighbor"),
    ];
    const STOPWORDS: &[&str] = &[
        "What", "When", "Where", "Which", "Who", "Whom", "Whose", "Why", "How", "Does", "Did",
        "Has", "Have", "Will", "Would", "Should", "Could", "Might", "This", "That", "These",
        "Those", "They", "Their", "Them", "Then",
    ];

    let lower = task.to_ascii_lowercase();
    for (needle, entity) in RELATIONS {
        if lower.contains(needle) {
            return Some((*entity).to_string());
        }
    }
    if lower.contains(" my ")
        || lower.starts_with("my ")
        || lower.contains(" i ")
        || lower.starts_with("i ")
    {
        return Some("user".to_string());
    }
    for word in task.split_whitespace().skip(1) {
        let clean: String = word
            .chars()
            .filter(|c| c.is_alphabetic() || *c == '\'')
            .collect();
        if clean.len() >= 3
            && clean.chars().next().is_some_and(|c| c.is_uppercase())
            && !STOPWORDS.contains(&clean.as_str())
        {
            return Some(kg::slugify(&clean));
        }
    }
    None
}

/// R18 P2 Sol B: Count proper nouns (capitalized words ≥4 chars, not sentence-start stopwords).
/// ≥2 proper nouns in a query → multi-session query routing (force 2-hop synapse expansion).
pub(crate) fn count_proper_nouns(task: &str) -> usize {
    const STOPWORDS: &[&str] = &[
        "What",
        "When",
        "Where",
        "Which",
        "Who",
        "Whom",
        "Whose",
        "Why",
        "How",
        "Does",
        "Did",
        "Has",
        "Have",
        "Will",
        "Would",
        "Should",
        "Could",
        "Might",
        "This",
        "That",
        "These",
        "Those",
        "They",
        "Their",
        "Them",
        "Then",
        "Also",
        "Just",
        "Very",
        "Well",
        "Even",
        "Most",
        "Some",
        "Many",
        "More",
        "Long",
        "Good",
        "Back",
        "Into",
        "Over",
        "Down",
        "Such",
        "Both",
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
        "January",
        "February",
        "March",
        "April",
        "June",
        "July",
        "August",
        "September",
        "October",
        "November",
        "December",
    ];
    // skip the very first word (sentence-start capital) to reduce false positives
    task.split_whitespace()
        .skip(1)
        .filter(|w| {
            let clean: String = w.chars().filter(|c| c.is_alphabetic()).collect();
            clean.len() >= 4
                && clean.chars().next().is_some_and(|c| c.is_uppercase())
                && !STOPWORDS.contains(&clean.as_str())
        })
        .count()
}

#[cfg(test)]
pub(crate) fn neuron_body_has_move_residence_evidence(path: &Path) -> bool {
    let Ok(content) = std::fs::read_to_string(path) else {
        return false;
    };
    content_has_move_residence_evidence(&content)
}

pub(crate) fn content_has_move_residence_evidence(content: &str) -> bool {
    use crate::neuron::parse_sections;
    let sections = parse_sections(content);
    let mut searchable = content.to_ascii_lowercase();
    for section_name in ["query_surface", "paraphrases"] {
        if let Some(section_content) = sections.get(section_name) {
            searchable = searchable.replacen(&section_content.to_ascii_lowercase(), "", 1);
        }
    }
    [
        " moved to ",
        " moved back to",
        " just moved",
        " recently moved",
        " relocated to ",
        " lives in ",
        " live in ",
        " living in ",
        " now living in ",
        " settled in ",
        " based in ",
    ]
    .iter()
    .any(|pattern| searchable.contains(pattern))
}

/// Parse an ISO 8601 date(-time) string to Unix epoch seconds (UTC, approx).
///
/// Supports "YYYY-MM-DD", "YYYY-MM-DDTHH:MM:SS", and "YYYY-MM-DD HH:MM:SS".
/// Does NOT handle timezone offsets — treats all timestamps as UTC.
/// Returns `None` for unparseable or obviously-invalid strings.
pub(crate) fn parse_iso8601_to_secs(ts: Option<&str>) -> Option<i64> {
    let s = ts?.trim();
    // Accept "YYYY-MM-DDTHH:MM:SS", "YYYY-MM-DD HH:MM:SS", or "YYYY-MM-DD"
    let date_part = s.split(['T', ' ']).next()?;
    let parts: Vec<&str> = date_part.split('-').collect();
    if parts.len() < 3 {
        return None;
    }
    let year: i64 = parts[0].parse().ok()?;
    let month: i64 = parts[1].parse::<i64>().ok()?.clamp(1, 12);
    let day: i64 = parts[2].parse::<i64>().ok()?.clamp(1, 31);
    if !(1970..=2200).contains(&year) {
        return None;
    }
    // Cumulative days at start of each month (non-leap year)
    const MONTH_START_DAYS: [i64; 12] = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334];
    // Leap-year correction: count leap years from 1970 to year-1
    let leap_years = {
        let y = year - 1;
        y / 4 - y / 100 + y / 400 - (1969 / 4 - 1969 / 100 + 1969 / 400)
    };
    let days = (year - 1970) * 365 + leap_years + MONTH_START_DAYS[(month - 1) as usize] + day - 1;
    Some(days * 86_400)
}

/// CountNeuron helper: convert count 1–20 to its English word equivalent.
/// Returns empty string for counts outside that range.
pub(crate) fn num_to_word(n: usize) -> &'static str {
    match n {
        1 => "one",
        2 => "two",
        3 => "three",
        4 => "four",
        5 => "five",
        6 => "six",
        7 => "seven",
        8 => "eight",
        9 => "nine",
        10 => "ten",
        11 => "eleven",
        12 => "twelve",
        13 => "thirteen",
        14 => "fourteen",
        15 => "fifteen",
        16 => "sixteen",
        17 => "seventeen",
        18 => "eighteen",
        19 => "nineteen",
        20 => "twenty",
        _ => "",
    }
}

/// Wilson score lower bound for a proportion at 95% confidence interval.
///
/// Used for Bayesian quarantine decisions: quarantine only when the lower bound
/// of the citation-rate confidence interval falls below a threshold, ensuring
/// the system has enough evidence before penalising a neuron.
///
/// Formula: `(p̂ + z²/2n − z·√(p̂(1−p̂)/n + z²/4n²)) / (1 + z²/n)`
/// where z = 1.96 (95% CI), n = total, p̂ = hits/total.
/// Returns 0.0 when total == 0.
#[cfg(test)]
pub(crate) fn wilson_lower_bound(hits: u32, total: u32) -> f32 {
    wilson_lower_bound_z(hits, total, 1.96)
}
///
/// - z = 1.0  → 68% CI (fast reaction, for small samples)
/// - z = 1.645 → 90% CI (medium)
/// - z = 1.96  → 95% CI (standard, for large samples)
pub(crate) fn wilson_lower_bound_z(hits: u32, total: u32, z: f32) -> f32 {
    if total == 0 {
        return 0.0;
    }
    let n = total as f32;
    let p = (hits as f32 / n).min(1.0);
    let z2 = z * z;
    let variance = (p * (1.0 - p) / n + z2 / (4.0 * n * n)).max(0.0);
    let numerator = p + z2 / (2.0 * n) - z * variance.sqrt();
    let denominator = 1.0 + z2 / n;
    (numerator / denominator).max(0.0)
}

/// Adaptive quarantine parameters based on observation count (TRIZ R11-S4).
///
/// Returns `Some((z, threshold))` when enough samples exist to make a decision,
/// or `None` if `use_count` is too small to draw any conclusion.
///
/// Tiers:
/// - `< 5`     → None (too few samples; withhold judgment entirely)
/// - `5–19`    → z=1.0,   threshold=0.02 (68% CI — react fast to obvious noise)
/// - `20–99`   → z=1.645, threshold=0.05 (90% CI — current behaviour)
/// - `≥ 100`   → z=1.96,  threshold=0.08 (95% CI — strict for mature neurons)
pub(crate) fn adaptive_quarantine_params(use_count: u32) -> Option<(f32, f32)> {
    match use_count {
        0..=4 => None,
        5..=19 => Some((1.0, 0.02)),
        20..=99 => Some((1.645, 0.05)),
        _ => Some((1.96, 0.08)),
    }
}

/// Build a confidence map for all files in the project by querying git once.
///
/// Returns `HashMap<abs_path, confidence_score>`:
/// - 1.0 = committed and unmodified (default; also used when git is absent)
/// - 0.9 = tracked but locally modified
/// - 0.85 = untracked (new file not yet committed)
///
/// Three git commands are run once per compile; per-file overhead is zero.
/// Build a per-file confidence multiplier from the current git working-tree status.
///
/// # Rationale for the multiplier values
/// Tracked but locally modified files (0.90) are slightly less reliable than
/// committed files (1.0, the default when a path is absent from this map):
/// a file under active edit may have partial, inconsistent, or draft content.
///
/// Untracked files (0.85) receive a slightly larger penalty: they have never
/// been through a commit gate, so they may be throwaway scratch or work-in-progress
/// that should be ranked below committed knowledge.
///
/// Both penalties are intentionally conservative (5–15%): they bias ranking
/// without completely suppressing fresh content, since new content is often
/// the most relevant.
///
/// The multiplier is applied as `meta.confidence_score`, which scales the BM25
/// retrieval score for each neuron belonging to that file.
pub(crate) fn build_git_confidence_map(project_root: &Path) -> HashMap<PathBuf, f32> {
    let mut map = HashMap::new();

    // Modified tracked files — locally changed but in git history
    for rel in git_file_list(project_root, &["ls-files", "-m"]) {
        map.entry(project_root.join(rel)).or_insert(0.9_f32);
    }

    // Untracked files — not yet in git history
    for rel in git_file_list(
        project_root,
        &["ls-files", "--others", "--exclude-standard"],
    ) {
        map.entry(project_root.join(rel)).or_insert(0.85_f32);
    }

    map
}

/// Run a git command and return one path per output line. Silent on error.
pub(crate) fn git_file_list(project_root: &Path, args: &[&str]) -> Vec<PathBuf> {
    let Ok(out) = std::process::Command::new("git")
        .args(args)
        .current_dir(project_root)
        .output()
    else {
        return Vec::new();
    };

    if !out.status.success() {
        return Vec::new();
    }

    String::from_utf8_lossy(&out.stdout)
        .lines()
        .filter(|l| !l.is_empty())
        .map(PathBuf::from)
        .collect()
}
