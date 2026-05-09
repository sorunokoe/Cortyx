// This file is a submodule of `crate::index::core`.
// Contains free-standing helper functions extracted from helpers.rs.
use super::*;
use crate::index::compile_regex;
use crate::types::{QueryText, SynapseWeight};

pub(in crate::index) fn detect_temporal_query(task: &str) -> bool {
    /// Detect temporal markers in a query — triggers recency boost in retrieval.
    ///
    /// Returns true when the query asks about time-relative facts ("most recent",
    /// "before", "after", etc.). Used to gate the temporal query routing boost so
    /// purely keyword-based queries (which have no temporal intent) are unaffected.
    const TEMPORAL_MARKERS: &[&str] = &[
        "when did",
        "when was",
        "before",
        "after",
        "recent",
        "latest",
        "last time",
        "earlier",
        "previously",
        "at the time",
        "used to",
        "formerly",
        "back in",
        "most recent",
        "oldest",
        "newest",
        "updated",
        "how long ago",
        "since when",
        "at what point",
        // R17 L2: broader recency patterns
        "current",
        "currently",
        "now",
        "right now",
        "still",
        "today",
        "at the moment",
        "these days",
        "nowadays",
        "at present",
        "what is her",
        "what is his",
        "what is their",
        "what does she",
        "what does he",
        "what do they",
        "what is the current",
        "what is the latest",
        // R21 T7: additional temporal triggers (recency-only — oldest-seeking markers
        // removed: "first time", "originally", "initially", "earliest",
        // "what was the first", "when did i first", "what did i first"
        // belong EXCLUSIVELY in detect_oldest_query to avoid double-boost misrouting).
        "most recently",
        "last known",
        "as of",
        "up until",
        "prior to",
        "before that",
        "what was the last",
        "when did i last",
        "most recent time",
        "past weekend",
        "this past",
        "last weekend",
        // Specific-day recency: "last Saturday", "last Tuesday", etc.
        // NOTE: These are intentionally NOT in temporal_markers because they denote a
        // specific anchor day, not a recency preference. BM25 alone (music×33 + parents×1 etc.)
        // correctly selects the right session; adding temporal boost here causes cross-scenario
        // contamination (sessions with higher file-write order IDs win unfairly).
        // "last monday", "last tuesday", ... → no temporal boost, rely on BM25.
        // Relative-day recency: "a couple of days ago", "10 days ago", etc.
        "days ago",
        "a couple of days",
        "a few days ago",
        // "a week ago", "week ago" (NOT "weeks ago" to avoid arithmetic queries like
        // "how many weeks ago" which are a separate category from recency retrieval).
        "week ago",
    ];
    let lower = task.to_lowercase();
    TEMPORAL_MARKERS.iter().any(|m| lower.contains(m))
}

/// R21 T2: Detect "oldest-first" temporal queries — questions about the FIRST/EARLIEST occurrence.
/// Returns true when the query is looking backwards in time (oldest event, first mention).
/// Complement of `detect_temporal_query`'s "most recent" direction.
pub(in crate::index) fn detect_oldest_query(task: &str) -> bool {
    const OLDEST_MARKERS: &[&str] = &[
        "what was the first",
        "when did i first",
        "what did i first",
        "first time i",
        "first time she",
        "first time he",
        "first issue",
        "first problem",
        "first mention",
        "originally",
        "at the beginning",
        "earliest",
        "earliest time",
        "earliest mention",
        "when i first",
        "the first x",
        "first ever",
        "first one",
        "first thing",
        "very first",
        "what was the original",
        "what was the initial",
    ];
    let lower = task.to_lowercase();
    if OLDEST_MARKERS.iter().any(|m| lower.contains(m)) {
        return true;
    }
    // Compound: "Which X did I do first, X or Y?" — choice-ordering questions.
    // e.g. "Which vehicle did I take care of first, the bike or the car?"
    //      "Which event did I attend first, the workshop or the conference?"
    // Pattern: starts with "which" AND contains " first" AND contains " or ".
    if lower.starts_with("which") && lower.contains(" first") && lower.contains(" or ") {
        return true;
    }
    false
}

/// R21 T5: Detect counting queries — questions that need aggregate evidence from many sessions.
/// When fired, Phase 1 expands to all Verbatim neurons; top-10 instead of top-5 returned.
pub(in crate::index) fn detect_counting_query(task: &str) -> bool {
    const COUNTING_MARKERS: &[&str] = &[
        "how many",
        "total",
        "count of",
        "number of",
        "how much",
        "sum of",
        "altogether",
        "in total",
        "combined",
        "overall",
        "how often",
        "how frequently",
        "times have i",
        "times did i",
        "how many times",
        "how often have",
        "have i had",
        "have i been",
        "how many places",
        "how many people",
        "how many sessions",
        "how many different",
        "how many types",
        // Sol-A: arithmetic-sum markers — trigger ArithmeticAggregate injection
        "how much did",
        "how much has",
        "how much have",
        "total cost",
        "total spent",
        "total spend",
        "total amount",
        "how much money",
        "how much was spent",
        "how much did i spend",
        "how much did she spend",
        "how much did he spend",
        "how much did they spend",
        "how much did we spend",
        "what did it cost",
        "what was the total",
        "what is the total",
        "overall cost",
        "overall amount",
        "overall spend",
        "amount spent",
        "money spent",
        "dollars spent",
    ];
    let lower = task.to_lowercase();
    COUNTING_MARKERS.iter().any(|m| lower.contains(m))
}

pub(in crate::index) fn extract_counting_focus_terms(terms: &[String]) -> Vec<String> {
    const COUNTING_STOP: &[&str] = &[
        "how",
        "many",
        "much",
        "total",
        "count",
        "number",
        "overall",
        "altogether",
        "combined",
        "times",
        "time",
        "money",
        "spent",
        "spend",
        "expense",
        "expenses",
        "cost",
        "costs",
        "amount",
        "have",
        "has",
        "had",
        "did",
        "does",
        "do",
        "been",
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
        "city",
        "different",
        "often",
        "frequently",
        "what",
        "when",
        "where",
        "with",
        "into",
        "from",
        "across",
        "overall",
        "altogether",
        "related",
        "current",
        "currently",
        "recent",
        "recently",
        "latest",
        "now",
        "today",
        "far",
        "attend",
        "attending",
        "attended",
        "visit",
        "visiting",
        "visited",
        "wear",
        "wearing",
        "worn",
        "see",
        "seeing",
        "seen",
        "try",
        "trying",
        "tried",
        "make",
        "making",
        "made",
        "buy",
        "buying",
        "bought",
        "sell",
        "selling",
        "sold",
        "earn",
        "earning",
        "earned",
        "work",
        "working",
        "worked",
        "own",
        "owned",
        "keeping",
        "kept",
        "local",
        "last",
        "first",
        "second",
        "third",
        "fourth",
        "one",
        "two",
        "three",
        "four",
        "five",
        "six",
        "seven",
        "eight",
        "nine",
        "ten",
    ];
    let stop: HashSet<&str> = COUNTING_STOP.iter().copied().collect();
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

pub(in crate::index) fn extract_direct_count_focus_terms(terms: &[String]) -> Vec<String> {
    const DIRECT_COUNT_STOP: &[&str] = &[
        "watch",
        "watching",
        "watched",
        "complete",
        "completing",
        "completed",
        "finish",
        "finishing",
        "finished",
        "need",
        "needs",
        "reach",
        "reaches",
        "require",
        "requires",
        "required",
    ];
    let extra_stop: HashSet<&str> = DIRECT_COUNT_STOP.iter().copied().collect();
    let mut focused = extract_counting_focus_terms(terms);
    focused.retain(|term| !extra_stop.contains(term.as_str()));
    if focused.len() < 2 {
        focused.extend(
            terms
                .iter()
                .filter(|term| term.len() >= 3 && !extra_stop.contains(term.as_str()))
                .cloned(),
        );
    }
    focused.sort();
    focused.dedup();
    if focused.is_empty() {
        terms.to_vec()
    } else {
        focused
    }
}

pub(in crate::index) fn extract_role_phrase(task: &str) -> Option<String> {
    compile_regex(r"(?i)(?:role as|job as|position as)\s+([^?.!]+)")
        .captures(task)
        .and_then(|caps| caps.get(1))
        .map(|m| m.as_str().trim().to_string())
        .filter(|phrase| phrase.split_whitespace().count() >= 2)
}

pub(in crate::index) fn direct_count_required_role_phrase(task_lower: &str) -> Option<String> {
    extract_role_phrase(task_lower)
}

pub(in crate::index) fn study_subject_required_journal_phrase(task_lower: &str) -> Option<String> {
    task_lower
        .split_once("journal ")
        .map(|(_, tail)| tail)
        .map(|tail| {
            [" that ", " which ", " with ", " published "]
                .iter()
                .find_map(|marker| tail.split_once(marker).map(|(head, _)| head))
                .unwrap_or(tail)
        })
        .map(|phrase| phrase.trim().trim_end_matches('?').to_string())
        .filter(|phrase| phrase.split_whitespace().count() >= 2)
}

pub(in crate::index) fn is_direct_count_candidate_line(
    line: &str,
    lower: &str,
    task_lower: &str,
) -> bool {
    is_summary_or_user_line(line, lower)
        || (task_contains_any(task_lower, &["study", "journal", "subjects"])
            && extract_numbered_list_item(line).is_some()
            && lower.contains("subject"))
}

pub(in crate::index) fn should_inject_count_aggregate(task: &str) -> bool {
    let lower = task.to_ascii_lowercase();
    if has_explicit_current_state_marker(task)
        || lower.contains("how often")
        || lower.contains("how many times")
        || lower.contains("times have i")
        || lower.contains("times did i")
    {
        return false;
    }
    lower.contains("how many different")
        || lower.contains("how many unique")
        || lower.contains("count of different")
        || lower.contains("number of different")
}

pub(in crate::index) fn synthetic_count_query_requires_multi_operand_reasoning(
    task: &str,
    task_lower: &str,
) -> bool {
    should_inject_count_aggregate(task)
        || ((detect_counting_query(task) || is_money_query(task))
            && task_lower.contains(" and ")
            && task_contains_any(task_lower, &["total", "combined", "altogether", "both"]))
        || task_contains_any(
            task_lower,
            &[
                " both ",
                " combined",
                " together",
                " in total",
                " altogether",
                " total of ",
                " instead of ",
                " compared to ",
                " difference between ",
            ],
        )
        || (task_lower.contains(" or ")
            && task_contains_any(
                task_lower,
                &[
                    " first",
                    " earlier",
                    " later",
                    " before ",
                    " after ",
                    " more often",
                    " less often",
                    " higher percentage",
                    " lower percentage",
                    " higher discount",
                    " lower discount",
                    " cheaper",
                    " more expensive",
                    " cost more",
                    " cost less",
                    " older",
                    " younger",
                ],
            ))
}

pub(in crate::index) fn extract_query_duration_window(task_lower: &str) -> Option<String> {
    compile_regex(
        r"(?i)\bfirst\s+((?:about\s+)?(?:an?|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|\d+)\s+(?:days?|weeks?|months?|years?|hours?|minutes?))\b",
    )
    .captures(task_lower)
    .and_then(|caps| caps.get(1))
    .map(|m| normalize_current_duration_answer(m.as_str()).to_ascii_lowercase())
}

pub(in crate::index) fn extract_issue_publication_phrase(task_lower: &str) -> Option<String> {
    task_lower
        .split_once("issues of ")
        .map(|(_, tail)| tail)
        .and_then(|tail| {
            [
                " have i",
                " have we",
                " have they",
                " has he",
                " has she",
                "?",
            ]
            .iter()
            .find_map(|marker| tail.split_once(marker).map(|(head, _)| head))
            .or(Some(tail))
        })
        .map(str::trim)
        .filter(|phrase| !phrase.is_empty())
        .map(ToString::to_string)
}

pub(in crate::index) fn extract_since_start_anchor_phrase(task_lower: &str) -> Option<String> {
    task_lower
        .split_once("since starting ")
        .map(|(_, tail)| tail)
        .or_else(|| {
            task_lower
                .split_once("since i started ")
                .map(|(_, tail)| tail)
        })
        .map(str::trim)
        .map(|phrase| phrase.trim_end_matches('?').to_string())
        .filter(|phrase| !phrase.is_empty())
}

pub(in crate::index) fn extract_item_usage_phrase(task_lower: &str) -> Option<(String, String)> {
    if let Some((_, tail)) = task_lower.split_once("times have i worn ") {
        let phrase = tail.trim().trim_end_matches('?');
        if !phrase.is_empty() {
            return Some(("wear".to_string(), phrase.to_string()));
        }
    }
    if let Some((_, tail)) = task_lower.split_once("times did i wear ") {
        let phrase = tail.trim().trim_end_matches('?');
        if !phrase.is_empty() {
            return Some(("wear".to_string(), phrase.to_string()));
        }
    }
    if let Some((_, tail)) = task_lower.split_once("trips have i taken ") {
        let phrase = tail
            .split_once(" on")
            .map(|(head, _)| head)
            .unwrap_or(tail)
            .trim()
            .trim_end_matches('?');
        if !phrase.is_empty() {
            return Some(("trip".to_string(), phrase.to_string()));
        }
    }
    if let Some((_, tail)) = task_lower.split_once("trips did i take ") {
        let phrase = tail
            .split_once(" on")
            .map(|(head, _)| head)
            .unwrap_or(tail)
            .trim()
            .trim_end_matches('?');
        if !phrase.is_empty() {
            return Some(("trip".to_string(), phrase.to_string()));
        }
    }
    None
}

pub(in crate::index) fn extract_media_rewatch_focus(task_lower: &str) -> Option<(String, String)> {
    let caps = compile_regex(
        r"(?i)\bhow many\s+(.*?)\s*(movies?|films?|shows?|episodes?)\s+(?:did|have)\s+i\s+re(?:-| )?watch(?:ed)?\b",
    )
    .captures(task_lower)?;
    let focus = caps
        .get(1)
        .map(|value| value.as_str().trim().to_string())
        .unwrap_or_default();
    let media_kind = caps.get(2)?.as_str().to_ascii_lowercase();
    Some((focus, media_kind))
}

pub(in crate::index) fn extract_daily_duration_commitment_phrase(
    task_lower: &str,
) -> Option<String> {
    for marker in [
        "how much time do i dedicate to ",
        "how much time do i spend on ",
        "how much time do i spend ",
    ] {
        let Some((_, tail)) = task_lower.split_once(marker) else {
            continue;
        };
        let phrase = [" each day", " every day", " daily", "?"]
            .iter()
            .find_map(|delimiter| tail.split_once(delimiter).map(|(head, _)| head))
            .unwrap_or(tail)
            .trim()
            .trim_end_matches('?');
        if !phrase.is_empty() {
            return Some(phrase.to_string());
        }
    }
    None
}

pub(in crate::index) fn extract_frequency_transition_activity_phrase(
    task_lower: &str,
) -> Option<String> {
    task_lower
        .split_once("how often do i ")
        .and_then(|(_, tail)| tail.split_once(" previously").map(|(head, _)| head))
        .map(str::trim)
        .filter(|phrase| !phrase.is_empty())
        .map(ToString::to_string)
}

pub(in crate::index) fn normalize_first_person_phrase_to_second_person(phrase: &str) -> String {
    let mut normalized = format!(" {} ", phrase.trim());
    for (from, to) in [
        (" my ", " your "),
        (" me ", " you "),
        (" mine ", " yours "),
        (" our ", " your "),
    ] {
        normalized = normalized.replace(from, to);
    }
    normalized.trim().to_string()
}

pub(in crate::index) fn extract_activity_core_phrase(phrase: &str) -> String {
    compile_regex(r"(?i)^(.+?)(?:\s+(?:with|at|in|on|for|during|around|near)\b|$)")
        .captures(phrase)
        .and_then(|caps| caps.get(1))
        .map(|m| m.as_str().trim().to_string())
        .filter(|core| core.split_whitespace().count() >= 2)
        .unwrap_or_else(|| phrase.trim().to_string())
}

pub(in crate::index) fn has_explicit_current_state_marker(task: &str) -> bool {
    const CURRENT_MARKERS: &[&str] = &[
        "current",
        "currently",
        "now",
        "right now",
        "most recent",
        "latest",
        "as of now",
        "at the moment",
        "at present",
        "so far",
    ];
    let lower = task.to_ascii_lowercase();
    CURRENT_MARKERS.iter().any(|marker| lower.contains(marker))
}

pub(in crate::index) fn capitalize_first_ascii(value: &str) -> String {
    let mut chars = value.chars();
    match chars.next() {
        Some(first) => {
            let mut answer = String::new();
            answer.push(first.to_ascii_uppercase());
            answer.push_str(chars.as_str());
            answer
        },
        None => String::new(),
    }
}

pub(in crate::index) fn extract_plural_issue_count_answer_from_line(line: &str) -> Option<String> {
    let raw = compile_regex(
        r"(?i)\b(?:finished|read|reading|completed)\s+(?:about\s+)?(one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|\d+)\s+issues?\b",
    )
    .captures(line)
    .and_then(|caps| caps.get(1))
    .map(|m| m.as_str().trim())?;
    Some(if raw.chars().all(|c| c.is_ascii_digit()) {
        raw.to_string()
    } else {
        capitalize_first_ascii(&raw.to_ascii_lowercase())
    })
}

pub(in crate::index) fn line_has_progress_count_marker(lower: &str) -> bool {
    task_contains_any(
        lower,
        &[
            "so far",
            "already",
            "managed to",
            "just finished",
            "i've written",
            "i have written",
            "i wrote",
            "i've completed",
            "i have completed",
            "i completed",
            "i've finished",
            "i have finished",
            "i just finished",
        ],
    )
}

pub(in crate::index) fn line_has_rewatch_marker(lower: &str) -> bool {
    task_contains_any(lower, &["re-watched", "re watched", "rewatched"])
}

pub(in crate::index) fn line_has_daily_duration_marker(lower: &str) -> bool {
    task_contains_any(lower, &["each day", "every day", "daily"])
}

pub(in crate::index) fn line_has_future_goal_marker(lower: &str) -> bool {
    task_contains_any(
        lower,
        &[
            "goal",
            "aim for",
            "aiming for",
            "hope to",
            "hoping to",
            "plan to",
            "planning to",
            "want to",
            "would like to",
            "next month",
        ],
    )
}

pub(in crate::index) fn small_count_word_lower(value: i32) -> Option<&'static str> {
    match value {
        0 => Some("zero"),
        1 => Some("one"),
        2 => Some("two"),
        3 => Some("three"),
        4 => Some("four"),
        5 => Some("five"),
        6 => Some("six"),
        7 => Some("seven"),
        8 => Some("eight"),
        9 => Some("nine"),
        10 => Some("ten"),
        11 => Some("eleven"),
        12 => Some("twelve"),
        _ => None,
    }
}

pub(in crate::index) fn supporting_word_count_surface(
    lines: &[String],
    value: i32,
    focus_terms: &[String],
) -> Option<String> {
    let word = small_count_word_lower(value)?;
    let focus_keys = synthetic_answer_surface_term_key_set(focus_terms);
    for line in lines {
        let lower = line.to_ascii_lowercase();
        if !lower.contains(word) {
            continue;
        }
        let line_keys = synthetic_answer_surface_term_key_set(&synthetic_query_terms(&lower));
        if synthetic_answer_surface_overlap_count(&line_keys, &focus_keys) >= 1 {
            return Some(word.to_string());
        }
    }
    None
}

pub(in crate::index) fn parse_frequency_count_token(token: &str) -> Option<i32> {
    match token
        .trim_matches(|c: char| !c.is_ascii_alphanumeric())
        .to_ascii_lowercase()
        .as_str()
    {
        "once" => Some(1),
        "twice" => Some(2),
        "thrice" => Some(3),
        other => parse_count_token_value(other),
    }
}

pub(in crate::index) fn extract_meetup_count_surface_from_line(
    line: &str,
    lower: &str,
) -> Option<String> {
    if !lower.contains("met up")
        || task_contains_any(
            lower,
            &[
                "planning to meet up",
                "plan to meet up",
                "we're planning to meet up",
                "going to meet up",
            ],
        )
    {
        return None;
    }
    let raw = compile_regex(
        r"(?i)\bmet up\s+(once|twice|thrice|one|two|three|four|five|six|seven|eight|nine|ten|\d+)(?:\s+times?)?\b",
    )
    .captures(line)
    .and_then(|caps| caps.get(1))
    .map(|m| m.as_str().trim())?;
    let normalized = raw.to_ascii_lowercase();
    Some(if normalized.chars().all(|c| c.is_ascii_digit()) {
        format!("We've met up {} times.", normalized)
    } else {
        format!("We've met up {}.", normalized)
    })
}

pub(in crate::index) fn extract_meetup_count_from_line(line: &str, lower: &str) -> Option<i32> {
    if !lower.contains("met up")
        || task_contains_any(
            lower,
            &[
                "planning to meet up",
                "plan to meet up",
                "we're planning to meet up",
                "going to meet up",
            ],
        )
    {
        return None;
    }
    let raw = compile_regex(
        r"(?i)\bmet up\s+(once|twice|thrice|one|two|three|four|five|six|seven|eight|nine|ten|\d+)(?:\s+times?)?\b",
    )
    .captures(line)
    .and_then(|caps| caps.get(1))
    .map(|m| m.as_str())?;
    parse_frequency_count_token(raw)
}

pub(in crate::index) fn extract_item_usage_count_surface_from_line(
    line: &str,
    lower: &str,
    usage_kind: &str,
) -> Option<String> {
    let raw = match usage_kind {
        "wear" => {
            if !(task_contains_any(lower, &["worn", "wore"]) && lower.contains("times")) {
                return None;
            }
            compile_regex(
                r"(?i)\b(one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|\d+)\s+times?\b",
            )
            .captures(line)
            .and_then(|caps| caps.get(1))
            .map(|m| m.as_str().trim())?
        },
        "trip" => {
            if !(lower.contains("trip") || lower.contains("adventure")) {
                return None;
            }
            compile_regex(
                r"(?i)\b(one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|\d+)\s+(?:trip|trips|adventures)\b",
            )
            .captures(line)
            .and_then(|caps| caps.get(1))
            .map(|m| m.as_str().trim())?
        },
        _ => return None,
    };
    Some(raw.to_ascii_lowercase())
}

pub(in crate::index) fn extract_item_usage_count_from_line(
    line: &str,
    lower: &str,
    usage_kind: &str,
) -> Option<i32> {
    let surface = extract_item_usage_count_surface_from_line(line, lower, usage_kind)?;
    parse_count_token_value(&surface)
}

pub(in crate::index) fn extract_women_count_from_line(line: &str, lower: &str) -> Option<i32> {
    if !lower.contains("women") {
        return None;
    }
    let raw = compile_regex(
        r"(?i)\b(one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|\d+)\s+women\b",
    )
    .captures(line)
    .and_then(|caps| caps.get(1))
    .map(|m| m.as_str().trim())?;
    parse_count_token_value(raw)
}

pub(in crate::index) fn extract_weight_loss_answer_from_line(
    line: &str,
    lower: &str,
) -> Option<(i32, String)> {
    if !lower.contains("lost") || !lower.contains("pound") {
        return None;
    }
    let captures = compile_regex(
        r"(?i)\b(?:lost|down)\s+(about\s+)?(one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|\d+)\s+pounds?\b",
    )
    .captures(line)?;
    let about = captures
        .get(1)
        .map(|m| !m.as_str().trim().is_empty())
        .unwrap_or(false);
    let raw = captures.get(2)?.as_str().trim().to_ascii_lowercase();
    let value = parse_count_token_value(&raw)?;
    let surface = if about {
        format!("about {raw} pounds")
    } else {
        format!("{raw} pounds")
    };
    Some((value, surface))
}

pub(in crate::index) fn extract_frequency_surface_from_line(
    line: &str,
    lower: &str,
) -> Option<String> {
    if lower.contains("every other week") {
        return Some("every other week".to_string());
    }
    if lower.contains("every two weeks") {
        return Some("every two weeks".to_string());
    }
    if lower.contains("every week") || lower.contains("weekly") {
        return Some("every week".to_string());
    }
    if lower.contains("every day") || lower.contains("daily") {
        return Some("every day".to_string());
    }
    compile_regex(
        r"(?i)\b(once|twice|thrice|one|two|three|four|five|\d+)\s+times?\s+(?:a|per)\s+(day|week|month|year)\b",
    )
    .captures(line)
    .and_then(|caps| {
        let raw = caps.get(1)?.as_str().trim().to_ascii_lowercase();
        let unit = caps.get(2)?.as_str().trim().to_ascii_lowercase();
        Some(format!("{raw} times a {unit}"))
    })
}

pub(in crate::index) fn extract_time_answer_from_line(line: &str) -> Option<String> {
    [
        r"(?i)\b(\d{1,2}:\d{2}\s?(?:AM|PM))\b",
        r"(?i)\b(\d{1,2}\s?(?:AM|PM))\b",
    ]
    .into_iter()
    .find_map(|pattern| {
        compile_regex(pattern)
            .captures(line)
            .and_then(|caps| caps.get(1))
            .map(|m| m.as_str().trim().to_string())
    })
}

pub(in crate::index) fn extract_focus_aligned_time_answer_from_line(
    line: &str,
    lower: &str,
    focus_terms: &[String],
) -> Option<String> {
    let pattern = compile_regex(r"(?i)\b(\d{1,2}(?::\d{2})?\s?(?:AM|PM))\b");
    let matches = pattern
        .captures_iter(line)
        .filter_map(|caps| caps.get(1))
        .map(|m| (m.start(), m.as_str().trim().to_string()))
        .collect::<Vec<_>>();
    if matches.is_empty() {
        return None;
    }
    if matches.len() == 1 {
        return extract_time_answer_from_line(line);
    }
    let focus_positions = focus_terms
        .iter()
        .filter_map(|term| lower.find(term))
        .collect::<Vec<_>>();
    if focus_positions.is_empty() {
        return matches.last().map(|(_, value)| value.clone());
    }
    matches
        .into_iter()
        .min_by_key(|(time_idx, _)| {
            focus_positions
                .iter()
                .map(|focus_idx| focus_idx.abs_diff(*time_idx))
                .min()
                .unwrap_or(usize::MAX)
        })
        .map(|(_, value)| value)
}

pub(in crate::index) fn extract_schedule_slot_focus_phrase(task_lower: &str) -> Option<String> {
    for marker in [
        "what day of the week do i ",
        "which day do i ",
        "what time do i ",
    ] {
        let Some((_, tail)) = task_lower.split_once(marker) else {
            continue;
        };
        let phrase = tail.trim().trim_end_matches('?');
        if !phrase.is_empty() {
            return Some(phrase.to_string());
        }
    }
    None
}

pub(in crate::index) fn extract_points_answer_from_line(line: &str, lower: &str) -> Option<String> {
    if !(lower.contains("score") || lower.contains("points")) {
        return None;
    }
    let raw = compile_regex(r"(?i)\b(\d+)\s+points\b")
        .captures(line)
        .and_then(|caps| caps.get(1))
        .map(|m| m.as_str().trim())?;
    Some(format!("{raw} points"))
}

pub(in crate::index) fn extract_record_answer_from_line(line: &str, lower: &str) -> Option<String> {
    if !(lower.contains("record") || lower.contains("we're") || lower.contains("we are")) {
        return None;
    }
    compile_regex(r"\b(\d+\s*-\s*\d+)\b")
        .captures(line)
        .and_then(|caps| caps.get(1))
        .map(|m| m.as_str().trim().replace(' ', ""))
}

pub(in crate::index) fn extract_status_answer_from_line(line: &str, lower: &str) -> Option<String> {
    if !lower.contains("status") {
        return None;
    }
    compile_regex(r"(?i)\b(Premier\s+(?:Silver|Gold|Platinum|Bronze|Diamond|1K))\s+status\b")
        .captures(line)
        .and_then(|caps| caps.get(1))
        .map(|m| m.as_str().trim().to_string())
}

pub(in crate::index) fn extract_level_goal_answer_from_line(
    line: &str,
    lower: &str,
) -> Option<String> {
    if !lower.contains("level")
        || !(line_has_future_goal_marker(lower)
            || lower.contains("determined to reach")
            || lower.contains("aiming to hit")
            || lower.contains("current goal"))
    {
        return None;
    }
    compile_regex(r"(?i)\b(level\s+\d+)\b")
        .captures(line)
        .and_then(|caps| caps.get(1))
        .map(|m| m.as_str().trim().to_ascii_lowercase())
}

pub(in crate::index) fn extract_state_transition_surface_from_line(
    line: &str,
    lower: &str,
    state_kind: &str,
) -> Option<String> {
    match state_kind {
        "score" => extract_points_answer_from_line(line, lower),
        "record" => extract_record_answer_from_line(line, lower),
        "status" => extract_status_answer_from_line(line, lower),
        "goal" => extract_level_goal_answer_from_line(line, lower),
        _ => None,
    }
}

pub(in crate::index) fn extract_relative_purchase_current_item(task_lower: &str) -> Option<String> {
    [
        "before getting the ",
        "before getting ",
        "before i got the ",
        "before i got ",
        "before buying the ",
        "before buying ",
        "before i bought the ",
        "before i bought ",
        "before purchasing the ",
        "before purchasing ",
        "before i purchased the ",
        "before i purchased ",
    ]
    .into_iter()
    .find_map(|marker| {
        let (_, tail) = task_lower.split_once(marker)?;
        let item = normalize_query_item_surface(tail);
        (!item.is_empty()).then_some(item)
    })
}

pub(in crate::index) fn normalize_query_item_surface(value: &str) -> String {
    let trimmed = value
        .trim()
        .trim_end_matches('?')
        .trim_end_matches('.')
        .trim();
    for prefix in ["the ", "a ", "an "] {
        if let Some(stripped) = trimmed.strip_prefix(prefix) {
            return stripped.trim().to_string();
        }
    }
    trimmed.to_string()
}

pub(in crate::index) fn extract_purchase_family_item_from_line(
    line: &str,
    lower: &str,
    family: &str,
) -> Option<String> {
    match family {
        "gadget" => extract_gadget_purchase_item_from_line(line, lower),
        "lens" => extract_lens_purchase_item_from_line(line, lower),
        _ => None,
    }
}

pub(in crate::index) fn extract_gadget_purchase_item_from_line(
    line: &str,
    lower: &str,
) -> Option<String> {
    if !task_contains_any(
        lower,
        &[
            "my new ",
            "i got",
            "got yesterday",
            "bought",
            "purchased",
            "gift",
            "using the ",
            "using my ",
        ],
    ) {
        return None;
    }
    compile_regex(
        r"(?i)\b(?:my\s+new\s+|my\s+|the\s+)?((?:[a-z0-9][a-z0-9+-]*)(?:\s+[a-z0-9][a-z0-9+-]*){0,2}\s(?:pot|fryer|mixer|blender|processor|maker|oven|grill|toaster|microwave|cooker|skillet))\b",
    )
    .captures_iter(line)
    .filter_map(|caps| caps.get(1))
    .map(|m| m.as_str().trim().to_string())
    .last()
}

pub(in crate::index) fn extract_lens_purchase_item_from_line(
    line: &str,
    lower: &str,
) -> Option<String> {
    let has_ownership_marker = task_contains_any(
        lower,
        &[
            "i got",
            "got my ",
            "recently got",
            "just got",
            "bought my ",
            "bought a ",
            "bought an ",
            "purchased",
            "picked up",
            "my new ",
        ],
    );
    if !lower.contains("lens") || !has_ownership_marker {
        return None;
    }
    if task_contains_any(lower, &["haven't bought", "have not bought", "might buy"])
        && !task_contains_any(lower, &["got my ", "recently got", "just got", "my new "])
    {
        return None;
    }
    let phrase = compile_regex(
        r"(?i)\b(?:old\s+|new\s+)?((?:\d{1,3}(?:-\d{1,3})?mm|[a-z]+(?:-[a-z]+)?)(?:\s+[a-z]+(?:-[a-z]+)?){0,2}\s+lens)\b",
    )
    .captures_iter(line)
    .filter_map(|caps| caps.get(1))
    .map(|m| m.as_str().trim().to_string())
    .last()?;
    Some(render_with_indefinite_article(&phrase))
}

pub(in crate::index) fn render_with_indefinite_article(value: &str) -> String {
    let trimmed = value.trim();
    let lower = trimmed.to_ascii_lowercase();
    if lower.starts_with("a ") || lower.starts_with("an ") {
        return trimmed.to_string();
    }
    let article = match lower.chars().next() {
        Some('a' | 'e' | 'i' | 'o' | 'u') => "an",
        _ => "a",
    };
    format!("{article} {trimmed}")
}

pub(in crate::index) fn extract_trip_destination_from_query(task_lower: &str) -> Option<String> {
    for marker in ["trip to ", "vacation to ", "visit to "] {
        let Some((_, tail)) = task_lower.split_once(marker) else {
            continue;
        };
        let destination = tail.trim().trim_end_matches('?').trim().to_string();
        if !destination.is_empty() {
            return Some(destination);
        }
    }
    None
}

pub(in crate::index) fn extract_planned_stay_location_from_line(
    line: &str,
    lower: &str,
) -> Option<String> {
    let value = extract_phrase_after_any_index(
        line,
        lower,
        &[
            "planning to stay on ",
            "planning to stay in ",
            "planning to stay at ",
            "plan to stay on ",
            "plan to stay in ",
            "plan to stay at ",
            "staying on ",
            "staying in ",
            "staying at ",
            "stay on ",
            "stay in ",
            "stay at ",
        ],
        &[
            " for ",
            " because ",
            " and ",
            " but ",
            " while ",
            ".",
            ",",
            ";",
            " instead",
            " during ",
        ],
        1,
    )?;
    (value.split_whitespace().count() <= 6).then(|| normalize_location_kg_value(&value))
}

pub(in crate::index) fn line_has_current_company_marker(lower: &str) -> bool {
    task_contains_any(
        lower,
        &[
            "currently working at ",
            "currently at ",
            "current company is ",
            "works at ",
            "working at ",
            "employed at ",
        ],
    )
}
