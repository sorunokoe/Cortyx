use std::collections::HashSet;

use super::kg_extract::{assistant_segments, extract_phrase_fact_value, user_disclosure_segments};
use super::Turn;

pub(super) fn fact_summary_lines(turns: &[Turn]) -> Vec<String> {
    fn is_fact_like(line: &str) -> bool {
        let lower = line.to_ascii_lowercase();
        line.chars().any(|c| c.is_ascii_digit())
            || lower.starts_with("i ")
            || lower.starts_with("i'")
            || lower.starts_with("my ")
            || lower.starts_with("we ")
            || lower.contains(" i ")
            || lower.contains(" my ")
            || lower.contains(" we ")
            || lower.contains(" i'm ")
            || lower.contains(" i have ")
            || lower.contains(" i got ")
            || lower.contains(" i bought ")
            || lower.contains(" i use ")
            || lower.contains(" i used ")
            || lower.contains(" i went ")
            || lower.contains(" i visited ")
            || lower.contains(" i completed ")
            || lower.contains(" i finished ")
            || lower.contains(" i attended ")
            || lower.contains(" my sister ")
            || lower.contains(" my brother ")
            || lower.contains(" my mom ")
            || lower.contains(" my dad ")
            || lower.contains(" my cat ")
            || lower.contains(" my dog ")
    }

    let mut lines = Vec::new();
    let mut seen = HashSet::new();

    for turn in turns {
        for segment in user_disclosure_segments(turn) {
            for raw_line in segment.split(|c| matches!(c, '\n' | '.' | '!' | '?')) {
                let trimmed = raw_line
                    .trim()
                    .trim_start_matches(|c: char| {
                        c == '-' || c == '*' || c.is_ascii_digit() || c == ')' || c == '.'
                    })
                    .trim();
                if trimmed.len() < 12 || !is_fact_like(trimmed) {
                    continue;
                }
                let compact = trimmed.split_whitespace().collect::<Vec<_>>().join(" ");
                if compact.len() < 12 {
                    continue;
                }
                let capped = if compact.len() > 180 {
                    compact[..180].trim_end().to_string()
                } else {
                    compact
                };
                let key = capped.to_ascii_lowercase();
                if seen.insert(key) {
                    lines.push(capped);
                    if lines.len() >= 24 {
                        return lines;
                    }
                }
            }
        }
    }

    lines
}

pub(super) fn assistant_numeric_summary_lines(turns: &[Turn]) -> Vec<String> {
    fn looks_like_list_item(line: &str) -> bool {
        let trimmed = line.trim_start();
        let mut chars = trimmed.chars();
        match (chars.next(), chars.next()) {
            (Some(a), Some(b)) if a.is_ascii_digit() && (b == '.' || b == ')') => true,
            _ => trimmed.starts_with("* ") || trimmed.starts_with("- "),
        }
    }

    fn is_answer_like(line: &str) -> bool {
        const MARKERS: &[&str] = &[
            "total",
            "sum",
            "difference",
            "more",
            "less",
            "remaining",
            "left",
            "per ",
            "each",
            "discount",
            "followers",
            "pages",
            "comments",
            "episodes",
            "hours",
            "cost",
            "price",
            "spent",
            "earned",
            "save",
            "saved",
            "older",
            "dozen",
            "night",
            "quote",
            "market",
            "trip",
            "gift",
            "score",
            "count",
            "percent",
        ];
        let lower = line.to_ascii_lowercase();
        let has_numeric_signal =
            lower.contains('$') || lower.contains('%') || line.chars().any(|c| c.is_ascii_digit());
        has_numeric_signal && MARKERS.iter().any(|marker| lower.contains(marker))
    }

    let mut lines = Vec::new();
    let mut seen = HashSet::new();

    for turn in turns {
        for segment in assistant_segments(turn) {
            for raw_line in segment.split(|c| matches!(c, '\n' | '.' | '!' | '?')) {
                let trimmed = raw_line.trim();
                if trimmed.len() < 12 || looks_like_list_item(trimmed) || !is_answer_like(trimmed) {
                    continue;
                }
                let compact = trimmed.split_whitespace().collect::<Vec<_>>().join(" ");
                if compact.len() < 12 {
                    continue;
                }
                let capped = if compact.len() > 180 {
                    compact[..180].trim_end().to_string()
                } else {
                    compact
                };
                let key = capped.to_ascii_lowercase();
                if seen.insert(key) {
                    lines.push(capped);
                    if lines.len() >= 12 {
                        return lines;
                    }
                }
            }
        }
    }

    lines
}

pub(super) fn assistant_named_item_summary_lines(turns: &[Turn]) -> Vec<String> {
    fn compact_named_item(line: &str) -> Option<String> {
        let trimmed = line.trim_start();
        let numbered = trimmed
            .chars()
            .next()
            .is_some_and(|c| c.is_ascii_digit() || c == '-' || c == '*');
        if !numbered {
            return None;
        }

        let item = trimmed
            .trim_start_matches(|c: char| {
                c.is_ascii_digit() || matches!(c, '.' | ')' | '-' | '*' | ' ' | '\t')
            })
            .trim()
            .trim_matches('*')
            .trim();
        if item.is_empty() {
            return None;
        }
        let candidate = item
            .split(':')
            .next()
            .unwrap_or(item)
            .split(" - ")
            .next()
            .unwrap_or(item)
            .trim()
            .trim_matches('*')
            .trim();
        let words = candidate.split_whitespace().count();
        if words == 0 || words > 6 {
            return None;
        }
        candidate
            .split_whitespace()
            .any(|word| word.chars().next().is_some_and(|c| c.is_uppercase()))
            .then_some(candidate.to_string())
    }

    fn sentence_named_item(line: &str) -> Option<String> {
        let lower = line.to_ascii_lowercase();
        for marker in [
            "there's also ",
            "there is also ",
            "another option is ",
            "one popular option is ",
            "one example is ",
        ] {
            if let Some(idx) = lower.find(marker) {
                return extract_phrase_fact_value(
                    &line[idx + marker.len()..],
                    &[
                        "that", "which", "who", "serves", "offers", "is", "in", "near", "and",
                        "but",
                    ],
                    4,
                );
            }
        }

        let prefix = line
            .split(',')
            .next()
            .unwrap_or(line)
            .trim()
            .trim_matches('*')
            .trim();
        let words = prefix.split_whitespace().count();
        if (1..=4).contains(&words)
            && lower.contains(" is ")
            && prefix
                .split_whitespace()
                .any(|word| word.chars().next().is_some_and(|c| c.is_uppercase()))
        {
            return Some(prefix.to_string());
        }
        None
    }

    let mut lines = Vec::new();
    let mut seen = HashSet::new();

    for turn in turns {
        for segment in assistant_segments(turn) {
            for raw_line in segment.lines() {
                if let Some(item) = compact_named_item(raw_line) {
                    let key = item.to_ascii_lowercase();
                    if seen.insert(key) {
                        lines.push(item);
                        if lines.len() >= 12 {
                            return lines;
                        }
                    }
                }
            }

            for raw_line in segment.split(|c| matches!(c, '.' | '!' | '?')) {
                let trimmed = raw_line.trim();
                if trimmed.len() < 6 {
                    continue;
                }
                if let Some(item) = sentence_named_item(trimmed) {
                    let key = item.to_ascii_lowercase();
                    if seen.insert(key) {
                        lines.push(item);
                        if lines.len() >= 12 {
                            return lines;
                        }
                    }
                }
            }
        }
    }

    lines
}
