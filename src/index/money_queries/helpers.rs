//! Helper functions for money query parsing.

use super::super::money_support::{
    normalized_money_terms, SpendFocus, SpendFocusKind, SpendSumQuery,
};
use super::super::*;

const DAY_NAMES: &[&str] = &[
    "monday",
    "tuesday",
    "wednesday",
    "thursday",
    "friday",
    "saturday",
    "sunday",
];

pub(super) fn build_spend_sum_query(focuses: Vec<SpendFocus>) -> Option<SpendSumQuery> {
    if focuses.is_empty() {
        return None;
    }
    let required_terms = build_required_terms(
        focuses
            .iter()
            .flat_map(|focus| {
                focus
                    .required_terms
                    .iter()
                    .chain(focus.optional_terms.iter())
                    .cloned()
            })
            .collect(),
    );
    Some(SpendSumQuery {
        focuses,
        required_terms,
    })
}

pub(super) fn extract_spend_sum_tail(task_lower: &str) -> Option<&str> {
    if let Some(tail) = compile_regex_static(r"(?i)\b(?:spend|spent)\s+on\s+(.+?)\??$")
        .captures(task_lower)
        .and_then(|captures| captures.get(1))
        .map(|value| value.as_str().trim())
    {
        return Some(tail);
    }
    if !task_contains_any(task_lower, &["total cost of", "total amount of", "cost of"]) {
        return None;
    }
    compile_regex_static(r"(?i)\b(?:total\s+cost|total\s+amount|cost)\s+of\s+(.+?)\??$")
        .captures(task_lower)
        .and_then(|captures| captures.get(1))
        .map(|value| value.as_str().trim())
}

pub(super) fn build_required_terms(mut terms: Vec<String>) -> Vec<String> {
    terms.sort();
    terms.dedup();
    terms
}

pub(super) fn parse_gift_recipient_focuses(tail: &str) -> Option<Vec<SpendFocus>> {
    let lower = normalize_possessives(tail);
    let rest = ["gifts for my ", "gift for my ", "gifts for ", "gift for "]
        .into_iter()
        .find_map(|prefix| lower.strip_prefix(prefix).map(str::to_string))?;
    let focuses = rest
        .split(" and ")
        .filter_map(build_gift_recipient_focus)
        .collect::<Vec<_>>();
    (!focuses.is_empty()).then_some(focuses)
}

pub(super) fn parse_spend_focuses(tail: &str) -> Option<Vec<SpendFocus>> {
    parse_gift_recipient_focuses(tail).or_else(|| parse_item_focuses(tail))
}

pub(super) fn parse_item_focuses(tail: &str) -> Option<Vec<SpendFocus>> {
    let focuses = split_item_focus_segments(tail)
        .into_iter()
        .filter_map(|segment| build_item_focus(&segment))
        .collect::<Vec<_>>();
    (!focuses.is_empty()).then_some(focuses)
}

pub(super) fn split_item_focus_segments(tail: &str) -> Vec<String> {
    let normalized = tail.trim().replace(", and ", ", ");
    if normalized.contains(',') {
        return normalized
            .split(',')
            .map(str::trim)
            .filter(|segment| !segment.is_empty())
            .map(str::to_string)
            .collect();
    }
    normalized
        .split(" and ")
        .map(str::trim)
        .filter(|segment| !segment.is_empty())
        .map(str::to_string)
        .collect()
}

pub(super) fn build_gift_recipient_focus(recipient: &str) -> Option<SpendFocus> {
    let recipient_surface = recipient.trim();
    let recipient_terms = normalized_money_terms(recipient_surface);
    if recipient_terms.is_empty() {
        return None;
    }
    let required_terms = build_required_terms(
        recipient_terms
            .iter()
            .cloned()
            .chain(["gift".to_string()])
            .collect(),
    );
    Some(SpendFocus {
        kind: SpendFocusKind::GiftRecipient,
        key: normalized_synthetic_phrase_key(recipient_surface),
        display: format!("gift for {recipient_surface}"),
        required_terms,
        optional_terms: vec![],
    })
}

pub(super) fn build_item_focus(segment: &str) -> Option<SpendFocus> {
    let item_surface = strip_item_focus_trailing_context(segment.trim());
    let terms = normalized_money_terms(&item_surface);
    if terms.is_empty() {
        return None;
    }
    let required_terms = focus_core_terms(&terms);
    let optional_terms = terms
        .into_iter()
        .filter(|term| !required_terms.contains(term))
        .collect::<Vec<_>>();
    Some(SpendFocus {
        kind: SpendFocusKind::GenericItem,
        key: normalized_synthetic_phrase_key(&item_surface),
        display: item_surface,
        required_terms,
        optional_terms,
    })
}

fn strip_item_focus_trailing_context(surface: &str) -> String {
    compile_regex_static(
        r"(?i)\s+i\s+(?:got|purchased|bought|ordered|received|found|snagged|picked\s+up)\b.*",
    )
    .replace(surface, "")
    .trim()
    .to_string()
}

pub(super) fn focus_core_terms(terms: &[String]) -> Vec<String> {
    const DESCRIPTOR_STOP: &[&str] = &[
        "adorable", "antique", "designer", "end", "formal", "great", "high", "luxury", "new",
        "nice", "old", "pair", "premium", "product", "products", "set", "some", "vintage",
    ];
    let filtered = terms
        .iter()
        .filter(|term| !DESCRIPTOR_STOP.contains(&term.as_str()))
        .cloned()
        .collect::<Vec<_>>();
    if filtered.is_empty() {
        return build_required_terms(terms.to_vec());
    }
    let core = if filtered.len() > 2 {
        filtered[filtered.len() - 2..].to_vec()
    } else {
        filtered
    };
    build_required_terms(core)
}

pub(super) fn normalize_possessives(surface: &str) -> String {
    compile_regex_static(r"'s\b")
        .replace_all(surface, "")
        .into_owned()
        .to_ascii_lowercase()
}

pub(super) fn extract_relative_day_anchor_terms(task_lower: &str) -> Vec<String> {
    compile_regex_static(
        r"\b(last|this)\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
    )
    .captures(task_lower)
    .and_then(|captures| {
        Some(vec![
            captures.get(1)?.as_str().to_string(),
            captures.get(2)?.as_str().to_string(),
        ])
    })
    .or_else(|| {
        DAY_NAMES
            .iter()
            .find(|day| task_lower.contains(**day))
            .map(|day| vec![(*day).to_string()])
    })
    .unwrap_or_default()
}
