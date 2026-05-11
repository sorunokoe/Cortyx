use super::money_support::{
    extract_money_after_markers, extract_percent_basis_points, focus_match_count,
    line_matches_focus, normalize_quantity_unit, CashbackQuery, CashbackRateFact,
    KeyedMoneyAmountFact, MoneyAmountFact, QuantityFact, RaisedTotalQuery, RevenueQuery,
    SavingsQuery, UnitPriceFact,
};
use super::*;

const CHARITY_EVENT_PHRASES: &[&str] = &[
    "charity walk",
    "charity yoga event",
    "charity bake sale",
    "charity fitness challenge",
    "charity cycling event",
    "charity run",
    "bike-a-thon",
];

pub(super) fn extract_paid_price_fact_from_line(
    line: &str,
    lower: &str,
    query: &SavingsQuery,
) -> Option<MoneyAmountFact> {
    if !is_summary_or_user_line(line, lower) {
        return None;
    }
    let focus_score = savings_focus_score(lower, query)?;
    let amount_cents = extract_money_after_markers(
        line,
        &[
            r"(?i)\$([0-9][0-9,]*(?:\.\d{1,2})?)\s+after\s+(?:a|the)\s+discount\b",
            r"(?i)\$([0-9][0-9,]*(?:\.\d{1,2})?)\s+on\s+sale\b",
            r"(?i)\b(?:got|bought|paid|purchased|found|snagged|picked(?:\s+it)?\s+up)\b[^$\n]{0,80}?\$([0-9][0-9,]*(?:\.\d{1,2})?)",
            r"(?i)\b(?:cost|cost me)\b[^$\n]{0,20}?\$([0-9][0-9,]*(?:\.\d{1,2})?)",
        ],
    )?;
    Some(MoneyAmountFact::new(
        amount_cents,
        focus_score
            + usize::from(lower.starts_with("user:")) * 8
            + usize::from(task_contains_any(lower, &["outlet", "deal", "got for"])) * 4,
        line,
    ))
}

pub(super) fn extract_original_price_fact_from_line(
    line: &str,
    lower: &str,
    query: &SavingsQuery,
) -> Option<MoneyAmountFact> {
    if !is_summary_or_user_line(line, lower) {
        return None;
    }
    let focus_score = savings_focus_score(lower, query)?;
    let amount_cents = extract_money_after_markers(
        line,
        &[
            r"(?i)\boriginally\b[^$\n]{0,24}?\$([0-9][0-9,]*(?:\.\d{1,2})?)",
            r"(?i)\b(?:retailed|retail(?:ed)?(?: price)?|list price|full price)\b[^$\n]{0,24}?\$([0-9][0-9,]*(?:\.\d{1,2})?)",
        ],
    )?;
    Some(MoneyAmountFact::new(
        amount_cents,
        focus_score
            + usize::from(lower.starts_with("user:")) * 8
            + usize::from(task_contains_any(
                lower,
                &["originally", "retailed", "retail"],
            )) * 4,
        line,
    ))
}

pub(super) fn extract_cashback_purchase_fact_from_line(
    line: &str,
    lower: &str,
    query: &CashbackQuery,
) -> Option<MoneyAmountFact> {
    if !is_summary_or_user_line(line, lower)
        || !line_matches_focus(lower, &query.merchant_terms)
        || !query
            .anchor_terms
            .iter()
            .all(|term| lower.contains(term.as_str()))
    {
        return None;
    }
    let amount_cents = extract_money_after_markers(
        line,
        &[
            r"(?i)\b(?:spent|paid|purchased|bought)\b[^$\n]{0,48}?\$([0-9][0-9,]*(?:\.\d{1,2})?)",
            r"(?i)\b(?:cost|cost me)\b[^$\n]{0,20}?\$([0-9][0-9,]*(?:\.\d{1,2})?)",
        ],
    )?;
    Some(MoneyAmountFact::new(
        amount_cents,
        focus_match_count(lower, &query.merchant_terms) * 10
            + query.anchor_terms.len() * 8
            + usize::from(lower.starts_with("user:")) * 8,
        line,
    ))
}

pub(super) fn extract_cashback_rate_fact_from_line(
    line: &str,
    lower: &str,
    query: &CashbackQuery,
) -> Option<CashbackRateFact> {
    if !is_summary_or_user_line(line, lower)
        || !line_matches_focus(lower, &query.merchant_terms)
        || !lower.contains("cashback")
    {
        return None;
    }
    let basis_points = extract_percent_basis_points(line)?;
    Some(CashbackRateFact::new(
        basis_points,
        focus_match_count(lower, &query.merchant_terms) * 10
            + usize::from(lower.contains("cashback")) * 8
            + usize::from(lower.starts_with("user:")) * 4,
        line,
    ))
}

pub(super) fn extract_raised_total_fact_from_line(
    line: &str,
    lower: &str,
    query: &RaisedTotalQuery,
) -> Option<KeyedMoneyAmountFact> {
    if !is_summary_or_user_line(line, lower)
        || !lower.contains('$')
        || !task_contains_any(
            lower,
            &[
                "raised",
                "managed to raise",
                "helped raise",
                "helped to raise",
            ],
        )
        || (query.event_scoped && !has_explicit_charity_event_reference(line, lower))
    {
        return None;
    }
    let amount_cents = extract_money_after_markers(
        line,
        &[
            r"(?i)\b(?:managed to raise|helped(?:\s+to)?\s+raise|raised)\b[^$\n]{0,40}?(?:over\s+)?\$([0-9][0-9,]*(?:\.\d{1,2})?)",
        ],
    )?;
    let key = extract_raised_event_key(line, lower)?;
    Some(KeyedMoneyAmountFact::new(
        key,
        amount_cents,
        usize::from(lower.starts_with("user:")) * 8
            + usize::from(lower.contains("charity")) * 4
            + usize::from(extract_charity_cause_phrase(line).is_some()) * 4
            + usize::from(extract_charity_time_phrase(line).is_some()) * 2
            + if query.event_scoped {
                explicit_event_participation_score(lower)
            } else {
                0
            },
        line,
    ))
}

pub(super) fn extract_revenue_quantity_fact_from_line(
    line: &str,
    lower: &str,
    query: &RevenueQuery,
) -> Option<QuantityFact> {
    if !is_summary_or_user_line(line, lower)
        || !line_matches_focus(lower, &query.item_terms)
        || !task_contains_any(lower, &["sold", "selling"])
    {
        return None;
    }
    let captures = compile_regex(
        r"(?i)\b(?:sold|selling)\b[^0-9]{0,32}?(?:a total of\s+)?(\d+)\s+(dozen|dozens|pairs?|items?|crates?|boxes?)\b",
    )
    .captures(line)?;
    let quantity_units = captures.get(1)?.as_str().parse::<i64>().ok()?;
    let unit_key = normalize_quantity_unit(captures.get(2)?.as_str());
    Some(QuantityFact::new(
        quantity_units,
        unit_key,
        focus_match_count(lower, &query.item_terms) * 10
            + usize::from(lower.starts_with("user:")) * 8
            + usize::from(lower.contains("total")) * 4,
        line,
    ))
}

pub(super) fn extract_revenue_unit_price_fact_from_line(
    line: &str,
    lower: &str,
    query: &RevenueQuery,
) -> Option<UnitPriceFact> {
    if !is_summary_or_user_line(line, lower)
        || !line_matches_focus(lower, &query.item_terms)
        || !task_contains_any(lower, &["sell", "selling"])
    {
        return None;
    }
    let captures =
        compile_regex(r"(?i)\$([0-9][0-9,]*(?:\.\d{1,2})?)\s+(?:a|per)\s+(dozen|dozens|pairs?|items?|crates?|boxes?)\b")
            .captures(line)?;
    let unit_price_cents = super::money_support::parse_money_cents(captures.get(1)?.as_str())?;
    let unit_key = normalize_quantity_unit(captures.get(2)?.as_str());
    Some(UnitPriceFact::new(
        unit_price_cents,
        unit_key,
        focus_match_count(lower, &query.item_terms) * 10
            + usize::from(lower.starts_with("user:")) * 8
            + usize::from(lower.contains("selling")) * 4,
        line,
    ))
}

fn extract_raised_event_key(line: &str, lower: &str) -> Option<String> {
    let (window_line, window_lower) = charity_event_context_window(line, lower)?;
    let titled_event = compile_regex(r#""([^"]+)""#)
        .captures(window_line)
        .and_then(|captures| captures.get(1))
        .map(|value| value.as_str().trim().to_string());
    let event_phrase = titled_event
        .clone()
        .or_else(|| extract_named_charity_event_phrase(window_lower))
        .or_else(|| extract_named_charity_event_phrase(lower));
    let cause_phrase = extract_charity_cause_phrase(line);
    let time_phrase = extract_charity_time_phrase(line);
    let key_surface = match (cause_phrase, time_phrase, event_phrase) {
        (Some(cause), Some(time), _) => format!("{cause} {time}"),
        (Some(cause), None, Some(event)) => format!("{event} {cause}"),
        (Some(cause), None, None) => cause,
        (None, Some(time), Some(event)) => format!("{event} {time}"),
        (None, None, Some(event)) => event,
        (None, Some(time), None) => time,
        (None, None, None) => return None,
    };
    Some(normalized_synthetic_phrase_key(&key_surface))
}

fn extract_named_charity_event_phrase(lower: &str) -> Option<String> {
    CHARITY_EVENT_PHRASES
        .iter()
        .find(|phrase| lower.contains(**phrase))
        .map(|phrase| (*phrase).to_string())
}

fn has_explicit_charity_event_reference(line: &str, lower: &str) -> bool {
    let Some((window_line, window_lower)) = charity_event_context_window(line, lower) else {
        return false;
    };
    compile_regex(r#""([^"]+)""#)
        .captures(window_line)
        .is_some()
        || extract_named_charity_event_phrase(window_lower).is_some()
}

fn charity_event_context_window<'a>(line: &'a str, lower: &'a str) -> Option<(&'a str, &'a str)> {
    let raise_idx = [
        "managed to raise",
        "helped to raise",
        "helped raise",
        "raised",
    ]
    .into_iter()
    .filter_map(|marker| lower.find(marker))
    .min()?;
    let start = raise_idx.saturating_sub(120);
    let end = (raise_idx + 140).min(lower.len());
    Some((&line[start..end], &lower[start..end]))
}

fn explicit_event_participation_score(lower: &str) -> usize {
    if lower.contains("participated") {
        14
    } else if lower.contains("helped organize") {
        10
    } else if lower.contains("volunteered") {
        6
    } else if lower.contains("completed")
        || lower.contains(" just ran ")
        || lower.contains(" ran 5 kilometers ")
        || lower.contains(" ran 5 kilometres ")
    {
        5
    } else {
        0
    }
}

fn extract_charity_cause_phrase(line: &str) -> Option<String> {
    compile_regex(
        r"(?i)\$[0-9][0-9,]*(?:\.\d{1,2})?\s+for\s+(?:a\s+|the\s+)?([A-Za-z][A-Za-z' -]{2,}?)(?:[.!?,]|\s+(?:on|at|in|through|and)\b|$)",
    )
        .captures(line)
        .and_then(|captures| captures.get(1))
        .map(|value| value.as_str().trim().to_string())
}

fn extract_charity_time_phrase(line: &str) -> Option<String> {
    compile_regex(
        r"(?i)\b((?:on\s+)?(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}(?:st|nd|rd|th)?|in\s+(?:January|February|March|April|May|June|July|August|September|October|November|December))\b",
    )
        .captures(line)
        .and_then(|captures| captures.get(1))
        .map(|value| value.as_str().trim().to_string())
}

fn savings_focus_score(lower: &str, query: &SavingsQuery) -> Option<usize> {
    if !line_matches_focus(lower, &query.product_terms) {
        return None;
    }
    Some(
        focus_match_count(lower, &query.product_terms) * 10
            + focus_match_count(lower, &query.context_terms) * 4,
    )
}
