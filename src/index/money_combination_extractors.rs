//! Money combination extractors.

use super::money_support::*;
use super::*;

pub(super) fn extract_spend_focus_fact_from_line(
    line: &str,
    lower: &str,
    focus: &SpendFocus,
) -> Option<MoneyAmountFact> {
    if !is_summary_or_user_line(line, lower) || !line_matches_spend_focus(lower, focus) {
        return None;
    }
    let amount_cents = match focus.kind {
        SpendFocusKind::GenericItem => {
            extract_price_after_focus_terms(line, lower, &focus.required_terms)
        },
        SpendFocusKind::GiftRecipient => extract_purchase_price_from_line(line),
    }?;
    Some(MoneyAmountFact::new(
        amount_cents,
        focus.required_terms.len() * 8 + usize::from(lower.starts_with("user:")) * 8,
        line,
    ))
}

pub(super) fn extract_keyed_gift_recipient_fact_from_line(
    line: &str,
    lower: &str,
    focus: &SpendFocus,
) -> Option<KeyedMoneyAmountFact> {
    let fact = extract_spend_focus_fact_from_line(line, lower, focus)?;
    let key = lower.chars().take(120).collect::<String>();
    Some(KeyedMoneyAmountFact::new(
        key,
        fact.amount_cents,
        fact.score,
        line,
    ))
}

pub(super) fn extract_contextual_spend_followup_fact_from_line(
    line: &str,
    lower: &str,
) -> Option<MoneyAmountFact> {
    if !is_summary_or_user_line(line, lower) {
        return None;
    }
    let amount_cents = extract_purchase_price_from_line(line)?;
    Some(MoneyAmountFact::new(amount_cents, 4, line))
}

pub(super) fn extract_sale_value_fact_from_line(
    line: &str,
    lower: &str,
    focus: &SpendFocus,
) -> Option<MoneyAmountFact> {
    if !is_summary_or_user_line(line, lower) || !line_matches_spend_focus(lower, focus) {
        return None;
    }
    let amount_cents = extract_money_after_markers(
        line,
        &[
            r"(?i)\b(?:worth|valued at|value of)\b[^$\n]{0,20}?\$([0-9][0-9,]*(?:\.\d{1,2})?)",
            r"(?i)\bfor\s+(?:at\s+least\s+)?\$([0-9][0-9,]*(?:\.\d{1,2})?)",
            r"(?i)\b(?:sell|sold|selling)\b[^$\n]{0,40}?\$([0-9][0-9,]*(?:\.\d{1,2})?)",
            r"(?i)\b(?:is|was|are|were)\b[^$\n]{0,10}?\$([0-9][0-9,]*(?:\.\d{1,2})?)",
        ],
    )?;
    Some(MoneyAmountFact::new(
        amount_cents,
        focus.required_terms.len() * 8 + usize::from(lower.starts_with("user:")) * 8,
        line,
    ))
}

/// Extract price from text starting after the last occurrence of any required term.
/// Falls back to searching the full line when nothing is found after the terms.
fn extract_price_after_focus_terms(
    line: &str,
    lower: &str,
    required_terms: &[String],
) -> Option<i64> {
    let last_term_end = required_terms
        .iter()
        .filter_map(|term| lower.rfind(term.as_str()).map(|pos| pos + term.len()))
        .max()
        .unwrap_or(0);
    let search_start = last_term_end.min(line.len());
    extract_purchase_price_from_line(&line[search_start..])
        .or_else(|| extract_purchase_price_from_line(line))
}

/// Try standard purchase/cost patterns on a text slice, returning the first match.
fn extract_purchase_price_from_line(text: &str) -> Option<i64> {
    extract_money_after_markers(
        text,
        &[
            r"(?i)\bfor\s+\$([0-9][0-9,]*(?:\.\d{1,2})?)",
            r"(?i)\b(?:cost(?:\s+me)?|costed(?:\s+me)?)\b[^$\n]{0,20}?\$([0-9][0-9,]*(?:\.\d{1,2})?)",
            r"(?i)\b(?:are|is|was|were)\b[^$\n]{0,10}?\$([0-9][0-9,]*(?:\.\d{1,2})?)",
            r"(?i)\b(?:totaling|totalled|worth|invested)\b[^$\n]{0,20}?\$([0-9][0-9,]*(?:\.\d{1,2})?)",
            r"(?i)\b(?:got|get|bought|paid|purchased|found|snagged|picked(?:\s+it)?\s+up|ordered|treated)\b[^$\n]{0,80}?\$([0-9][0-9,]*(?:\.\d{1,2})?)",
        ],
    )
}
