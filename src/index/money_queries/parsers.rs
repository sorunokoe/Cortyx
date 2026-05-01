//! Money query parsers for specific query types.

use super::super::money_support::{
    normalized_money_terms, CashbackQuery, RaisedTotalQuery, RecipientGiftTotalQuery,
    RevenueQuery, SaleMinimumQuery, SavingsQuery, SpendFocus, SpendSumQuery,
};
use super::super::*;
use super::helpers::*;

pub(super) fn parse_cashback_query(task_lower: &str) -> Option<CashbackQuery> {
    if !task_lower.contains("cashback") || !task_lower.contains("earn") {
        return None;
    }
    let merchant_phrase = Regex::new(
        r"\bat\s+([a-z0-9][a-z0-9&' -]*?)(?:\s+(?:last|this|next|on|from|during|in)\b|[?]|$)",
    )
    .unwrap()
    .captures(task_lower)
    .and_then(|captures| captures.get(1))
    .map(|value| value.as_str().trim().to_string())?;
    let merchant_terms = normalized_money_terms(&merchant_phrase);
    if merchant_terms.is_empty() {
        return None;
    }
    let anchor_terms = extract_relative_day_anchor_terms(task_lower);
    if anchor_terms.is_empty() {
        return None;
    }
    let required_terms = build_required_terms(
        merchant_terms
            .iter()
            .cloned()
            .chain(anchor_terms.iter().cloned())
            .chain(["cashback".to_string()])
            .collect(),
    );
    Some(CashbackQuery {
        merchant_terms,
        anchor_terms,
        required_terms,
    })
}

pub(super) fn parse_savings_query(task_lower: &str) -> Option<SavingsQuery> {
    if !task_contains_any(task_lower, &["save on", "saved on"]) || task_lower.contains("cashback") {
        return None;
    }
    let tail = Regex::new(r"(?i)\bsav(?:e|ed)\s+on\s+(.+?)\??$")
        .unwrap()
        .captures(task_lower)
        .and_then(|captures| captures.get(1))
        .map(|value| value.as_str().trim())?;
    let (product_phrase, context_phrase) = tail.split_once(" at ").unwrap_or((tail, ""));
    let product_terms = normalized_money_terms(product_phrase);
    if product_terms.is_empty() {
        return None;
    }
    let context_terms = normalized_money_terms(context_phrase);
    let required_terms = build_required_terms(
        product_terms
            .iter()
            .cloned()
            .chain(context_terms.iter().cloned())
            .collect(),
    );
    Some(SavingsQuery {
        product_terms,
        context_terms,
        required_terms,
    })
}

pub(super) fn parse_discount_percent_query(task_lower: &str) -> Option<SavingsQuery> {
    if !task_contains_any(
        task_lower,
        &[
            "what percentage discount did i get on",
            "what percent discount did i get on",
        ],
    ) {
        return None;
    }
    let tail = Regex::new(
        r"(?i)\bwhat\s+(?:percentage|percent)\s+discount\s+did\s+i\s+get\s+on\s+(.+?)\??$",
    )
    .unwrap()
    .captures(task_lower)
    .and_then(|captures| captures.get(1))
    .map(|value| value.as_str().trim())?;
    let (product_phrase, context_phrase) = tail.split_once(" at ").unwrap_or((tail, ""));
    let product_terms = normalized_money_terms(product_phrase);
    if product_terms.is_empty() {
        return None;
    }
    let context_terms = normalized_money_terms(context_phrase);
    let required_terms = build_required_terms(
        product_terms
            .iter()
            .cloned()
            .chain(context_terms.iter().cloned())
            .chain(["discount".to_string()])
            .collect(),
    );
    Some(SavingsQuery {
        product_terms,
        context_terms,
        required_terms,
    })
}

pub(super) fn parse_spend_sum_query(task_lower: &str) -> Option<SpendSumQuery> {
    let tail = extract_spend_sum_tail(task_lower)?;
    let focuses = parse_spend_focuses(tail)?;
    if focuses.len() < 2 {
        return None;
    }
    build_spend_sum_query(focuses)
}

pub(super) fn parse_single_recipient_gift_total_query(
    task_lower: &str,
) -> Option<RecipientGiftTotalQuery> {
    let tail = extract_spend_sum_tail(task_lower)?;
    let focuses = parse_gift_recipient_focuses(tail)?;
    if focuses.len() != 1 {
        return None;
    }
    let focus = focuses.into_iter().next()?;
    let required_terms = build_required_terms(
        focus
            .required_terms
            .iter()
            .chain(focus.optional_terms.iter())
            .cloned()
            .collect(),
    );
    Some(RecipientGiftTotalQuery {
        focus,
        required_terms,
    })
}

pub(super) fn parse_sale_minimum_query(task_lower: &str) -> Option<SaleMinimumQuery> {
    if !task_contains_any(task_lower, &["minimum amount", "least amount"])
        || !task_contains_any(task_lower, &["if i sold", "if i sell"])
    {
        return None;
    }
    let tail = Regex::new(r"(?i)\bif i (?:sold|sell)\s+(.+?)\??$")
        .unwrap()
        .captures(task_lower)
        .and_then(|captures| captures.get(1))
        .map(|value| value.as_str().trim())?;
    let focuses = parse_item_focuses(tail)?;
    if focuses.len() < 2 {
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
    Some(SaleMinimumQuery {
        focuses,
        required_terms,
    })
}

pub(super) fn parse_raised_total_query(task_lower: &str) -> Option<RaisedTotalQuery> {
    if !task_contains_any(
        task_lower,
        &["how much money did i raise", "how much did i raise"],
    ) {
        return None;
    }
    if !task_contains_any(task_lower, &["charity", "fundrais"])
        || !task_contains_any(task_lower, &["total", "through all"])
    {
        return None;
    }
    let event_scoped = task_lower.contains("through all the charity events")
        || task_lower.contains("charity events i participated in");
    let required_terms = if event_scoped {
        vec![
            "charity".to_string(),
            "raise".to_string(),
            "participated".to_string(),
            "event".to_string(),
        ]
    } else {
        vec!["charity".to_string(), "raise".to_string()]
    };
    Some(RaisedTotalQuery {
        event_scoped,
        required_terms,
    })
}

pub(super) fn parse_revenue_query(task_lower: &str) -> Option<RevenueQuery> {
    if !task_contains_any(task_lower, &["made from selling", "earned from selling"]) {
        return None;
    }
    let item_phrase = Regex::new(r"(?i)\b(?:made|earned)\s+from\s+selling\s+(.+?)\??$")
        .unwrap()
        .captures(task_lower)
        .and_then(|captures| captures.get(1))
        .map(|value| strip_revenue_time_window(value.as_str()))?;
    let item_terms = normalized_money_terms(&item_phrase);
    if item_terms.is_empty() {
        return None;
    }
    let required_terms = build_required_terms(
        item_terms
            .iter()
            .cloned()
            .chain(["sold".to_string(), "selling".to_string()])
            .collect(),
    );
    Some(RevenueQuery {
        item_terms,
        required_terms,
    })
}

fn strip_revenue_time_window(surface: &str) -> String {
    let without_window = Regex::new(r"(?i)\s+(?:this|last|next)\s+(?:day|week|month|year)s?\b.*$")
        .unwrap()
        .replace(surface, "")
        .into_owned();
    Regex::new(r"(?i)\s+so\s+far\b.*$")
        .unwrap()
        .replace(&without_window, "")
        .trim()
        .to_string()
}
