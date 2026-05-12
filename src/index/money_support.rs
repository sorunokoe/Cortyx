use super::*;

const MONEY_QUERY_STOP: &[&str] = &[
    "a", "all", "an", "and", "at", "by", "cashback", "compare", "compared", "did", "earn",
    "earned", "for", "from", "have", "how", "i", "in", "made", "make", "money", "much", "my", "of",
    "on", "save", "saved", "sell", "selling", "spend", "spent", "the", "this", "through", "to",
    "total", "week", "weeks", "with", "year", "years", "month", "months",
];

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) enum MoneyQuery {
    Cashback(CashbackQuery),
    Savings(SavingsQuery),
    DiscountPercent(SavingsQuery),
    RecipientGiftTotal(RecipientGiftTotalQuery),
    SpendSum(SpendSumQuery),
    SaleMinimum(SaleMinimumQuery),
    RaisedTotal(RaisedTotalQuery),
    Revenue(RevenueQuery),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct CashbackQuery {
    pub(super) merchant_terms: Vec<String>,
    pub(super) anchor_terms: Vec<String>,
    pub(super) required_terms: Vec<String>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct SavingsQuery {
    pub(super) product_terms: Vec<String>,
    pub(super) context_terms: Vec<String>,
    pub(super) required_terms: Vec<String>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct SpendSumQuery {
    pub(super) focuses: Vec<SpendFocus>,
    pub(super) required_terms: Vec<String>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct RecipientGiftTotalQuery {
    pub(super) focus: SpendFocus,
    pub(super) required_terms: Vec<String>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct SaleMinimumQuery {
    pub(super) focuses: Vec<SpendFocus>,
    pub(super) required_terms: Vec<String>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct RaisedTotalQuery {
    pub(super) event_scoped: bool,
    pub(super) required_terms: Vec<String>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct RevenueQuery {
    pub(super) item_terms: Vec<String>,
    pub(super) required_terms: Vec<String>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct SpendFocus {
    pub(super) kind: SpendFocusKind,
    pub(super) key: String,
    pub(super) display: String,
    pub(super) required_terms: Vec<String>,
    pub(super) optional_terms: Vec<String>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum SpendFocusKind {
    GenericItem,
    GiftRecipient,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct MoneyAmountFact {
    pub(super) amount_cents: i64,
    pub(super) score: usize,
    pub(super) evidence: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct KeyedMoneyAmountFact {
    pub(super) key: String,
    pub(super) amount_cents: i64,
    pub(super) score: usize,
    pub(super) evidence: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct CashbackRateFact {
    pub(super) basis_points: i64,
    pub(super) score: usize,
    pub(super) evidence: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct QuantityFact {
    pub(super) quantity_units: i64,
    pub(super) unit_key: String,
    pub(super) score: usize,
    pub(super) evidence: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct UnitPriceFact {
    pub(super) unit_price_cents: i64,
    pub(super) unit_key: String,
    pub(super) score: usize,
    pub(super) evidence: String,
}

impl MoneyAmountFact {
    pub(super) fn new(amount_cents: i64, score: usize, evidence: &str) -> Self {
        Self {
            amount_cents,
            score,
            evidence: evidence.trim().to_string(),
        }
    }
}

impl KeyedMoneyAmountFact {
    pub(super) fn new(
        key: impl Into<String>,
        amount_cents: i64,
        score: usize,
        evidence: &str,
    ) -> Self {
        Self {
            key: key.into(),
            amount_cents,
            score,
            evidence: evidence.trim().to_string(),
        }
    }
}

impl CashbackRateFact {
    pub(super) fn new(basis_points: i64, score: usize, evidence: &str) -> Self {
        Self {
            basis_points,
            score,
            evidence: evidence.trim().to_string(),
        }
    }
}

impl QuantityFact {
    pub(super) fn new(
        quantity_units: i64,
        unit_key: impl Into<String>,
        score: usize,
        evidence: &str,
    ) -> Self {
        Self {
            quantity_units,
            unit_key: unit_key.into(),
            score,
            evidence: evidence.trim().to_string(),
        }
    }
}

impl UnitPriceFact {
    pub(super) fn new(
        unit_price_cents: i64,
        unit_key: impl Into<String>,
        score: usize,
        evidence: &str,
    ) -> Self {
        Self {
            unit_price_cents,
            unit_key: unit_key.into(),
            score,
            evidence: evidence.trim().to_string(),
        }
    }
}

pub(super) fn format_money_cents(amount_cents: i64) -> String {
    let sign = if amount_cents < 0 { "-" } else { "" };
    let absolute = amount_cents.unsigned_abs();
    let dollars = absolute / 100;
    let cents = absolute % 100;
    let grouped_dollars = format_grouped_dollars(dollars);
    if cents == 0 {
        format!("{sign}${grouped_dollars}")
    } else {
        format!("{sign}${grouped_dollars}.{cents:02}")
    }
}

pub(super) fn dedupe_evidence<I>(evidence: I) -> Vec<String>
where
    I: IntoIterator<Item = String>,
{
    let mut deduped = Vec::new();
    for line in evidence {
        if !line.is_empty() && !deduped.iter().any(|existing| existing == &line) {
            deduped.push(line);
        }
    }
    deduped
}

pub(super) fn normalized_money_terms(surface: &str) -> Vec<String> {
    let mut terms = synthetic_query_terms(surface)
        .into_iter()
        .filter(|term| !MONEY_QUERY_STOP.contains(&term.as_str()))
        .collect::<Vec<_>>();
    terms.sort();
    terms.dedup();
    terms
}

pub(super) fn line_matches_focus(lower: &str, terms: &[String]) -> bool {
    !terms.is_empty() && focus_match_count(lower, terms) >= required_focus_match_count(terms)
}

pub(super) fn focus_match_count(lower: &str, terms: &[String]) -> usize {
    terms
        .iter()
        .filter(|term| lower.contains(term.as_str()))
        .count()
}

pub(super) fn line_matches_spend_focus(lower: &str, focus: &SpendFocus) -> bool {
    match focus.kind {
        SpendFocusKind::GenericItem => {
            !focus.required_terms.is_empty()
                && focus
                    .required_terms
                    .iter()
                    .all(|term| lower.contains(term.as_str()))
        },
        SpendFocusKind::GiftRecipient => line_matches_gift_recipient_focus(lower, focus),
    }
}

pub(super) fn extract_money_after_markers(line: &str, patterns: &[&str]) -> Option<i64> {
    patterns.iter().find_map(|pattern| {
        compile_regex(pattern)
            .captures(line)
            .and_then(|captures| captures.get(1))
            .and_then(|value| parse_money_cents(value.as_str()))
    })
}

pub(super) fn parse_money_cents(raw: &str) -> Option<i64> {
    let cleaned = raw.trim().trim_start_matches('$').replace(',', "");
    let (whole, fractional) = cleaned.split_once('.').unwrap_or((cleaned.as_str(), ""));
    let dollars = whole.parse::<i64>().ok()?;
    let cents = match fractional.len() {
        0 => 0,
        1 => fractional.parse::<i64>().ok()? * 10,
        _ => fractional.get(..2)?.parse::<i64>().ok()?,
    };
    Some(dollars * 100 + cents)
}

pub(super) fn extract_percent_basis_points(line: &str) -> Option<i64> {
    let raw = compile_regex(r"(?i)(\d+(?:\.\d+)?)%")
        .captures(line)
        .and_then(|captures| captures.get(1))
        .map(|value| value.as_str())?;
    let percent = raw.parse::<f64>().ok()?;
    Some((percent * 100.0).round() as i64)
}

pub(super) fn normalize_quantity_unit(raw: &str) -> String {
    match raw.trim().to_ascii_lowercase().as_str() {
        "dozen" | "dozens" => "dozen".to_string(),
        "egg" | "eggs" => "egg".to_string(),
        "pair" | "pairs" => "pair".to_string(),
        "item" | "items" => "item".to_string(),
        "crate" | "crates" => "crate".to_string(),
        "box" | "boxes" => "box".to_string(),
        other => other.trim_end_matches('s').to_string(),
    }
}

fn required_focus_match_count(terms: &[String]) -> usize {
    terms.len().clamp(1, 2)
}

fn line_matches_gift_recipient_focus(lower: &str, focus: &SpendFocus) -> bool {
    let recipient_terms = if focus.optional_terms.is_empty() {
        focus
            .required_terms
            .iter()
            .filter(|term| term.as_str() != "gift")
            .collect::<Vec<_>>()
    } else {
        focus.optional_terms.iter().collect::<Vec<_>>()
    };
    if recipient_terms.is_empty()
        || !recipient_terms
            .iter()
            .all(|term| gift_recipient_term_matches(lower, term))
    {
        return false;
    }

    if task_contains_any(
        lower,
        &["gift", "gifts", "gift card", "present", "presents"],
    ) {
        return true;
    }

    (task_contains_any(
        lower,
        &[
            "spent",
            "paid",
            "bought",
            "purchased",
            "got",
            "get",
            "picked up",
            "ordered",
            "treated",
        ],
    ) && lower.contains(" for ")
        && task_contains_any(
            lower,
            &[
                "birthday",
                "baby shower",
                "graduation",
                "wedding",
                "anniversary",
                "christmas",
                "holiday",
                "bridal shower",
                "housewarming",
                "valentine",
                "valentine's",
                "mother's day",
                "mothers day",
                "father's day",
                "fathers day",
                "new baby",
            ],
        ))
        || recipient_terms.iter().any(|term| {
            let escaped = regex::escape(term.as_str());
            compile_regex(&format!(
                r"(?i)\b(?:got|bought|purchased|gave|ordered|treated)\b.{{0,20}}\b{escaped}\b"
            ))
            .is_match(lower)
        })
}

fn gift_recipient_term_matches(lower: &str, term: &str) -> bool {
    if term.is_empty() {
        return false;
    }
    let escaped = regex::escape(term);
    compile_regex(&format!(r"(?i)\b{}(?:'s)?\b", escaped)).is_match(lower)
}

fn format_grouped_dollars(dollars: u64) -> String {
    let mut remaining = dollars;
    let mut groups = Vec::new();
    loop {
        groups.push(remaining % 1_000);
        remaining /= 1_000;
        if remaining == 0 {
            break;
        }
    }
    let mut rendered = groups.pop().unwrap_or(0).to_string();
    while let Some(group) = groups.pop() {
        rendered.push_str(&format!(",{group:03}"));
    }
    rendered
}
