//! Money combination extractors (stub - TODO: restore from backup)

use super::*;
use super::money_support::*;

pub(super) fn extract_spend_focus_fact_from_line(
    _line: &str,
    _lower: &str,
    _focus: &SpendFocus,
) -> Option<MoneyAmountFact> {
    None // Stub implementation
}

pub(super) fn extract_keyed_gift_recipient_fact_from_line(
    _line: &str,
    _lower: &str,
    _focus: &SpendFocus,
) -> Option<KeyedMoneyAmountFact> {
    None // Stub implementation
}

pub(super) fn extract_contextual_spend_followup_fact_from_line(
    _line: &str,
    _lower: &str,
) -> Option<MoneyAmountFact> {
    None // Stub implementation
}

pub(super) fn extract_sale_value_fact_from_line(
    _line: &str,
    _lower: &str,
    _focus: &SpendFocus,
) -> Option<MoneyAmountFact> {
    None // Stub implementation
}
