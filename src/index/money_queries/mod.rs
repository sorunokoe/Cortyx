//! Money query parsing and dispatch.

use super::money_support::MoneyQuery;

mod helpers;
mod parsers;

use parsers::*;

pub(super) fn parse_money_query(_task: &str, task_lower: &str) -> Option<MoneyQuery> {
    if let Some(query) = parse_cashback_query(task_lower) {
        return Some(MoneyQuery::Cashback(query));
    }
    if let Some(query) = parse_discount_percent_query(task_lower) {
        return Some(MoneyQuery::DiscountPercent(query));
    }
    if let Some(query) = parse_savings_query(task_lower) {
        return Some(MoneyQuery::Savings(query));
    }
    if let Some(query) = parse_single_recipient_gift_total_query(task_lower) {
        return Some(MoneyQuery::RecipientGiftTotal(query));
    }
    if let Some(query) = parse_spend_sum_query(task_lower) {
        return Some(MoneyQuery::SpendSum(query));
    }
    if let Some(query) = parse_sale_minimum_query(task_lower) {
        return Some(MoneyQuery::SaleMinimum(query));
    }
    if let Some(query) = parse_raised_total_query(task_lower) {
        return Some(MoneyQuery::RaisedTotal(query));
    }
    parse_revenue_query(task_lower).map(MoneyQuery::Revenue)
}
