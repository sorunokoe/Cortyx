//! Newtype wrappers for common primitives to prevent type confusion.
//!
//! These types enforce domain invariants at construction time:
//! - `TokenCount`: Non-negative token counts
//! - `TokenBudget`: Token budget with validation
//! - `QueryText`: Non-empty, trimmed query strings
//! - `TermFrequency`: Non-negative term frequencies

use crate::error::Result;
use serde::{Deserialize, Serialize};
use std::fmt;

// ─── TokenCount ──────────────────────────────────────────────────────────────

/// A validated token count (non-negative).
///
/// Prevents confusing token counts with other numeric types and ensures
/// counts are never negative.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct TokenCount(usize);

impl TokenCount {
    pub const ZERO: Self = Self(0);

    #[must_use]
    pub fn new(count: usize) -> Self {
        Self(count)
    }

    #[must_use]
    pub fn get(self) -> usize {
        self.0
    }

    #[must_use]
    pub fn is_zero(self) -> bool {
        self.0 == 0
    }

    #[must_use]
    pub fn saturating_add(self, other: Self) -> Self {
        Self(self.0.saturating_add(other.0))
    }

    #[must_use]
    pub fn saturating_sub(self, other: Self) -> Self {
        Self(self.0.saturating_sub(other.0))
    }
}

impl fmt::Display for TokenCount {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl From<usize> for TokenCount {
    fn from(count: usize) -> Self {
        Self::new(count)
    }
}

// ─── TokenBudget ─────────────────────────────────────────────────────────────

/// A validated token budget with optional upper limit.
///
/// Prevents requesting unreasonably large token budgets (>100k).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct TokenBudget(usize);

impl TokenBudget {
    /// Maximum reasonable token budget (100k tokens).
    pub const MAX: usize = 100_000;

    /// # Errors
    ///
    /// Returns an error if the underlying operation fails.
    pub fn new(budget: usize) -> Result<Self> {
        if budget > Self::MAX {
            crate::cortyx_bail!("Token budget {} exceeds maximum {}", budget, Self::MAX);
        }
        Ok(Self(budget))
    }

    /// Create without validation (for internal use).
    #[must_use]
    pub fn new_unchecked(budget: usize) -> Self {
        Self(budget)
    }

    #[must_use]
    pub fn get(self) -> usize {
        self.0
    }

    #[must_use]
    pub fn remaining(self, used: TokenCount) -> TokenCount {
        TokenCount::new(self.0.saturating_sub(used.get()))
    }

    #[must_use]
    pub fn can_fit(self, tokens: TokenCount) -> bool {
        tokens.get() <= self.0
    }
}

impl fmt::Display for TokenBudget {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl TryFrom<usize> for TokenBudget {
    type Error = crate::error::CortyxError;

    fn try_from(budget: usize) -> Result<Self> {
        Self::new(budget)
    }
}

// ─── QueryText ───────────────────────────────────────────────────────────────

/// A validated, non-empty query string.
///
/// Ensures queries are trimmed and not empty. Prevents passing empty
/// queries to search functions.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct QueryText(String);

impl QueryText {
    /// # Errors
    ///
    /// Returns an error if the underlying operation fails.
    pub fn new(text: impl Into<String>) -> Result<Self> {
        let text = text.into();
        let trimmed = text.trim().to_string();

        if trimmed.is_empty() {
            crate::cortyx_bail!("Query text must not be empty");
        }

        Ok(Self(trimmed))
    }

    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }

    #[must_use]
    pub fn into_string(self) -> String {
        self.0
    }
}

impl fmt::Display for QueryText {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl AsRef<str> for QueryText {
    fn as_ref(&self) -> &str {
        &self.0
    }
}

impl TryFrom<String> for QueryText {
    type Error = crate::error::CortyxError;

    fn try_from(text: String) -> Result<Self> {
        Self::new(text)
    }
}

impl TryFrom<&str> for QueryText {
    type Error = crate::error::CortyxError;

    fn try_from(text: &str) -> Result<Self> {
        Self::new(text)
    }
}

// ─── TermFrequency ───────────────────────────────────────────────────────────

/// A non-negative term frequency value.
///
/// Used in BM25 scoring. Prevents negative frequencies which would be
/// nonsensical.
#[derive(Debug, Default, Clone, Copy, PartialEq, PartialOrd, Serialize, Deserialize)]
pub struct TermFrequency(f32);

impl TermFrequency {
    pub const ZERO: Self = Self(0.0);

    #[must_use]
    pub fn new(freq: f32) -> Self {
        Self(freq.max(0.0))
    }

    #[must_use]
    pub fn get(self) -> f32 {
        self.0
    }

    #[must_use]
    pub fn is_zero(self) -> bool {
        self.0 == 0.0
    }
}

impl std::ops::AddAssign<f32> for TermFrequency {
    fn add_assign(&mut self, rhs: f32) {
        self.0 = (self.0 + rhs).max(0.0);
    }
}

impl fmt::Display for TermFrequency {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:.4}", self.0)
    }
}

impl From<f32> for TermFrequency {
    fn from(freq: f32) -> Self {
        Self::new(freq)
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn token_count_operations() {
        let a = TokenCount::new(100);
        let b = TokenCount::new(50);

        assert_eq!(a.saturating_add(b), TokenCount::new(150));
        assert_eq!(a.saturating_sub(b), TokenCount::new(50));
        assert_eq!(b.saturating_sub(a), TokenCount::ZERO);
    }

    #[test]
    fn token_budget_validates_max() {
        assert!(TokenBudget::new(TokenBudget::MAX).is_ok());
        assert!(TokenBudget::new(TokenBudget::MAX + 1).is_err());
    }

    #[test]
    fn token_budget_remaining() {
        let budget = TokenBudget::new(1000).unwrap();
        let used = TokenCount::new(300);

        assert_eq!(budget.remaining(used), TokenCount::new(700));
    }

    #[test]
    fn token_budget_can_fit() {
        let budget = TokenBudget::new(1000).unwrap();

        assert!(budget.can_fit(TokenCount::new(500)));
        assert!(budget.can_fit(TokenCount::new(1000)));
        assert!(!budget.can_fit(TokenCount::new(1001)));
    }

    #[test]
    fn query_text_rejects_empty() {
        assert!(QueryText::new("").is_err());
        assert!(QueryText::new("   ").is_err());
    }

    #[test]
    fn query_text_trims() {
        let query = QueryText::new("  hello  ").unwrap();
        assert_eq!(query.as_str(), "hello");
    }

    #[test]
    fn term_frequency_never_negative() {
        assert_eq!(TermFrequency::new(-1.0).get(), 0.0);
        assert_eq!(TermFrequency::new(5.0).get(), 5.0);
    }

    #[test]
    fn serde_round_trip() {
        let count = TokenCount::new(42);
        let json = serde_json::to_string(&count).unwrap();
        let back: TokenCount = serde_json::from_str(&json).unwrap();
        assert_eq!(count, back);

        let query = QueryText::new("test query").unwrap();
        let json = serde_json::to_string(&query).unwrap();
        let back: QueryText = serde_json::from_str(&json).unwrap();
        assert_eq!(query, back);
    }
}
