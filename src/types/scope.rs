//! Module scope types — replaces stringly-typed `module: Option<String>` at API boundaries.
//!
//! The `GetContextsInput.module` field accepts `"@alice"` (person scope),
//! a plain tag string, or nothing (project-wide). Previously this was a raw
//! `Option<String>` that callers had to pattern-match manually, with no guarantee
//! that `"@alice"` and `"alice"` were normalised consistently.
//!
//! `ModuleScope::from_api_str` is the single normalisation point.

use std::fmt;

use crate::error::Result;
use serde::{Deserialize, Serialize};

// ─── PersonSlug ──────────────────────────────────────────────────────────────

/// A normalised person identifier: lowercase alphanumeric plus underscores.
///
/// Stored _without_ the `@` prefix. At MCP API boundaries the `@` is accepted
/// and stripped; internally the slug is always bare (`"alice"`, not `"@alice"`).
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct PersonSlug(String);

impl PersonSlug {
    /// Parse a person slug, accepting an optional leading `@`.
    ///
    /// Valid chars after stripping `@`: `[a-z0-9_]`, non-empty.
    ///
    /// # Errors
    ///
    /// Returns an error if the underlying operation fails.
    pub fn parse(s: &str) -> Result<Self> {
        let s = s.strip_prefix('@').unwrap_or(s);
        if s.is_empty() {
            crate::cortyx_bail!("PersonSlug must not be empty");
        }
        if !s
            .chars()
            .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_')
        {
            crate::cortyx_bail!(
                "PersonSlug must contain only lowercase letters, digits, or underscores: {:?}",
                s
            );
        }
        Ok(Self(s.to_string()))
    }

    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// Returns the slug with its `@` prefix, as used in neuron paths and display.
    #[must_use]
    pub fn with_at_prefix(&self) -> String {
        format!("@{}", self.0)
    }
}

impl fmt::Display for PersonSlug {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "@{}", self.0)
    }
}

// ─── ModuleScope ─────────────────────────────────────────────────────────────

/// The scope of a context retrieval or neuron filter operation.
///
/// Replaces the raw `Option<String>` `module` field that mixed `@alice`,
/// `alice`, and plain tags without a single normalisation path.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ModuleScope {
    /// No module filter — search the entire project.
    ProjectWide,
    /// Filter to a specific person's memory namespace (`@alice`).
    Person(PersonSlug),
    /// Filter to a named module or tag (arbitrary string, non-empty).
    Tag(String),
}

impl ModuleScope {
    /// Parse from an API string value.
    ///
    /// - `None` or empty → `ProjectWide`
    /// - Starts with `@` (e.g. `"@alice"`) → `Person`
    /// - Anything else → `Tag`
    pub fn from_api_str(s: Option<&str>) -> Self {
        match s {
            None | Some("") => Self::ProjectWide,
            Some(s) if s.starts_with('@') => PersonSlug::parse(s)
                .map(Self::Person)
                .unwrap_or_else(|_| Self::Tag(s.to_string())),
            Some(s) => Self::Tag(s.to_string()),
        }
    }

    /// Return the string representation used in neuron path filters.
    ///
    /// `ProjectWide` returns `None` (no filter applied).
    #[must_use]
    pub fn as_filter_str(&self) -> Option<&str> {
        match self {
            Self::ProjectWide => None,
            Self::Person(slug) => Some(slug.as_str()),
            Self::Tag(tag) => Some(tag.as_str()),
        }
    }

    #[must_use]
    pub fn is_project_wide(&self) -> bool {
        matches!(self, Self::ProjectWide)
    }
}

impl fmt::Display for ModuleScope {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ProjectWide => write!(f, "(project-wide)"),
            Self::Person(slug) => write!(f, "{slug}"),
            Self::Tag(tag) => write!(f, "{tag}"),
        }
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn person_slug_strips_at() {
        let s = PersonSlug::parse("@alice").unwrap();
        assert_eq!(s.as_str(), "alice");
        assert_eq!(s.with_at_prefix(), "@alice");
    }

    #[test]
    fn person_slug_without_at() {
        let s = PersonSlug::parse("alice").unwrap();
        assert_eq!(s.as_str(), "alice");
    }

    #[test]
    fn person_slug_rejects_uppercase() {
        assert!(PersonSlug::parse("Alice").is_err());
    }

    #[test]
    fn person_slug_rejects_empty() {
        assert!(PersonSlug::parse("").is_err());
        assert!(PersonSlug::parse("@").is_err());
    }

    #[test]
    fn person_slug_allows_underscore_digits() {
        assert!(PersonSlug::parse("alice_1").is_ok());
        assert!(PersonSlug::parse("user_42").is_ok());
    }

    #[test]
    fn module_scope_from_none_is_project_wide() {
        assert_eq!(ModuleScope::from_api_str(None), ModuleScope::ProjectWide);
        assert_eq!(
            ModuleScope::from_api_str(Some("")),
            ModuleScope::ProjectWide
        );
    }

    #[test]
    fn module_scope_from_at_is_person() {
        let scope = ModuleScope::from_api_str(Some("@alice"));
        assert!(matches!(scope, ModuleScope::Person(_)));
        assert_eq!(scope.as_filter_str(), Some("alice"));
    }

    #[test]
    fn module_scope_from_tag() {
        let scope = ModuleScope::from_api_str(Some("auth"));
        assert_eq!(scope, ModuleScope::Tag("auth".to_string()));
        assert_eq!(scope.as_filter_str(), Some("auth"));
    }

    #[test]
    fn project_wide_has_no_filter() {
        assert_eq!(ModuleScope::ProjectWide.as_filter_str(), None);
        assert!(ModuleScope::ProjectWide.is_project_wide());
    }

    #[test]
    fn serde_round_trip() {
        let scope = ModuleScope::from_api_str(Some("@bob"));
        let json = serde_json::to_string(&scope).unwrap();
        let back: ModuleScope = serde_json::from_str(&json).unwrap();
        assert_eq!(scope, back);
    }
}
