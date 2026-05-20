use std::fmt;

/// A single fact triple with optional temporal validity window.
#[derive(Debug, Clone, PartialEq)]
pub struct KgFact {
    pub predicate: String,
    pub value: String,
    /// ISO-8601 date/datetime string (e.g. "2024-01-15") or empty string if unknown.
    pub valid_from: String,
    /// ISO-8601 date/datetime string when this fact was superseded/ended, or empty.
    pub ended: String,
}

impl KgFact {
    /// Returns `true` if this fact is active as of `as_of` (ISO-8601 date string).
    ///
    /// Rules:
    /// - If `ended` is non-empty and `as_of >= ended`, the fact is inactive.
    /// - With no `as_of`, only open-ended facts are considered currently active.
    /// - Otherwise the fact is considered active.
    #[must_use]
    pub fn is_active(&self, as_of: Option<&str>) -> bool {
        if self.ended.is_empty() {
            return true;
        }
        match as_of {
            Some(d) => d < self.ended.as_str(),
            None => false,
        }
    }
}

impl fmt::Display for KgFact {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "| {} | {} | {} | {} |",
            escape_table_cell(&self.predicate),
            escape_table_cell(&self.value),
            escape_table_cell(&self.valid_from),
            escape_table_cell(&self.ended)
        )
    }
}

pub(super) fn escape_table_cell(value: &str) -> String {
    let mut escaped = String::with_capacity(value.len());
    for ch in value.chars() {
        match ch {
            '\\' => escaped.push_str("\\\\"),
            '|' => escaped.push_str("\\|"),
            _ => escaped.push(ch),
        }
    }
    escaped
}

pub(super) fn split_table_row(row: &str) -> Vec<String> {
    let mut cols = Vec::new();
    let mut current = String::new();
    let mut escaped = false;

    for ch in row.trim().trim_matches('|').chars() {
        if escaped {
            match ch {
                '\\' | '|' => current.push(ch),
                _ => {
                    current.push('\\');
                    current.push(ch);
                },
            }
            escaped = false;
            continue;
        }

        match ch {
            '\\' => escaped = true,
            '|' => {
                cols.push(current.trim().to_string());
                current.clear();
            },
            _ => current.push(ch),
        }
    }

    if escaped {
        current.push('\\');
    }
    cols.push(current.trim().to_string());
    cols
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kg_is_active_temporal_window() {
        let fact = KgFact {
            predicate: "lead".into(),
            value: "Alice".into(),
            valid_from: "2024-01-01".into(),
            ended: "2024-06-01".into(),
        };
        assert!(fact.is_active(Some("2024-03-01")));
        assert!(!fact.is_active(Some("2024-06-01")));
        assert!(!fact.is_active(Some("2025-01-01")));
    }
}
