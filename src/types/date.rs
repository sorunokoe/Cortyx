//! Typed ISO-8601 date — replaces bare `String` date fields in `KgFact` and temporal reasoning.
//!
//! `IsoDate` is either a valid `YYYY-MM-DD` string or the sentinel empty string
//! representing "open-ended" (no end date, or no start date known).
//! Lexicographic ordering is correct for `YYYY-MM-DD` strings, so `PartialOrd`/`Ord`
//! are derived directly from the inner `String`.

use std::fmt;

use crate::error::Result;
use serde::{Deserialize, Serialize};

// ─── IsoDate ─────────────────────────────────────────────────────────────────

/// A validated ISO-8601 date (`YYYY-MM-DD`) or the open-ended sentinel (`""`).
///
/// Comparison operators work correctly: `"2024-01-01" < "2024-06-15"`.
/// An open date (`""`) compares as _less than_ any concrete date, which means
/// `fact.ended.is_open()` means the fact is still active.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct IsoDate(String);

impl IsoDate {
    /// Parse a `YYYY-MM-DD` string. Returns an error on wrong length or non-digit characters.
    /// Accepts `""` as the open-ended sentinel.
    pub fn parse(s: &str) -> Result<Self> {
        if s.is_empty() {
            return Ok(Self(String::new()));
        }
        if s.len() != 10 {
            crate::cortyx_bail!(
                "IsoDate must be YYYY-MM-DD (10 chars) or empty, got {:?}",
                s
            );
        }
        let bytes = s.as_bytes();
        if bytes[4] != b'-' || bytes[7] != b'-' {
            crate::cortyx_bail!("IsoDate must be YYYY-MM-DD, got {:?}", s);
        }
        for (i, &b) in bytes.iter().enumerate() {
            if i == 4 || i == 7 {
                continue;
            }
            if !b.is_ascii_digit() {
                crate::cortyx_bail!("IsoDate non-digit at position {i} in {:?}", s);
            }
        }
        Ok(Self(s.to_string()))
    }

    /// The open-ended sentinel (empty string).
    pub fn open() -> Self {
        Self(String::new())
    }

    /// Today's date in `YYYY-MM-DD` format, computed from `SystemTime`.
    pub fn today() -> Self {
        let secs = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        let days = secs / 86400;
        let (y, mo, d) = civil_from_days(days as i64);
        Self(format!("{y:04}-{mo:02}-{d:02}"))
    }

    /// Returns `true` if this is the open-ended sentinel.
    pub fn is_open(&self) -> bool {
        self.0.is_empty()
    }

    /// Borrow the inner `YYYY-MM-DD` string, or `""` for open.
    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// Returns `true` if `as_of` falls within `[valid_from, ended)`.
    ///
    /// `ended.is_open()` means "still active". `valid_from.is_open()` means
    /// "active from the beginning of time".
    pub fn is_active_at(valid_from: &Self, ended: &Self, as_of: &Self) -> bool {
        if !ended.is_open() && as_of >= ended {
            return false;
        }
        if !valid_from.is_open() && as_of < valid_from {
            return false;
        }
        true
    }
}

impl fmt::Display for IsoDate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(if self.0.is_empty() { "(open)" } else { &self.0 })
    }
}

impl Default for IsoDate {
    fn default() -> Self {
        Self::open()
    }
}

/// Hinnant civil-from-days algorithm: maps days-since-epoch to `(year, month, day)`.
fn civil_from_days(z: i64) -> (i64, i64, i64) {
    let z = z + 719468;
    let era = if z >= 0 { z } else { z - 146096 } / 146097;
    let doe = z - era * 146097;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    (y, m, d)
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_valid() {
        let d = IsoDate::parse("2024-06-15").unwrap();
        assert_eq!(d.as_str(), "2024-06-15");
        assert!(!d.is_open());
    }

    #[test]
    fn parse_empty_is_open() {
        let d = IsoDate::parse("").unwrap();
        assert!(d.is_open());
    }

    #[test]
    fn parse_rejects_bad_format() {
        assert!(IsoDate::parse("20240615").is_err());
        assert!(IsoDate::parse("2024-6-15").is_err());
        assert!(IsoDate::parse("2024-06-1X").is_err());
    }

    #[test]
    fn ordering_is_chronological() {
        let a = IsoDate::parse("2024-01-01").unwrap();
        let b = IsoDate::parse("2024-06-15").unwrap();
        let c = IsoDate::parse("2025-01-01").unwrap();
        assert!(a < b);
        assert!(b < c);
    }

    #[test]
    fn open_is_less_than_any_date() {
        let open = IsoDate::open();
        let d = IsoDate::parse("2000-01-01").unwrap();
        assert!(open < d);
    }

    #[test]
    fn is_active_at_closed_range() {
        let from = IsoDate::parse("2024-01-01").unwrap();
        let end = IsoDate::parse("2024-12-31").unwrap();
        let mid = IsoDate::parse("2024-06-01").unwrap();
        let before = IsoDate::parse("2023-12-31").unwrap();
        let after = IsoDate::parse("2025-01-01").unwrap();

        assert!(IsoDate::is_active_at(&from, &end, &mid));
        assert!(!IsoDate::is_active_at(&from, &end, &before));
        assert!(!IsoDate::is_active_at(&from, &end, &after));
    }

    #[test]
    fn is_active_at_open_ended() {
        let from = IsoDate::parse("2024-01-01").unwrap();
        let open = IsoDate::open();
        let future = IsoDate::parse("2099-01-01").unwrap();
        assert!(IsoDate::is_active_at(&from, &open, &future));
    }

    #[test]
    fn today_has_correct_format() {
        let t = IsoDate::today();
        assert_eq!(t.as_str().len(), 10);
        assert_eq!(t.as_str().as_bytes()[4], b'-');
        assert_eq!(t.as_str().as_bytes()[7], b'-');
    }

    #[test]
    fn serde_round_trip() {
        let d = IsoDate::parse("2024-03-22").unwrap();
        let json = serde_json::to_string(&d).unwrap();
        let back: IsoDate = serde_json::from_str(&json).unwrap();
        assert_eq!(d, back);
    }
}
