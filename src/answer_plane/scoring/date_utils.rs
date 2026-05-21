use super::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ExplicitDateMatch {
    Day { year: i32, month: u32, day: u32 },
    Month { year: i32, month: u32 },
}

type Ymd = (i32, u32, u32);
type YmdRange = (Ymd, Ymd);

impl ExplicitDateMatch {
    fn start(self) -> (i32, u32, u32) {
        match self {
            Self::Day { year, month, day } => (year, month, day),
            Self::Month { year, month } => (year, month, 1),
        }
    }

    fn end(self) -> (i32, u32, u32) {
        match self {
            Self::Day { year, month, day } => (year, month, day),
            Self::Month { year, month } => (year, month, days_in_month(year, month)),
        }
    }
}

pub(crate) fn extract_session_base_date(content: &str) -> Option<(i32, u32, u32)> {
    content
        .lines()
        .take(8)
        .filter(|line| !is_session_metadata_line(line))
        .find_map(|line| extract_explicit_date(line, None))
}

fn is_session_metadata_line(line: &str) -> bool {
    let trimmed = line.trim_start();
    trimmed.starts_with("<!--") && trimmed.contains("timestamp:")
}

pub(crate) fn extract_temporal_rank(line: &str, base_date: Option<(i32, u32, u32)>) -> Option<i32> {
    if let Some(date) = extract_explicit_date(line, base_date) {
        return Some(ymd_to_days(date.0, date.1, date.2));
    }
    if let Some(days_ago) = extract_relative_days(line) {
        if let Some(base) = base_date {
            let base_days = ymd_to_days(base.0, base.1, base.2);
            Some(base_days - days_ago)
        } else {
            Some(-days_ago)
        }
    } else {
        None
    }
}

#[must_use]
pub(crate) fn extract_explicit_date_range(text: &str, base_date: Option<Ymd>) -> Option<YmdRange> {
    let date = extract_explicit_date_match(text, base_date)?;
    Some((date.start(), date.end()))
}

/// Extract the first explicit date mention in a line of text.
#[must_use]
pub fn extract_explicit_date(text: &str, base_date: Option<Ymd>) -> Option<Ymd> {
    match extract_explicit_date_match(text, base_date)? {
        ExplicitDateMatch::Day { year, month, day } => Some((year, month, day)),
        ExplicitDateMatch::Month { year, month } => Some((year, month, 15)),
    }
}

pub(crate) fn extract_explicit_date_match(
    text: &str,
    base_date: Option<(i32, u32, u32)>,
) -> Option<ExplicitDateMatch> {
    let lower = text.to_ascii_lowercase();
    let year_hint = base_date.map(|(year, _, _)| year);
    if let Some(date) = extract_numeric_slash_date(text, year_hint) {
        return Some(ExplicitDateMatch::Day {
            year: date.0,
            month: date.1,
            day: date.2,
        });
    }
    if let Some(date) = extract_iso8601_date(text) {
        return Some(ExplicitDateMatch::Day {
            year: date.0,
            month: date.1,
            day: date.2,
        });
    }
    for (month_number, month) in (1_u32..).zip([
        "january",
        "february",
        "march",
        "april",
        "may",
        "june",
        "july",
        "august",
        "september",
        "october",
        "november",
        "december",
    ]) {
        if let Some(pos) = lower.find(month) {
            let before = &lower[..pos];
            let after = &lower[pos + month.len()..];
            let year = extract_year_near(after).or(year_hint).unwrap_or(2023);
            if let Some(day) = extract_nearest_day(before, after, &lower, pos) {
                return Some(ExplicitDateMatch::Day {
                    year,
                    month: month_number,
                    day,
                });
            }
            if let Some(day) = extract_month_qualifier_day(before, &lower, pos) {
                return Some(ExplicitDateMatch::Day {
                    year,
                    month: month_number,
                    day,
                });
            }
            return Some(ExplicitDateMatch::Month {
                year,
                month: month_number,
            });
        }
    }
    if let Some(date) = extract_named_holiday_date(&lower, year_hint) {
        return Some(ExplicitDateMatch::Day {
            year: date.0,
            month: date.1,
            day: date.2,
        });
    }
    None
}

fn extract_iso8601_date(text: &str) -> Option<(i32, u32, u32)> {
    text.split_whitespace().find_map(|raw| {
        let clean = raw.trim_matches(|c: char| {
            !c.is_ascii_alphanumeric() && !matches!(c, '-' | 'T' | ':' | 'Z')
        });
        parse_iso8601_date_prefix(clean)
    })
}

fn parse_iso8601_date_prefix(text: &str) -> Option<(i32, u32, u32)> {
    let prefix = text.get(..10)?;
    let [year, month, day] = prefix.split('-').collect::<Vec<_>>()[..] else {
        return None;
    };
    let year = year.parse::<i32>().ok()?;
    let month = month.parse::<u32>().ok()?;
    let day = day.parse::<u32>().ok()?;
    ((1..=12).contains(&month) && (1..=31).contains(&day)).then_some((year, month, day))
}

fn extract_month_qualifier_day(before: &str, lower: &str, month_pos: usize) -> Option<u32> {
    if before.ends_with("mid-") || before.ends_with("mid ") {
        return Some(15);
    }
    if before.ends_with("early-") || before.ends_with("early ") {
        return Some(5);
    }
    if before.ends_with("late-") || before.ends_with("late ") {
        return Some(25);
    }
    let around = safe_slice(
        lower,
        month_pos.saturating_sub(8),
        (month_pos + 20).min(lower.len()),
    );
    if around.contains("mid-") || around.contains("mid ") {
        Some(15)
    } else if around.contains("early-") || around.contains("early ") {
        Some(5)
    } else if around.contains("late-") || around.contains("late ") {
        Some(25)
    } else {
        None
    }
}

pub(crate) fn extract_numeric_slash_date(
    text: &str,
    year_hint: Option<i32>,
) -> Option<(i32, u32, u32)> {
    for raw in text.split_whitespace() {
        let clean = raw.trim_matches(|c: char| !c.is_ascii_digit() && c != '/');
        if clean.len() < 3 || !clean.contains('/') {
            continue;
        }
        let parts = clean
            .split('/')
            .filter(|part| !part.is_empty())
            .collect::<Vec<_>>();
        if parts.len() < 2 || parts.len() > 3 {
            continue;
        }
        let Some(month) = parts[0].parse::<u32>().ok() else {
            continue;
        };
        let Some(day) = parts[1].parse::<u32>().ok() else {
            continue;
        };
        if !(1..=12).contains(&month) || !(1..=31).contains(&day) {
            continue;
        }
        let year = parts
            .get(2)
            .and_then(|part| {
                if part.len() == 4 {
                    part.parse::<i32>().ok()
                } else {
                    None
                }
            })
            .or(year_hint)
            .unwrap_or(2023);
        return Some((year, month, day));
    }
    None
}

pub(crate) fn extract_named_holiday_date(
    lower: &str,
    year_hint: Option<i32>,
) -> Option<(i32, u32, u32)> {
    let year = year_hint.unwrap_or(2023);
    if lower.contains("black friday") {
        return Some(black_friday_date(year));
    }
    if lower.contains("thanksgiving") {
        return Some(thanksgiving_date(year));
    }
    if lower.contains("christmas eve") {
        return Some((year, 12, 24));
    }
    if lower.contains("christmas") {
        return Some((year, 12, 25));
    }
    if lower.contains("maundy thursday") {
        return Some(shift_date_by_days(easter_sunday_date(year), -3));
    }
    if lower.contains("good friday") {
        return Some(shift_date_by_days(easter_sunday_date(year), -2));
    }
    if lower.contains("ash wednesday") {
        return Some(shift_date_by_days(easter_sunday_date(year), -46));
    }
    if lower.contains("easter monday") {
        return Some(shift_date_by_days(easter_sunday_date(year), 1));
    }
    if lower.contains("easter sunday") || contains_standalone_token(lower, "easter") {
        return Some(easter_sunday_date(year));
    }
    if lower.contains("holi") {
        return Some(match year {
            2023 => (2023, 3, 8),
            2024 => (2024, 3, 25),
            2025 => (2025, 3, 14),
            2026 => (2026, 3, 3),
            _ => (year, 3, 8),
        });
    }
    None
}

pub(crate) fn thanksgiving_date(year: i32) -> (i32, u32, u32) {
    let november_first = ymd_to_days(year, 11, 1);
    let november_first_weekday = (4 + november_first).rem_euclid(7);
    let days_until_thursday = (4 - november_first_weekday).rem_euclid(7);
    let thanksgiving_day = 1 + days_until_thursday.cast_unsigned() + 21;
    (year, 11, thanksgiving_day)
}

pub(crate) fn black_friday_date(year: i32) -> (i32, u32, u32) {
    shift_date_by_days(thanksgiving_date(year), 1)
}

pub(crate) fn easter_sunday_date(year: i32) -> (i32, u32, u32) {
    let a = year % 19;
    let b = year / 100;
    let c = year % 100;
    let d = b / 4;
    let e = b % 4;
    let f = (b + 8) / 25;
    let g = (b - f + 1) / 3;
    let h = (19 * a + b - d - g + 15) % 30;
    let i = c / 4;
    let k = c % 4;
    let l = (32 + 2 * e + 2 * i - h - k) % 7;
    let m = (a + 11 * h + 22 * l) / 451;
    let month = (h + l - 7 * m + 114) / 31;
    let day = ((h + l - 7 * m + 114) % 31) + 1;
    (year, month.cast_unsigned(), day.cast_unsigned())
}

pub(crate) fn extract_nearest_day(
    before: &str,
    after: &str,
    _lower: &str,
    _month_pos: usize,
) -> Option<u32> {
    extract_last_number(before)
        .or_else(|| extract_first_number(after))
        .and_then(|value| (1..=31).contains(&value).then_some(value.cast_unsigned()))
}

pub(crate) fn extract_year_near(after: &str) -> Option<i32> {
    after
        .split(|c: char| !c.is_ascii_digit())
        .find_map(|token| {
            if token.len() == 4 {
                token.parse::<i32>().ok()
            } else {
                None
            }
        })
}

pub(crate) fn extract_last_number(text: &str) -> Option<i32> {
    text.split(|c: char| !c.is_ascii_digit())
        .filter(|token| !token.is_empty())
        .filter_map(|token| token.parse::<i32>().ok())
        .next_back()
}

pub(crate) fn extract_first_number(text: &str) -> Option<i32> {
    text.split(|c: char| !c.is_ascii_digit()).find_map(|token| {
        (!token.is_empty())
            .then(|| token.parse::<i32>().ok())
            .flatten()
    })
}

pub(crate) fn extract_relative_date(
    text: &str,
    base_date: (i32, u32, u32),
) -> Option<(i32, u32, u32)> {
    let lower = text.to_ascii_lowercase();
    if lower.contains("yesterday") {
        return Some(shift_date_by_days(base_date, -1));
    }
    if lower.contains("today") {
        return Some(base_date);
    }
    if lower.contains("tomorrow") {
        return Some(shift_date_by_days(base_date, 1));
    }
    if lower.contains("a couple of days ago") {
        return Some(shift_date_by_days(base_date, -2));
    }
    if lower.contains("a few days ago") {
        return Some(shift_date_by_days(base_date, -3));
    }
    if let Some(target_weekday) = extract_last_weekday(&lower) {
        return Some(resolve_last_weekday(base_date, target_weekday));
    }

    for unit in ["day", "week", "month"] {
        for marker in [format!("{unit} ago"), format!("{unit}s ago")] {
            if !lower.contains(&marker) {
                continue;
            }
            let Some(prefix) = lower.split(&marker).next() else {
                continue;
            };
            let Some(amount) = extract_trailing_count(prefix) else {
                continue;
            };
            return Some(match unit {
                "day" => shift_date_by_days(base_date, -amount),
                "week" => shift_date_by_days(base_date, -(amount * 7)),
                "month" => shift_date_by_months(base_date, -amount),
                _ => base_date,
            });
        }
    }
    None
}

pub(crate) fn extract_relative_days(text: &str) -> Option<i32> {
    let lower = text.to_ascii_lowercase();
    if lower.contains("yesterday") {
        return Some(1);
    }
    if lower.contains("a couple of days ago") {
        return Some(2);
    }
    if lower.contains("a few days ago") {
        return Some(3);
    }
    if lower.contains("last week") {
        return Some(7);
    }
    if lower.contains("last month") {
        return Some(30);
    }
    if [
        "monday",
        "tuesday",
        "wednesday",
        "thursday",
        "friday",
        "saturday",
        "sunday",
    ]
    .iter()
    .any(|day| lower.contains(&format!("last {day}")))
    {
        return Some(7);
    }

    for unit in ["day", "week", "month"] {
        for marker in [format!("{unit} ago"), format!("{unit}s ago")] {
            if !lower.contains(&marker) {
                continue;
            }
            if let Some(prefix) = lower.split(&marker).next() {
                if let Some(amount) = extract_trailing_count(prefix) {
                    let scale = match unit {
                        "day" => 1,
                        "week" => 7,
                        "month" => 30,
                        _ => 1,
                    };
                    return Some(amount * scale);
                }
            }
        }
    }
    None
}

fn extract_last_weekday(lower: &str) -> Option<i32> {
    (0_i32..)
        .zip([
            "sunday",
            "monday",
            "tuesday",
            "wednesday",
            "thursday",
            "friday",
            "saturday",
        ])
        .find_map(|(weekday, name)| lower.contains(&format!("last {name}")).then_some(weekday))
}

fn resolve_last_weekday(base_date: (i32, u32, u32), target_weekday: i32) -> (i32, u32, u32) {
    let current_weekday = weekday_number(base_date.0, base_date.1, base_date.2);
    let mut delta = (current_weekday - target_weekday).rem_euclid(7);
    if delta == 0 {
        delta = 7;
    }
    shift_date_by_days(base_date, -delta)
}

fn shift_date_by_months(base_date: (i32, u32, u32), delta_months: i32) -> (i32, u32, u32) {
    let (year, month) = shift_month(base_date.0, base_date.1, delta_months);
    (year, month, base_date.2.min(days_in_month(year, month)))
}

fn days_in_month(year: i32, month: u32) -> u32 {
    match month {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 if (year % 4 == 0 && year % 100 != 0) || year % 400 == 0 => 29,
        2 => 28,
        _ => 30,
    }
}

fn weekday_number(year: i32, month: u32, day: u32) -> i32 {
    (ymd_to_days(year, month, day) + 4).rem_euclid(7)
}

pub(crate) fn extract_trailing_count(prefix: &str) -> Option<i32> {
    let token = prefix
        .split_whitespace()
        .rev()
        .find(|token| !token.is_empty())?;
    parse_count_token(token)
}

pub(crate) fn parse_count_token(token: &str) -> Option<i32> {
    let clean = token
        .trim_matches(|c: char| !c.is_ascii_alphanumeric() && c != '+')
        .trim_end_matches('+');
    if let Ok(value) = clean.parse::<i32>() {
        return Some(value);
    }
    match clean {
        "a" | "an" | "one" => Some(1),
        "two" => Some(2),
        "three" => Some(3),
        "four" => Some(4),
        "five" => Some(5),
        "six" => Some(6),
        "seven" => Some(7),
        "eight" => Some(8),
        "nine" => Some(9),
        "ten" => Some(10),
        "eleven" => Some(11),
        "twelve" => Some(12),
        "couple" => Some(2),
        "few" => Some(3),
        _ => None,
    }
}

pub(crate) fn ymd_to_days(year: i32, month: u32, day: u32) -> i32 {
    const MONTH_START_DAYS: [i32; 12] = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334];
    let leap_years = {
        let y = year - 1;
        y / 4 - y / 100 + y / 400 - (1969 / 4 - 1969 / 100 + 1969 / 400)
    };
    let month_index = usize::try_from(month.saturating_sub(1)).unwrap_or_default();
    let day = i32::try_from(day).unwrap_or(i32::MAX);
    (year - 1970) * 365 + leap_years + MONTH_START_DAYS[month_index] + day - 1
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extracts_iso8601_datetime_as_day() {
        assert_eq!(
            extract_explicit_date("Deployment completed at 2023-01-19T14:30:00Z.", None),
            Some((2023, 1, 19))
        );
    }

    #[test]
    fn extracts_ordinal_named_dates() {
        assert_eq!(
            extract_explicit_date("The kickoff was on January 19th, 2023.", None),
            Some((2023, 1, 19))
        );
        assert_eq!(
            extract_explicit_date("The retrofit finished on 19th of January 2023.", None),
            Some((2023, 1, 19))
        );
    }

    #[test]
    fn extracts_month_year_as_full_month_range() {
        assert_eq!(
            extract_explicit_date_range("The migration happened in January 2023.", None),
            Some(((2023, 1, 1), (2023, 1, 31)))
        );
    }

    #[test]
    fn resolves_relative_dates_with_anchor() {
        assert_eq!(
            extract_relative_date("We met last Tuesday.", (2024, 3, 15)),
            Some((2024, 3, 12))
        );
        assert_eq!(
            extract_relative_date("The rollout started 3 months ago.", (2024, 5, 31)),
            Some((2024, 2, 29))
        );
    }
}
