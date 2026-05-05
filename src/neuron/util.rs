use crate::types::TokenCount;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};

pub(super) static NEURON_UUID_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Rough token count estimate tuned for mixed code/prose and non-ASCII text.
///
/// ASCII-heavy text averages ~4 chars/token, while CJK scripts are closer to
/// ~1 char/token. This keeps mixed-language neurons from being severely undercounted.
///
/// # Error bounds
/// This is a heuristic, not a byte-pair encoding (BPE) count. Measured error
/// on English prose is ±15–25% vs. OpenAI `cl100k_base` / `o200k_base`; code
/// with heavy punctuation (Rust, C++, regex) can under-count by up to 50%
/// because short symbols tokenize to individual tokens rather than sharing a
/// 4-char slot. CJK estimates are generally accurate (±5%).
///
/// Budgets derived from this estimate should include a conservative headroom
/// (e.g., multiply by 1.3) before comparing against a hard token limit.
pub fn estimate_tokens(text: &str) -> TokenCount {
    let mut ascii_chars = 0usize;
    let mut cjk_chars = 0usize;
    let mut other_unicode_chars = 0usize;

    for ch in text.chars() {
        if ch.is_ascii() {
            ascii_chars += 1;
        } else if is_cjk_char(ch) {
            cjk_chars += 1;
        } else {
            other_unicode_chars += 1;
        }
    }

    let ascii_tokens = ascii_chars.div_ceil(4);
    let other_unicode_tokens = other_unicode_chars.div_ceil(2);
    TokenCount::new((ascii_tokens + cjk_chars + other_unicode_tokens).max(1))
}

pub fn estimate_context_tokens(text: &str) -> TokenCount {
    estimate_tokens(&strip_context_only_sections(text))
}

fn strip_context_only_sections(content: &str) -> String {
    let without_query = strip_named_token_section(content, "query_surface");
    strip_named_token_section(&without_query, "answer_surface")
}

fn strip_named_token_section(content: &str, section_name: &str) -> String {
    let header = format!("## {section_name}");
    let marker = format!("<!-- SECTION: {section_name} -->");
    let end_marker = "<!-- /SECTION -->";
    let Some(header_start) = content.find(&header) else {
        return content.to_string();
    };
    let Some(section_start_rel) = content[header_start..].find(&marker) else {
        return content.to_string();
    };
    let section_start = header_start + section_start_rel;
    let Some(section_end_rel) = content[section_start..].find(end_marker) else {
        return content.to_string();
    };
    let section_end = section_start + section_end_rel + end_marker.len();

    let mut stripped = String::with_capacity(content.len());
    stripped.push_str(content[..header_start].trim_end());
    if !stripped.ends_with('\n') {
        stripped.push('\n');
    }
    let tail = content[section_end..].trim_start_matches('\n');
    if !tail.is_empty() {
        stripped.push('\n');
        stripped.push_str(tail);
    }
    stripped
}

fn is_cjk_char(ch: char) -> bool {
    matches!(
        ch as u32,
        0x3040..=0x30ff
            | 0x3400..=0x4dbf
            | 0x4e00..=0x9fff
            | 0xac00..=0xd7af
            | 0xf900..=0xfaff
            | 0x20000..=0x2a6df
            | 0x2a700..=0x2b73f
            | 0x2b740..=0x2b81f
            | 0x2b820..=0x2ceaf
            | 0x2f800..=0x2fa1f
    )
}

/// BLAKE3 hash of a file's contents, returned as a 16-char hex prefix.
/// Returns `None` on error (file may not exist yet).
pub fn hash_file(path: &Path) -> Option<String> {
    let data = std::fs::read(path).ok()?;
    let hash = blake3::hash(&data);
    Some(hash.to_hex()[..16].to_string())
}

/// Current time as RFC 3339 / ISO 8601 string (UTC, second precision).
///
/// Uses Hinnant's civil calendar algorithm — no external crate required.
pub fn now_iso8601() -> String {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    let (y, mo, d, h, mi, s) = unix_secs_to_datetime(secs);
    format!("{y:04}-{mo:02}-{d:02}T{h:02}:{mi:02}:{s:02}Z")
}

/// S-XI (R16): Generate a stable neuron UUID from source path + current nanoseconds.
///
/// Uses BLAKE3 over `{path}:{nanos}` to produce a 32-char hex string.
/// Called once at neuron creation; thereafter the UUID is loaded from sidecar JSON.
pub fn generate_neuron_uuid(source: &Path) -> String {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    let nonce = NEURON_UUID_COUNTER.fetch_add(1, Ordering::Relaxed);
    let input = format!("{}:{nanos}:{nonce}", source.display());
    let hash = blake3::hash(input.as_bytes());
    hash.to_hex()[..32].to_string()
}

/// Decompose Unix epoch seconds into `(year, month, day, hour, minute, second)`.
pub fn unix_secs_to_datetime(secs: u64) -> (u32, u32, u32, u32, u32, u32) {
    let h = (secs / 3600) % 24;
    let mi = (secs / 60) % 60;
    let s = secs % 60;
    let (y, mo, d) = days_to_ymd((secs / 86400) as i64);
    (y as u32, mo as u32, d as u32, h as u32, mi as u32, s as u32)
}

/// Convert days since Unix epoch to (year, month, day) using Hinnant's algorithm.
pub fn days_to_ymd(z: i64) -> (i32, i32, i32) {
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
    (y as i32, m as i32, d as i32)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn estimate_tokens_counts_cjk_by_character() {
        assert_eq!(estimate_tokens("你好世界").get(), 4);
        assert_eq!(estimate_tokens("abcd").get(), 1);
    }

    #[test]
    fn estimate_context_tokens_ignores_hidden_answer_surfaces() {
        let content = "# Note\n\nVisible body.\n\n## answer_surface\n<!-- SECTION: answer_surface -->\n| question_pattern | answer_span | confidence |\n| --- | --- | --- |\n| role | reviewer | 0.90 |\n<!-- /SECTION -->\n";
        assert_eq!(
            estimate_context_tokens(content),
            estimate_tokens("# Note\n\nVisible body.\n")
        );
    }

    #[test]
    fn now_iso8601_format() {
        let s = now_iso8601();
        assert!(s.ends_with('Z'), "should be UTC: {s}");
        assert_eq!(s.len(), 20, "YYYY-MM-DDTHH:MM:SSZ: {s}");
    }

    #[test]
    fn days_to_ymd_known_dates() {
        assert_eq!(days_to_ymd(0), (1970, 1, 1));
        assert_eq!(days_to_ymd(20089), (2025, 1, 1));
        assert_eq!(days_to_ymd(11016), (2000, 2, 29));
    }

    #[test]
    fn generate_neuron_uuid_changes_on_repeated_calls() {
        let path = Path::new("/tmp/example.context.md");
        let a = generate_neuron_uuid(path);
        let b = generate_neuron_uuid(path);
        assert_ne!(a, b);
        assert_eq!(a.len(), 32);
        assert_eq!(b.len(), 32);
    }
}
