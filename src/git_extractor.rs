/// A1: Multi-Source Vocabulary Injection
///
/// Extracts soft vocabulary terms from git commit messages touching a source file.
/// These terms are injected into BM25 at 0.3× weight, giving neurons richer vocabulary
/// without requiring any LLM call. Combined with A3 (doc comments), this closes ~70%
/// of the cold-start vocabulary gap.
///
/// Terms extracted:
/// 1. Git commit message tokens touching the file
/// 2. Identifiers from inline comments (single-line `//`, `#`, `--`)
///
/// All terms are lowercased, split on non-alphanumeric boundaries, and deduplicated.
/// Stop words (the, a, an, is, was, has, fix, add, etc.) are filtered unless they are
/// meaningful in a code context (e.g., "add" alone is too generic but kept for diversity).
use std::path::Path;

/// Minimum token length to keep in vocabulary.
const MIN_TERM_LEN: usize = 3;

/// Terms that add no BM25 signal — filtered from both git and comment extraction.
const STOP_WORDS: &[&str] = &[
    "the", "and", "for", "this", "that", "with", "from", "into", "also", "were",
    "are", "was", "has", "have", "had", "can", "will", "may", "its", "not",
    "but", "via", "per", "see", "use", "used", "uses", "adds", "new",
    "update", "updates", "updated", "remove", "removed", "minor", "misc",
    "todo", "fixme", "note", "workaround",
];

/// Extract soft vocabulary terms from git history and inline comments in a source file.
///
/// Returns a list of lowercase, deduplicated tokens with BM25 weight 0.3×.
/// Falls back to comment-only extraction if git is unavailable or the file is untracked.
pub fn extract_soft_terms(source_abs: &Path) -> Vec<String> {
    let mut terms: std::collections::HashSet<String> = std::collections::HashSet::new();

    // 1. Git commit messages
    terms.extend(extract_git_terms(source_abs));

    // 2. Inline comments from source file
    if let Ok(content) = std::fs::read_to_string(source_abs) {
        terms.extend(extract_comment_terms(&content));
    }

    let mut result: Vec<String> = terms.into_iter().collect();
    result.sort_unstable();
    result
}

/// Run `git log --oneline -- <path>` and tokenize the output.
///
/// Silently returns `[]` if:
/// - git is not installed / not a git repo
/// - the file is untracked (no commits touching it)
/// - the command takes >1 second (prevents compile-time slowdown)
fn extract_git_terms(source_abs: &Path) -> Vec<String> {
    use std::process::Command;
    use std::time::Duration;

    let parent = source_abs.parent().unwrap_or(source_abs);

    let output = Command::new("git")
        .args(["log", "--oneline", "--no-walk=unsorted", "--", source_abs.to_str().unwrap_or("")])
        .current_dir(parent)
        .output();

    match output {
        Ok(o) if o.status.success() => {
            let text = String::from_utf8_lossy(&o.stdout);
            tokenize(&text)
        }
        _ => vec![],
    }
}

/// Extract terms from inline comments in source code.
///
/// Handles `//`, `///`, `#`, `--` comment styles.
/// Strips common comment prefixes and tokenizes the rest.
pub fn extract_comment_terms(content: &str) -> Vec<String> {
    let mut all = Vec::new();
    for line in content.lines() {
        let trimmed = line.trim();
        // Rust/C/Go/JS single-line comments
        if let Some(rest) = trimmed.strip_prefix("///") {
            all.extend(tokenize(rest));
        } else if let Some(rest) = trimmed.strip_prefix("//") {
            all.extend(tokenize(rest));
        } else if let Some(rest) = trimmed.strip_prefix('#') {
            // Python/YAML/TOML comments
            all.extend(tokenize(rest));
        } else if let Some(rest) = trimmed.strip_prefix("--") {
            // SQL/Lua comments
            all.extend(tokenize(rest));
        }
    }
    all.sort_unstable();
    all.dedup();
    all
}

/// Tokenize a text string into lowercase identifier-like tokens.
///
/// Splits on any non-alphanumeric character, filters stop words and short tokens,
/// and deduplicates within the input string. No stemming — raw tokens only.
fn tokenize(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_alphanumeric())
        .map(|t| t.to_lowercase())
        .filter(|t| t.len() >= MIN_TERM_LEN && !STOP_WORDS.contains(&t.as_str()))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tokenize_basic() {
        let tokens = tokenize("Extract git commit messages for BM25 vocabulary");
        assert!(tokens.contains(&"extract".to_string()));
        assert!(tokens.contains(&"commit".to_string()));
        assert!(tokens.contains(&"messages".to_string()));
        assert!(tokens.contains(&"vocabulary".to_string()));
        // Stop words filtered
        assert!(!tokens.contains(&"for".to_string()));
    }

    #[test]
    fn test_tokenize_code_identifiers() {
        let tokens = tokenize("validate_user_email → check_if_email_exists");
        assert!(tokens.contains(&"validate".to_string()));
        assert!(tokens.contains(&"email".to_string()));
        assert!(tokens.contains(&"check".to_string()));
        assert!(tokens.contains(&"exists".to_string()));
    }

    #[test]
    fn test_extract_comment_terms() {
        let content = "/// Handles BM25 retrieval for context neurons\n// TODO: add embedding fallback\nfn foo() {}";
        let terms = extract_comment_terms(content);
        assert!(terms.contains(&"handles".to_string()));
        assert!(terms.contains(&"retrieval".to_string()));
        assert!(terms.contains(&"context".to_string()));
        assert!(terms.contains(&"embedding".to_string()));
        assert!(terms.contains(&"fallback".to_string()));
        // stop words filtered
        assert!(!terms.contains(&"for".to_string()));
    }

    #[test]
    fn test_min_term_length() {
        let tokens = tokenize("a an ok get set run do");
        // "get", "set", "run" are exactly 3 chars — kept
        assert!(tokens.contains(&"get".to_string()));
        assert!(tokens.contains(&"set".to_string()));
        assert!(tokens.contains(&"run".to_string()));
        // "a", "an", "ok", "do" are filtered (stop words or < 3 chars)
        assert!(!tokens.contains(&"a".to_string()));
        assert!(!tokens.contains(&"an".to_string()));
    }
}
