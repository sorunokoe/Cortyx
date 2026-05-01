//! Special-purpose AST extractors (Jupyter notebooks, universal fallback).

use super::super::AstSummary;

/// Jupyter notebook extractor — parses `cells[]` from `.ipynb` JSON, extracts code
/// cells, and runs the Python extractor on the concatenated source.
pub fn extract_jupyter(content: &str) -> AstSummary {
    let Ok(json) = serde_json::from_str::<serde_json::Value>(content) else {
        return extract_universal_fallback(content);
    };
    let Some(cells) = json.get("cells").and_then(|v| v.as_array()) else {
        return extract_universal_fallback(content);
    };

    let mut code = String::new();
    for cell in cells {
        if cell.get("cell_type").and_then(|t| t.as_str()) != Some("code") {
            continue;
        }
        match cell.get("source") {
            Some(serde_json::Value::String(s)) => {
                code.push_str(s);
                code.push('\n');
            },
            Some(serde_json::Value::Array(lines)) => {
                for line in lines {
                    if let Some(s) = line.as_str() {
                        code.push_str(s);
                    }
                }
                code.push('\n');
            },
            _ => {},
        }
    }

    if code.trim().is_empty() {
        extract_universal_fallback(content)
    } else {
        super::super::lang_core::extract_python(&code)
    }
}

/// Universal Level-0 AST harvester — extracts soft vocabulary for any unknown file type.
pub fn extract_universal_fallback(content: &str) -> AstSummary {
    const COMMENT_PREFIXES: &[&str] = &["//", "/*", "#", "--", "%%", ";;", "(*", "<!--"];
    const PROG_KEYWORDS: &[&str] = &[
        "def",
        "fn",
        "func",
        "fun",
        "function",
        "sub",
        "proc",
        "method",
        "class",
        "struct",
        "type",
        "interface",
        "module",
        "package",
        "namespace",
        "import",
        "export",
        "use",
        "require",
        "include",
        "public",
        "private",
        "protected",
        "static",
        "async",
        "await",
        "return",
        "self",
        "this",
    ];

    let mut vocab: std::collections::HashSet<String> = std::collections::HashSet::new();
    let mut doc_lines: Vec<String> = Vec::new();

    for line in content.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }

        let is_comment = COMMENT_PREFIXES
            .iter()
            .any(|prefix| trimmed.starts_with(prefix));
        if is_comment && doc_lines.len() < 5 {
            let stripped = trimmed.trim_start_matches(|c: char| !c.is_alphabetic());
            if stripped.len() >= 4 {
                doc_lines.push(stripped.chars().take(120).collect());
            }
        }

        for token in trimmed.split(|c: char| !c.is_alphanumeric() && c != '_') {
            if token.len() >= 3 && token.len() <= 40 && !token.chars().all(|c| c.is_ascii_digit()) {
                vocab.insert(token.to_lowercase());
            }
        }
    }

    let detected_keywords: Vec<String> = PROG_KEYWORDS
        .iter()
        .filter(|&&kw| vocab.contains(kw))
        .map(|s| s.to_string())
        .collect();

    let mut extra_vocab: Vec<String> = vocab
        .into_iter()
        .filter(|t| {
            !matches!(
                t.as_str(),
                "the"
                    | "and"
                    | "for"
                    | "this"
                    | "that"
                    | "with"
                    | "from"
                    | "into"
                    | "are"
                    | "was"
                    | "has"
                    | "not"
                    | "can"
                    | "will"
                    | "its"
            )
        })
        .collect();
    extra_vocab.sort_unstable();
    extra_vocab.truncate(100);

    AstSummary {
        doc_lines,
        extra_vocab: if !detected_keywords.is_empty() || !extra_vocab.is_empty() {
            let mut combined = detected_keywords;
            combined.extend(extra_vocab);
            combined.dedup();
            combined
        } else {
            vec![]
        },
        ..Default::default()
    }
}
