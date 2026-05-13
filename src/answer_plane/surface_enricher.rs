//! Retroactive answer-surface enrichment from LLM responses.
//!
//! TRIZ Concepts A + D: when a neuron is hard-cited in an LLM response,
//! extract overlapping sentence spans and append them to the neuron's
//! `## answer_surface` section, converting each response into training
//! data for future zero-inference answer mode.

use std::collections::HashSet;
use std::path::{Path, PathBuf};

use crate::index::tokenize;
use crate::neuron::{atomic_write, parse_sections, replace_section};

/// An extracted answer span from an LLM response.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AnswerSpan {
    /// Sentence or short paragraph extracted from the response.
    pub text: String,
    /// Number of neuron vocabulary terms overlapping with this span.
    pub overlap: usize,
}

/// For each hard-cited neuron, extract sentence spans from `response_text`
/// that overlap with the neuron's vocabulary and append them to the neuron's
/// `## answer_surface` section.
///
/// - `hard_cited_paths`: absolute paths to `.context.md` neuron files
///   (pass only neurons with ≥30 BM25 term overlap — "hard cited")
/// - `response_text`: the full LLM response text
///
/// Returns the number of neuron files successfully updated.
pub fn enrich_neuron_answer_surfaces(response_text: &str, hard_cited_paths: &[PathBuf]) -> usize {
    if response_text.is_empty() || hard_cited_paths.is_empty() {
        return 0;
    }

    let response_spans = split_into_spans(response_text);
    if response_spans.is_empty() {
        return 0;
    }

    let mut updated = 0usize;
    for neuron_path in hard_cited_paths {
        if enrich_one(neuron_path, &response_spans).unwrap_or(0) > 0 {
            updated += 1;
        }
    }
    updated
}

/// Enrich a single neuron file. Returns `Some(spans_written)` on success, `None` on I/O error.
fn enrich_one(neuron_path: &Path, response_spans: &[String]) -> Option<usize> {
    let content = std::fs::read_to_string(neuron_path).ok()?;

    // Build neuron vocabulary from its full text content.
    let vocab: HashSet<String> = tokenize(&content).into_iter().collect();
    if vocab.is_empty() {
        return Some(0);
    }

    let spans = extract_matching_spans(response_spans, &vocab, 4, 5);
    if spans.is_empty() {
        return Some(0);
    }

    // Append new spans to any existing answer_surface section (dedup by prefix).
    let existing_surface = parse_sections(&content)
        .remove("answer_surface")
        .unwrap_or_default();

    let new_lines: Vec<String> = spans
        .into_iter()
        .filter(|span| {
            // Skip if first 40 chars already present (dedup guard).
            let prefix: String = span.text.chars().take(40).collect();
            !existing_surface.contains(prefix.as_str())
        })
        .map(|span| format!("- {}", span.text.trim()))
        .collect();

    if new_lines.is_empty() {
        return Some(0);
    }

    let merged = if existing_surface.trim().is_empty() {
        new_lines.join("\n")
    } else {
        format!("{}\n{}", existing_surface.trim_end(), new_lines.join("\n"))
    };

    let new_content = replace_section(&content, "answer_surface", &merged);
    atomic_write(neuron_path, new_content.as_bytes())
        .ok()
        .map(|()| new_lines.len())
}

/// Split response text into sentence/paragraph-level spans.
/// Keeps lines ≥20 characters; collapses internal whitespace.
fn split_into_spans(text: &str) -> Vec<String> {
    text.split('\n')
        .map(str::trim)
        .filter(|line| line.len() >= 20)
        .map(|line| line.split_whitespace().collect::<Vec<_>>().join(" "))
        .collect()
}

/// Return up to `max_results` spans sharing ≥ `min_overlap` tokens with `vocab`,
/// sorted by overlap descending then text ascending for determinism.
fn extract_matching_spans(
    spans: &[String],
    vocab: &HashSet<String>,
    min_overlap: usize,
    max_results: usize,
) -> Vec<AnswerSpan> {
    let mut scored: Vec<AnswerSpan> = spans
        .iter()
        .filter_map(|span| {
            let span_tokens: HashSet<String> = tokenize(span).into_iter().collect();
            let overlap = span_tokens.intersection(vocab).count();
            if overlap >= min_overlap {
                Some(AnswerSpan {
                    text: span.clone(),
                    overlap,
                })
            } else {
                None
            }
        })
        .collect();

    scored.sort_by(|a, b| b.overlap.cmp(&a.overlap).then_with(|| a.text.cmp(&b.text)));
    scored.truncate(max_results);
    scored
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn split_into_spans_filters_short_lines() {
        let text = "short\n\nThis is a longer sentence that should be kept.\nok";
        let spans = split_into_spans(text);
        assert!(
            spans.iter().all(|s| s.len() >= 20),
            "all spans must be ≥20 chars"
        );
        assert!(
            spans.iter().any(|s| s.contains("longer sentence")),
            "the qualifying line must be present"
        );
    }

    #[test]
    fn extract_matching_spans_filters_by_overlap() {
        let vocab: HashSet<String> = ["foo", "bar", "baz", "qux"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        let spans = vec![
            "foo bar baz qux extra words here to make it long enough".to_string(),
            "completely unrelated words that do not match the vocab at all".to_string(),
            "foo only one match but this is a longer line with more content here".to_string(),
        ];
        let results = extract_matching_spans(&spans, &vocab, 4, 5);
        assert_eq!(results.len(), 1, "only the 4-overlap span should pass");
        assert!(results[0].text.contains("foo bar baz qux"));
    }

    #[test]
    fn extract_matching_spans_respects_max_results() {
        let vocab: HashSet<String> = (0..20).map(|i| format!("word{i}")).collect();
        let spans: Vec<String> = (0..10)
            .map(|i| {
                format!(
                    "word{} word{} word{} word{} word{}",
                    i,
                    i + 1,
                    i + 2,
                    i + 3,
                    i + 4
                )
            })
            .collect();
        let results = extract_matching_spans(&spans, &vocab, 4, 3);
        assert!(results.len() <= 3, "must respect max_results cap");
    }

    #[test]
    fn enrich_answer_surfaces_returns_zero_for_empty_input() {
        assert_eq!(enrich_neuron_answer_surfaces("", &[]), 0);
        assert_eq!(enrich_neuron_answer_surfaces("some text", &[]), 0);
        assert_eq!(
            enrich_neuron_answer_surfaces("", &[PathBuf::from("a.md")]),
            0
        );
    }
}
