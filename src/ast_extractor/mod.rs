//! AST Bootstrap — extracts function signatures, types, and doc comments from
//! source files at compile time using language-specific regex patterns.
//!
//! Extracted content is inserted into Core neuron stubs so BM25 has real
//! vocabulary from day 1, before the LLM curates any neurons. The extractor
//! is intentionally lightweight: regex patterns capture the public API surface
//! without building a full parse tree.

mod call_sites;
mod lang_core;
mod lang_ext;

pub use call_sites::{extract_call_sites, CallEdge};

/// Extracted public API surface of a source file.
#[derive(Debug, Default)]
pub struct AstSummary {
    pub functions: Vec<String>,
    pub types: Vec<String>,
    pub doc_lines: Vec<String>,
    /// Extra vocabulary terms injected at soft BM25 weight (0.3×) for unknown
    /// file types. NOT included in `compute_sig_hash()` — these are soft terms
    /// only and do not trigger staleness events when they change.
    pub extra_vocab: Vec<String>,
}

impl AstSummary {
    pub fn is_empty(&self) -> bool {
        self.functions.is_empty() && self.types.is_empty() && self.doc_lines.is_empty()
    }
}

/// Extract the public API surface from `content` using `source_rel` for language detection.
pub fn extract_signatures(source_rel: &str, content: &str) -> AstSummary {
    let ext = std::path::Path::new(source_rel)
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("");

    match ext {
        "rs" => lang_core::extract_rust(content),
        "py" | "pyw" => lang_core::extract_python(content),
        "ts" | "tsx" | "js" | "jsx" | "mjs" | "cjs" => lang_core::extract_typescript(content),
        "go" => lang_core::extract_go(content),
        "swift" => lang_core::extract_swift(content),
        "kt" | "kts" => lang_core::extract_kotlin(content),
        "java" => lang_core::extract_java(content),
        "cs" => lang_core::extract_csharp(content),
        "rb" | "rake" => lang_core::extract_ruby(content),
        "c" | "cpp" | "cc" | "cxx" | "h" | "hpp" | "hxx" => lang_core::extract_c(content),
        "php" => lang_ext::extract_php(content),
        "lua" => lang_ext::extract_lua(content),
        "r" | "rmd" => lang_ext::extract_r(content),
        "jl" => lang_ext::extract_julia(content),
        "ex" | "exs" => lang_ext::extract_elixir(content),
        "zig" => lang_ext::extract_zig(content),
        "dart" => lang_ext::extract_dart(content),
        "sh" | "bash" | "zsh" | "fish" => lang_ext::extract_shell(content),
        "sql" => lang_ext::extract_sql(content),
        "tf" | "hcl" => lang_ext::extract_hcl(content),
        "proto" => lang_ext::extract_proto(content),
        "graphql" | "gql" => lang_ext::extract_graphql(content),
        "ipynb" => lang_ext::extract_jupyter(content),
        _ => lang_ext::extract_universal_fallback(content),
    }
}

/// Compute a stable 16-hex-char BLAKE3 hash of the public API surface.
///
/// Sorts function and type names before hashing so that reordering declarations
/// does not change the hash. Only names, not bodies, are included — this hash
/// changes only when the public API surface changes, not on whitespace or
/// doc-comment edits. Used by the compile pass (S1) to distinguish cosmetic
/// source changes from semantic ones.
pub fn compute_sig_hash(summary: &AstSummary) -> String {
    let mut sigs: Vec<&str> = Vec::with_capacity(summary.functions.len() + summary.types.len());
    sigs.extend(summary.functions.iter().map(String::as_str));
    sigs.extend(summary.types.iter().map(String::as_str));
    sigs.sort_unstable();
    let joined = sigs.join("\n");
    blake3::hash(joined.as_bytes()).to_hex()[..16].to_string()
}

/// Format an `AstSummary` as a compact markdown block for insertion into neuron stubs.
///
/// Returns `""` if the summary is empty. Output is capped at 600 chars.
pub fn format_for_stub(summary: &AstSummary) -> String {
    if summary.functions.is_empty() && summary.types.is_empty() {
        return String::new();
    }

    let mut parts: Vec<String> = Vec::new();
    if !summary.functions.is_empty() {
        let fns: Vec<&str> = summary
            .functions
            .iter()
            .take(10)
            .map(String::as_str)
            .collect();
        parts.push(format!("Functions: {}", fns.join(", ")));
    }
    if !summary.types.is_empty() {
        let types: Vec<&str> = summary.types.iter().take(5).map(String::as_str).collect();
        parts.push(format!("Types: {}", types.join(", ")));
    }

    let result = parts.join("\n");
    if result.len() > 600 {
        let truncated: String = result.chars().take(597).collect();
        format!("{}…", truncated)
    } else {
        result
    }
}

/// Format doc comment lines as a purpose hint for pre-populating the purpose section (A3).
pub fn format_purpose_hint(summary: &AstSummary) -> String {
    if summary.doc_lines.is_empty() {
        return String::new();
    }
    let lines: Vec<&str> = summary
        .doc_lines
        .iter()
        .take(5)
        .map(String::as_str)
        .collect();
    let result = lines.join("\n");
    if result.len() > 400 {
        let truncated: String = result.chars().take(397).collect();
        format!("{}…", truncated)
    } else {
        result
    }
}

/// Format `extra_vocab` terms for inclusion in the neuron stub as a hidden vocab section.
pub fn format_extra_vocab_for_stub(summary: &AstSummary) -> String {
    if summary.extra_vocab.is_empty() {
        return String::new();
    }
    let terms: Vec<&str> = summary
        .extra_vocab
        .iter()
        .filter(|t| t.len() >= 3)
        .take(80)
        .map(String::as_str)
        .collect();
    if terms.is_empty() {
        return String::new();
    }
    format!(
        "\n<!-- SECTION: vocab -->\n{}\n<!-- /SECTION -->\n",
        terms.join(" ")
    )
}

#[cfg(test)]
mod tests;
