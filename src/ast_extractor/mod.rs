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
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.functions.is_empty() && self.types.is_empty() && self.doc_lines.is_empty()
    }
}

/// Extract the public API surface from `content` using `source_rel` for language detection.
#[must_use]
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

/// Generate a `## Relevant For` section populated with natural-language task descriptions.
///
/// Derived purely from the AST summary and source path — zero LLM calls. Gives BM25
/// rich task vocabulary from session 0, resolving the cold-start contradiction (TRIZ R5).
///
/// Examples of generated phrases:
/// - "authenticating users with JWT tokens"
/// - "validating request credentials"
/// - "parsing configuration from environment"
#[must_use]
pub fn format_relevant_for_stub(summary: &AstSummary, source_rel: &str) -> String {
    let mut phrases: Vec<String> = Vec::new();

    // 1. Phrases from function names (verb + noun pairs from snake_case)
    for fn_name in summary.functions.iter().take(12) {
        // Strip type prefix (e.g. "fn " or "async fn ")
        let bare = fn_name
            .trim_start_matches("async ")
            .trim_start_matches("fn ")
            .split('(')
            .next()
            .unwrap_or(fn_name)
            .trim();
        let words = split_identifier(bare);
        if words.len() >= 2 {
            phrases.push(words.join(" "));
        }
    }

    // 2. Phrases from type names (decomposed CamelCase → words)
    for type_name in summary.types.iter().take(6) {
        let bare = type_name
            .trim_start_matches("struct ")
            .trim_start_matches("enum ")
            .trim_start_matches("trait ")
            .trim_start_matches("class ")
            .trim_start_matches("interface ")
            .split('<')
            .next()
            .unwrap_or(type_name)
            .trim();
        let words = split_camel_case(bare);
        if words.len() >= 2 {
            phrases.push(words.join(" ").to_lowercase());
        }
    }

    // 3. Phrases from module path segments
    let path_phrases = path_task_hints(source_rel);
    phrases.extend(path_phrases);

    // 4. First doc line as a direct phrase
    if let Some(first_doc) = summary.doc_lines.first() {
        let cleaned = first_doc
            .trim_start_matches("///")
            .trim_start_matches("//!")
            .trim_start_matches('#')
            .trim();
        if cleaned.len() > 10 {
            phrases.push(cleaned.to_string());
        }
    }

    // Deduplicate while preserving order, cap at 10 phrases
    let mut seen = std::collections::HashSet::new();
    let phrases: Vec<String> = phrases
        .into_iter()
        .filter(|p| p.len() > 4 && seen.insert(p.to_lowercase()))
        .take(10)
        .collect();

    if phrases.is_empty() {
        return String::new();
    }

    let body: String = phrases
        .iter()
        .map(|p| format!("- {p}"))
        .collect::<Vec<_>>()
        .join("\n");
    format!("\n<!-- SECTION: relevant_for -->\n{body}\n<!-- /SECTION -->\n")
}

/// Split a snake_case identifier into words (handles digits as word breaks too).
fn split_identifier(s: &str) -> Vec<String> {
    s.split(['_', '-'])
        .filter(|w| !w.is_empty() && w.len() > 1)
        .map(|w| w.to_lowercase())
        .collect()
}

/// Split a CamelCase or PascalCase identifier into words.
fn split_camel_case(s: &str) -> Vec<String> {
    let mut words = Vec::new();
    let mut current = String::new();
    for (i, ch) in s.chars().enumerate() {
        if ch.is_uppercase() && i > 0 && !current.is_empty() {
            words.push(current.clone());
            current.clear();
        }
        current.push(ch.to_ascii_lowercase());
    }
    if !current.is_empty() {
        words.push(current);
    }
    words.into_iter().filter(|w| w.len() > 1).collect()
}

/// Derive task hint phrases from path segments (e.g. "src/auth/mod.rs" → ["authentication", "auth"]).
fn path_task_hints(source_rel: &str) -> Vec<String> {
    // Common path segment → expanded meanings
    let expansions: &[(&str, &[&str])] = &[
        ("auth", &["authentication", "authorization", "auth"]),
        ("login", &["user login", "sign in"]),
        ("token", &["token handling", "JWT"]),
        ("config", &["configuration", "settings"]),
        ("db", &["database access", "persistence"]),
        ("api", &["API endpoint", "REST"]),
        ("parse", &["parsing", "deserialization"]),
        ("index", &["indexing", "search index"]),
        ("cache", &["caching", "memoization"]),
        ("error", &["error handling", "error types"]),
        ("util", &["utilities", "helpers"]),
        ("test", &["testing", "test fixtures"]),
        ("compile", &["compilation", "code generation"]),
        ("sync", &["synchronization", "sync"]),
        ("transport", &["data transport", "serialization"]),
        ("server", &["server setup", "request handling"]),
        ("client", &["client connection", "HTTP client"]),
        ("model", &["data model", "domain model"]),
        ("migrate", &["migration", "schema migration"]),
        ("render", &["rendering", "output formatting"]),
        ("search", &["search", "full text search"]),
        ("neuron", &["neuron management", "context storage"]),
        ("memory", &["memory", "episodic memory"]),
        ("mcp", &["MCP tools", "MCP server"]),
        ("graph", &["graph traversal", "knowledge graph"]),
    ];

    let mut hints = Vec::new();
    let lower = source_rel.to_lowercase();
    let segments: Vec<&str> = lower
        .split(['/', '\\', '_', '-', '.'])
        .filter(|s| !s.is_empty() && *s != "rs" && *s != "ts" && *s != "py")
        .collect();

    for seg in &segments {
        for (key, expanded) in expansions {
            if seg.contains(key) {
                for &phrase in *expanded {
                    hints.push(phrase.to_string());
                }
                break;
            }
        }
    }
    hints
}

#[cfg(test)]
mod tests;
