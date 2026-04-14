//! AST Bootstrap — extracts function signatures, types, and doc comments from
//! source files at compile time using language-specific regex patterns.
//!
//! Extracted content is inserted into Core neuron stubs so BM25 has real
//! vocabulary from day 1, before the LLM curates any neurons. The extractor
//! is intentionally lightweight: regex patterns capture the public API surface
//! without building a full parse tree.

use regex::Regex;
use std::sync::OnceLock;

// ─── Regex accessors (compiled once via OnceLock) ────────────────────────────

fn rust_fn_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"(?m)^\s*pub(?:\s+\w+)*\s+fn\s+(\w+)").unwrap())
}

fn rust_type_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"(?m)^\s*pub\s+(?:struct|enum|trait|type)\s+(\w+)").unwrap())
}

fn rust_doc_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"(?m)^\s*///\s*(.+)").unwrap())
}

fn py_fn_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"(?m)^\s*(?:async\s+)?def\s+(\w+)").unwrap())
}

fn py_class_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"(?m)^class\s+(\w+)").unwrap())
}

/// Captures the first triple-quoted docstring in a Python file.
/// Python convention: module docstring is the first string literal in the file.
fn py_docstring_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        // Matches """...""" or '''...''' (non-greedy, DOTALL via [\s\S])
        Regex::new(r#"(?:"""([\s\S]*?)"""|'''([\s\S]*?)''')"#).unwrap()
    })
}

/// Captures Go doc comments — lines of `// text` immediately before a func/type declaration.
fn go_doc_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"(?m)^//\s*(.+)").unwrap())
}

fn ts_fn_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(r"(?m)^export\s+(?:default\s+)?(?:async\s+)?function\s+(\w+)").unwrap()
    })
}

fn ts_arrow_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(r"(?m)^export\s+const\s+(\w+)\s*=\s*(?:async\s+)?\(").unwrap()
    })
}

fn ts_type_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(r"(?m)^export\s+(?:class|interface|type|enum)\s+(\w+)").unwrap()
    })
}

fn go_fn_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    // Only exported (uppercase) identifiers — unexported functions are internal
    RE.get_or_init(|| Regex::new(r"(?m)^func\s+(?:\([^)]+\)\s+)?([A-Z]\w*)").unwrap())
}

fn go_type_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"(?m)^type\s+([A-Z]\w*)").unwrap())
}

// ── Swift ──────────────────────────────────────────────────────────────────────

fn swift_fn_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"(?m)^\s*(?:public\s+|open\s+|internal\s+)?(?:override\s+)?(?:static\s+)?func\s+(\w+)").unwrap())
}

fn swift_type_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"(?m)^\s*(?:public\s+|open\s+)?(?:final\s+)?(?:class|struct|protocol|enum|extension|actor)\s+(\w+)").unwrap())
}

fn swift_doc_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"(?m)^\s*///\s*(.+)").unwrap())
}

// ── Kotlin ────────────────────────────────────────────────────────────────────

fn kotlin_fn_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"(?m)^\s*(?:public\s+|protected\s+|internal\s+)?(?:override\s+)?(?:suspend\s+)?fun\s+(\w+)").unwrap())
}

fn kotlin_type_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"(?m)^\s*(?:data\s+|sealed\s+|abstract\s+)?(?:class|interface|object|enum\s+class)\s+(\w+)").unwrap())
}

// ── Java ──────────────────────────────────────────────────────────────────────

fn java_fn_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        // public/protected [static] [final] ReturnType methodName(
        Regex::new(r"(?m)^\s*(?:public|protected)\s+(?:static\s+)?(?:final\s+)?[\w<>\[\]]+\s+(\w+)\s*\(").unwrap()
    })
}

fn java_type_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"(?m)^\s*(?:public\s+)?(?:abstract\s+|final\s+)?(?:class|interface|enum|record)\s+(\w+)").unwrap())
}

fn java_doc_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"(?m)^\s*\*\s+(.+)").unwrap())
}

// ── C# ────────────────────────────────────────────────────────────────────────

fn csharp_fn_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        // public/protected [static] [async] ReturnType MethodName(
        Regex::new(r"(?m)^\s*(?:public|protected)\s+(?:static\s+)?(?:async\s+)?(?:override\s+)?[\w<>\[\]?]+\s+([A-Z]\w+)\s*[\(<]").unwrap()
    })
}

fn csharp_type_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"(?m)^\s*(?:public\s+)?(?:abstract\s+|sealed\s+|static\s+|partial\s+)?(?:class|interface|struct|enum|record)\s+(\w+)").unwrap())
}

fn csharp_doc_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"(?m)^\s*///\s*<summary>\s*(.+?)\s*(?:</summary>|$)").unwrap())
}

// ── Ruby ──────────────────────────────────────────────────────────────────────

fn ruby_fn_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"(?m)^\s*def\s+(\w+[\?!]?)").unwrap())
}

fn ruby_type_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"(?m)^\s*(?:class|module)\s+(\w+)").unwrap())
}

// ── C / C++ ───────────────────────────────────────────────────────────────────

fn c_fn_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    // Match C/C++ function definitions: ReturnType functionName(
    // Captures functions at column 0 or minimal indent to avoid matching inner lambdas.
    RE.get_or_init(|| Regex::new(r"(?m)^[\w\s\*]+\s+(\w+)\s*\([^;{]*\)\s*(?:const\s*)?\{").unwrap())
}

fn c_type_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"(?m)^(?:typedef\s+)?(?:struct|enum|class|union)\s+(\w+)").unwrap())
}

fn c_doc_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"(?m)^\s*(?:///?|\*)\s*(.+)").unwrap())
}

// ─── Call-site detection ──────────────────────────────────────────────────────

/// A detected call-site edge: this file calls `callee_fn` which is defined in `callee_file`.
#[derive(Debug, Clone)]
pub struct CallEdge {
    /// Relative path of the file that *defines* the callee function.
    pub callee_file: std::path::PathBuf,
    /// Name of the called function (for logging / debugging).
    #[allow(dead_code)]
    pub callee_fn: String,
}

/// Scan `content` (a source file at `source_rel`) for calls to public functions
/// defined in *other* files of the project.
///
/// `vocab` maps `function_name → relative_source_path` built from all neurons'
/// `AstSummary.functions` during the compile pass.  Only calls to functions that
/// appear in the vocabulary are emitted — external / stdlib calls are ignored.
///
/// One `CallEdge` per unique callee file is returned (duplicates collapsed).
pub fn extract_call_sites(
    source_rel: &str,
    content: &str,
    vocab: &std::collections::HashMap<String, std::path::PathBuf>,
) -> Vec<CallEdge> {
    if vocab.is_empty() {
        return vec![];
    }

    // Collect the function names defined *in this file* so we don't emit self-loops.
    let self_fns: std::collections::HashSet<String> = {
        let summary = extract_signatures(source_rel, content);
        summary.functions.into_iter().collect()
    };

    // Regex: any word boundary `\bname\s*(` — a direct function call.
    // We iterate over all vocab entries and test for presence in content.
    // This is O(|vocab| × |content_len|) in the worst case but vocab is typically
    // ≤ 500 entries and content ≤ 50 KB, so runtime is negligible per-file.
    let mut seen_files: std::collections::HashSet<std::path::PathBuf> = std::collections::HashSet::new();
    let mut edges: Vec<CallEdge> = Vec::new();

    for (fn_name, callee_path) in vocab {
        // Skip self-defined functions to avoid self-loops.
        if self_fns.contains(fn_name) {
            continue;
        }
        // Skip the trivial case where callee_path == source_rel.
        if callee_path == std::path::Path::new(source_rel) {
            continue;
        }
        // Skip if we already emitted an edge to this file.
        if seen_files.contains(callee_path.as_path()) {
            continue;
        }
        // Fast substring check before compiling per-name regex.
        // The function name must appear followed by `(` somewhere in the file.
        let needle = format!("{fn_name}(");
        if content.contains(&needle) {
            seen_files.insert(callee_path.clone());
            edges.push(CallEdge {
                callee_file: callee_path.clone(),
                callee_fn: fn_name.clone(),
            });
        }
    }

    edges
}

// ─── Summary ─────────────────────────────────────────────────────────────────

/// Extracted public API surface of a source file.
#[derive(Debug, Default)]
pub struct AstSummary {
    pub functions: Vec<String>,
    pub types: Vec<String>,
    pub doc_lines: Vec<String>,
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
        "rs" => extract_rust(content),
        "py" => extract_python(content),
        "ts" | "tsx" | "js" | "jsx" | "mjs" | "cjs" => extract_typescript(content),
        "go" => extract_go(content),
        "swift" => extract_swift(content),
        "kt" | "kts" => extract_kotlin(content),
        "java" => extract_java(content),
        "cs" => extract_csharp(content),
        "rb" => extract_ruby(content),
        "c" | "cpp" | "cc" | "cxx" | "h" | "hpp" | "hxx" => extract_c(content),
        _ => AstSummary::default(),
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
/// Note: doc_lines are NOT included here — they go into the purpose section via
/// `format_purpose_hint`. Only functions and types are formatted for the api section.
pub fn format_for_stub(summary: &AstSummary) -> String {
    if summary.functions.is_empty() && summary.types.is_empty() {
        return String::new();
    }

    let mut parts: Vec<String> = Vec::new();
    if !summary.functions.is_empty() {
        let fns: Vec<&str> = summary.functions.iter().take(10).map(String::as_str).collect();
        parts.push(format!("Functions: {}", fns.join(", ")));
    }
    if !summary.types.is_empty() {
        let types: Vec<&str> = summary.types.iter().take(5).map(String::as_str).collect();
        parts.push(format!("Types: {}", types.join(", ")));
    }

    let result = parts.join("\n");
    if result.len() > 600 {
        // Truncate at a char boundary — slicing at a byte index would panic on
        // multibyte characters (Cyrillic, CJK, emoji) that appear in doc comments.
        let truncated: String = result.chars().take(597).collect();
        format!("{}…", truncated)
    } else {
        result
    }
}

/// Format doc comment lines as a purpose hint for pre-populating the purpose section (A3).
///
/// Returns `""` if no doc_lines are available. Used to create Level-1 neurons that
/// have meaningful purpose vocabulary from day 1 — before any LLM call.
pub fn format_purpose_hint(summary: &AstSummary) -> String {
    if summary.doc_lines.is_empty() {
        return String::new();
    }
    // Use up to 5 doc lines — enough for BM25 vocabulary without flooding the section
    let lines: Vec<&str> = summary.doc_lines.iter().take(5).map(String::as_str).collect();
    let result = lines.join("\n");
    // Cap at 400 chars — purpose section should be concise
    if result.len() > 400 {
        let truncated: String = result.chars().take(397).collect();
        format!("{}…", truncated)
    } else {
        result
    }
}

// ─── Language extractors ──────────────────────────────────────────────────────

fn extract_rust(content: &str) -> AstSummary {
    let functions = rust_fn_re()
        .captures_iter(content)
        .filter_map(|c| c.get(1))
        .map(|m| m.as_str().to_string())
        .collect();

    let types = rust_type_re()
        .captures_iter(content)
        .filter_map(|c| c.get(1))
        .map(|m| m.as_str().to_string())
        .collect();

    let doc_lines: Vec<String> = rust_doc_re()
        .captures_iter(content)
        .filter_map(|c| c.get(1))
        .map(|m| m.as_str().trim().to_string())
        .filter(|s| !s.is_empty())
        .take(5)
        .collect();

    AstSummary { functions, types, doc_lines }
}

fn extract_python(content: &str) -> AstSummary {
    let functions = py_fn_re()
        .captures_iter(content)
        .filter_map(|c| c.get(1))
        .map(|m| m.as_str().to_string())
        // Skip private helpers and dunder methods — uninformative for BM25 vocabulary.
        .filter(|name| !name.starts_with('_'))
        .collect();

    let types = py_class_re()
        .captures_iter(content)
        .filter_map(|c| c.get(1))
        .map(|m| m.as_str().to_string())
        .collect();

    // Extract first module-level docstring — gives BM25 high-signal vocabulary.
    let doc_lines: Vec<String> = py_docstring_re()
        .captures_iter(content)
        .take(1)
        .flat_map(|c| {
            // Group 1 = """, group 2 = '''
            let text = c.get(1).or_else(|| c.get(2))
                .map(|m| m.as_str())
                .unwrap_or("");
            text.lines()
                .map(str::trim)
                .filter(|l| !l.is_empty())
                .take(3)
                .map(String::from)
                .collect::<Vec<_>>()
        })
        .collect();

    AstSummary { functions, types, doc_lines }
}

fn extract_typescript(content: &str) -> AstSummary {
    let mut functions: Vec<String> = ts_fn_re()
        .captures_iter(content)
        .filter_map(|c| c.get(1))
        .map(|m| m.as_str().to_string())
        .collect();

    let arrow: Vec<String> = ts_arrow_re()
        .captures_iter(content)
        .filter_map(|c| c.get(1))
        .map(|m| m.as_str().to_string())
        .collect();
    functions.extend(arrow);

    let types = ts_type_re()
        .captures_iter(content)
        .filter_map(|c| c.get(1))
        .map(|m| m.as_str().to_string())
        .collect();

    AstSummary { functions, types, doc_lines: Vec::new() }
}

fn extract_go(content: &str) -> AstSummary {
    let functions = go_fn_re()
        .captures_iter(content)
        .filter_map(|c| c.get(1))
        .map(|m| m.as_str().to_string())
        .collect();

    let types = go_type_re()
        .captures_iter(content)
        .filter_map(|c| c.get(1))
        .map(|m| m.as_str().to_string())
        .collect();

    // Collect leading `//` comments before exported declarations as doc lines.
    // Strategy: gather all `// text` lines; Go doc comments naturally appear
    // before func/type, so the first few comments are the package-level docs.
    let doc_lines: Vec<String> = go_doc_re()
        .captures_iter(content)
        .filter_map(|c| c.get(1))
        .map(|m| m.as_str().trim().to_string())
        .filter(|s| !s.is_empty() && !s.starts_with("//"))
        .take(3)
        .collect();

    AstSummary { functions, types, doc_lines }
}

fn extract_swift(content: &str) -> AstSummary {
    let functions = swift_fn_re()
        .captures_iter(content)
        .filter_map(|c| c.get(1))
        .map(|m| m.as_str().to_string())
        .filter(|name| !name.starts_with("init") || name == "init")
        .collect();

    let types = swift_type_re()
        .captures_iter(content)
        .filter_map(|c| c.get(1))
        .map(|m| m.as_str().to_string())
        .collect();

    let doc_lines: Vec<String> = swift_doc_re()
        .captures_iter(content)
        .filter_map(|c| c.get(1))
        .map(|m| m.as_str().trim().to_string())
        .filter(|s| !s.is_empty())
        .take(5)
        .collect();

    AstSummary { functions, types, doc_lines }
}

fn extract_kotlin(content: &str) -> AstSummary {
    let functions = kotlin_fn_re()
        .captures_iter(content)
        .filter_map(|c| c.get(1))
        .map(|m| m.as_str().to_string())
        .filter(|name| !name.starts_with('_'))
        .collect();

    let types = kotlin_type_re()
        .captures_iter(content)
        .filter_map(|c| c.get(1))
        .map(|m| m.as_str().to_string())
        .collect();

    AstSummary { functions, types, doc_lines: Vec::new() }
}

fn extract_java(content: &str) -> AstSummary {
    let functions = java_fn_re()
        .captures_iter(content)
        .filter_map(|c| c.get(1))
        .map(|m| m.as_str().to_string())
        .collect();

    let types = java_type_re()
        .captures_iter(content)
        .filter_map(|c| c.get(1))
        .map(|m| m.as_str().to_string())
        .collect();

    // Javadoc comments — first `* text` lines after `/**`
    let doc_lines: Vec<String> = java_doc_re()
        .captures_iter(content)
        .filter_map(|c| c.get(1))
        .map(|m| m.as_str().trim().to_string())
        .filter(|s| !s.is_empty() && !s.starts_with('@') && !s.starts_with('*'))
        .take(3)
        .collect();

    AstSummary { functions, types, doc_lines }
}

fn extract_csharp(content: &str) -> AstSummary {
    let functions = csharp_fn_re()
        .captures_iter(content)
        .filter_map(|c| c.get(1))
        .map(|m| m.as_str().to_string())
        .collect();

    let types = csharp_type_re()
        .captures_iter(content)
        .filter_map(|c| c.get(1))
        .map(|m| m.as_str().to_string())
        .collect();

    let doc_lines: Vec<String> = csharp_doc_re()
        .captures_iter(content)
        .filter_map(|c| c.get(1))
        .map(|m| m.as_str().trim().to_string())
        .filter(|s| !s.is_empty())
        .take(3)
        .collect();

    AstSummary { functions, types, doc_lines }
}

fn extract_ruby(content: &str) -> AstSummary {
    let functions = ruby_fn_re()
        .captures_iter(content)
        .filter_map(|c| c.get(1))
        .map(|m| m.as_str().to_string())
        .filter(|name| !name.starts_with('_'))
        .collect();

    let types = ruby_type_re()
        .captures_iter(content)
        .filter_map(|c| c.get(1))
        .map(|m| m.as_str().to_string())
        .collect();

    AstSummary { functions, types, doc_lines: Vec::new() }
}

fn extract_c(content: &str) -> AstSummary {
    let functions = c_fn_re()
        .captures_iter(content)
        .filter_map(|c| c.get(1))
        .map(|m| m.as_str().to_string())
        // Skip common false positives: main, and single-char names
        .filter(|name| name.len() > 1 && name != "main")
        .collect();

    let types = c_type_re()
        .captures_iter(content)
        .filter_map(|c| c.get(1))
        .map(|m| m.as_str().to_string())
        .filter(|name| name.len() > 1)
        .collect();

    let doc_lines: Vec<String> = c_doc_re()
        .captures_iter(content)
        .filter_map(|c| c.get(1))
        .map(|m| m.as_str().trim().to_string())
        .filter(|s| !s.is_empty() && !s.starts_with('*') && !s.starts_with('/'))
        .take(3)
        .collect();

    AstSummary { functions, types, doc_lines }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extract_rust_functions_and_types() {
        let src = r#"
/// Parse the session state.
pub fn parse_session(input: &str) -> Result<Session> { todo!() }
pub async fn load_model(cfg: Config) -> Model { todo!() }
fn not_pub() {}
struct Private;
pub struct Session { id: u64 }
pub enum Status { Active, Idle }
pub trait Runnable { fn run(&self); }
"#;
        let s = extract_signatures("src/engine.rs", src);
        assert!(s.functions.contains(&"parse_session".to_string()), "{:?}", s.functions);
        assert!(s.functions.contains(&"load_model".to_string()));
        assert!(!s.functions.contains(&"not_pub".to_string()), "private fn excluded");
        assert!(s.types.contains(&"Session".to_string()));
        assert!(s.types.contains(&"Status".to_string()));
        assert!(s.types.contains(&"Runnable".to_string()));
        let formatted = format_for_stub(&s);
        assert!(formatted.contains("parse_session"), "formatted: {formatted}");
        // Doc comments now go into purpose hint (A3), not the api stub
        let purpose = format_purpose_hint(&s);
        assert!(purpose.contains("Parse the session state"), "doc comment: {purpose}");
    }

    #[test]
    fn extract_python_classes_and_functions() {
        let src = "class AuthService:\n    def login(self): pass\ndef logout(): pass\n";
        let s = extract_signatures("auth.py", src);
        assert!(s.types.contains(&"AuthService".to_string()), "{:?}", s.types);
        assert!(s.functions.contains(&"login".to_string()), "{:?}", s.functions);
        assert!(s.functions.contains(&"logout".to_string()));
    }

    #[test]
    fn extract_python_docstring() {
        let src = r#""""
Authentication service for JWT tokens.
Handles login, logout, and refresh.
"""
class AuthService:
    def login(self): pass
"#;
        let s = extract_signatures("auth.py", src);
        assert!(!s.doc_lines.is_empty(), "should capture docstring");
        assert!(
            s.doc_lines.iter().any(|l| l.contains("Authentication")),
            "doc lines: {:?}", s.doc_lines
        );
    }

    #[test]
    fn extract_go_doc_comments() {
        let src = "// Package server provides HTTP handling.\n// Use NewServer to initialise.\nfunc NewServer(cfg Config) *Server {}\n";
        let s = extract_signatures("server.go", src);
        assert!(!s.doc_lines.is_empty(), "should capture go doc comments");
        assert!(
            s.doc_lines.iter().any(|l| l.contains("server")),
            "doc lines: {:?}", s.doc_lines
        );
    }

    #[test]
    fn extract_typescript_exports_only() {
        let src = r#"
export function fetchUser(id: string): Promise<User> {}
export const handler = async (req: Request) => {}
export interface UserProfile { name: string }
export class UserService {}
function internal() {}
const privateConst = () => {}
"#;
        let s = extract_signatures("api/user.ts", src);
        assert!(s.functions.contains(&"fetchUser".to_string()), "{:?}", s.functions);
        assert!(s.functions.contains(&"handler".to_string()));
        assert!(!s.functions.contains(&"internal".to_string()), "non-export excluded");
        assert!(!s.functions.contains(&"privateConst".to_string()), "private arrow excluded");
        assert!(s.types.contains(&"UserProfile".to_string()));
        assert!(s.types.contains(&"UserService".to_string()));
    }

    #[test]
    fn extract_go_exported_only() {
        let src = "func NewServer(cfg Config) *Server {}\nfunc (s *Server) Start() error {}\nfunc internal() {}\ntype Server struct {}\n";
        let s = extract_signatures("server.go", src);
        assert!(s.functions.contains(&"NewServer".to_string()), "{:?}", s.functions);
        assert!(s.functions.contains(&"Start".to_string()), "method Start exported");
        assert!(!s.functions.contains(&"internal".to_string()), "lowercase fn excluded");
        assert!(s.types.contains(&"Server".to_string()));
    }

    #[test]
    fn unknown_extension_returns_empty() {
        let s = extract_signatures("data.csv", "hello,world\n1,2\n");
        assert!(s.is_empty());
        assert_eq!(format_for_stub(&s), "");
    }

    #[test]
    fn format_for_stub_caps_at_600_chars() {
        let long_fns: Vec<String> = (0..50).map(|i| format!("very_long_function_name_{i}")).collect();
        let summary = AstSummary {
            functions: long_fns,
            types: vec!["T".to_string()],
            doc_lines: Vec::new(),
        };
        let out = format_for_stub(&summary);
        assert!(out.len() <= 601, "output length: {}", out.len()); // 600 + possible ellipsis
    }

    #[test]
    fn extract_swift_functions_and_types() {
        let src = r#"
/// Authentication service.
public class AuthService {
    public func login(user: String) -> Bool { false }
    open func logout() {}
    private func reset() {}
}
public struct Token {}
public protocol Authenticatable {}
"#;
        let s = extract_signatures("Auth.swift", src);
        assert!(s.functions.contains(&"login".to_string()), "{:?}", s.functions);
        assert!(s.functions.contains(&"logout".to_string()));
        assert!(s.types.contains(&"AuthService".to_string()));
        assert!(s.types.contains(&"Token".to_string()));
        assert!(s.types.contains(&"Authenticatable".to_string()));
        assert!(!s.doc_lines.is_empty(), "should capture doc comment");
    }

    #[test]
    fn extract_kotlin_functions_and_types() {
        let src = "class UserService {\n    fun getUser(id: Int): User = TODO()\n    suspend fun fetchAsync(): List<User> = TODO()\n    private fun helper() {}\n}\ndata class User(val id: Int)\n";
        let s = extract_signatures("UserService.kt", src);
        assert!(s.functions.contains(&"getUser".to_string()), "{:?}", s.functions);
        assert!(s.functions.contains(&"fetchAsync".to_string()));
        assert!(s.types.contains(&"UserService".to_string()));
        assert!(s.types.contains(&"User".to_string()));
    }

    #[test]
    fn extract_java_public_methods_and_classes() {
        let src = r#"
/**
 * Handles user authentication.
 */
public class AuthController {
    public static User authenticate(String token) { return null; }
    protected void logout(HttpRequest req) {}
    private void helper() {}
}
public interface Repository {}
"#;
        let s = extract_signatures("AuthController.java", src);
        assert!(s.functions.contains(&"authenticate".to_string()), "{:?}", s.functions);
        assert!(s.functions.contains(&"logout".to_string()));
        assert!(!s.functions.contains(&"helper".to_string()), "private excluded");
        assert!(s.types.contains(&"AuthController".to_string()));
        assert!(s.types.contains(&"Repository".to_string()));
    }

    #[test]
    fn extract_csharp_methods_and_types() {
        let src = r#"
/// <summary>Authentication service</summary>
public class AuthService {
    public static async Task<User> LoginAsync(string token) { return null; }
    protected void Logout() {}
}
public interface IRepository {}
public enum Status { Active, Idle }
"#;
        let s = extract_signatures("AuthService.cs", src);
        assert!(s.functions.contains(&"LoginAsync".to_string()), "{:?}", s.functions);
        assert!(s.types.contains(&"AuthService".to_string()));
        assert!(s.types.contains(&"IRepository".to_string()));
        assert!(!s.doc_lines.is_empty(), "should capture XML doc comment");
    }

    #[test]
    fn extract_ruby_methods_and_classes() {
        let src = "module Auth\n  class UserService\n    def login(token)\n      true\n    end\n    def logout; end\n    def _private_helper; end\n  end\nend\n";
        let s = extract_signatures("user_service.rb", src);
        assert!(s.functions.contains(&"login".to_string()), "{:?}", s.functions);
        assert!(s.functions.contains(&"logout".to_string()));
        assert!(!s.functions.contains(&"_private_helper".to_string()), "private excluded");
        assert!(s.types.contains(&"UserService".to_string()));
        assert!(s.types.contains(&"Auth".to_string()));
    }

    #[test]
    fn extract_c_functions_and_structs() {
        let src = r#"
// Authentication utilities
struct AuthToken {
    int id;
};
typedef struct User User;
int authenticate(const char* token) {
    return 0;
}
void cleanup(AuthToken* t) {}
"#;
        let s = extract_signatures("auth.c", src);
        assert!(s.functions.contains(&"authenticate".to_string()), "{:?}", s.functions);
        assert!(s.functions.contains(&"cleanup".to_string()));
        assert!(s.types.contains(&"AuthToken".to_string()) || s.types.contains(&"User".to_string()), "{:?}", s.types);
    }

    #[test]
    fn extract_call_sites_detects_cross_file_calls() {
        use std::collections::HashMap;
        use std::path::PathBuf;

        // Simulate: index.rs calls parse_session() which is defined in engine.rs
        let caller_content = r#"
fn activate(&self) {
    let session = parse_session(input);
    let model = load_model(cfg);
}
"#;
        let mut vocab: HashMap<String, PathBuf> = HashMap::new();
        vocab.insert("parse_session".to_string(), PathBuf::from("src/engine.rs"));
        vocab.insert("load_model".to_string(), PathBuf::from("src/model.rs"));
        vocab.insert("unrelated_fn".to_string(), PathBuf::from("src/other.rs"));

        let edges = extract_call_sites("src/index.rs", caller_content, &vocab);
        let callee_files: Vec<&std::path::Path> = edges.iter().map(|e| e.callee_file.as_path()).collect();
        assert!(
            callee_files.contains(&std::path::Path::new("src/engine.rs")),
            "should detect parse_session call: {callee_files:?}"
        );
        assert!(
            callee_files.contains(&std::path::Path::new("src/model.rs")),
            "should detect load_model call: {callee_files:?}"
        );
        assert!(
            !callee_files.contains(&std::path::Path::new("src/other.rs")),
            "unrelated_fn not called: {callee_files:?}"
        );
    }

    #[test]
    fn extract_call_sites_empty_vocab_returns_empty() {
        use std::collections::HashMap;
        let edges = extract_call_sites("src/main.rs", "fn main() { do_something(); }", &HashMap::new());
        assert!(edges.is_empty());
    }

    #[test]
    fn extract_call_sites_no_self_loops() {
        use std::collections::HashMap;
        use std::path::PathBuf;

        // parse_session is defined in the same file we're scanning
        let content = "pub fn parse_session(s: &str) -> () { parse_session(s) }";
        let mut vocab: HashMap<String, PathBuf> = HashMap::new();
        vocab.insert("parse_session".to_string(), PathBuf::from("src/engine.rs"));

        // extract_call_sites should skip self-defined functions
        let edges = extract_call_sites("src/engine.rs", content, &vocab);
        // Self-loops suppressed — callee_path == source_rel guard prevents it
        let self_loops: Vec<_> = edges
            .iter()
            .filter(|e| e.callee_file == std::path::Path::new("src/engine.rs"))
            .collect();
        assert!(self_loops.is_empty(), "no self-loops expected: {self_loops:?}");
    }

    #[test]
    fn compute_sig_hash_is_16_hex_chars() {
        let summary = extract_signatures("src/lib.rs", "pub fn foo() {}\npub struct Bar {}");
        let hash = compute_sig_hash(&summary);
        assert_eq!(hash.len(), 16, "sig_hash must be 16 hex chars");
        assert!(hash.chars().all(|c| c.is_ascii_hexdigit()), "sig_hash must be hex");
    }

    #[test]
    fn compute_sig_hash_empty_summary_is_stable() {
        let empty = extract_signatures("src/empty.unknown", "no recognizable signatures here");
        let h1 = compute_sig_hash(&empty);
        let h2 = compute_sig_hash(&empty);
        assert_eq!(h1, h2, "empty summary hash must be deterministic");
    }
}
