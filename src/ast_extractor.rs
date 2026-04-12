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
        _ => AstSummary::default(),
    }
}

/// Format an `AstSummary` as a compact markdown block for insertion into neuron stubs.
///
/// Returns `""` if the summary is empty. Output is capped at 600 chars.
pub fn format_for_stub(summary: &AstSummary) -> String {
    if summary.is_empty() {
        return String::new();
    }

    let mut parts: Vec<String> = Vec::new();

    if !summary.doc_lines.is_empty() {
        let docs: Vec<&str> = summary.doc_lines.iter().take(3).map(String::as_str).collect();
        parts.push(format!("Docs: {}", docs.join(" / ")));
    }
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

    AstSummary { functions, types, doc_lines: Vec::new() }
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

    AstSummary { functions, types, doc_lines: Vec::new() }
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
        assert!(formatted.contains("Parse the session state"), "doc comment: {formatted}");
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
}
