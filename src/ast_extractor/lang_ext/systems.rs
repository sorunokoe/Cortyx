//! Systems programming language extractors (Zig, Dart).

use super::super::AstSummary;
use super::compile_regex;
use regex::Regex;

pub fn extract_zig(content: &str) -> AstSummary {
    use std::sync::OnceLock;
    static FN_RE: OnceLock<Regex> = OnceLock::new();
    static TYPE_RE: OnceLock<Regex> = OnceLock::new();
    let fn_re = FN_RE.get_or_init(|| compile_regex(r"(?m)^\s*pub\s+fn\s+(\w+)"));
    let type_re = TYPE_RE
        .get_or_init(|| compile_regex(r"(?m)^\s*pub\s+const\s+(\w+)\s*=\s*(?:struct|union|enum)"));
    AstSummary {
        functions: fn_re
            .captures_iter(content)
            .filter_map(|c| c.get(1))
            .map(|m| m.as_str().to_string())
            .collect(),
        types: type_re
            .captures_iter(content)
            .filter_map(|c| c.get(1))
            .map(|m| m.as_str().to_string())
            .collect(),
        ..Default::default()
    }
}

pub fn extract_dart(content: &str) -> AstSummary {
    use std::sync::OnceLock;
    static FN_RE: OnceLock<Regex> = OnceLock::new();
    static TYPE_RE: OnceLock<Regex> = OnceLock::new();
    let fn_re = FN_RE.get_or_init(|| compile_regex(r"(?m)^\s*(?:[\w<>\[\]?]+\s+)+([a-z]\w*)\s*\("));
    let type_re = TYPE_RE.get_or_init(|| {
        compile_regex(r"(?m)^\s*(?:abstract\s+)?(?:class|mixin|extension|enum)\s+(\w+)")
    });
    AstSummary {
        functions: fn_re
            .captures_iter(content)
            .filter_map(|c| c.get(1))
            .map(|m| m.as_str().to_string())
            .filter(|s| !s.starts_with('_'))
            .take(20)
            .collect(),
        types: type_re
            .captures_iter(content)
            .filter_map(|c| c.get(1))
            .map(|m| m.as_str().to_string())
            .collect(),
        ..Default::default()
    }
}
