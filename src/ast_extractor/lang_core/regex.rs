//! Regex accessors for core language AST extraction (compiled once via OnceLock).

use regex::Regex;
use std::sync::OnceLock;

fn compile_regex(pattern: &str) -> Regex {
    match Regex::new(pattern) {
        Ok(regex) => regex,
        Err(err) => {
            tracing::error!("invalid AST extractor regex {pattern:?}: {err}");
            match Regex::new(r"$^") {
                Ok(fallback) => fallback,
                Err(_) => unreachable!("fallback regex must compile"),
            }
        },
    }
}

// ── Rust ──────────────────────────────────────────────────────────────────────

pub(super) fn rust_fn_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| compile_regex(r"(?m)^\s*pub(?:\s+\w+)*\s+fn\s+(\w+)"))
}

pub(super) fn rust_type_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| compile_regex(r"(?m)^\s*pub\s+(?:struct|enum|trait|type)\s+(\w+)"))
}

pub(super) fn rust_doc_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| compile_regex(r"(?m)^\s*///\s*(.+)"))
}

// ── Python ────────────────────────────────────────────────────────────────────

pub(super) fn py_fn_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| compile_regex(r"(?m)^\s*(?:async\s+)?def\s+(\w+)"))
}

pub(super) fn py_class_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| compile_regex(r"(?m)^class\s+(\w+)"))
}

/// Captures the first triple-quoted docstring in a Python file.
pub(super) fn py_docstring_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| compile_regex(r#"(?:"""([\s\S]*?)"""|'''([\s\S]*?)''')"#))
}

// ── TypeScript/JavaScript ─────────────────────────────────────────────────────

pub(super) fn ts_fn_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| compile_regex(r"(?m)^export\s+(?:default\s+)?(?:async\s+)?function\s+(\w+)"))
}

pub(super) fn ts_arrow_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| compile_regex(r"(?m)^export\s+const\s+(\w+)\s*=\s*(?:async\s+)?\("))
}

pub(super) fn ts_type_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| compile_regex(r"(?m)^export\s+(?:class|interface|type|enum)\s+(\w+)"))
}

// ── Go ────────────────────────────────────────────────────────────────────────

/// Captures Go doc comments — lines of `// text` immediately before a func/type declaration.
pub(super) fn go_doc_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| compile_regex(r"(?m)^//\s*(.+)"))
}

pub(super) fn go_fn_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| compile_regex(r"(?m)^func\s+(?:\([^)]+\)\s+)?([A-Z]\w*)"))
}

pub(super) fn go_type_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| compile_regex(r"(?m)^type\s+([A-Z]\w*)"))
}

// ── Swift ─────────────────────────────────────────────────────────────────────

pub(super) fn swift_fn_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        compile_regex(
            r"(?m)^\s*(?:public\s+|open\s+|internal\s+)?(?:override\s+)?(?:static\s+)?func\s+(\w+)",
        )
    })
}

pub(super) fn swift_type_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| compile_regex(r"(?m)^\s*(?:public\s+|open\s+)?(?:final\s+)?(?:class|struct|protocol|enum|extension|actor)\s+(\w+)"))
}

pub(super) fn swift_doc_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| compile_regex(r"(?m)^\s*///\s*(.+)"))
}

// ── Kotlin ────────────────────────────────────────────────────────────────────

pub(super) fn kotlin_fn_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| compile_regex(r"(?m)^\s*(?:public\s+|protected\s+|internal\s+)?(?:override\s+)?(?:suspend\s+)?fun\s+(\w+)"))
}

pub(super) fn kotlin_type_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| compile_regex(r"(?m)^\s*(?:data\s+|sealed\s+|abstract\s+)?(?:class|interface|object|enum\s+class)\s+(\w+)"))
}

// ── Java ──────────────────────────────────────────────────────────────────────

pub(super) fn java_fn_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        compile_regex(
            r"(?m)^\s*(?:public|protected)\s+(?:static\s+)?(?:final\s+)?[\w<>\[\]]+\s+(\w+)\s*\(",
        )
    })
}

pub(super) fn java_type_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| compile_regex(r"(?m)^\s*(?:public\s+)?(?:abstract\s+|final\s+)?(?:class|interface|enum|record)\s+(\w+)"))
}

pub(super) fn java_doc_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| compile_regex(r"(?m)^\s*\*\s+(.+)"))
}

// ── C# ────────────────────────────────────────────────────────────────────────

pub(super) fn csharp_fn_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        compile_regex(r"(?m)^\s*(?:public|protected)\s+(?:static\s+)?(?:async\s+)?(?:override\s+)?[\w<>\[\]?]+\s+([A-Z]\w+)\s*[\(<]")
    })
}

pub(super) fn csharp_type_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| compile_regex(r"(?m)^\s*(?:public\s+)?(?:abstract\s+|sealed\s+|static\s+|partial\s+)?(?:class|interface|struct|enum|record)\s+(\w+)"))
}

pub(super) fn csharp_doc_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| compile_regex(r"(?m)^\s*///\s*<summary>\s*(.+?)\s*(?:</summary>|$)"))
}

// ── Ruby ──────────────────────────────────────────────────────────────────────

pub(super) fn ruby_fn_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| compile_regex(r"(?m)^\s*def\s+(\w+[\?!]?)"))
}

pub(super) fn ruby_type_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| compile_regex(r"(?m)^\s*(?:class|module)\s+(\w+)"))
}

// ── C / C++ ───────────────────────────────────────────────────────────────────

pub(super) fn c_fn_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| compile_regex(r"(?m)^[\w\s\*]+\s+(\w+)\s*\([^;{]*\)\s*(?:const\s*)?\{"))
}

pub(super) fn c_type_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| compile_regex(r"(?m)^(?:typedef\s+)?(?:struct|enum|class|union)\s+(\w+)"))
}

pub(super) fn c_doc_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| compile_regex(r"(?m)^\s*(?:///?|\*)\s*(.+)"))
}
