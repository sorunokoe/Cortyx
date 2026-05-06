//! Compiled regex patterns for import statement parsing.

use regex::Regex;
use std::sync::OnceLock;

pub(super) fn compile_regex(pattern: &str) -> Regex {
    match Regex::new(pattern) {
        Ok(regex) => regex,
        Err(err) => {
            tracing::error!("invalid import parser regex {pattern:?}: {err}");
            match Regex::new(r"$^") {
                Ok(fallback) => fallback,
                Err(_) => unreachable!("fallback regex must compile"),
            }
        },
    }
}

pub(super) fn rust_use_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    // `use crate::module` or `use crate::module::sub` — capture first path segment
    RE.get_or_init(|| compile_regex(r"(?m)^use crate::([a-z_]+)"))
}

pub(super) fn py_relative_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    // `from .module import X` or `from ..pkg.sub import Y`
    RE.get_or_init(|| compile_regex(r"(?m)^from (\.+[\w.]*)"))
}

pub(super) fn ts_relative_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    // `import ... from './path'` or `from "../path"`
    RE.get_or_init(|| compile_regex(r#"(?m)from ['"](\.[^'"]+)['"]"#))
}

pub(super) fn go_single_import_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    // `import "module/pkg"` or `import alias "module/pkg"`
    RE.get_or_init(|| compile_regex(r#"(?m)^import\s+(?:[\w.]+\s+)?"([^"]+)""#))
}

pub(super) fn go_block_import_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| compile_regex(r#"(?s)import\s*\((?P<body>.*?)\)"#))
}

pub(super) fn quoted_import_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| compile_regex(r#""([^"]+)""#))
}
