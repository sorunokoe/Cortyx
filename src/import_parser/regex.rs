//! Compiled regex patterns for import statement parsing.

use regex::Regex;
use std::sync::OnceLock;

pub(super) fn rust_use_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    // `use crate::module` or `use crate::module::sub` — capture first path segment
    RE.get_or_init(|| Regex::new(r"(?m)^use crate::([a-z_]+)").unwrap())
}

pub(super) fn py_relative_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    // `from .module import X` or `from ..pkg.sub import Y`
    RE.get_or_init(|| Regex::new(r"(?m)^from (\.+[\w.]*)").unwrap())
}

pub(super) fn ts_relative_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    // `import ... from './path'` or `from "../path"`
    RE.get_or_init(|| Regex::new(r#"(?m)from ['"](\.[^'"]+)['"]"#).unwrap())
}

pub(super) fn go_single_import_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    // `import "module/pkg"` or `import alias "module/pkg"`
    RE.get_or_init(|| Regex::new(r#"(?m)^import\s+(?:[\w.]+\s+)?"([^"]+)""#).unwrap())
}

pub(super) fn go_block_import_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r#"(?s)import\s*\((?P<body>.*?)\)"#).unwrap())
}

pub(super) fn quoted_import_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r#""([^"]+)""#).unwrap())
}
