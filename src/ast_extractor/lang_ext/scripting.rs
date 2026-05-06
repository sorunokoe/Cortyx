//! Scripting language AST extractors (PHP, Lua, R, Julia, Elixir).

use super::super::AstSummary;
use super::compile_regex;
use regex::Regex;

pub fn extract_php(content: &str) -> AstSummary {
    use std::sync::OnceLock;
    static FN_RE: OnceLock<Regex> = OnceLock::new();
    static TYPE_RE: OnceLock<Regex> = OnceLock::new();
    let fn_re = FN_RE.get_or_init(|| {
        compile_regex(r"(?m)^\s*(?:public|protected|private)?\s*(?:static\s+)?function\s+(\w+)")
    });
    let type_re = TYPE_RE.get_or_init(|| {
        compile_regex(r"(?m)^\s*(?:abstract\s+|final\s+)?(?:class|interface|trait|enum)\s+(\w+)")
    });
    AstSummary {
        functions: fn_re
            .captures_iter(content)
            .filter_map(|c| c.get(1))
            .map(|m| m.as_str().to_string())
            .filter(|s| !s.starts_with('_'))
            .collect(),
        types: type_re
            .captures_iter(content)
            .filter_map(|c| c.get(1))
            .map(|m| m.as_str().to_string())
            .collect(),
        ..Default::default()
    }
}

pub fn extract_lua(content: &str) -> AstSummary {
    use std::sync::OnceLock;
    static FN_RE: OnceLock<Regex> = OnceLock::new();
    let fn_re = FN_RE.get_or_init(|| compile_regex(r"(?m)^(?:local\s+)?function\s+(\w[\w.]*)"));
    AstSummary {
        functions: fn_re
            .captures_iter(content)
            .filter_map(|c| c.get(1))
            .map(|m| m.as_str().to_string())
            .collect(),
        ..Default::default()
    }
}

pub fn extract_r(content: &str) -> AstSummary {
    use std::sync::OnceLock;
    static FN_RE: OnceLock<Regex> = OnceLock::new();
    let fn_re = FN_RE.get_or_init(|| compile_regex(r"(?m)^(\w[\w.]*)\s*<-\s*function\s*\("));
    AstSummary {
        functions: fn_re
            .captures_iter(content)
            .filter_map(|c| c.get(1))
            .map(|m| m.as_str().to_string())
            .collect(),
        ..Default::default()
    }
}

pub fn extract_julia(content: &str) -> AstSummary {
    use std::sync::OnceLock;
    static FN_RE: OnceLock<Regex> = OnceLock::new();
    static TYPE_RE: OnceLock<Regex> = OnceLock::new();
    let fn_re = FN_RE.get_or_init(|| compile_regex(r"(?m)^(?:function|macro)\s+(\w+)"));
    let type_re = TYPE_RE
        .get_or_init(|| compile_regex(r"(?m)^(?:abstract type|struct|mutable struct)\s+(\w+)"));
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

pub fn extract_elixir(content: &str) -> AstSummary {
    use std::sync::OnceLock;
    static FN_RE: OnceLock<Regex> = OnceLock::new();
    static MOD_RE: OnceLock<Regex> = OnceLock::new();
    let fn_re = FN_RE.get_or_init(|| compile_regex(r"(?m)^\s*def(?:p)?\s+(\w+)"));
    let mod_re = MOD_RE.get_or_init(|| compile_regex(r"(?m)^defmodule\s+([\w.]+)"));
    AstSummary {
        functions: fn_re
            .captures_iter(content)
            .filter_map(|c| c.get(1))
            .map(|m| m.as_str().to_string())
            .collect(),
        types: mod_re
            .captures_iter(content)
            .filter_map(|c| c.get(1))
            .map(|m| m.as_str().to_string())
            .collect(),
        ..Default::default()
    }
}
