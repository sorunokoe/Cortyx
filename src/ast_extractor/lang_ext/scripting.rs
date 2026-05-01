//! Scripting language AST extractors (PHP, Lua, R, Julia, Elixir).

use super::super::AstSummary;
use regex::Regex;

pub fn extract_php(content: &str) -> AstSummary {
    use std::sync::OnceLock;
    static FN_RE: OnceLock<Regex> = OnceLock::new();
    static TYPE_RE: OnceLock<Regex> = OnceLock::new();
    let fn_re = FN_RE.get_or_init(|| {
        Regex::new(r"(?m)^\s*(?:public|protected|private)?\s*(?:static\s+)?function\s+(\w+)")
            .unwrap()
    });
    let type_re = TYPE_RE.get_or_init(|| {
        Regex::new(r"(?m)^\s*(?:abstract\s+|final\s+)?(?:class|interface|trait|enum)\s+(\w+)")
            .unwrap()
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
    let fn_re =
        FN_RE.get_or_init(|| Regex::new(r"(?m)^(?:local\s+)?function\s+(\w[\w.]*)").unwrap());
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
    let fn_re = FN_RE.get_or_init(|| Regex::new(r"(?m)^(\w[\w.]*)\s*<-\s*function\s*\(").unwrap());
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
    let fn_re = FN_RE.get_or_init(|| Regex::new(r"(?m)^(?:function|macro)\s+(\w+)").unwrap());
    let type_re = TYPE_RE.get_or_init(|| {
        Regex::new(r"(?m)^(?:abstract type|struct|mutable struct)\s+(\w+)").unwrap()
    });
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
    let fn_re = FN_RE.get_or_init(|| Regex::new(r"(?m)^\s*def(?:p)?\s+(\w+)").unwrap());
    let mod_re = MOD_RE.get_or_init(|| Regex::new(r"(?m)^defmodule\s+([\w.]+)").unwrap());
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
