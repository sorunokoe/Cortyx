//! Config and data language AST extractors (Shell, SQL, HCL, Protocol Buffers, GraphQL).

use super::super::AstSummary;
use super::compile_regex;
use regex::Regex;

pub fn extract_shell(content: &str) -> AstSummary {
    use std::sync::OnceLock;
    static FN_RE: OnceLock<Regex> = OnceLock::new();
    let fn_re = FN_RE.get_or_init(|| compile_regex(r"(?m)^(?:function\s+)?(\w+)\s*\(\s*\)\s*\{"));
    AstSummary {
        functions: fn_re
            .captures_iter(content)
            .filter_map(|c| c.get(1))
            .map(|m| m.as_str().to_string())
            .filter(|s| !matches!(s.as_str(), "if" | "while" | "for" | "case" | "do"))
            .collect(),
        ..Default::default()
    }
}

pub fn extract_sql(content: &str) -> AstSummary {
    use std::sync::OnceLock;
    static FN_RE: OnceLock<Regex> = OnceLock::new();
    static TYPE_RE: OnceLock<Regex> = OnceLock::new();
    let fn_re = FN_RE.get_or_init(|| {
        compile_regex(r"(?mi)^\s*(?:CREATE|ALTER)\s+(?:OR\s+REPLACE\s+)?(?:FUNCTION|PROCEDURE|TRIGGER)\s+(\w+)")
    });
    let type_re = TYPE_RE.get_or_init(|| {
        compile_regex(r"(?mi)^\s*CREATE\s+(?:TABLE|VIEW|MATERIALIZED\s+VIEW|INDEX)\s+(?:IF\s+NOT\s+EXISTS\s+)?(\w+)")
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

pub fn extract_hcl(content: &str) -> AstSummary {
    use std::sync::OnceLock;
    static BLOCK_RE: OnceLock<Regex> = OnceLock::new();
    let block_re = BLOCK_RE.get_or_init(|| {
        compile_regex(r#"(?m)^\s*(?:resource|data|module|variable|output|locals)\s+"([^"]+)""#)
    });
    let types: Vec<String> = block_re
        .captures_iter(content)
        .filter_map(|c| c.get(1))
        .map(|m| m.as_str().to_string())
        .collect();
    AstSummary {
        types,
        ..Default::default()
    }
}

pub fn extract_proto(content: &str) -> AstSummary {
    use std::sync::OnceLock;
    static MSG_RE: OnceLock<Regex> = OnceLock::new();
    static RPC_RE: OnceLock<Regex> = OnceLock::new();
    let msg_re = MSG_RE.get_or_init(|| compile_regex(r"(?m)^\s*(?:message|enum|service)\s+(\w+)"));
    let rpc_re = RPC_RE.get_or_init(|| compile_regex(r"(?m)^\s*rpc\s+(\w+)\s*\("));
    AstSummary {
        functions: rpc_re
            .captures_iter(content)
            .filter_map(|c| c.get(1))
            .map(|m| m.as_str().to_string())
            .collect(),
        types: msg_re
            .captures_iter(content)
            .filter_map(|c| c.get(1))
            .map(|m| m.as_str().to_string())
            .collect(),
        ..Default::default()
    }
}

pub fn extract_graphql(content: &str) -> AstSummary {
    use std::sync::OnceLock;
    static TYPE_RE: OnceLock<Regex> = OnceLock::new();
    static FIELD_RE: OnceLock<Regex> = OnceLock::new();
    let type_re = TYPE_RE.get_or_init(|| {
        compile_regex(r"(?m)^\s*(?:type|interface|input|enum|union|scalar)\s+(\w+)")
    });
    let field_re = FIELD_RE.get_or_init(|| compile_regex(r"(?m)^\s+(\w+)(?:\([^)]*\))?\s*:"));
    AstSummary {
        functions: field_re
            .captures_iter(content)
            .filter_map(|c| c.get(1))
            .map(|m| m.as_str().to_string())
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
