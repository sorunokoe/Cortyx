//! Core language AST extractors using tree-sitter and/or regex patterns.

use super::super::AstSummary;
use super::regex::*;

pub fn extract_rust(content: &str) -> AstSummary {
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

    AstSummary {
        functions,
        types,
        doc_lines,
        ..Default::default()
    }
}

pub fn extract_python(content: &str) -> AstSummary {
    let functions = py_fn_re()
        .captures_iter(content)
        .filter_map(|c| c.get(1))
        .map(|m| m.as_str().to_string())
        .filter(|name| !name.starts_with('_'))
        .collect();

    let types = py_class_re()
        .captures_iter(content)
        .filter_map(|c| c.get(1))
        .map(|m| m.as_str().to_string())
        .collect();

    let doc_lines: Vec<String> = py_docstring_re()
        .captures_iter(content)
        .take(1)
        .flat_map(|c| {
            let text = c
                .get(1)
                .or_else(|| c.get(2))
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

    AstSummary {
        functions,
        types,
        doc_lines,
        ..Default::default()
    }
}

pub fn extract_typescript(content: &str) -> AstSummary {
    let mut functions: Vec<String> = ts_fn_re()
        .captures_iter(content)
        .filter_map(|c| c.get(1))
        .map(|m| m.as_str().to_string())
        .collect();
    functions.extend(
        ts_arrow_re()
            .captures_iter(content)
            .filter_map(|c| c.get(1))
            .map(|m| m.as_str().to_string()),
    );

    let types = ts_type_re()
        .captures_iter(content)
        .filter_map(|c| c.get(1))
        .map(|m| m.as_str().to_string())
        .collect();

    AstSummary {
        functions,
        types,
        ..Default::default()
    }
}

pub fn extract_go(content: &str) -> AstSummary {
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

    let doc_lines: Vec<String> = go_doc_re()
        .captures_iter(content)
        .filter_map(|c| c.get(1))
        .map(|m| m.as_str().trim().to_string())
        .filter(|s| !s.is_empty())
        .take(5)
        .collect();

    AstSummary {
        functions,
        types,
        doc_lines,
        ..Default::default()
    }
}

pub fn extract_swift(content: &str) -> AstSummary {
    let functions = swift_fn_re()
        .captures_iter(content)
        .filter_map(|c| c.get(1))
        .map(|m| m.as_str().to_string())
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

    AstSummary {
        functions,
        types,
        doc_lines,
        ..Default::default()
    }
}

pub fn extract_kotlin(content: &str) -> AstSummary {
    let functions = kotlin_fn_re()
        .captures_iter(content)
        .filter_map(|c| c.get(1))
        .map(|m| m.as_str().to_string())
        .collect();

    let types = kotlin_type_re()
        .captures_iter(content)
        .filter_map(|c| c.get(1))
        .map(|m| m.as_str().to_string())
        .collect();

    AstSummary {
        functions,
        types,
        ..Default::default()
    }
}

pub fn extract_java(content: &str) -> AstSummary {
    let functions = java_fn_re()
        .captures_iter(content)
        .filter_map(|c| c.get(1))
        .map(|m| m.as_str().to_string())
        .take(20)
        .collect();

    let types = java_type_re()
        .captures_iter(content)
        .filter_map(|c| c.get(1))
        .map(|m| m.as_str().to_string())
        .collect();

    let doc_lines: Vec<String> = java_doc_re()
        .captures_iter(content)
        .filter_map(|c| c.get(1))
        .map(|m| m.as_str().trim().to_string())
        .filter(|s| !s.is_empty() && !s.starts_with('@'))
        .take(3)
        .collect();

    AstSummary {
        functions,
        types,
        doc_lines,
        ..Default::default()
    }
}

pub fn extract_csharp(content: &str) -> AstSummary {
    let functions = csharp_fn_re()
        .captures_iter(content)
        .filter_map(|c| c.get(1))
        .map(|m| m.as_str().to_string())
        .take(20)
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

    AstSummary {
        functions,
        types,
        doc_lines,
        ..Default::default()
    }
}

pub fn extract_ruby(content: &str) -> AstSummary {
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

    AstSummary {
        functions,
        types,
        ..Default::default()
    }
}

pub fn extract_c(content: &str) -> AstSummary {
    let functions = c_fn_re()
        .captures_iter(content)
        .filter_map(|c| c.get(1))
        .map(|m| m.as_str().to_string())
        .filter(|name| !matches!(name.as_str(), "if" | "while" | "for" | "switch" | "main"))
        .take(20)
        .collect();

    let types = c_type_re()
        .captures_iter(content)
        .filter_map(|c| c.get(1))
        .map(|m| m.as_str().to_string())
        .collect();

    let doc_lines: Vec<String> = c_doc_re()
        .captures_iter(content)
        .filter_map(|c| c.get(1))
        .map(|m| m.as_str().trim().to_string())
        .filter(|s| !s.is_empty())
        .take(5)
        .collect();

    AstSummary {
        functions,
        types,
        doc_lines,
        ..Default::default()
    }
}
