//! Language-specific import statement parsers.

use super::regex::*;
use regex::Regex;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

/// Rust: `use crate::module` → `src/module.rs` (Cargo convention)
pub(super) fn rust_imports(content: &str) -> Vec<PathBuf> {
    rust_use_re()
        .captures_iter(content)
        .filter_map(|c| c.get(1))
        .map(|m| PathBuf::from("src").join(m.as_str()))
        .collect()
}

/// Python: `from .module import X` → sibling in the same package
pub(super) fn python_imports(content: &str, source_rel: &Path) -> Vec<PathBuf> {
    let source_dir = source_rel.parent().unwrap_or(Path::new(""));

    py_relative_re()
        .captures_iter(content)
        .filter_map(|c| c.get(1))
        .filter_map(|m| {
            let raw = m.as_str();
            let dots = raw.chars().take_while(|&c| c == '.').count();
            let module_path = raw.trim_start_matches('.');

            // Navigate up `dots - 1` directories from source_dir
            let mut base = source_dir.to_path_buf();
            for _ in 1..dots {
                base = base.parent().unwrap_or(Path::new("")).to_path_buf();
            }

            if module_path.is_empty() {
                Some(base.join("__init__"))
            } else {
                Some(base.join(module_path.replace('.', "/")))
            }
        })
        .collect()
}

/// TypeScript/JS: `from './relative/path'` → resolve relative to source dir
pub(super) fn ts_imports(content: &str, source_rel: &Path) -> Vec<PathBuf> {
    let source_dir = source_rel.parent().unwrap_or(Path::new(""));

    ts_relative_re()
        .captures_iter(content)
        .filter_map(|c| c.get(1))
        .map(|m| source_dir.join(m.as_str()))
        .collect()
}

/// Go: `import "example.com/project/pkg/auth"` → `pkg/auth/auth.go` (when resolvable)
pub(super) fn go_imports(content: &str, project_root: &Path) -> Vec<PathBuf> {
    let Some(module_path) = read_go_module_path(project_root) else {
        return Vec::new();
    };

    let mut imports = Vec::new();
    for captures in go_single_import_re().captures_iter(content) {
        if let Some(path) = captures
            .get(1)
            .and_then(|m| strip_go_module_prefix(m.as_str(), &module_path))
        {
            imports.push(PathBuf::from(path));
        }
    }
    for captures in go_block_import_re().captures_iter(content) {
        let Some(body) = captures.name("body") else {
            continue;
        };
        for quoted in quoted_import_re().captures_iter(body.as_str()) {
            if let Some(path) = quoted
                .get(1)
                .and_then(|m| strip_go_module_prefix(m.as_str(), &module_path))
            {
                imports.push(PathBuf::from(path));
            }
        }
    }
    imports.sort();
    imports.dedup();
    imports
}

fn read_go_module_path(project_root: &Path) -> Option<String> {
    let content = fs::read_to_string(project_root.join("go.mod")).ok()?;
    content
        .lines()
        .map(str::trim)
        .find_map(|line| line.strip_prefix("module ").map(str::trim))
        .filter(|line| !line.is_empty())
        .map(ToOwned::to_owned)
}

fn strip_go_module_prefix(import_path: &str, module_path: &str) -> Option<String> {
    import_path
        .strip_prefix(module_path)
        .and_then(|path| path.strip_prefix('/'))
        .map(ToOwned::to_owned)
}

/// C/C++: `#include "relative/path.h"` (quotes only — angle brackets are stdlib).
pub(super) fn c_include_imports(content: &str, source_rel: &Path) -> Vec<PathBuf> {
    static RE: OnceLock<Regex> = OnceLock::new();
    let re = RE.get_or_init(|| Regex::new(r#"(?m)^#include\s+"([^"]+)""#).unwrap());
    let dir = source_rel.parent().unwrap_or(Path::new(""));
    re.captures_iter(content)
        .filter_map(|c| c.get(1))
        .map(|m| dir.join(m.as_str()))
        .collect()
}

/// Ruby: `require_relative 'sibling'` → sibling file in the same directory.
pub(super) fn ruby_imports(content: &str, source_rel: &Path) -> Vec<PathBuf> {
    static RE: OnceLock<Regex> = OnceLock::new();
    let re = RE.get_or_init(|| Regex::new(r#"(?m)^require_relative\s+['"]([^'"]+)['"]"#).unwrap());
    let dir = source_rel.parent().unwrap_or(Path::new(""));
    re.captures_iter(content)
        .filter_map(|c| c.get(1))
        .map(|m| dir.join(m.as_str()))
        .collect()
}

/// Swift / Kotlin / Dart: `import PackageName` — we try to resolve
/// the last identifier component as a same-directory source file.
pub(super) fn simple_import_imports(content: &str, source_rel: &Path, ext: &str) -> Vec<PathBuf> {
    static RE: OnceLock<Regex> = OnceLock::new();
    let re = RE.get_or_init(|| Regex::new(r"(?m)^import\s+([\w.]+)").unwrap());
    let dir = source_rel.parent().unwrap_or(Path::new(""));
    re.captures_iter(content)
        .filter_map(|c| c.get(1))
        .map(|m| {
            // Take the last path component (e.g. "com.example.auth" → "auth")
            let last = m.as_str().rsplit('.').next().unwrap_or(m.as_str());
            dir.join(last).with_extension(ext)
        })
        .collect()
}

/// Elixir: `alias MyApp.Module`, `import MyApp.Module`, `use MyApp.Module`
/// Maps dot-joined module name to a lib-convention file path.
pub(super) fn elixir_imports(content: &str, _source_rel: &Path) -> Vec<PathBuf> {
    static RE: OnceLock<Regex> = OnceLock::new();
    let re = RE.get_or_init(|| Regex::new(r"(?m)^\s*(?:alias|import|use)\s+([\w.]+)").unwrap());
    re.captures_iter(content)
        .filter_map(|c| c.get(1))
        .map(|m| {
            // MyApp.Auth.Service → lib/my_app/auth/service.ex
            let parts: Vec<String> = m.as_str().split('.').map(|s| to_snake_case(s)).collect();
            PathBuf::from("lib")
                .join(parts.join("/"))
                .with_extension("ex")
        })
        .collect()
}

/// Simple CamelCase → snake_case (for Elixir module path convention).
fn to_snake_case(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 4);
    for (i, ch) in s.chars().enumerate() {
        if ch.is_uppercase() && i > 0 {
            out.push('_');
        }
        out.push(ch.to_ascii_lowercase());
    }
    out
}
