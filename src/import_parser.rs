//! Auto-Synapse — infers `Imports`-typed synapse edges from import statements at
//! compile time, so the synapse graph has real structure from day 1.
//!
//! Only local/relative imports are resolved; stdlib and third-party packages are
//! silently skipped (they have no neurons). Each resolved path is returned as
//! an absolute source file path — callers compute the target neuron path.

use regex::Regex;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

// ─── Regex accessors ──────────────────────────────────────────────────────────

fn rust_use_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    // `use crate::module` or `use crate::module::sub` — capture first path segment
    RE.get_or_init(|| Regex::new(r"(?m)^use crate::([a-z_]+)").unwrap())
}

fn py_relative_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    // `from .module import X` or `from ..pkg.sub import Y`
    RE.get_or_init(|| Regex::new(r"(?m)^from (\.+[\w.]*)").unwrap())
}

fn ts_relative_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    // `import ... from './path'` or `from "../path"`
    RE.get_or_init(|| Regex::new(r#"(?m)from ['"](\.[^'"]+)['"]"#).unwrap())
}

// ─── Public API ───────────────────────────────────────────────────────────────

/// Parse import statements in `content` and resolve them to existing project files.
///
/// Returns absolute paths of source files imported by `source_abs`. Only files
/// that actually exist on disk are returned — stdlib and third-party imports
/// are silently ignored.
pub fn parse_imports(source_abs: &Path, content: &str, project_root: &Path) -> Vec<PathBuf> {
    let source_rel = source_abs.strip_prefix(project_root).unwrap_or(source_abs);
    let ext = source_rel.extension().and_then(|e| e.to_str()).unwrap_or("");

    let candidates = match ext {
        "rs" => rust_imports(content),
        "py" => python_imports(content, source_rel),
        "ts" | "tsx" | "js" | "jsx" | "mjs" | "cjs" => ts_imports(content, source_rel),
        _ => Vec::new(),
    };

    candidates
        .into_iter()
        .filter_map(|c| resolve_to_existing(c, project_root))
        // Exclude self-referential imports
        .filter(|p| p != source_abs)
        .collect()
}

// ─── Language parsers ─────────────────────────────────────────────────────────

/// Rust: `use crate::module` → `src/module.rs` (Cargo convention)
fn rust_imports(content: &str) -> Vec<PathBuf> {
    rust_use_re()
        .captures_iter(content)
        .filter_map(|c| c.get(1))
        .map(|m| PathBuf::from("src").join(m.as_str()))
        .collect()
}

/// Python: `from .module import X` → sibling in the same package
fn python_imports(content: &str, source_rel: &Path) -> Vec<PathBuf> {
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
fn ts_imports(content: &str, source_rel: &Path) -> Vec<PathBuf> {
    let source_dir = source_rel.parent().unwrap_or(Path::new(""));

    ts_relative_re()
        .captures_iter(content)
        .filter_map(|c| c.get(1))
        .map(|m| source_dir.join(m.as_str()))
        .collect()
}

// ─── Path resolution ──────────────────────────────────────────────────────────

/// Try to resolve a candidate path (possibly without extension) to an existing file.
///
/// Tries the path as-is, then with common source extensions, then as a directory
/// index file.
///
/// Security: canonicalizes the resolved path and verifies it remains inside
/// `project_root`. This prevents path-traversal attacks via crafted import strings
/// such as `../../etc/passwd` in TypeScript or Python sources.
fn resolve_to_existing(candidate: PathBuf, project_root: &Path) -> Option<PathBuf> {
    // Compute a canonical project root once for containment checks.
    // Fall back to the raw path on systems where canonicalize fails (e.g. missing dirs).
    let root_canonical = project_root.canonicalize().unwrap_or_else(|_| project_root.to_path_buf());

    // Helper: return `abs` only if it exists, is a file, and is inside root.
    let check = |abs: PathBuf| -> Option<PathBuf> {
        if !abs.exists() || !abs.is_file() {
            return None;
        }
        match abs.canonicalize() {
            Ok(c) if c.starts_with(&root_canonical) => Some(c),
            _ => None,
        }
    };

    if let Some(p) = check(project_root.join(&candidate)) {
        return Some(p);
    }

    // Try adding common source extensions
    for ext in &["rs", "py", "ts", "tsx", "js", "jsx", "go"] {
        if let Some(p) = check(project_root.join(candidate.with_extension(ext))) {
            return Some(p);
        }
    }

    // Directory index files
    for index in &["index.ts", "index.js", "index.tsx", "__init__.py", "mod.rs"] {
        if let Some(p) = check(project_root.join(&candidate).join(index)) {
            return Some(p);
        }
    }

    None
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    #[test]
    fn rust_import_resolves_sibling_module() {
        let dir = TempDir::new().unwrap();
        let src = dir.path().join("src");
        fs::create_dir_all(&src).unwrap();
        fs::write(src.join("index.rs"), b"").unwrap();
        fs::write(src.join("engine.rs"), b"").unwrap();

        let source_abs = src.join("main.rs");
        fs::write(&source_abs, b"use crate::index;\nuse crate::engine;\nuse std::fmt;").unwrap();
        let content = fs::read_to_string(&source_abs).unwrap();

        let imports = parse_imports(&source_abs, &content, dir.path());
        let names: Vec<&str> = imports
            .iter()
            .filter_map(|p| p.file_stem().and_then(|s| s.to_str()))
            .collect();

        assert!(names.contains(&"index"), "expected index in {names:?}");
        assert!(names.contains(&"engine"));
        // std::fmt should not appear — no src/fmt.rs
    }

    #[test]
    fn ts_relative_import_resolves() {
        let dir = TempDir::new().unwrap();
        let api = dir.path().join("api");
        fs::create_dir_all(&api).unwrap();
        fs::write(api.join("auth.ts"), b"").unwrap();

        let source_abs = api.join("handler.ts");
        fs::write(&source_abs, b"import { login } from './auth';").unwrap();
        let content = fs::read_to_string(&source_abs).unwrap();

        let imports = parse_imports(&source_abs, &content, dir.path());
        let names: Vec<&str> = imports
            .iter()
            .filter_map(|p| p.file_stem().and_then(|s| s.to_str()))
            .collect();

        assert!(names.contains(&"auth"), "expected auth in {names:?}");
    }

    #[test]
    fn nonexistent_import_is_ignored() {
        let dir = TempDir::new().unwrap();
        let source_abs = dir.path().join("main.rs");
        fs::write(&source_abs, b"use crate::nonexistent;").unwrap();
        let content = fs::read_to_string(&source_abs).unwrap();

        let imports = parse_imports(&source_abs, &content, dir.path());
        assert!(imports.is_empty(), "nonexistent file should not appear: {imports:?}");
    }

    #[test]
    fn self_import_is_excluded() {
        let dir = TempDir::new().unwrap();
        let src = dir.path().join("src");
        fs::create_dir_all(&src).unwrap();
        // source_abs = src/main.rs, resolves to src/ which contains main.rs
        // Rust can't self-import via crate::main, but guard against it anyway
        let source_abs = src.join("main.rs");
        fs::write(&source_abs, b"use crate::main;").unwrap();
        let content = fs::read_to_string(&source_abs).unwrap();

        let imports = parse_imports(&source_abs, &content, dir.path());
        // src/main.rs exists but is the source itself — filtered out
        assert!(!imports.contains(&source_abs), "self-import must be excluded");
    }
}
