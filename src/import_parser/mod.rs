//! Auto-Synapse — infers `Imports`-typed synapse edges from import statements at
//! compile time, so the synapse graph has real structure from day 1.
//!
//! Only local/relative imports are resolved; stdlib and third-party packages are
//! silently skipped (they have no neurons). Each resolved path is returned as
//! an absolute source file path — callers compute the target neuron path.

mod parsers;
mod regex;
mod resolution;

use parsers::*;
use resolution::*;
use std::path::{Path, PathBuf};

/// Parse import statements in `content` and resolve them to existing project files.
///
/// Returns absolute paths of source files imported by `source_abs`. Only files
/// that actually exist on disk are returned — stdlib and third-party imports
/// are silently ignored.
pub fn parse_imports(source_abs: &Path, content: &str, project_root: &Path) -> Vec<PathBuf> {
    let source_rel = source_abs.strip_prefix(project_root).unwrap_or(source_abs);
    let ext = source_rel
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("");

    let candidates = match ext {
        "rs" => rust_imports(content),
        "py" | "pyw" => python_imports(content, source_rel),
        "ts" | "tsx" | "js" | "jsx" | "mjs" | "cjs" => ts_imports(content, source_rel),
        "go" => go_imports(content, project_root),
        // C/C++: #include "relative/path.h"
        "c" | "cc" | "cpp" | "cxx" | "h" | "hh" | "hpp" => c_include_imports(content, source_rel),
        // Ruby: require_relative 'sibling'
        "rb" => ruby_imports(content, source_rel),
        // Swift / Kotlin / Dart: import Module  — resolve same-dir sources
        "swift" | "kt" | "kts" | "dart" => simple_import_imports(content, source_rel, ext),
        // Elixir: alias/import/use MyApp.Module
        "ex" | "exs" => elixir_imports(content, source_rel),
        _ => Vec::new(),
    };

    candidates
        .into_iter()
        .filter_map(|c| resolve_to_existing(c, project_root))
        // Exclude self-referential imports
        .filter(|p| p != source_abs)
        .collect()
}

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
        fs::write(
            &source_abs,
            b"use crate::index;\nuse crate::engine;\nuse std::fmt;",
        )
        .unwrap();
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
        assert!(
            imports.is_empty(),
            "nonexistent file should not appear: {imports:?}"
        );
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
        assert!(
            !imports.contains(&source_abs),
            "self-import must be excluded"
        );
    }

    #[test]
    fn go_module_import_resolves_package_file() {
        let dir = TempDir::new().unwrap();
        fs::write(dir.path().join("go.mod"), "module example.com/cortyx-go\n").unwrap();

        let pkg = dir.path().join("pkg").join("auth");
        fs::create_dir_all(&pkg).unwrap();
        fs::write(pkg.join("auth.go"), b"package auth\n").unwrap();

        let source_abs = dir.path().join("main.go");
        fs::write(
            &source_abs,
            br#"
import (
    "fmt"
    "example.com/cortyx-go/pkg/auth"
)
"#,
        )
        .unwrap();
        let content = fs::read_to_string(&source_abs).unwrap();

        let imports = parse_imports(&source_abs, &content, dir.path());
        assert!(
            imports
                .iter()
                .any(|path| path.ends_with("pkg/auth/auth.go")),
            "expected pkg/auth/auth.go in {imports:?}"
        );
    }
}
