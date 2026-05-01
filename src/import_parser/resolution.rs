//! Path resolution utilities for import candidates.

use std::fs;
use std::path::{Path, PathBuf};

/// Try to resolve a candidate path (possibly without extension) to an existing file.
///
/// Tries the path as-is, then with common source extensions, then as a directory
/// index file.
///
/// Security: canonicalizes the resolved path and verifies it remains inside
/// `project_root`. This prevents path-traversal attacks via crafted import strings
/// such as `../../etc/passwd` in TypeScript or Python sources.
pub(super) fn resolve_to_existing(candidate: PathBuf, project_root: &Path) -> Option<PathBuf> {
    // Compute a canonical project root once for containment checks.
    // Fall back to the raw path on systems where canonicalize fails (e.g. missing dirs).
    let root_canonical = project_root
        .canonicalize()
        .unwrap_or_else(|_| project_root.to_path_buf());

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

    let candidate_dir = project_root.join(&candidate);
    if candidate_dir.is_dir() {
        if let Some(name) = candidate.file_name().and_then(|s| s.to_str()) {
            if let Some(p) = check(candidate_dir.join(format!("{name}.go"))) {
                return Some(p);
            }
        }
        for preferred in ["main.go", "doc.go"] {
            if let Some(p) = check(candidate_dir.join(preferred)) {
                return Some(p);
            }
        }
        let go_files: Vec<PathBuf> = fs::read_dir(&candidate_dir)
            .ok()
            .into_iter()
            .flatten()
            .filter_map(|entry| {
                let path = entry.ok()?.path();
                (path.extension().and_then(|ext| ext.to_str()) == Some("go")).then_some(path)
            })
            .collect();
        if go_files.len() == 1 {
            if let Some(p) = check(go_files[0].clone()) {
                return Some(p);
            }
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
