use crate::error::{Result, SecurityError};
use std::path::{Path, PathBuf};

/// Returns `true` for paths that should not get neurons.
///
/// Expects a **relative** path from the project root. Calling with an
/// absolute path risks false positives (e.g. macOS tempdirs start with `.`).
#[must_use]
pub fn should_skip(rel: &Path) -> bool {
    const SKIP_DIRS: &[&str] = &[
        "target",
        "node_modules",
        "__pycache__",
        "vendor",
        "dist",
        ".next",
        "build",
        "out",
        ".venv",
        "venv",
        "env",
    ];
    for component in rel.components() {
        let s = component.as_os_str().to_string_lossy();
        if s.starts_with('.') || SKIP_DIRS.contains(&s.as_ref()) {
            return true;
        }
    }

    let s = rel.to_string_lossy();
    if s.contains(".cortyx") || s.ends_with(".context.md") || s.ends_with(".context.json") {
        return true;
    }

    // `Path::extension()` returns only the last component ("js" for "bundle.min.js"),
    // so minified assets need a dedicated filename-level check.
    if let Some(name) = rel.file_name().map(|n| n.to_string_lossy().to_lowercase()) {
        if name.ends_with(".min.js") || name.ends_with(".min.css") {
            return true;
        }
    }

    const SKIP_EXT: &[&str] = &[
        "png", "jpg", "jpeg", "gif", "svg", "ico", "webp", "woff", "woff2", "ttf", "eot", "mp3",
        "mp4", "wav", "ogg", "zip", "tar", "gz", "bz2", "xz", "7z", "pdf", "doc", "docx", "xls",
        "xlsx", "exe", "dll", "so", "dylib", "a", "o", "class", "pyc", "pyo", "bin", "dat", "db",
        "sqlite", "sqlite3", "map",
    ];
    if let Some(ext) = rel.extension().map(|e| e.to_string_lossy().to_lowercase()) {
        if SKIP_EXT.contains(&ext.as_str()) {
            return true;
        }
    }

    const SKIP_EXACT: &[&str] = &[
        "Cargo.lock",
        "package-lock.json",
        "yarn.lock",
        "pnpm-lock.yaml",
        "poetry.lock",
        "go.sum",
        "composer.lock",
        "Gemfile.lock",
        "uv.lock",
    ];
    if let Some(name) = rel.file_name().map(|n| n.to_string_lossy()) {
        if SKIP_EXACT.contains(&name.as_ref()) {
            return true;
        }
        if name.ends_with(".lock") || name.ends_with(".log") {
            return true;
        }
    }

    false
}

/// Validate that a user-supplied path is safe to use relative to the project root.
///
/// Rejects: absolute paths, `..` components, and components starting with `.`.
///
/// # Errors
///
/// Returns an error if the underlying operation fails.
pub fn validate_relative_path(raw: &str) -> Result<PathBuf> {
    let path = PathBuf::from(raw);
    if path.is_absolute() {
        return Err(SecurityError::PathEscape {
            path: raw.to_string(),
        }
        .into());
    }
    for component in path.components() {
        use std::path::Component;
        match component {
            Component::Normal(s) => {
                let s = s.to_string_lossy();
                if s.starts_with('.') {
                    return Err(SecurityError::HiddenPath {
                        path: raw.to_string(),
                    }
                    .into());
                }
            },
            _ => {
                return Err(SecurityError::PathEscape {
                    path: raw.to_string(),
                }
                .into())
            },
        }
    }
    Ok(path)
}

/// Validate a neuron-to-neuron synapse target path.
///
/// Less strict than `validate_relative_path`: allows `.cortyx/neurons/...` targets for
/// stored neuron links, but rejects every other hidden component as well as traversal.
///
/// # Errors
///
/// Returns an error if the underlying operation fails.
pub fn validate_synapse_path(raw: &str) -> Result<PathBuf> {
    let path = PathBuf::from(raw);
    if path.is_absolute() {
        return Err(SecurityError::PathEscape {
            path: raw.to_string(),
        }
        .into());
    }
    let mut parts = Vec::new();
    for component in path.components() {
        use std::path::Component;
        match component {
            Component::Normal(part) => parts.push(part.to_string_lossy().into_owned()),
            _ => {
                return Err(SecurityError::PathEscape {
                    path: raw.to_string(),
                }
                .into())
            },
        }
    }
    if parts.is_empty() {
        return Err(SecurityError::PathEscape {
            path: raw.to_string(),
        }
        .into());
    }
    if parts[0].starts_with('.') {
        if parts[0] != ".cortyx" || parts.get(1).map(String::as_str) != Some("neurons") {
            return Err(SecurityError::HiddenPath {
                path: raw.to_string(),
            }
            .into());
        }
        if parts.iter().skip(2).any(|part| part.starts_with('.')) {
            return Err(SecurityError::HiddenPath {
                path: raw.to_string(),
            }
            .into());
        }
    } else if parts.iter().any(|part| part.starts_with('.')) {
        return Err(SecurityError::HiddenPath {
            path: raw.to_string(),
        }
        .into());
    }
    Ok(path)
}

/// Confirm that `rel`, when joined with `root`, does not escape `root` via symlinks.
///
/// Call this after [`validate_relative_path`] in any write/read path where the joined
/// path must physically reside within `root`. The lexical check in `validate_relative_path`
/// is necessary but not sufficient when `root` contains symlinks.
///
/// Returns:
/// - `Ok(canonical)` if the path exists and resolves within `root`
/// - `Ok(joined)` (non-canonical) if the path does not yet exist (new neuron) —
///   creation is safe because the lexical check already rejected traversal components
/// - `Err(SecurityError::PathEscape)` if the canonicalized path escapes `root`
/// - `Err(CortyxError::Io)` for unexpected IO errors
///
/// # Errors
///
/// Returns an error if the underlying operation fails.
pub fn validate_within_root(
    root: &std::path::Path,
    rel: &std::path::Path,
) -> crate::error::Result<std::path::PathBuf> {
    // Canonicalize root first to handle symlinks (e.g. /tmp → /private/tmp on macOS)
    let canonical_root = root.canonicalize().unwrap_or_else(|_| root.to_path_buf());
    let joined = canonical_root.join(rel);
    match joined.canonicalize() {
        Ok(canonical) => {
            if canonical.starts_with(&canonical_root) {
                Ok(canonical)
            } else {
                Err(crate::error::SecurityError::PathEscape {
                    path: rel.to_string_lossy().into_owned(),
                }
                .into())
            }
        },
        // Path doesn't exist yet (new neuron write) — return relative to original root
        // so callers see a consistent path regardless of whether root is a symlink.
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(root.join(rel)),
        Err(e) => Err(crate::error::CortyxError::Io(e)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;

    #[test]
    fn should_skip_target_dir() {
        assert!(should_skip(Path::new("target/debug/cortyx")));
    }

    #[test]
    fn should_skip_hidden_dir() {
        assert!(should_skip(Path::new(".git/HEAD")));
    }

    #[test]
    fn should_skip_node_modules() {
        assert!(should_skip(Path::new("node_modules/react/index.js")));
    }

    #[test]
    fn should_skip_neuron_files() {
        assert!(should_skip(Path::new(".cortyx/neurons/foo.context.md")));
    }

    #[test]
    fn should_skip_binary_extensions() {
        assert!(should_skip(Path::new("assets/logo.png")));
        assert!(should_skip(Path::new("dist/bundle.min.js")));
    }

    #[test]
    fn should_skip_lock_files() {
        assert!(should_skip(Path::new("Cargo.lock")));
        assert!(should_skip(Path::new("package-lock.json")));
        assert!(should_skip(Path::new("yarn.lock")));
    }

    #[test]
    fn should_skip_log_files() {
        assert!(should_skip(Path::new("logs/app.log")));
        assert!(should_skip(Path::new("debug.log")));
    }

    #[test]
    fn should_not_skip_source_files() {
        assert!(!should_skip(Path::new("src/main.rs")));
        assert!(!should_skip(Path::new("lib/auth.py")));
        assert!(!should_skip(Path::new("README.md")));
    }

    #[test]
    fn validate_relative_path_ok() {
        let p = validate_relative_path("src/engine.rs").unwrap();
        assert_eq!(p, PathBuf::from("src/engine.rs"));
    }

    #[test]
    fn validate_relative_path_rejects_absolute() {
        assert!(validate_relative_path("/etc/passwd").is_err());
    }

    #[test]
    fn validate_relative_path_rejects_parent_dir() {
        assert!(validate_relative_path("../../etc/passwd").is_err());
    }

    #[test]
    fn validate_relative_path_rejects_hidden() {
        let result = validate_relative_path(".hidden/file");
        assert!(result.is_err());
        assert!(
            matches!(
                result.unwrap_err(),
                crate::error::CortyxError::Security(crate::error::SecurityError::HiddenPath { .. })
            ),
            "dot-prefixed path should return HiddenPath, not PathEscape"
        );
    }

    #[test]
    fn validate_synapse_path_allows_neuron_store_prefix() {
        let path = validate_synapse_path(".cortyx/neurons/src/auth_rs.context.md").unwrap();
        assert_eq!(
            path,
            PathBuf::from(".cortyx/neurons/src/auth_rs.context.md")
        );
    }

    #[test]
    fn validate_synapse_path_rejects_other_hidden_components() {
        assert!(validate_synapse_path(".cache/secret.md").is_err());
        assert!(validate_synapse_path("src/.hidden/context.md").is_err());
        assert!(validate_synapse_path(".cortyx/private/context.md").is_err());
    }

    #[test]
    fn validate_within_root_nonexistent_path_ok() {
        let root = std::path::Path::new("/tmp");
        let rel = std::path::Path::new("nonexistent_cortyx_test_file.md");
        assert!(validate_within_root(root, rel).is_ok());
    }

    #[test]
    fn validate_within_root_existing_path_ok() {
        let root = std::path::Path::new("/tmp");
        let tmp = NamedTempFile::new_in("/tmp").ok();
        if let Some(f) = tmp {
            let rel = f
                .path()
                .file_name()
                .map(std::path::Path::new)
                .unwrap_or(std::path::Path::new("test"));
            assert!(validate_within_root(root, rel).is_ok());
        }
    }
}
