use crate::error::Result;
use std::path::{Path, PathBuf};

/// Returns `true` for paths that should not get neurons.
///
/// Expects a **relative** path from the project root. Calling with an
/// absolute path risks false positives (e.g. macOS tempdirs start with `.`).
pub fn should_skip(rel: &Path) -> bool {
    for component in rel.components() {
        let s = component.as_os_str().to_string_lossy();
        if s.starts_with('.') {
            return true;
        }
    }

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
        if SKIP_DIRS.contains(&s.as_ref()) {
            return true;
        }
    }

    let s = rel.to_string_lossy();
    if s.contains(".cortyx") || s.ends_with(".context.md") || s.ends_with(".context.json") {
        return true;
    }

    const SKIP_EXT: &[&str] = &[
        "png", "jpg", "jpeg", "gif", "svg", "ico", "webp", "woff", "woff2", "ttf", "eot", "mp3",
        "mp4", "wav", "ogg", "zip", "tar", "gz", "bz2", "xz", "7z", "pdf", "doc", "docx", "xls",
        "xlsx", "exe", "dll", "so", "dylib", "a", "o", "class", "pyc", "pyo", "bin", "dat", "db",
        "sqlite", "sqlite3", "min.js", "min.css", "map",
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
pub fn validate_relative_path(raw: &str) -> Result<PathBuf> {
    let path = PathBuf::from(raw);
    if path.is_absolute() {
        crate::cortyx_bail!("path must be relative, got absolute: {raw}");
    }
    for component in path.components() {
        use std::path::Component;
        match component {
            Component::Normal(s) => {
                let s = s.to_string_lossy();
                if s.starts_with('.') {
                    crate::cortyx_bail!("hidden component not allowed: {s} in {raw}");
                }
            },
            other => crate::cortyx_bail!("unsafe path component {:?} in: {raw}", other),
        }
    }
    Ok(path)
}

/// Validate a neuron-to-neuron synapse target path.
///
/// Less strict than `validate_relative_path`: allows `.cortyx/neurons/...` targets for
/// stored neuron links, but rejects every other hidden component as well as traversal.
pub fn validate_synapse_path(raw: &str) -> Result<PathBuf> {
    let path = PathBuf::from(raw);
    if path.is_absolute() {
        crate::cortyx_bail!("synapse target must be relative, got absolute: {raw}");
    }
    let mut parts = Vec::new();
    for component in path.components() {
        use std::path::Component;
        match component {
            Component::Normal(part) => parts.push(part.to_string_lossy().into_owned()),
            other => {
                crate::cortyx_bail!("unsafe path component {:?} in synapse target: {raw}", other)
            },
        }
    }
    if parts.is_empty() {
        crate::cortyx_bail!("synapse target must not be empty");
    }
    if parts[0].starts_with('.') {
        if parts[0] != ".cortyx" || parts.get(1).map(String::as_str) != Some("neurons") {
            crate::cortyx_bail!(
                "hidden synapse target paths must stay under .cortyx/neurons: {raw}"
            );
        }
        if parts.iter().skip(2).any(|part| part.starts_with('.')) {
            crate::cortyx_bail!("hidden component not allowed in synapse target: {raw}");
        }
    } else if parts.iter().any(|part| part.starts_with('.')) {
        crate::cortyx_bail!("hidden component not allowed in synapse target: {raw}");
    }
    Ok(path)
}

#[cfg(test)]
mod tests {
    use super::*;

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
        assert!(validate_relative_path(".hidden/file").is_err());
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
}
