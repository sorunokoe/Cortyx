//! Centralized git binary resolution.

use std::path::{Path, PathBuf};
use std::sync::OnceLock;

static GIT_BINARY: OnceLock<PathBuf> = OnceLock::new();

/// Returns the path to the `git` binary, resolved once at first call.
///
/// Resolution order:
/// 1. `GIT_PATH` env var (must be absolute and exist)
/// 2. `which git` output (canonicalized)
/// 3. `/usr/bin/git` (canonicalized)
/// 4. `"git"` — bare name, relies on `PATH` at command-invocation time (last resort)
///
/// # Panics
///
/// Does not panic.
pub fn git_binary() -> &'static Path {
    git_binary_from_lock(&GIT_BINARY)
}

fn git_binary_from_lock(lock: &OnceLock<PathBuf>) -> &Path {
    lock.get_or_init(resolve_git_binary).as_path()
}

fn resolve_git_binary() -> PathBuf {
    if let Ok(explicit) = std::env::var("GIT_PATH") {
        if let Some(path) = sanitize_git_path(PathBuf::from(explicit)) {
            return path;
        }
    }

    if let Ok(output) = std::process::Command::new("which").arg("git").output() {
        if output.status.success() {
            let path = PathBuf::from(String::from_utf8_lossy(&output.stdout).trim());
            if let Some(path) = sanitize_git_path(path) {
                return path;
            }
        }
    }

    // If the canonical path doesn't exist, fall back to bare "git" so the OS
    // PATH resolves it at invocation time. This is preferable to returning a
    // known-invalid absolute path (e.g. /usr/bin/git on Apple Silicon where git
    // lives at /opt/homebrew/bin/git and which-git already failed above).
    sanitize_git_path(PathBuf::from("/usr/bin/git")).unwrap_or_else(|| PathBuf::from("git"))
}

fn sanitize_git_path(path: PathBuf) -> Option<PathBuf> {
    if !path.is_absolute() || !path.exists() {
        return None;
    }

    path.canonicalize().ok().or(Some(path))
}

#[cfg(test)]
mod tests {
    use super::*;

    static GIT_PATH_ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    #[test]
    fn git_binary_resolves_to_something() {
        // On most systems `which git` succeeds and produces an absolute path.
        // On systems without git at a known absolute path the fallback is bare "git"
        // (relies on PATH at invocation time). Either outcome is acceptable.
        let p = git_binary();
        assert!(!p.as_os_str().is_empty(), "git_binary must not be empty");
    }

    #[test]
    fn git_binary_env_override() {
        let _env_guard = GIT_PATH_ENV_LOCK.lock().unwrap();
        let original = std::env::var_os("GIT_PATH");
        std::env::set_var("GIT_PATH", "/usr/bin/git");

        let test_lock = OnceLock::new();
        let resolved = git_binary_from_lock(&test_lock);

        assert_eq!(resolved, Path::new("/usr/bin/git"));

        match original {
            Some(value) => std::env::set_var("GIT_PATH", value),
            None => std::env::remove_var("GIT_PATH"),
        }
    }
}
