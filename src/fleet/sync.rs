//! Git-backed fleet node synchronisation.
//!
//! A fleet node with a `git_url` is a shared corpus hosted in a git repository.
//! Cortyx clones it on first registration and fetches updates at serve startup.
//!
//! # URL allowlist
//! Only `https://github.com/`, `https://gitlab.com/`, `git@github.com:`, and
//! `git@gitlab.com:` URLs are accepted. This matches the existing allowlist used
//! by the global concept library, adding zero new attack surface.
//!
//! # Offline safety
//! A `git fetch` failure is logged as a warning and does not prevent the local
//! clone from being used. The fleet continues to serve the last-synced state.

use std::path::{Path, PathBuf};
use std::process::Command;

use crate::error::Result;
use crate::neuron::now_iso8601;

use super::types::FleetNode;

/// Accepted URL prefixes for git-backed fleet nodes.
const ALLOWED_PREFIXES: &[&str] = &[
    "https://github.com/",
    "https://gitlab.com/",
    "git@github.com:",
    "git@gitlab.com:",
];

/// Validate that `url` is in the git fleet allowlist.
#[must_use]
pub fn is_allowed_git_url(url: &str) -> bool {
    ALLOWED_PREFIXES
        .iter()
        .any(|prefix| url.starts_with(prefix))
}

/// Local clone directory for a git-backed fleet node.
///
/// # Errors
///
/// Returns an error if the underlying operation fails.
pub fn git_fleet_cache_dir(alias: &str) -> Result<PathBuf> {
    let home =
        dirs::home_dir().ok_or_else(|| crate::cortyx_err!("could not determine home directory"))?;
    Ok(home.join(".cortyx").join("fleet").join(alias))
}

/// Clone or fetch a git-backed fleet node.
///
/// - If the local clone does not exist: `git clone <url> <path>`.
/// - If it exists: `git fetch --ff-only`.
/// - On fetch failure: log a warning and return `Ok(())` (offline-safe).
///
/// Returns the local path of the clone.
///
/// # Errors
///
/// Returns an error if the underlying operation fails.
pub fn sync_fleet_node(node: &FleetNode) -> Result<()> {
    let url = match &node.git_url {
        Some(u) => u,
        None => return Ok(()), // local node — nothing to sync
    };

    if !is_allowed_git_url(url) {
        return Err(crate::cortyx_err!(
            "Git fleet URL not in allowlist: {url}. \
             Accepted prefixes: https://github.com/, https://gitlab.com/, \
             git@github.com:, git@gitlab.com:"
        ));
    }

    let local_path = &node.path;

    if !local_path.exists() {
        clone_repo(url, local_path)
    } else {
        fetch_repo(local_path, url)
    }
}

/// Clone a git repository to `dest`.
fn clone_repo(url: &str, dest: &Path) -> Result<()> {
    if let Some(parent) = dest.parent() {
        std::fs::create_dir_all(parent)?;
    }
    tracing::info!(url, dest = %dest.display(), "Fleet: cloning git-backed node");
    let status = Command::new("git")
        .args(["clone", "--depth=1", url, &dest.to_string_lossy()])
        .status()?;
    if status.success() {
        Ok(())
    } else {
        Err(crate::cortyx_err!(
            "git clone failed for fleet node (url={url}, dest={})",
            dest.display()
        ))
    }
}

/// Fetch updates for an existing local clone, falling back gracefully on failure.
fn fetch_repo(path: &Path, url: &str) -> Result<()> {
    tracing::debug!(url, path = %path.display(), "Fleet: fetching git-backed node");
    let status = Command::new("git")
        .args([
            "-C",
            &path.to_string_lossy(),
            "fetch",
            "--ff-only",
            "origin",
        ])
        .status();
    match status {
        Ok(s) if s.success() => {},
        Ok(s) => tracing::warn!(
            url,
            code = ?s.code(),
            "Fleet git fetch failed — using cached clone"
        ),
        Err(e) => tracing::warn!(url, "Fleet git fetch error ({e}) — using cached clone"),
    }
    Ok(())
}

/// Update `last_fetched` timestamp on a node in the registry after a successful sync.
pub fn update_last_fetched(node: &mut FleetNode) {
    if node.git_url.is_some() {
        node.last_fetched = Some(now_iso8601());
    }
}
