//! Utility functions for installer module.

use std::path::PathBuf;

/// Get the user's home directory.
pub(super) fn dirs_home() -> PathBuf {
    std::env::var("HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("."))
}
