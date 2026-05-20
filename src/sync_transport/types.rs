//! Typed newtypes for sync transport domain.

use std::fmt;
use std::path::{Path, PathBuf};

/// A validated repository path.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct RepoPath(PathBuf);

impl RepoPath {
    /// Create a new RepoPath, validating it exists.
    ///
    /// # Errors
    ///
    /// Returns an error if the underlying operation fails.
    pub fn new(path: impl AsRef<Path>) -> Result<Self, String> {
        let path = path.as_ref();
        if !path.exists() {
            return Err(format!(
                "Repository path does not exist: {}",
                path.display()
            ));
        }
        Ok(Self(path.to_path_buf()))
    }

    /// Create a RepoPath without validation (unsafe - use only when path is known to exist).
    pub fn new_unchecked(path: impl Into<PathBuf>) -> Self {
        Self(path.into())
    }

    /// Get the inner PathBuf.
    #[must_use]
    pub fn as_path(&self) -> &Path {
        &self.0
    }

    /// Convert into the inner PathBuf.
    #[must_use]
    pub fn into_inner(self) -> PathBuf {
        self.0
    }
}

impl AsRef<Path> for RepoPath {
    fn as_ref(&self) -> &Path {
        &self.0
    }
}

impl fmt::Display for RepoPath {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0.display())
    }
}

/// A Git commit hash (40-character hex string).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct CommitHash(String);

impl CommitHash {
    /// Create a new CommitHash, validating the format.
    ///
    /// # Errors
    ///
    /// Returns an error if the underlying operation fails.
    pub fn new(hash: impl Into<String>) -> Result<Self, String> {
        let hash = hash.into();
        if hash.len() != 40 {
            return Err(format!(
                "Invalid commit hash length: {} (expected 40)",
                hash.len()
            ));
        }
        if !hash.chars().all(|c| c.is_ascii_hexdigit()) {
            return Err("Invalid commit hash: contains non-hex characters".to_string());
        }
        Ok(Self(hash))
    }

    /// Create a CommitHash without validation (unsafe - use only for trusted sources).
    pub fn new_unchecked(hash: impl Into<String>) -> Self {
        Self(hash.into())
    }

    /// Get the hash as a string slice.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// Convert into the inner String.
    #[must_use]
    pub fn into_inner(self) -> String {
        self.0
    }

    /// Get the short hash (first 7 characters).
    #[must_use]
    pub fn short(&self) -> &str {
        &self.0[..7]
    }
}

impl fmt::Display for CommitHash {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl AsRef<str> for CommitHash {
    fn as_ref(&self) -> &str {
        &self.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn commit_hash_valid() {
        let hash = "a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2";
        let commit = CommitHash::new(hash).unwrap();
        assert_eq!(commit.as_str(), hash);
        assert_eq!(commit.short(), "a1b2c3d");
    }

    #[test]
    fn commit_hash_invalid_length() {
        let hash = "tooshort";
        assert!(CommitHash::new(hash).is_err());
    }

    #[test]
    fn commit_hash_invalid_chars() {
        let hash = "gggggggggggggggggggggggggggggggggggggggg"; // 'g' is not hex
        assert!(CommitHash::new(hash).is_err());
    }

    #[test]
    fn repo_path_display() {
        let path = RepoPath::new_unchecked("/tmp/test");
        assert!(path.to_string().contains("/tmp/test"));
    }
}
