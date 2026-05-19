//! Domain-specific error types for Cortyx.
//!
//! This module provides typed errors to replace `anyhow::Error` in public APIs,
//! enabling better error handling and clearer error semantics.

use std::io;
use std::path::PathBuf;
use thiserror::Error;

#[macro_export]
macro_rules! cortyx_err {
    ($($arg:tt)*) => {
        $crate::error::CortyxError::other(format!($($arg)*))
    };
}

#[macro_export]
macro_rules! cortyx_bail {
    ($($arg:tt)*) => {
        return Err($crate::cortyx_err!($($arg)*))
    };
}

#[macro_export]
macro_rules! cortyx_ensure {
    ($cond:expr, $($arg:tt)*) => {
        if !($cond) {
            $crate::cortyx_bail!($($arg)*)
        }
    };
}

/// Errors related to the neuron index.
#[non_exhaustive]
#[derive(Error, Debug)]
pub enum IndexError {
    /// Index file not found or inaccessible.
    #[error("Index not found at path: {path}")]
    NotFound { path: PathBuf },

    /// Index file is corrupted or has invalid format.
    #[error("Corrupted index at {path}: {reason}")]
    Corrupted { path: PathBuf, reason: String },

    /// Index version mismatch requires migration.
    #[error("Index version mismatch: expected {expected}, found {found}")]
    VersionMismatch { expected: u32, found: u32 },

    /// Failed to serialize or deserialize index data.
    #[error("Serialization error: {0}")]
    Serialization(#[from] bincode::Error),

    /// IO error while reading/writing index.
    #[error("IO error: {0}")]
    Io(#[from] io::Error),

    /// Index is locked by another process.
    #[error("Index is locked by another process")]
    Locked,

    /// Index rebuild in progress.
    #[error("Index rebuild in progress")]
    Rebuilding,
}

/// Errors related to individual neurons.
#[non_exhaustive]
#[derive(Error, Debug)]
pub enum NeuronError {
    /// Neuron file not found.
    #[error("Neuron not found: {path}")]
    NotFound { path: PathBuf },

    /// Neuron file has invalid format or missing required sections.
    #[error("Invalid neuron format at {path}: {reason}")]
    InvalidFormat { path: PathBuf, reason: String },

    /// Neuron metadata is missing or invalid.
    #[error("Invalid neuron metadata at {path}: {reason}")]
    InvalidMetadata { path: PathBuf, reason: String },

    /// Neuron file is empty or too small.
    #[error("Empty neuron file: {path}")]
    Empty { path: PathBuf },

    /// IO error while reading/writing neuron.
    #[error("IO error: {0}")]
    Io(#[from] io::Error),

    /// UTF-8 decoding error.
    #[error("UTF-8 decoding error: {0}")]
    Utf8(#[from] std::string::FromUtf8Error),
}

/// Errors related to sync operations.
#[non_exhaustive]
#[derive(Error, Debug)]
pub enum SyncError {
    /// Git repository not found.
    #[error("Git repository not found at: {path}")]
    NoRepository { path: PathBuf },

    /// Git operation failed.
    #[error("Git operation failed: {operation} - {reason}")]
    GitFailed { operation: String, reason: String },

    /// Sync conflict detected.
    #[error("Sync conflict at {path}: {reason}")]
    Conflict { path: PathBuf, reason: String },

    /// Remote repository is unreachable.
    #[error("Remote unreachable: {remote}")]
    RemoteUnreachable { remote: String },

    /// Authentication failed.
    #[error("Authentication failed: {reason}")]
    AuthFailed { reason: String },

    /// IO error during sync.
    #[error("IO error: {0}")]
    Io(#[from] io::Error),
}

/// Errors raised when a security boundary is violated.
#[non_exhaustive]
#[derive(Error, Debug)]
pub enum SecurityError {
    /// A path component would escape the allowed root directory.
    #[error("Path traversal denied: {path}")]
    PathEscape { path: String },

    /// A path component is a hidden (dot-prefixed) file — rejected by policy.
    #[error("Hidden path rejected by policy: {path}")]
    HiddenPath { path: String },

    /// A URL-based remote is not in the configured allowlist.
    #[error("Remote URL not in allowlist: {url}")]
    UntrustedRemote { url: String },

    /// Input exceeds a configured size limit.
    #[error("Input exceeds size limit ({limit} bytes): {context}")]
    SizeExceeded { limit: usize, context: String },
}

/// Errors related to query processing.
#[non_exhaustive]
#[derive(Error, Debug)]
pub enum QueryError {
    /// Query text is empty or invalid.
    #[error("Invalid query: {reason}")]
    Invalid { reason: String },

    /// Query exceeded token budget.
    #[error("Query exceeded token budget: {used} > {budget}")]
    BudgetExceeded { used: usize, budget: usize },

    /// No results found for query.
    #[error("No results found for query: {query}")]
    NoResults { query: String },

    /// Query timeout exceeded.
    #[error("Query timeout exceeded: {timeout_ms}ms")]
    Timeout { timeout_ms: u64 },

    /// Index error during query.
    #[error("Index error: {0}")]
    Index(#[from] IndexError),
}

/// Errors related to embeddings (optional feature).
#[cfg(feature = "embed")]
#[non_exhaustive]
#[derive(Error, Debug)]
pub enum EmbedError {
    /// Embedding model not loaded.
    #[error("Embedding model not loaded")]
    ModelNotLoaded,

    /// Embedding generation failed.
    #[error("Embedding generation failed: {reason}")]
    GenerationFailed { reason: String },

    /// Incompatible embedding dimensions.
    #[error("Incompatible embedding dimensions: expected {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },

    /// IO error.
    #[error("IO error: {0}")]
    Io(#[from] io::Error),
}

/// Top-level result type using domain errors.
pub type Result<T, E = CortyxError> = std::result::Result<T, E>;

/// Unified error type for Cortyx operations.
#[non_exhaustive]
#[derive(Error, Debug)]
pub enum CortyxError {
    /// Index-related error.
    #[error(transparent)]
    Index(#[from] IndexError),

    /// Neuron-related error.
    #[error(transparent)]
    Neuron(#[from] NeuronError),

    /// Sync-related error.
    #[error(transparent)]
    Sync(#[from] SyncError),

    /// Security boundary violation.
    #[error(transparent)]
    Security(#[from] SecurityError),

    /// Query-related error.
    #[error(transparent)]
    Query(#[from] QueryError),

    /// Embedding-related error (optional feature).
    #[cfg(feature = "embed")]
    #[error(transparent)]
    Embed(#[from] EmbedError),

    /// Generic IO error.
    #[error("IO error: {0}")]
    Io(#[from] io::Error),

    /// JSON serialization or deserialization error.
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    /// Regex compilation error.
    #[error("Regex error: {0}")]
    Regex(#[from] regex::Error),

    /// Directory walking error.
    #[error("Walkdir error: {0}")]
    Walkdir(#[from] walkdir::Error),

    /// File watching error.
    #[error("Notify error: {0}")]
    Notify(#[from] notify::Error),

    /// UTF-8 decoding error.
    #[error("UTF-8 error: {0}")]
    Utf8(#[from] std::str::Utf8Error),

    /// Other errors (for internal use, migration path from anyhow).
    #[error("{0}")]
    Other(String),
}

impl CortyxError {
    /// Create an "other" error from any displayable value.
    pub fn other(msg: impl std::fmt::Display) -> Self {
        Self::Other(msg.to_string())
    }
}

impl From<bincode::Error> for CortyxError {
    fn from(err: bincode::Error) -> Self {
        Self::Index(IndexError::Serialization(err))
    }
}

#[cfg(feature = "embed")]
impl From<anyhow::Error> for CortyxError {
    fn from(err: anyhow::Error) -> Self {
        Self::other(err)
    }
}

#[cfg(feature = "rerank")]
impl From<ort::Error> for CortyxError {
    fn from(err: ort::Error) -> Self {
        Self::other(err)
    }
}

impl From<std::array::TryFromSliceError> for CortyxError {
    fn from(err: std::array::TryFromSliceError) -> Self {
        Self::other(err)
    }
}

/// Extension trait for converting `Result<T, E>` into `Result<T, CortyxError>` during
/// the migration away from `anyhow`.
///
/// This is a transitional helper. Prefer typed `From` implementations for new code.
#[deprecated(
    note = "Use typed From impls or anyhow::Context instead; this trait bypasses the type system"
)]
pub trait AnyhowCompat<T> {
    /// Wrap any error with additional context and convert to `CortyxError::Other`.
    fn context_cortyx(self, msg: &str) -> std::result::Result<T, CortyxError>;
}

#[allow(deprecated)]
impl<T, E: std::fmt::Display> AnyhowCompat<T> for std::result::Result<T, E> {
    fn context_cortyx(self, msg: &str) -> std::result::Result<T, CortyxError> {
        self.map_err(|e| CortyxError::other(format!("{msg}: {e}")))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn index_error_display() {
        let err = IndexError::NotFound {
            path: PathBuf::from("/test/path"),
        };
        assert!(err.to_string().contains("/test/path"));
    }

    #[test]
    fn cortyx_error_from_index_error() {
        let index_err = IndexError::Locked;
        let cortyx_err: CortyxError = index_err.into();
        assert!(matches!(cortyx_err, CortyxError::Index(_)));
    }

    #[test]
    fn cortyx_error_other() {
        let err = CortyxError::other("test error");
        assert!(err.to_string().contains("test error"));
    }
}
