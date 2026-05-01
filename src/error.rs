//! Domain-specific error types for Cortyx.
//!
//! This module provides typed errors to replace `anyhow::Error` in public APIs,
//! enabling better error handling and clearer error semantics.

use std::io;
use std::path::PathBuf;
use thiserror::Error;

/// Errors related to the neuron index.
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

/// Errors related to query processing.
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

/// Extension trait for converting anyhow errors during migration.
pub trait AnyhowCompat {
    fn context_cortyx(self, msg: &str) -> CortyxError;
}

impl<T, E: std::fmt::Display> AnyhowCompat for std::result::Result<T, E> {
    fn context_cortyx(self, msg: &str) -> CortyxError {
        match self {
            Ok(_) => unreachable!("context_cortyx called on Ok"),
            Err(e) => CortyxError::other(format!("{}: {}", msg, e)),
        }
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
