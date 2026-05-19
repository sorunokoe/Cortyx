//! Typed identifiers — eliminates raw `String` IDs that can be confused at call sites.
//!
//! All types are validated at construction; callers that hold a value already
//! have a proof that it satisfies the format invariant.

use std::fmt;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use crate::error::Result;
use serde::{Deserialize, Serialize};

static UUID_COUNTER: AtomicU64 = AtomicU64::new(0);

// ─── NeuronUuid ──────────────────────────────────────────────────────────────

/// A 32-character lowercase hexadecimal neuron identifier.
///
/// Generated from a blake3 hash of `path:nanos:nonce`; globally unique in
/// practice and stable across renames (the UUID travels with the neuron file).
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct NeuronUuid(String);

impl NeuronUuid {
    /// Parse a pre-existing UUID string, e.g. from a `.meta.json` file.
    ///
    /// Returns an error if `s` is not exactly 32 lowercase hex characters.
    pub fn parse(s: &str) -> Result<Self> {
        if s.len() != 32 || !s.chars().all(|c| c.is_ascii_hexdigit()) {
            crate::cortyx_bail!("invalid NeuronUuid (expected 32 hex chars): {:?}", s);
        }
        Ok(Self(s.to_ascii_lowercase()))
    }

    /// Generate a fresh UUID from a source path hint (used during compilation).
    pub fn generate(source_hint: &std::path::Path) -> Self {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        let nonce = UUID_COUNTER.fetch_add(1, Ordering::Relaxed);
        let input = format!("{}:{nanos}:{nonce}", source_hint.display());
        let hash = blake3::hash(input.as_bytes());
        Self(hash.to_hex()[..32].to_string())
    }

    /// Borrow the inner string slice.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for NeuronUuid {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl AsRef<str> for NeuronUuid {
    fn as_ref(&self) -> &str {
        &self.0
    }
}

// ─── EditId ──────────────────────────────────────────────────────────────────

/// An opaque, non-empty edit identifier used in provenance chains.
///
/// Typically a timestamp-nonce string; the only structural requirement
/// is non-emptiness (an empty edit ID is meaningless as a chain pointer).
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct EditId(String);

impl EditId {
    /// Construct from an existing string, rejecting empty values.
    pub fn new(s: impl Into<String>) -> Result<Self> {
        let s = s.into();
        if s.is_empty() {
            crate::cortyx_bail!("EditId must not be empty");
        }
        Ok(Self(s))
    }

    /// Generate a fresh edit ID based on current time and a monotonic counter.
    pub fn generate() -> Self {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        let nonce = UUID_COUNTER.fetch_add(1, Ordering::Relaxed);
        Self(format!("{nanos:032x}{nonce:016x}"))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for EditId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl AsRef<str> for EditId {
    fn as_ref(&self) -> &str {
        &self.0
    }
}

// ─── AuthorId ────────────────────────────────────────────────────────────────

/// A non-empty agent or user identifier attached to provenance edits.
///
/// No format beyond non-emptiness is enforced here; callers use free-form
/// agent names like `"assistant"`, `"human"`, or `"@alice"`.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct AuthorId(String);

impl AuthorId {
    pub fn new(s: impl Into<String>) -> Result<Self> {
        let s = s.into();
        if s.is_empty() {
            crate::cortyx_bail!("AuthorId must not be empty");
        }
        Ok(Self(s))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for AuthorId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl AsRef<str> for AuthorId {
    fn as_ref(&self) -> &str {
        &self.0
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

// ─── NeuronId ────────────────────────────────────────────────────────────────

/// Canonical project-relative identity of a neuron.
///
/// Wraps a `PathBuf` that is always relative to the project root (never
/// absolute). Prevents mixing absolute and relative paths in indexing APIs.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct NeuronId(PathBuf);

impl NeuronId {
    /// Construct from a project-relative path. Returns an error if `path` is
    /// absolute.
    pub fn new(path: impl Into<PathBuf>) -> Result<Self> {
        let path = path.into();
        if path.is_absolute() {
            crate::cortyx_bail!(
                "NeuronId must be a project-relative path, got absolute: {:?}",
                path
            );
        }
        Ok(Self(path))
    }

    /// Construct without validation — only use when the caller has already
    /// established the path is project-relative (e.g. after `strip_prefix`).
    pub fn from_relative_unchecked(path: PathBuf) -> Self {
        debug_assert!(!path.is_absolute(), "NeuronId must be relative");
        Self(path)
    }

    pub fn as_path(&self) -> &Path {
        &self.0
    }

    pub fn into_path(self) -> PathBuf {
        self.0
    }
}

impl fmt::Display for NeuronId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0.display())
    }
}

impl AsRef<Path> for NeuronId {
    fn as_ref(&self) -> &Path {
        &self.0
    }
}

// ─── NeuronRelPath ───────────────────────────────────────────────────────────

/// A project-relative file path — guaranteed never to be absolute.
///
/// Lighter than `NeuronId` (no identity semantics), used wherever a path
/// component must stay relative to the project root (synapse targets, source
/// file references, sidecar paths).
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct NeuronRelPath(PathBuf);

impl NeuronRelPath {
    /// Construct from a relative path. Returns an error if `path` is absolute
    /// or contains `..` components (path-escape defence).
    pub fn new(path: impl Into<PathBuf>) -> Result<Self> {
        let path = path.into();
        if path.is_absolute() {
            crate::cortyx_bail!("NeuronRelPath must be relative, got absolute: {:?}", path);
        }
        if path.components().any(|c| {
            matches!(
                c,
                std::path::Component::ParentDir | std::path::Component::CurDir
            )
        }) {
            crate::cortyx_bail!(
                "NeuronRelPath must not contain .. or . components: {:?}",
                path
            );
        }
        Ok(Self(path))
    }

    pub fn as_path(&self) -> &Path {
        &self.0
    }

    pub fn into_path(self) -> PathBuf {
        self.0
    }
}

impl fmt::Display for NeuronRelPath {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0.display())
    }
}

impl AsRef<Path> for NeuronRelPath {
    fn as_ref(&self) -> &Path {
        &self.0
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn neuron_uuid_parse_accepts_32_hex() {
        let s = "a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4";
        assert!(NeuronUuid::parse(s).is_ok());
    }

    #[test]
    fn neuron_uuid_parse_rejects_short() {
        assert!(NeuronUuid::parse("abc").is_err());
    }

    #[test]
    fn neuron_uuid_parse_rejects_non_hex() {
        assert!(NeuronUuid::parse("z1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4").is_err());
    }

    #[test]
    fn neuron_uuid_generate_unique() {
        let path = std::path::Path::new("src/foo.rs");
        let a = NeuronUuid::generate(path);
        let b = NeuronUuid::generate(path);
        assert_ne!(a, b, "two generates must differ");
        assert_eq!(a.as_str().len(), 32);
    }

    #[test]
    fn edit_id_rejects_empty() {
        assert!(EditId::new("").is_err());
    }

    #[test]
    fn edit_id_generate_unique() {
        let a = EditId::generate();
        let b = EditId::generate();
        assert_ne!(a, b);
    }

    #[test]
    fn author_id_rejects_empty() {
        assert!(AuthorId::new("").is_err());
    }

    #[test]
    fn serde_round_trip() {
        let uuid = NeuronUuid::parse("deadbeefdeadbeefdeadbeefdeadbeef").unwrap();
        let json = serde_json::to_string(&uuid).unwrap();
        let back: NeuronUuid = serde_json::from_str(&json).unwrap();
        assert_eq!(uuid, back);
    }

    #[test]
    fn neuron_id_rejects_absolute() {
        assert!(NeuronId::new("/absolute/path.md").is_err());
    }

    #[test]
    fn neuron_id_accepts_relative() {
        let id = NeuronId::new("src/auth.md").unwrap();
        assert_eq!(id.as_path(), std::path::Path::new("src/auth.md"));
    }

    #[test]
    fn neuron_rel_path_rejects_absolute() {
        assert!(NeuronRelPath::new("/bad/path").is_err());
    }

    #[test]
    fn neuron_rel_path_rejects_parent_dir() {
        assert!(NeuronRelPath::new("../escape").is_err());
    }

    #[test]
    fn neuron_rel_path_accepts_normal_relative() {
        let p = NeuronRelPath::new("neurons/auth.context.md").unwrap();
        assert_eq!(p.as_path(), std::path::Path::new("neurons/auth.context.md"));
    }
}
