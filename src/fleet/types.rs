//! Fleet-specific types for optional multi-project context routing.

use std::fmt;
use std::path::PathBuf;

use serde::{Deserialize, Serialize};

/// Stable identifier for a registered fleet node.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct FleetNodeId(String);

impl FleetNodeId {
    /// Create a new fleet node identifier.
    pub fn new(id: impl Into<String>) -> Self {
        Self(id.into())
    }

    /// Borrow the raw identifier string.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for FleetNodeId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

/// A registered Cortyx project participating in the local fleet.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FleetNode {
    pub id: FleetNodeId,
    pub path: PathBuf,
    pub alias: String,
    pub modules: Vec<String>,
    pub last_registered: String,
    /// Git remote URL when this node is a git-backed shared corpus.
    /// When `Some`, `path` is a managed local clone under `~/.cortyx/fleet/{alias}/`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub git_url: Option<String>,
    /// ISO 8601 timestamp of the last successful `git fetch` for git-backed nodes.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub last_fetched: Option<String>,
}

/// Persisted fleet registry stored under `~/.cortyx/fleet/`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FleetRegistry {
    pub version: u32,
    pub nodes: Vec<FleetNode>,
}

impl Default for FleetRegistry {
    fn default() -> Self {
        Self {
            version: super::FLEET_REGISTRY_VERSION,
            nodes: Vec::new(),
        }
    }
}

/// Simplified response returned from a routed fleet query.
#[derive(Debug, Clone, PartialEq)]
pub struct FleetQueryResult {
    pub node_id: FleetNodeId,
    pub node_alias: String,
    pub contexts: String,
    pub top_score: f32,
}

/// Why a query escalated beyond the local project.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FleetRouteReason {
    LowLocalConfidence,
    ExplicitRequest,
    ModuleMatch(String),
}
