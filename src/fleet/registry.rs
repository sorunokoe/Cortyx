//! Registry persistence for optional local fleet coordination.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use crate::error::Result;
use crate::index::NeuronIndex;
use crate::neuron::{atomic_write, now_iso8601};

use super::router::{cache_fleet_index, invalidate_fleet_index};
use super::{FleetNode, FleetNodeId, FleetRegistry, FLEET_REGISTRY_VERSION};

/// Path to the fleet registry file.
pub fn fleet_registry_path() -> Result<PathBuf> {
    let home_dir =
        dirs::home_dir().ok_or_else(|| crate::cortyx_err!("could not determine home directory"))?;
    Ok(home_dir.join(".cortyx").join("fleet").join("nodes.json"))
}

/// Load the fleet registry from disk, returning an empty registry when absent.
pub fn load_registry() -> Result<FleetRegistry> {
    let path = fleet_registry_path()?;
    if !path.exists() {
        return Ok(FleetRegistry::default());
    }

    let data = std::fs::read_to_string(&path)?;
    let mut registry: FleetRegistry = serde_json::from_str(&data)?;
    registry.version = FLEET_REGISTRY_VERSION;
    Ok(registry)
}

/// Save the fleet registry atomically.
pub fn save_registry(registry: &FleetRegistry) -> Result<()> {
    let path = fleet_registry_path()?;
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }

    let stored = FleetRegistry {
        version: FLEET_REGISTRY_VERSION,
        nodes: registry.nodes.clone(),
    };
    let json = serde_json::to_string_pretty(&stored)?;
    atomic_write(&path, json.as_bytes())?;
    Ok(())
}

/// Register or update a project as a fleet node.
pub fn register_node(project_path: &Path, alias: Option<String>) -> Result<FleetNode> {
    let canonical_path = std::fs::canonicalize(project_path)?;
    let index = match NeuronIndex::load_or_create(&canonical_path) {
        Ok(index) => Arc::new(index),
        Err(err) => {
            tracing::warn!(
                path = %canonical_path.display(),
                "Failed to load neuron index during fleet registration: {err}"
            );
            return Err(err);
        },
    };
    let modules = index
        .list_modules()
        .into_iter()
        .map(|module| module.name)
        .collect();

    let alias = alias.unwrap_or_else(|| {
        canonical_path
            .file_name()
            .map(|name| name.to_string_lossy().to_string())
            .unwrap_or_else(|| canonical_path.display().to_string())
    });
    let node = FleetNode {
        id: FleetNodeId::new(format!(
            "node-{}",
            blake3_short(&canonical_path.to_string_lossy())
        )),
        path: canonical_path.clone(),
        alias,
        modules,
        last_registered: now_iso8601(),
    };

    let mut registry = load_registry()?;
    registry
        .nodes
        .retain(|existing| existing.path != canonical_path);
    registry.nodes.push(node.clone());
    registry.nodes.sort_by(|left, right| {
        left.alias
            .cmp(&right.alias)
            .then_with(|| left.path.cmp(&right.path))
    });
    save_registry(&registry)?;
    cache_fleet_index(&canonical_path, index);

    Ok(node)
}

/// Remove a fleet node by alias or path string.
pub fn deregister_node(alias_or_path: &str) -> Result<bool> {
    let mut registry = load_registry()?;
    let canonical_match = std::fs::canonicalize(alias_or_path).ok();
    let mut removed_paths = Vec::new();

    registry.nodes.retain(|node| {
        let alias_matches = node.alias == alias_or_path;
        let path_matches = node.path.to_string_lossy() == alias_or_path
            || canonical_match
                .as_ref()
                .is_some_and(|path| node.path == *path);
        let should_remove = alias_matches || path_matches;
        if should_remove {
            removed_paths.push(node.path.clone());
        }
        !should_remove
    });

    if removed_paths.is_empty() {
        return Ok(false);
    }

    save_registry(&registry)?;
    for path in removed_paths {
        invalidate_fleet_index(&path);
    }
    Ok(true)
}

fn blake3_short(s: &str) -> String {
    let hex = blake3::hash(s.as_bytes()).to_hex();
    hex.as_str()[..8].to_owned()
}
