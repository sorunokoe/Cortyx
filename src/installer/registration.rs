//! MCP server registration logic.

use crate::error::Result;
use crate::installer::client::{ClientConfig, ConfigKind};
use std::io::ErrorKind;
use std::path::{Path, PathBuf};

/// Load a JSON file or return an empty object if it doesn't exist.
pub(super) fn load_json_object_or_default(path: &Path, label: &str) -> Result<serde_json::Value> {
    match std::fs::read_to_string(path) {
        Ok(content) => serde_json::from_str(&content).map_err(|err| {
            crate::cortyx_err!("cannot parse {label} {} as JSON: {err}", path.display())
        }),
        Err(err) if err.kind() == ErrorKind::NotFound => Ok(serde_json::json!({})),
        Err(err) => Err(crate::cortyx_err!(
            "cannot read {label} {}: {err}",
            path.display()
        )),
    }
}

/// Ensure parent directory exists for a path.
pub(super) fn ensure_parent_dir(path: &Path) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    Ok(())
}

/// Write the MCP server entry to a client config.
/// Returns `Ok(true)` if written, `Ok(false)` if already present, `Err` on failure.
pub(super) fn register_mcp_server(client: &ClientConfig, exe: &PathBuf) -> Result<bool> {
    let ConfigKind::McpServersJson = client.kind;

    let mut json =
        load_json_object_or_default(&client.config_path, &format!("{} config", client.name))?;
    let root = json.as_object_mut().ok_or_else(|| {
        crate::cortyx_err!(
            "{} config {} must contain a top-level JSON object",
            client.name,
            client.config_path.display()
        )
    })?;

    // Check if cortyx is already registered
    if let Some(servers) = root.get("mcpServers") {
        if servers.get("cortyx").is_some() {
            return Ok(false);
        }
    }

    // Add the entry
    let entry = serde_json::json!({
        "command": exe.to_string_lossy().as_ref(),
        "args": ["serve"]
    });
    match root.entry("mcpServers".to_string()) {
        serde_json::map::Entry::Occupied(mut existing) => {
            let servers = existing.get_mut().as_object_mut().ok_or_else(|| {
                crate::cortyx_err!(
                    "{} config {} has non-object mcpServers",
                    client.name,
                    client.config_path.display()
                )
            })?;
            servers.insert("cortyx".to_string(), entry);
        },
        serde_json::map::Entry::Vacant(slot) => {
            slot.insert(serde_json::json!({ "cortyx": entry }));
        },
    }

    ensure_parent_dir(&client.config_path)?;
    crate::neuron::atomic_write_json(&client.config_path, &json).map_err(|err| {
        crate::cortyx_err!(
            "cannot write {} config {}: {err}",
            client.name,
            client.config_path.display()
        )
    })?;
    Ok(true)
}
