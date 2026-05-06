//! Client configuration detection and metadata.

use std::path::PathBuf;

/// A detected LLM client config location.
#[derive(Debug)]
pub(super) struct ClientConfig {
    pub name: &'static str,
    pub config_path: PathBuf,
    pub kind: ConfigKind,
}

#[derive(Debug)]
pub(super) enum ConfigKind {
    /// JSON object with "mcpServers" key (Claude Code, Cursor, Windsurf, Codex)
    McpServersJson,
}
