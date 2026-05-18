use super::super::*;
use rmcp::{handler::server::wrapper::Parameters, tool, tool_router};

#[derive(schemars::JsonSchema, serde::Deserialize)]
pub(super) struct FleetQueryInput {
    pub task: String,
    pub module: Option<String>,
    pub max_tokens: Option<usize>,
}

#[derive(schemars::JsonSchema, serde::Deserialize)]
pub(super) struct FleetStatusInput {}

#[derive(schemars::JsonSchema, serde::Deserialize)]
pub(super) struct FleetRegisterInput {
    /// Local filesystem path to a Cortyx project directory.
    /// Mutually exclusive with `git_url`.
    pub path: Option<String>,
    /// Git remote URL of a shared corpus to clone and register.
    /// Accepted: https://github.com/, https://gitlab.com/, git@github.com:, git@gitlab.com:
    /// Requires `alias`.
    pub git_url: Option<String>,
    /// Human-readable alias for this fleet node.
    /// Required when `git_url` is provided.
    pub alias: Option<String>,
}

#[tool_router(router = fleet_tool_router, vis = "pub(super)")]
impl CortyxServer {
    #[tool(
        name = "cortyx_fleet_query",
        description = "Query the local fleet of registered Cortyx projects for supplementary context. Used automatically when local confidence is low, or explicitly to search other projects in the fleet."
    )]
    pub(in crate::mcp) async fn fleet_query(
        &self,
        Parameters(input): Parameters<FleetQueryInput>,
    ) -> String {
        let registry = match self.fleet_registry.as_ref() {
            Some(registry) => registry,
            None => return "Fleet not configured. Run: cortyx fleet register <path>".to_string(),
        };

        let results = crate::fleet::route_fleet_query(
            &input.task,
            input.module.as_deref(),
            input.max_tokens.unwrap_or(4000),
            registry.as_ref(),
        )
        .await;
        // C7: Dynamic fleet weight — use max top_score as proxy for fleet quality.
        // High-scoring fleet results (top_score ≥ 8.0) get weight → 0.50;
        // low-scoring results (top_score ≈ 0.0) are suppressed to weight → 0.10.
        let fleet_weight = results.iter().map(|r| r.top_score).fold(0.0_f32, f32::max);
        let fleet_weight = crate::fleet::dynamic_fleet_weight(fleet_weight);
        let merged = crate::fleet::rrf_merge("", 0.0, results, 0.7, fleet_weight);
        if merged.trim().is_empty() {
            "No relevant fleet context found.".to_string()
        } else {
            merged
        }
    }

    #[tool(
        name = "cortyx_fleet_status",
        description = "List all registered fleet nodes with their alias, path, module count, and last registration time."
    )]
    pub(in crate::mcp) async fn fleet_status(
        &self,
        Parameters(_input): Parameters<FleetStatusInput>,
    ) -> String {
        let registry = match self.fleet_registry.as_ref() {
            Some(registry) => registry,
            None => return "Fleet not configured.".to_string(),
        };

        if registry.nodes.is_empty() {
            return "No registered fleet nodes.".to_string();
        }

        let mut out = String::new();
        for node in &registry.nodes {
            let kind = if node.git_url.is_some() {
                "git"
            } else {
                "local"
            };
            out.push_str(&format!(
                "- **[{}] {}** — `{}` (module count: {}, last_registered: {})\n",
                kind,
                node.alias,
                node.path.display(),
                node.modules.len(),
                node.last_registered
            ));
        }
        out
    }

    #[tool(
        name = "cortyx_fleet_register",
        description = "Register a local project directory or a git-backed shared corpus as a fleet node. \
            For a local path, provide `path` (and optionally `alias`). \
            For a shared git corpus, provide `git_url` and `alias`. \
            Accepted git URL prefixes: https://github.com/, https://gitlab.com/, git@github.com:, git@gitlab.com:"
    )]
    pub(in crate::mcp) async fn fleet_register(
        &self,
        Parameters(input): Parameters<FleetRegisterInput>,
    ) -> String {
        match (input.git_url, input.path) {
            (Some(url), _) => {
                let alias = match input.alias {
                    Some(a) => a,
                    None => {
                        return "alias is required when registering a git-backed fleet node"
                            .to_string()
                    },
                };
                match crate::fleet::register_git_node(&url, &alias) {
                    Ok(node) => format!(
                        "✓ Registered git fleet node '{}'\n  URL: {}\n  Path: {}\n  Modules: {}",
                        node.alias,
                        url,
                        node.path.display(),
                        if node.modules.is_empty() {
                            "(none)".to_string()
                        } else {
                            node.modules.join(", ")
                        }
                    ),
                    Err(e) => format!("Failed to register git fleet node: {e}"),
                }
            },
            (None, Some(path)) => {
                let project_path = std::path::PathBuf::from(path);
                match crate::fleet::register_node(&project_path, input.alias) {
                    Ok(node) => format!(
                        "✓ Registered fleet node '{}'\n  Path: {}\n  Modules: {}",
                        node.alias,
                        node.path.display(),
                        if node.modules.is_empty() {
                            "(none)".to_string()
                        } else {
                            node.modules.join(", ")
                        }
                    ),
                    Err(e) => format!("Failed to register fleet node: {e}"),
                }
            },
            (None, None) => {
                "Provide either 'path' for a local node or 'git_url' for a git-backed node."
                    .to_string()
            },
        }
    }
}
