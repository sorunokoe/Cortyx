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
        let merged = crate::fleet::rrf_merge("", 0.0, results, 0.7, 0.3);
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
            out.push_str(&format!(
                "- **{}** — `{}` (module count: {}, last_registered: {})\n",
                node.alias,
                node.path.display(),
                node.modules.len(),
                node.last_registered
            ));
        }
        out
    }
}
