use super::super::*;
use rmcp::{handler::server::wrapper::Parameters, tool, tool_router};

#[tool_router(router = collaboration_tool_router, vis = "pub(super)")]
impl CortyxServer {
    #[tool(
        name = "cortyx_agent_status",
        description = "Show the latest structured collaboration snapshot for an agent by combining recent @agent diary entries, mirrored temporal KG facts, and shared-sync status. Useful for specialist-agent handoff, wake-up, and coordination."
    )]
    pub(in crate::mcp) async fn agent_status(
        &self,
        Parameters(input): Parameters<AgentStatusInput>,
    ) -> String {
        if input.agent.trim().is_empty() {
            return "ERROR: agent name must not be empty".to_string();
        }
        let idx = self.index.read().await;
        render_agent_status_report(
            &idx,
            &self.project_root,
            input.agent.trim(),
            input.include_timeline.unwrap_or(false),
        )
        .unwrap_or_else(|| {
            format!(
                "No structured agent memory found for agent '{}'.",
                input.agent
            )
        })
    }

    #[tool(
        name = "cortyx_collaboration_status",
        description = "Summarize collaboration-kernel state across agents, shared modules, and sync activity. Optionally scope to one agent or one module, and append recent collaboration timeline events."
    )]
    pub(in crate::mcp) async fn collaboration_status(
        &self,
        Parameters(input): Parameters<CollaborationStatusInput>,
    ) -> String {
        if input
            .agent
            .as_deref()
            .is_some_and(|agent| agent.trim().is_empty())
        {
            return "ERROR: agent name must not be empty".to_string();
        }
        if input
            .module
            .as_deref()
            .is_some_and(|module| module.trim().is_empty())
        {
            return "ERROR: module name must not be empty".to_string();
        }
        if input.agent.is_some() && input.module.is_some() {
            return "ERROR: agent and module filters are mutually exclusive".to_string();
        }

        let idx = self.index.read().await;
        let projection = build_collaboration_projection(&idx, &self.project_root);
        render_collaboration_status_report(
            &projection,
            input.agent.as_deref(),
            input.module.as_deref(),
            input.include_timeline.unwrap_or(false),
        )
        .unwrap_or_else(|| {
            if let Some(agent) = input.agent.as_deref() {
                format!("No collaboration state found for agent '{}'.", agent.trim())
            } else if let Some(module) = input.module.as_deref() {
                format!(
                    "No collaboration state found for module '{}'.",
                    module.trim()
                )
            } else {
                "No collaboration state found.".to_string()
            }
        })
    }
}
