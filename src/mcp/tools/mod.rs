use super::CortyxServer;
use rmcp::handler::server::router::tool::ToolRouter;

mod admin;
mod collaboration;
mod context;
mod knowledge;
mod memory;

pub(super) fn tool_router() -> ToolRouter<CortyxServer> {
    ToolRouter::<CortyxServer>::new()
        + CortyxServer::context_tool_router()
        + CortyxServer::memory_tool_router()
        + CortyxServer::knowledge_tool_router()
        + CortyxServer::collaboration_tool_router()
        + CortyxServer::admin_tool_router()
}
