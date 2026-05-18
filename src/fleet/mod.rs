//! Optional fleet orchestration for cross-project Cortyx context lookup.

pub mod merge;
pub mod registry;
pub mod router;
pub mod sync;
pub mod types;

#[cfg(test)]
mod tests;

pub use merge::{dynamic_fleet_weight, rrf_merge, MergedFleetResult};
pub use registry::{
    deregister_node, fleet_registry_path, load_registry, register_git_node, register_node,
    save_registry, sync_git_nodes,
};
pub use router::{route_fleet_query, FLEET_LOW_CONFIDENCE_THRESHOLD, FLEET_QUERY_TIMEOUT_MS};
pub use sync::{is_allowed_git_url, sync_fleet_node};
pub use types::{FleetNode, FleetNodeId, FleetQueryResult, FleetRegistry, FleetRouteReason};

pub const FLEET_REGISTRY_VERSION: u32 = 1;
