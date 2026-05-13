//! Optional fleet orchestration for cross-project Cortyx context lookup.

pub mod merge;
pub mod registry;
pub mod router;
pub mod types;

#[cfg(test)]
mod tests;

pub use merge::{rrf_merge, MergedFleetResult};
pub use registry::{
    deregister_node, fleet_registry_path, load_registry, register_node, save_registry,
};
pub use router::{route_fleet_query, FLEET_LOW_CONFIDENCE_THRESHOLD, FLEET_QUERY_TIMEOUT_MS};
pub use types::{FleetNode, FleetNodeId, FleetQueryResult, FleetRegistry, FleetRouteReason};

pub const FLEET_REGISTRY_VERSION: u32 = 1;
