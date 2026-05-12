// This file is a submodule of `crate::index::core`.
// Contains `impl NeuronIndex` synthetic answer methods extracted from synthetic.rs.
use super::*;

pub mod counts;
pub mod knowledge_update;
pub mod lifestyle;
pub mod transport_schedule;

pub(super) use self::counts::*;
pub(super) use self::knowledge_update::*;
pub(super) use self::lifestyle::*;
pub(super) use self::transport_schedule::*;
