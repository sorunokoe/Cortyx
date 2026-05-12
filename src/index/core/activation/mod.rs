// This file is a submodule of `crate::index::core`.
// It contains `impl NeuronIndex` methods extracted from helpers.rs.
// All visibility is relative to `crate::index` (the parent of `core`).
use super::*;
use crate::types::{QueryText, SynapseWeight};

pub mod overflow;
pub mod search;

pub(super) use self::overflow::*;
pub(super) use self::search::*;
