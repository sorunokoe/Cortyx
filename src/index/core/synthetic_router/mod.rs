// This file is a submodule of `crate::index::core`.
// Contains `impl NeuronIndex` synthetic answer methods extracted from synthetic.rs.
use super::*;

pub mod matchers;

pub(super) use self::matchers::*;

include!("dispatch.rs");
