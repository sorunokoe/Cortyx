use std::collections::HashSet;

pub mod aliases;
pub mod categories_basic;
pub mod categories_benchmark;
pub mod categories_lifestyle;
pub mod categories_personal;
pub mod surface;

pub(super) type QuerySurfacePattern = (&'static [&'static str], &'static [&'static str]);

use self::categories_basic::*;
use self::categories_benchmark::*;
use self::categories_lifestyle::*;
use self::categories_personal::*;

pub(crate) use self::aliases::fact_alias_lines;
pub(crate) use self::surface::generate_query_surface;
