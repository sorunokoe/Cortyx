//! S4 (NE3): Temporal Knowledge Graph stored as Markdown pipe-tables.
//!
//! KG entities live as Concept neurons at `.cortyx/neurons/_kg_{entity}.context.md`.
//! Each file contains a `## facts` section with a pipe-delimited table:
//!
//! ```markdown
//! | predicate | value | valid_from | ended |
//! |---|---|---|---|
//! | language | Rust | 2024-01-01 | |
//! | lead | Alice | 2024-01-01 | 2024-06-01 |
//! ```
//!
//! Benefits over a SQLite KG:
//! - Git-trackable (diff, history, blame)
//! - Human-readable and hand-editable
//! - BM25-indexed by Cortyx — KG facts are searchable alongside code context
//! - Zero new dependencies

pub mod entity;
pub mod fact;
mod parse;
pub mod render;
pub mod stats;

pub use entity::KgEntity;
pub use fact::KgFact;
pub use render::{kg_neuron_path, list_kg_paths, slugify};
pub use stats::{compute_stats, KgStats};
