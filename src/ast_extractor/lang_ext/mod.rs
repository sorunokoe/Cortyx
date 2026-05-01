//! Additional language extractors (Level-1 regex).
//!
//! Organizes 14 language-specific AST extractors by category:
//! - Scripting: PHP, Lua, R, Julia, Elixir
//! - Systems: Zig, Dart
//! - Config/Data: Shell, SQL, HCL, Protocol Buffers, GraphQL
//! - Special: Jupyter, Universal fallback

mod config;
mod scripting;
mod special;
mod systems;

// Re-export all extractors for parent module
pub(super) use config::{extract_graphql, extract_hcl, extract_proto, extract_shell, extract_sql};
pub(super) use scripting::{extract_elixir, extract_julia, extract_lua, extract_php, extract_r};
pub(super) use special::{extract_jupyter, extract_universal_fallback};
pub(super) use systems::{extract_dart, extract_zig};
