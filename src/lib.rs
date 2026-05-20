#![allow(missing_docs)]
#![cfg_attr(not(test), deny(clippy::unwrap_used, clippy::expect_used))]

// Cortyx public library API.
//

#[cfg(all(feature = "embed", target_os = "macos"))]
extern crate blas_src;

// Exposes `index` and `miner` for in-process use by integration tests and benchmarks.
// This eliminates the need for subprocess spawning (500 × binary-startup overhead)
// and allows tests to load the NeuronIndex once and query it N times in-process.
// TRIZ P10 (Preliminary Action) + P20 (Continuity of Useful Action).

pub mod error;
pub mod types;
pub mod verify_gate;

pub mod agent_memory;
pub mod alias_gen;
pub mod answer_plane;
pub mod ast_extractor;
pub mod cli;
pub mod collaboration_kernel;
pub mod commands;
pub mod embedder;
pub mod export;
pub mod fleet;
pub mod git_extractor;
pub mod global_index;
pub mod import_parser;
pub mod index;
pub mod installer;
pub mod kg;
pub mod mcp;
pub mod miner;
pub mod neuron;
pub mod reasoner;
pub mod reranker;
pub mod sync_transport;
pub mod watcher;
