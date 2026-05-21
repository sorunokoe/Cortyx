//! Cortyx public library API.

#![cfg_attr(not(test), deny(clippy::unwrap_used, clippy::expect_used))]

#[cfg(all(feature = "embed", target_os = "macos"))]
extern crate blas_src;

// Exposes `index` and `miner` for in-process use by integration tests and benchmarks.
// This eliminates the need for subprocess spawning (500 × binary-startup overhead)
// and allows tests to load the NeuronIndex once and query it N times in-process.
// TRIZ P10 (Preliminary Action) + P20 (Continuity of Useful Action).

/// Error types and result helpers.
#[allow(missing_docs)]
pub mod error;
/// Shared data types used across Cortyx.
#[allow(missing_docs)]
pub mod types;
/// Verification gates for task routing and command safety.
pub mod verify_gate;

/// Agent memory storage and retrieval helpers.
#[allow(missing_docs)]
pub mod agent_memory;
/// Alias generation utilities.
pub mod alias_gen;
/// Answer planning and response assembly.
#[allow(missing_docs)]
pub mod answer_plane;
/// AST extraction helpers for source analysis.
#[allow(missing_docs)]
pub mod ast_extractor;
/// Command-line interface types and parsing.
#[allow(missing_docs)]
pub mod cli;
/// Collaboration kernel orchestration logic.
#[allow(missing_docs)]
pub mod collaboration_kernel;
/// Top-level command implementations.
#[allow(missing_docs)]
pub mod commands;
/// Embedding generation and model integration.
#[allow(missing_docs)]
pub mod embedder;
/// Export pipeline helpers.
pub mod export;
/// Fleet coordination utilities.
#[allow(missing_docs)]
pub mod fleet;
/// Git-backed extraction helpers.
pub mod git_extractor;
/// Shared Git utility functions.
pub mod git_util;
/// Global index management.
#[allow(missing_docs)]
pub mod global_index;
/// Import parsing helpers.
pub mod import_parser;
/// Core indexing and retrieval engine.
#[allow(missing_docs)]
pub mod index;
/// Installation and setup helpers.
pub mod installer;
/// Knowledge graph construction and queries.
#[allow(missing_docs)]
pub mod kg;
/// MCP server implementation and handlers.
#[allow(missing_docs)]
pub mod mcp;
/// Mining and ingestion workflows.
#[allow(missing_docs)]
pub mod miner;
/// Neuron data structures and I/O.
#[allow(missing_docs)]
pub mod neuron;
/// Reasoning and inference helpers.
#[allow(missing_docs)]
pub mod reasoner;
/// Retrieval reranking components.
#[allow(missing_docs)]
pub mod reranker;
/// Sync transport protocol types.
#[allow(missing_docs)]
pub mod sync_transport;
/// Filesystem watch and refresh helpers.
#[allow(missing_docs)]
pub mod watcher;
