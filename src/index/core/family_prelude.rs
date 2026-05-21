//! Explicit re-export mediator for `index/core` submodules (TRIZ P24).
//!
//! Instead of `use super::*` which imports the entire god module, production code
//! in `src/index/core/*.rs` should use `use super::family_prelude::*` to import
//! only the explicitly listed subset needed by most direct submodules.
//!
//! This makes the dependency surface visible and reduces cascade recompilation
//! when the god module changes.

#[cfg(feature = "embed")]
pub(crate) use crate::embedder::load_embeddings;
pub(crate) use crate::error::Result;
pub(in crate::index) use crate::index::core::activation_cache_path;
pub(crate) use crate::index::core::bm25::BM25Entry;
pub(in crate::index::core) use crate::index::core::config::INDEX_VERSION;
pub(crate) use crate::index::core::domain::{
    FeedbackState, PersistenceState, RetrievalState, WatcherState,
};
pub(in crate::index) use crate::index::core::helpers::{
    build_module_capsule_content, index_path, load_coactivation_counts,
    read_index_cache_generation, safe_module_name, save_coactivation_counts, sidecar_module_for,
};
pub(in crate::index) use crate::index::core::types::CompiledFile;
pub(crate) use crate::index::core::{
    ModuleSummary, NeuronIndex, NeuronSummary, PublishReadySummary,
};
pub(crate) use crate::neuron::{
    atomic_write, atomic_write_json, core_neuron_path, meta_path, neuron_dir, NeuronKind,
    NeuronMeta, NeuronStatus, Synapse,
};
pub(crate) use crate::types::TermFrequency;
pub(crate) use serde::{Deserialize, Serialize};
pub(crate) use std::collections::{HashMap, HashSet};
pub(crate) use std::path::{Path, PathBuf};
pub(crate) use std::sync::atomic::{AtomicUsize, Ordering};
pub(crate) use walkdir::WalkDir;
