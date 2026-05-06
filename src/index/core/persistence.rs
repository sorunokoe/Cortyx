//! Persistence types for index serialization.

use crate::index::core::bm25::BM25Entry;
use crate::neuron::Synapse;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::path::PathBuf;

/// Borrowed view used for serialization — avoids cloning the entire entry vector
/// on every save() call (which would otherwise be O(n) allocation per MCP mutation).
#[derive(Serialize)]
pub(crate) struct PersistedIndexRef<'a> {
    pub version: u32,
    pub cache_generation: u64,
    pub entries: &'a [BM25Entry],
    #[serde(skip_serializing_if = "<[[usize; 2]]>::is_empty")]
    pub session_utilization: &'a [[usize; 2]],
    #[serde(skip_serializing_if = "<[String]>::is_empty")]
    pub shards: &'a [String],
}

/// Binary activation cache persisted alongside index.json.
///
/// TRIZ P10 (Preliminary Action): precompute and persist the query-hot derived
/// structures at save time, so CLI startup does not have to rebuild them on
/// every `status` / `get-contexts` invocation.
#[derive(Serialize, Deserialize)]
pub(crate) struct PersistedActivationCache {
    pub version: u32,
    pub index_generation: u64,
    pub entries: Vec<BM25Entry>,
    pub concept_clouds: Vec<Vec<String>>,
    pub summaries: Vec<String>,
    pub adjacency: HashMap<PathBuf, Vec<Synapse>>,
    pub path_index: HashMap<PathBuf, usize>,
    pub parent_index: HashMap<PathBuf, Vec<usize>>,
    pub df_cache: HashMap<String, usize>,
    pub posting_list: HashMap<String, Vec<usize>>,
    pub avg_doc_len: f32,
    pub avg_verbatim_doc_len: f32,
    pub module_index: HashMap<String, Vec<usize>>,
    pub vocab_bridge: HashMap<String, HashSet<String>>,
    pub morpheme_map: HashMap<String, Vec<String>>,
    pub session_utilization: Vec<[usize; 2]>,
    pub session_index: HashMap<String, Vec<usize>>,
    pub pmi_neighbors: HashMap<String, Vec<String>>,
    pub idf_n: usize,
    /// S4-WAL: entries.len() at last full index.json write. 0 = no full save yet.
    pub wal_base: usize,
}

#[derive(Serialize)]
pub(crate) struct PersistedActivationCacheRef<'a> {
    pub version: u32,
    pub index_generation: u64,
    pub entries: &'a [BM25Entry],
    pub concept_clouds: Vec<&'a [String]>,
    pub summaries: Vec<&'a str>,
    pub adjacency: &'a HashMap<PathBuf, Vec<Synapse>>,
    pub path_index: &'a HashMap<PathBuf, usize>,
    pub parent_index: &'a HashMap<PathBuf, Vec<usize>>,
    pub df_cache: &'a HashMap<String, usize>,
    pub posting_list: &'a HashMap<String, Vec<usize>>,
    pub avg_doc_len: f32,
    pub avg_verbatim_doc_len: f32,
    pub module_index: &'a HashMap<String, Vec<usize>>,
    pub vocab_bridge: &'a HashMap<String, HashSet<String>>,
    pub morpheme_map: &'a HashMap<String, Vec<String>>,
    pub session_utilization: &'a Vec<[usize; 2]>,
    pub session_index: &'a HashMap<String, Vec<usize>>,
    pub pmi_neighbors: &'a HashMap<String, Vec<String>>,
    pub idf_n: usize,
    /// S4-WAL: entries.len() at last full index.json write. 0 = no full save yet.
    pub wal_base: usize,
}
