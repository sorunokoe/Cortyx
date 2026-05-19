#[cfg(feature = "embed")]
use crate::embedder::{load_embeddings, EmbeddingStore};

use crate::error::Result;
use crate::types::TermFrequency;
use rayon::prelude::*;
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use walkdir::WalkDir;

use crate::alias_gen;
use crate::ast_extractor;
use crate::git_extractor;
use crate::global_index;
use crate::import_parser;
use crate::kg;
use crate::neuron::{
    atomic_write, atomic_write_json, core_neuron_path, estimate_context_tokens, estimate_tokens,
    meta_path, neuron_dir, now_iso8601, replace_section, should_skip, stub_core_neuron,
    stub_function_neuron, stub_project_neuron, sub_neuron_path, update_neuron_header, NeuronKind,
    NeuronMeta, NeuronStatus, Synapse, SynapseType,
};
use crate::reasoner::{
    GraphReasoner, ReasonerNeuron, ReasonerSeed, ReasoningReport, TraversalOptions,
};

mod answer_surface;
use answer_surface::*;

mod activation;
mod bm25;
pub(super) use bm25::BM25Entry;

mod compile;
mod config;
use config::*;
pub use config::{
    HIGH_ACTIVATION_THRESHOLD, MAX_CORE_NEURONS, MAX_USE_CASE_PER_CORE, SYNAPSE_RELEVANCE_THRESHOLD,
};

mod helpers;
#[cfg(test)]
pub use self::helpers::simple_overlap_score;
pub(super) use self::helpers::*;
pub use self::helpers::{
    dirty_path, infer_module, is_capsule_module, module_capsule_path, tokenize,
};

mod helpers_detect;
pub(super) use self::helpers_detect::*;

mod helpers_extract;
pub(super) use self::helpers_extract::*;

mod helpers_phrase;
pub(super) use self::helpers_phrase::*;

mod helpers_surface;
pub(super) use self::helpers_surface::*;

mod helpers_temporal2;
pub(super) use self::helpers_temporal2::*;

mod helpers_title;
pub(super) use self::helpers_title::*;

mod hierarchy;
mod impl_helpers;
mod impl_helpers2;
mod invalidation;
mod lsh;
use lsh::{hamming_distance, simhash_256, simhash_with_seed, LSH_SEEDS};

mod persistence;
use persistence::*;

mod query;
pub(super) use query::{
    adaptive_quarantine_params, build_git_confidence_map, content_has_move_residence_evidence,
    count_proper_nouns, detect_knowledge_update_query, detect_personal_fact_entity,
    detect_personal_fact_query, extract_knowledge_update_focus_terms, extract_numbered_list_item,
    extract_pet_name, extract_query_ordinal, extract_single_word_after_marker, git_file_list,
    is_book_query, is_commute_query, is_education_query, is_fitness_record_query,
    is_list_style_query, is_location_query, is_major_query, is_named_move_query,
    is_occupation_query, is_partner_query, is_pet_query, is_phone_query, is_project_name_query,
    num_to_word, parse_iso8601_to_secs, synthetic_query_terms, task_contains_all,
    task_contains_any, term_overlap_count, wilson_lower_bound_z,
};
#[cfg(test)]
use query::{neuron_body_has_move_residence_evidence, wilson_lower_bound};

mod stats;
mod synthetic_count;
mod synthetic_kg;
mod synthetic_router;
mod synthetic_session;
mod synthetic_temporal;
mod synthetic_temporal2;
mod synthetic_temporal3;

mod pipeline;
pub(super) use pipeline::*;

mod types;
pub use types::*;

#[cfg(test)]
mod tests;
#[cfg(test)]
mod tests_b;
#[cfg(test)]
mod tests_c;
#[cfg(test)]
mod tests_d;
