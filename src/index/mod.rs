// These imports are intentionally kept here so child (family) modules can
// access them via `use super::*;` without individual re-imports.
#![allow(unused_imports)]

#[cfg(feature = "embed")]
use crate::embedder::{load_embeddings, EmbeddingStore};

use crate::error::Result;
use rayon::prelude::*;
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
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
    NeuronMeta, NeuronStatus, Synapse, SynapseType, DEFAULT_CONFIDENCE,
};
use crate::reasoner::{
    GraphReasoner, ReasonerNeuron, ReasonerSeed, ReasoningReport, TraversalOptions,
};

/// Compile a regex from a hardcoded, known-good pattern.
///
/// # Panics
///
/// Panics if `pattern` fails to compile. This is intentional — a hardcoded regex
/// that fails to compile is a programming error, not a runtime condition.
pub(crate) fn compile_regex_static(pattern: &str) -> Regex {
    compile_regex(pattern)
        .unwrap_or_else(|err| panic!("hardcoded regex failed to compile: {pattern:?}: {err}"))
}

/// Compile a regex from a runtime or user-supplied pattern.
///
/// Returns an error if the pattern is invalid. Callers must handle the error
/// explicitly — no silent fallback.
///
/// # Errors
///
/// Returns [`crate::error::CortyxError::Regex`] if the pattern fails to compile.
pub(crate) fn compile_regex(pattern: &str) -> crate::error::Result<Regex> {
    Regex::new(pattern).map_err(crate::error::CortyxError::Regex)
}

mod age_event_extractors;
mod age_event_families;
#[cfg(test)]
mod age_event_family_tests;
mod anchored_time_extractors;
mod anchored_time_families;
#[cfg(test)]
mod anchored_time_family_tests;
mod assistant_fact_extractors;
mod assistant_fact_families;
#[cfg(test)]
mod assistant_fact_family_tests;
mod assistant_fact_query_support;
mod assistant_fact_support;
mod assistant_recall_extractors;
mod assistant_recall_families;
#[cfg(test)]
mod assistant_recall_family_tests;
mod assistant_resource_extractors;
mod assistant_resource_families;
#[cfg(test)]
mod assistant_resource_family_tests;
mod assistant_structured_extractors;
mod assistant_structured_families;
#[cfg(test)]
mod assistant_structured_family_tests;
mod average_value_extractors;
mod average_value_families;
#[cfg(test)]
mod average_value_family_tests;
mod community_relation_extractors;
mod community_relation_families;
#[cfg(test)]
mod community_relation_family_tests;
mod comparison_delta_extractors;
mod comparison_delta_families;
#[cfg(test)]
mod comparison_delta_family_tests;
mod conversation_scan_support;
mod count_extractors;
mod count_families;
mod count_support;
mod count_total_extractors;
mod count_total_families;
#[cfg(test)]
mod count_total_family_tests;
mod event_extractors;
mod event_families;
#[cfg(test)]
mod event_family_tests;
mod gathering_count_extractors;
mod gathering_count_families;
#[cfg(test)]
mod gathering_count_family_tests;
#[cfg(feature = "personal-families")]
mod instagram_delta_extractors;
#[cfg(feature = "personal-families")]
mod instagram_delta_families;
#[cfg(all(test, feature = "personal-families"))]
mod instagram_delta_family_tests;
mod money_combination_extractors;
mod money_combination_families;
#[cfg(test)]
mod money_combination_family_tests;
mod money_extractors;
mod money_families;
#[cfg(test)]
mod money_family_tests;
mod money_queries;
mod money_support;
mod money_total_families;
mod numeric_delta_extractors;
mod numeric_delta_families;
#[cfg(test)]
mod numeric_delta_family_tests;
#[cfg(feature = "personal-families")]
mod paper_submission_extractors;
#[cfg(feature = "personal-families")]
mod paper_submission_families;
#[cfg(all(test, feature = "personal-families"))]
mod paper_submission_family_tests;
#[cfg(feature = "personal-families")]
mod podcast_count_extractors;
#[cfg(feature = "personal-families")]
mod podcast_count_families;
#[cfg(all(test, feature = "personal-families"))]
mod podcast_count_family_tests;
mod preference_profile_advice_families;
#[cfg(test)]
mod preference_profile_advice_family_tests;
mod preference_profile_context_families;
#[cfg(test)]
mod preference_profile_dynamic_family_tests;
mod preference_profile_extractors;
mod preference_profile_families;
#[cfg(test)]
mod preference_profile_family_tests;
mod quantity_total_extractors;
mod quantity_total_families;
#[cfg(test)]
mod quantity_total_family_tests;
mod quantity_total_support;
mod ratio_weight_extractors;
mod ratio_weight_families;
#[cfg(test)]
mod ratio_weight_family_tests;
mod reading_progress_extractors;
mod reading_progress_families;
#[cfg(test)]
mod reading_progress_family_tests;
mod scalar_total_extractors;
mod scalar_total_families;
#[cfg(test)]
mod scalar_total_family_tests;
mod social_metric_extractors;
mod social_metric_families;
#[cfg(test)]
mod social_metric_family_tests;
mod temporal_anchor_extractors;
mod temporal_anchor_families;
#[cfg(test)]
mod temporal_anchor_family_tests;
#[cfg(test)]
mod temporal_elapsed_gap_fixture_tests;
mod temporal_relative_recall_families;
mod time_delta_extractors;
mod time_delta_families;
#[cfg(test)]
mod time_delta_family_tests;
#[cfg(feature = "personal-families")]
mod travel_packing_families;
#[cfg(all(test, feature = "personal-families"))]
mod travel_packing_family_tests;

pub mod core;
#[cfg(not(feature = "personal-families"))]
mod family_stubs;

// Re-export everything from core so family modules continue to use `super::*`
// and `super::some_fn()` without modification.
#[cfg(test)]
pub use core::simple_overlap_score;
pub use core::NeuronIndex;
pub use core::{dirty_path, infer_module, tokenize};
pub use core::{is_capsule_module, module_capsule_path};
pub use core::{ContextMetadata, PublishReadySummary};
pub use core::{
    HIGH_ACTIVATION_THRESHOLD, MAX_CORE_NEURONS, MAX_USE_CASE_PER_CORE, SYNAPSE_RELEVANCE_THRESHOLD,
};
// Bring all pub(super) free functions into crate::index so family modules find
// them via `super::function_name()`.
use core::*;
