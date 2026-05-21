//! Core neuron types, I/O, and knowledge-graph primitives.

pub mod filter;
pub mod io;
pub mod kind;
pub mod meta;
pub mod provenance;
pub mod section;
pub mod synapse;
pub mod synapse_parser;
pub mod sync;
pub mod templates;
pub mod util;

// ─── Public re-exports ────────────────────────────────────────────────────────

pub use filter::{should_skip, validate_relative_path, validate_synapse_path};
pub use io::{
    atomic_write, atomic_write_json, core_neuron_path, meta_path, neuron_dir, sidecar_path,
    strip_private_blocks, sub_neuron_path,
};
pub use kind::{NeuronKind, NeuronStatus};
pub use meta::{latest_shadow, pop_shadow, push_shadow, NeuronMeta, DEFAULT_CONFIDENCE};
pub use section::{parse_sections, replace_section, update_neuron_header};
pub use synapse::{Synapse, SynapseConfidenceTier, SynapseType};
pub use synapse_parser::parse_synapses_from_content;
pub use templates::{stub_core_neuron, stub_function_neuron, stub_project_neuron};
pub use util::{
    days_to_ymd, estimate_context_tokens, estimate_tokens, generate_neuron_uuid, hash_file,
    now_iso8601, unix_secs_to_datetime,
};
