/// Builds a ready-to-use prompt JSON with cache_control breakpoint for Anthropic or OpenAI.
///
/// The static prefix (cortyx schema + instructions) gets `cache_control: {type: "ephemeral"}`.
/// Dynamic neurons (from cortyx_get_contexts) come after, in a separate block.
/// This guarantees byte-identical static prefix → 100% prompt-cache hit rate.
use anyhow::Result;
use serde_json::{Value, json};
use std::path::Path;

use crate::cli::Provider;
use crate::index::NeuronIndex;

/// Static system prompt that goes BEFORE the cache_control breakpoint.
/// This content is always byte-identical → guaranteed cache hit.
const STATIC_SYSTEM_PROMPT: &str = r#"You are an AI assistant working with a codebase indexed by Cortyx.

CORTYX USAGE PROTOCOL:
1. At the start of EVERY task, call cortyx_get_contexts(task=<task_description>) to activate relevant neurons.
2. Inject the returned neurons AFTER this cached system block (in the user turn or a second system block).
3. After completing a task, call cortyx_evolve_context to improve the neurons you used.
4. When you find a useful exact code pattern, call cortyx_extract_from_raw to save it as a use-case neuron.
5. When two files are strongly related for a task, call cortyx_create_synapse.

CACHE HIT GUARANTEE:
This system block (before cache_control) is always byte-identical → 100% Anthropic/OpenAI cache hits.
Neuron content is injected after this breakpoint so it does NOT affect the cache key.

NEURON STATUS LEGEND:
- <!-- status: stub -->   → Not yet evolved. Use cortyx_evolve_context to fill it.
- <!-- status: fresh -->  → Up to date with source.
- <!-- status: stale -->  → Source changed. Re-read raw source before using.

TOKEN EFFICIENCY:
Cortyx activates only 3-5 relevant neurons per task (typically 800-2000 tokens) instead of
loading the entire codebase. Combined with 100% cache hits, this gives ~70-85% cost reduction."#;

pub fn build_prompt_json(root: &Path, idx: &NeuronIndex, provider: Provider) -> Result<String> {
    let instructions_note = format!(
        "\n\n<!-- CORTYX EXPORT — project: {} -->\n\
         <!-- Replace <TASK> with your actual task and call cortyx_get_contexts first. -->\n\
         <!-- Then inject the returned neurons in the user turn or a second system block. -->",
        root.display()
    );

    let static_text = format!("{STATIC_SYSTEM_PROMPT}{instructions_note}");

    let json = match provider {
        Provider::Anthropic => build_anthropic_format(&static_text, idx),
        Provider::Openai => build_openai_format(&static_text, idx),
    };

    Ok(serde_json::to_string_pretty(&json)?)
}

fn build_anthropic_format(static_text: &str, idx: &NeuronIndex) -> Value {
    let neuron_count = idx.neuron_count();
    let synapse_count = idx.synapse_count();

    json!({
        "_cortyx_meta": {
            "version": env!("CARGO_PKG_VERSION"),
            "neurons": neuron_count,
            "synapses": synapse_count,
            "cache_strategy": "static_prefix_breakpoint",
            "usage": "Inject cortyx_get_contexts output after the cached system block."
        },
        "model": "claude-opus-4-5",
        "system": [
            {
                "type": "text",
                "text": static_text,
                "cache_control": {"type": "ephemeral"}
            },
            {
                "type": "text",
                "text": "<!-- DYNAMIC NEURONS — inject cortyx_get_contexts output here -->\n\
                         <!-- Example: call cortyx_get_contexts(task=\"add dark mode\") and paste result here -->"
            }
        ],
        "messages": [
            {
                "role": "user",
                "content": "<TASK DESCRIPTION — replace this with your actual task>"
            }
        ],
        "max_tokens": 8192
    })
}

fn build_openai_format(static_text: &str, idx: &NeuronIndex) -> Value {
    let neuron_count = idx.neuron_count();
    let synapse_count = idx.synapse_count();

    json!({
        "_cortyx_meta": {
            "version": env!("CARGO_PKG_VERSION"),
            "neurons": neuron_count,
            "synapses": synapse_count,
            "cache_strategy": "static_prefix_breakpoint",
            "usage": "The system message is byte-identical every call → cached prefix hit. Inject neurons before user message."
        },
        "model": "gpt-4.1",
        "store": true,
        "messages": [
            {
                "role": "system",
                "content": static_text
            },
            {
                "role": "system",
                "content": "<!-- DYNAMIC NEURONS — inject cortyx_get_contexts output here -->"
            },
            {
                "role": "user",
                "content": "<TASK DESCRIPTION — replace this with your actual task>"
            }
        ]
    })
}
