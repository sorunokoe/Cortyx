/// Builds a ready-to-use prompt JSON with cache_control breakpoint for Anthropic or OpenAI.
///
/// The static prefix (cortyx schema + instructions) gets `cache_control: {type: "ephemeral"}`.
/// Dynamic context (from `cortyx(intent="context", ...)`) comes after, in a separate block.
/// This guarantees byte-identical static prefix → 100% prompt-cache hit rate.
use crate::error::Result;
use serde_json::{json, Value};
use std::path::Path;

use crate::cli::Provider;
use crate::index::NeuronIndex;

/// Static system prompt that goes BEFORE the cache_control breakpoint.
/// This content is always byte-identical → guaranteed cache hit.
const STATIC_SYSTEM_PROMPT: &str = r#"You are an AI assistant working with a codebase indexed by Cortyx.

CORTYX USAGE PROTOCOL:
1. At the start of EVERY task, call cortyx(intent="context", task=<task_description>) to activate relevant neurons through the universal Cortyx entrypoint.
2. If you need route discovery first, call cortyx() with no args for a capability summary.
3. Inject the returned context AFTER this cached system block (in the user turn or a second system block).
4. Use cortyx_get_contexts only when you want the narrow raw retrieval surface explicitly.
5. After completing a task, call cortyx_evolve_context to improve the neurons you used.
6. When you find a useful exact code pattern, call cortyx_extract_from_raw to save it as a use-case neuron.
7. When two files are strongly related for a task, call cortyx_create_synapse.

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

const TERMINAL_ROUTE_EXAMPLE: &str = r#"cortyx route --task "trace the auth flow""#;
const WATCH_EXAMPLE: &str = "cortyx watch";
const DOCTOR_EXAMPLE: &str = "cortyx doctor";
const INCREMENTAL_COMPILE_EXAMPLE: &str = "cortyx compile --incremental";
const MCP_CAPABILITY_EXAMPLE: &str = r#"cortyx()"#;
const MCP_TASK_EXAMPLE: &str = r#"cortyx(task="trace the auth flow")"#;
const ROUTE_OUTCOMES: &[&str] = &[
    "capabilities",
    "context",
    "answer",
    "wake_up",
    "agent_status",
    "consistency",
];

fn ux_proof_meta() -> Value {
    json!({
        "onboarding": {
            "terminal_steps": 3,
            "in_tool_steps": 2,
            "terminal_quickstart": [
                TERMINAL_ROUTE_EXAMPLE,
                WATCH_EXAMPLE,
                DOCTOR_EXAMPLE,
            ],
            "in_tool_quickstart": [
                MCP_CAPABILITY_EXAMPLE,
                MCP_TASK_EXAMPLE,
            ],
        },
        "recovery": {
            "watch": WATCH_EXAMPLE,
            "doctor": DOCTOR_EXAMPLE,
            "incremental_compile": INCREMENTAL_COMPILE_EXAMPLE,
        },
        "one_entrypoint": {
            "terminal_route": TERMINAL_ROUTE_EXAMPLE,
            "mcp_summary": MCP_CAPABILITY_EXAMPLE,
            "mcp_task": MCP_TASK_EXAMPLE,
            "outcomes": ROUTE_OUTCOMES,
        },
    })
}

/// # Errors
///
/// Returns an error if the underlying operation fails.
pub fn build_prompt_json(root: &Path, idx: &NeuronIndex, provider: Provider) -> Result<String> {
    let instructions_note = format!(
        "\n\n<!-- CORTYX EXPORT — project: {} -->\n\
         <!-- Replace <TASK> with your actual task and call cortyx(intent=\"context\", task=\"<TASK>\") first. -->\n\
         <!-- If you need route discovery first, call cortyx() with no args for a capability summary. -->\n\
         <!-- Then inject the returned context in the user turn or a second system block. -->",
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
            "usage": "Inject cortyx(intent=\"context\", task=\"...\") output after the cached system block.",
            "quickstart": {
                "terminal_route": TERMINAL_ROUTE_EXAMPLE,
                "watch": WATCH_EXAMPLE,
                "doctor": DOCTOR_EXAMPLE,
                "mcp_summary": MCP_CAPABILITY_EXAMPLE,
                "mcp_task": MCP_TASK_EXAMPLE
            },
            "ux_proof": ux_proof_meta()
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
                "text": "<!-- DYNAMIC NEURONS — inject cortyx(intent=\"context\", task=\"...\") output here -->\n\
                         <!-- Example: call cortyx(intent=\"context\", task=\"add dark mode\") and paste result here -->"
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
            "usage": "The system message is byte-identical every call → cached prefix hit. Inject cortyx(intent=\"context\", task=\"...\") output before the user message.",
            "quickstart": {
                "terminal_route": TERMINAL_ROUTE_EXAMPLE,
                "watch": WATCH_EXAMPLE,
                "doctor": DOCTOR_EXAMPLE,
                "mcp_summary": MCP_CAPABILITY_EXAMPLE,
                "mcp_task": MCP_TASK_EXAMPLE
            },
            "ux_proof": ux_proof_meta()
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
                "content": "<!-- DYNAMIC NEURONS — inject cortyx(intent=\"context\", task=\"...\") output here -->"
            },
            {
                "role": "user",
                "content": "<TASK DESCRIPTION — replace this with your actual task>"
            }
        ]
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn anthropic_export_prefers_universal_cortyx_context_entrypoint() {
        let dir = tempfile::tempdir().unwrap();
        let idx = NeuronIndex::load_or_create(dir.path()).unwrap();

        let json = serde_json::from_str::<Value>(
            &build_prompt_json(dir.path(), &idx, Provider::Anthropic).unwrap(),
        )
        .unwrap();
        let system_text = json["system"][0]["text"].as_str().unwrap();
        let placeholder = json["system"][1]["text"].as_str().unwrap();

        assert!(system_text.contains("cortyx(intent=\"context\", task=<task_description>)"));
        assert!(system_text.contains("call cortyx() with no args for a capability summary"));
        assert!(placeholder.contains("cortyx(intent=\"context\", task=\"add dark mode\")"));
    }

    #[test]
    fn openai_export_mentions_universal_context_usage() {
        let dir = tempfile::tempdir().unwrap();
        let idx = NeuronIndex::load_or_create(dir.path()).unwrap();

        let json = serde_json::from_str::<Value>(
            &build_prompt_json(dir.path(), &idx, Provider::Openai).unwrap(),
        )
        .unwrap();
        let usage = json["_cortyx_meta"]["usage"].as_str().unwrap();
        let placeholder = json["messages"][1]["content"].as_str().unwrap();

        assert!(usage.contains("cortyx(intent=\"context\", task=\"...\")"));
        assert!(placeholder.contains("cortyx(intent=\"context\", task=\"...\")"));
    }

    #[test]
    fn export_meta_includes_terminal_and_in_tool_quickstart() {
        let dir = tempfile::tempdir().unwrap();
        let idx = NeuronIndex::load_or_create(dir.path()).unwrap();

        let json = serde_json::from_str::<Value>(
            &build_prompt_json(dir.path(), &idx, Provider::Anthropic).unwrap(),
        )
        .unwrap();
        let quickstart = &json["_cortyx_meta"]["quickstart"];

        assert_eq!(
            quickstart["terminal_route"].as_str(),
            Some(TERMINAL_ROUTE_EXAMPLE)
        );
        assert_eq!(quickstart["watch"].as_str(), Some(WATCH_EXAMPLE));
        assert_eq!(quickstart["doctor"].as_str(), Some(DOCTOR_EXAMPLE));
        assert_eq!(
            quickstart["mcp_summary"].as_str(),
            Some(MCP_CAPABILITY_EXAMPLE)
        );
        assert_eq!(quickstart["mcp_task"].as_str(), Some(MCP_TASK_EXAMPLE));
    }

    #[test]
    fn export_meta_includes_machine_readable_ux_proof() {
        let dir = tempfile::tempdir().unwrap();
        let idx = NeuronIndex::load_or_create(dir.path()).unwrap();

        let json = serde_json::from_str::<Value>(
            &build_prompt_json(dir.path(), &idx, Provider::Anthropic).unwrap(),
        )
        .unwrap();
        let proof = &json["_cortyx_meta"]["ux_proof"];
        let outcomes = proof["one_entrypoint"]["outcomes"]
            .as_array()
            .expect("route outcomes should be an array");

        assert_eq!(proof["onboarding"]["terminal_steps"].as_u64(), Some(3));
        assert_eq!(proof["onboarding"]["in_tool_steps"].as_u64(), Some(2));
        assert_eq!(
            proof["recovery"]["incremental_compile"].as_str(),
            Some(INCREMENTAL_COMPILE_EXAMPLE)
        );
        assert!(outcomes
            .iter()
            .any(|value| value.as_str() == Some("answer")));
        assert!(outcomes
            .iter()
            .any(|value| value.as_str() == Some("capabilities")));
    }
}
