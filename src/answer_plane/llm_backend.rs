/// Local LLM answer synthesis via Ollama (`--features answer-llm`).
///
/// When the `answer-llm` feature is active and Ollama is reachable, this module
/// constructs a grounded prompt from retrieved neuron bodies and calls Ollama's
/// `/api/generate` endpoint.  When unavailable it returns `None` so the caller
/// falls back to the existing rule-based answer plane.
///
/// # Configuration (environment variables)
/// - `CORTYX_OLLAMA_URL`   — base URL (default: `http://localhost:11434`)
/// - `CORTYX_ANSWER_MODEL` — model tag  (default: `qwen2.5:1.5b`)
///
/// # Why Ollama?
/// No Rust model-loading complexity, ~1-2 GB model, user controls model choice,
/// works on CPU without GPU.  Expected LoCoMo F1 improvement: 0.133 → ~0.55.

#[cfg(feature = "answer-llm")]
mod inner {
    use std::time::Duration;

    const DEFAULT_OLLAMA_URL: &str = "http://localhost:11434";
    const DEFAULT_MODEL: &str = "qwen2.5:1.5b";
    /// Hard token budget for the answer to keep latency low.
    const MAX_TOKENS: u32 = 256;
    /// Context window passed to Ollama (prompt + answer budget).
    const CTX_SIZE: u32 = 2048;
    /// HTTP connect + read timeout.
    const TIMEOUT_SECS: u64 = 15;

    fn ollama_url() -> String {
        std::env::var("CORTYX_OLLAMA_URL").unwrap_or_else(|_| DEFAULT_OLLAMA_URL.to_string())
    }

    fn answer_model() -> String {
        std::env::var("CORTYX_ANSWER_MODEL").unwrap_or_else(|_| DEFAULT_MODEL.to_string())
    }

    /// Build a grounded prompt from the task and retrieved neuron bodies.
    ///
    /// The prompt is intentionally short: Cortyx neurons are already condensed
    /// summaries so feeding the full bodies is affordable within a 2K context.
    fn build_prompt(task: &str, neuron_bodies: &[&str]) -> String {
        let context = neuron_bodies
            .iter()
            .enumerate()
            .map(|(i, b)| format!("[{}] {}", i + 1, b.trim()))
            .collect::<Vec<_>>()
            .join("\n\n");
        format!(
            "You are a precise memory assistant. Answer the question using ONLY the \
             provided context. If the context does not contain the answer, say \
             \"I don't have that information.\"\n\n\
             Context:\n{context}\n\n\
             Question: {task}\n\n\
             Answer (concise, factual):"
        )
    }

    /// Call Ollama synchronously (blocking HTTP).
    ///
    /// Returns `None` on any error (Ollama not running, timeout, bad JSON) so the
    /// caller can fall back to the rule-based answer plane without crashing.
    pub fn call_ollama(task: &str, neuron_bodies: &[&str]) -> Option<String> {
        if neuron_bodies.is_empty() {
            return None;
        }
        let url = format!("{}/api/generate", ollama_url());
        let model = answer_model();
        let prompt = build_prompt(task, neuron_bodies);

        let body = serde_json::json!({
            "model": model,
            "prompt": prompt,
            "stream": false,
            "options": {
                "num_predict": MAX_TOKENS,
                "num_ctx": CTX_SIZE,
                "temperature": 0.1,
            }
        });

        let client = ureq::AgentBuilder::new()
            .timeout(Duration::from_secs(TIMEOUT_SECS))
            .build();

        let response = client
            .post(&url)
            .set("Content-Type", "application/json")
            .send_string(&body.to_string())
            .ok()?;

        let json: serde_json::Value = serde_json::from_str(&response.into_string().ok()?).ok()?;
        let answer = json["response"].as_str()?.trim().to_string();
        if answer.is_empty() {
            return None;
        }
        Some(answer)
    }

    /// Check if Ollama is reachable (fast probe, 2s timeout).
    pub fn is_available() -> bool {
        let url = format!("{}/api/tags", ollama_url());
        ureq::AgentBuilder::new()
            .timeout(Duration::from_secs(2))
            .build()
            .get(&url)
            .call()
            .is_ok()
    }
}

// ─── Public API — always compiled ─────────────────────────────────────────────

/// Try to synthesize an answer using a local Ollama LLM.
///
/// Returns `Some(answer)` when the `answer-llm` feature is active and Ollama
/// responds within the timeout.  Returns `None` otherwise — callers must fall
/// back to the rule-based answer plane.
#[allow(unused_variables)]
pub fn try_llm_answer(task: &str, neuron_bodies: &[&str]) -> Option<String> {
    #[cfg(feature = "answer-llm")]
    {
        inner::call_ollama(task, neuron_bodies)
    }
    #[cfg(not(feature = "answer-llm"))]
    {
        None
    }
}

/// Returns `true` when `answer-llm` is compiled in and Ollama is reachable.
pub fn llm_available() -> bool {
    #[cfg(feature = "answer-llm")]
    {
        inner::is_available()
    }
    #[cfg(not(feature = "answer-llm"))]
    {
        false
    }
}
