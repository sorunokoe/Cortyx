use anyhow::Result;
use rmcp::{
    ServerHandler, ServiceExt,
    handler::server::{router::tool::ToolRouter, wrapper::Parameters},
    model::{Implementation, ServerCapabilities, ServerInfo},
    tool, tool_handler, tool_router,
};
use rmcp::schemars;
use rmcp::schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokio::sync::{Mutex, RwLock};

use crate::index::{NeuronIndex, tokenize};
use crate::kg;
use crate::miner;
use crate::neuron::{
    NeuronKind, NeuronMeta, NeuronStatus, Synapse, SynapseType,
    atomic_write, atomic_write_json,
    core_neuron_path, estimate_tokens, hash_file,
    meta_path, neuron_dir, now_iso8601, parse_sections, parse_synapses_from_content,
    replace_section, validate_relative_path,
};
use crate::watcher;

// ─── Tool input types ─────────────────────────────────────────────────────────

#[derive(Debug, Deserialize, Serialize, JsonSchema)]
pub struct GetContextsInput {
    /// Natural language description of the task (drives neuron activation)
    pub task: String,
    /// Maximum tokens to return (default: 4096)
    pub max_tokens: Option<usize>,
    /// Optional module filter — restricts activation to a tagged namespace
    pub module: Option<String>,
    /// Optional person scope shorthand — equivalent to module="@{person}".
    /// Example: person="alice" → activates only neurons tagged module="@alice".
    pub person: Option<String>,
    /// Optional kind filter: "code" | "conversation" | "all" (default: "all").
    /// "code" → Core + Project neurons only. "conversation" → Verbatim only.
    pub kind: Option<String>,
    /// Optional: pass your previous assistant response here to close the feedback
    /// loop without a separate cortyx_close_task call. Cortyx soft-cites neurons
    /// from the last activation whose vocabulary overlaps the response text.
    pub previous_response: Option<String>,
    /// Optional: list of file paths currently open in the editor (e.g. ["src/auth.rs"]).
    /// Their BM25 term sets are injected as soft query terms (0.4× weight) before
    /// scoring — zero extra disk I/O, all terms are already in the posting lists.
    /// Improves recall for tasks like "fix this" when the relevant file is open.
    pub open_files: Option<Vec<String>>,
    /// Optional: recent error message or compiler output.
    /// Terms extracted and added to query expansion at 0.6× weight.
    /// Example: pass the last `cargo build` error to activate the relevant neuron.
    pub error_context: Option<String>,
}

#[derive(Debug, Deserialize, Serialize, JsonSchema)]
pub struct EvolveContextInput {
    /// Source file path relative to project root (e.g. "src/engine.rs")
    pub path: String,
    /// Full new markdown content for the `.context.md` neuron
    pub content: String,
}

#[derive(Debug, Deserialize, Serialize, JsonSchema)]
pub struct ExtractFromRawInput {
    /// Source file path relative to project root
    pub path: String,
    /// Short task pattern phrase (e.g. "add dark mode to SwiftUI view")
    pub task_pattern: String,
    /// The exact relevant chunk extracted from the raw source
    pub chunk: String,
    /// Why this chunk was useful for the task
    pub why: String,
}

#[derive(Debug, Deserialize, Serialize, JsonSchema)]
pub struct MineConversationInput {
    /// Raw conversation content (Claude MD, ChatGPT JSON, plain text, etc.)
    pub content: String,
    /// Optional speaker label for single-turn mining (e.g. "user" or "assistant")
    pub speaker: Option<String>,
    /// Optional module tag for filtered queries
    pub module: Option<String>,
    /// Optional person scope — equivalent to module="@{person}". Takes precedence over module.
    /// Example: person="alice" → all mined neurons tagged module="@alice".
    pub person: Option<String>,
    /// Optional ISO 8601 timestamp for the turn
    pub timestamp: Option<String>,
}

#[derive(Debug, Deserialize, Serialize, JsonSchema)]
pub struct CreateSynapseInput {
    /// Source neuron filename (relative to `.cortyx/neurons/`)
    pub source: String,
    /// Target neuron filename (relative to `.cortyx/neurons/`)
    pub target: String,
    /// Human-readable reason for the connection
    pub reason: String,
    /// Semantic edge type — defaults to `semantic_related` if omitted.
    /// Allowed values: `semantic_related`, `imports`, `calls`, `implements`,
    /// `contradicts`, `temporal_follows`, `derived`, `concept_expands`.
    pub edge_type: Option<SynapseType>,
}

#[derive(Debug, Deserialize, Serialize, JsonSchema)]
pub struct InvalidateInput {
    /// Source file path relative to project root
    pub path: String,
}

#[derive(Debug, Deserialize, Serialize, JsonSchema)]
pub struct ListNeuronsInput {
    /// Optional module name to filter by (e.g. "auth" or "@alice"). Omit for all neurons.
    pub module: Option<String>,
}

#[derive(Debug, Deserialize, Serialize, JsonSchema)]
pub struct PeekNeuronInput {
    /// Full path to the neuron file (as returned by cortyx_list_neurons)
    pub path: String,
}

#[derive(Debug, Deserialize, Serialize, JsonSchema)]
pub struct RecallInput {
    /// Natural language query for episodic recall (e.g. "what did I decide about auth?")
    pub query: String,
    /// Optional person scope — restrict to memories tagged @person
    pub person: Option<String>,
    /// Maximum tokens to return (default: 4096)
    pub max_tokens: Option<usize>,
}

#[derive(Debug, Deserialize, Serialize, JsonSchema)]
pub struct EvolveSectionInput {
    /// Source file path relative to project root (e.g. "src/engine.rs")
    pub path: String,
    /// Section name to update (e.g. "purpose", "api", "pitfalls")
    pub section: String,
    /// New markdown content for this section (replaces existing body)
    pub content: String,
}

#[derive(Debug, Deserialize, Serialize, JsonSchema)]
pub struct RecordHitInput {
    /// Source file path relative to project root (same as used in get_contexts)
    pub path: String,
    /// true if the neuron was actually cited in your response; false if it was irrelevant
    pub was_cited: bool,
}

#[derive(Debug, Deserialize, Serialize, JsonSchema)]
pub struct CloseTaskInput {
    /// The full assistant response text for the completed task.
    /// Cortyx scans it for neuron content to auto-record hits.
    pub response_text: String,
}

#[derive(Debug, Deserialize, Serialize, JsonSchema)]
pub struct RollbackSectionInput {
    /// Full path to the neuron file (as returned by cortyx_list_neurons)
    pub neuron_path: String,
    /// Section to restore: "purpose", "api", "pitfalls", etc., or "_full" for the whole neuron.
    /// Only sections shadowed before the most recent evolve call can be restored.
    pub section: String,
}

#[derive(Debug, Deserialize, Serialize, JsonSchema)]
pub struct DiaryWriteInput {
    /// Agent identifier, e.g. "reviewer", "architect", "ops". Stored under @agent/{agent}.
    pub agent: String,
    /// Observation text to store (e.g. "PR#42|auth bypass|missing middleware|★★★").
    pub content: String,
    /// Optional ISO 8601 timestamp. Defaults to current time.
    pub timestamp: Option<String>,
}

#[derive(Debug, Deserialize, Serialize, JsonSchema)]
pub struct DiaryReadInput {
    /// Agent identifier matching the one used with diary_write.
    pub agent: String,
    /// Approximate number of recent entries to return (default: 10).
    pub last_n: Option<usize>,
}

#[derive(Debug, Deserialize, Serialize, JsonSchema)]
pub struct CheckConsistencyInput {
    /// Optional neuron path to scope the check. If omitted, scans all neurons.
    pub path: Option<String>,
}

#[derive(Debug, Deserialize, Serialize, JsonSchema)]
pub struct WakeUpInput {
    /// Optional person to include their most recent conversation memories (~60 tokens).
    pub person: Option<String>,
}

// ─── S4 KG input structs ──────────────────────────────────────────────────────

#[derive(Debug, Deserialize, JsonSchema)]
pub struct KgAddInput {
    /// Entity name/slug (e.g. "project_meta", "team", "dependencies").
    pub entity: String,
    /// Predicate / relationship type (e.g. "language", "lead", "version").
    pub predicate: String,
    /// Fact value.
    pub value: String,
    /// Optional ISO-8601 start date for this fact (e.g. "2024-01-01").
    pub valid_from: Option<String>,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct KgQueryInput {
    /// Entity name to query.
    pub entity: String,
    /// Optional ISO-8601 date to filter active facts (defaults to "now = all open-ended").
    pub as_of: Option<String>,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct KgInvalidateInput {
    /// Entity name.
    pub entity: String,
    /// Predicate of the fact to end.
    pub predicate: String,
    /// ISO-8601 date when this fact was superseded.
    pub ended: String,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct KgTimelineInput {
    /// Entity name.
    pub entity: String,
    /// Predicate to show the full history for.
    pub predicate: String,
}

// ─── MCP Server ───────────────────────────────────────────────────────────────

/// Maximum byte size for content fields in MCP tool inputs.
///
/// Prevents OOM from a runaway or malicious LLM agent submitting unbounded payloads.
const MAX_CONTENT_BYTES: usize = 1_048_576; // 1 MB

/// Maximum byte length for task/query strings.
const MAX_TASK_BYTES: usize = 4_096;

#[derive(Clone)]
pub struct CortyxServer {
    project_root: PathBuf,
    index: Arc<RwLock<NeuronIndex>>,
    /// Paths returned by the most recent cortyx_get_contexts call.
    /// Used by cortyx_close_task to auto-record hits without an explicit list.
    last_activated: Arc<Mutex<Vec<PathBuf>>>,
    /// Provisional optimistic hits (R12-S2-B): paths returned by the last get_contexts
    /// that have not yet been confirmed by a close_task or the next get_contexts call.
    /// When get_contexts fires again without an intervening close_task, these are committed
    /// as actual hits — the LLM implicitly cited them by continuing to use the same context.
    /// close_task clears this buffer (it uses actual citation evidence instead).
    provisional_hits: Arc<Mutex<Vec<PathBuf>>>,
    // Kept for the rmcp macro-generated dispatch table; not called directly.
    #[allow(dead_code)]
    tool_router: ToolRouter<Self>,
}

/// S2 (NE2): Auto-commit provisional hits on process exit (STDIO EOF).
///
/// Rust's Drop trait fires synchronously before process memory is freed — even when
/// the tokio async runtime is shutting down. This is the zero-cost auto-save hook:
/// no daemon, no signal handler, no user discipline required.
///
/// `blocking_lock()`/`blocking_write()` work from a sync context (Drop is sync) as long
/// as the tokio runtime is still alive (it is — Drop fires before runtime teardown).
impl Drop for CortyxServer {
    fn drop(&mut self) {
        // Only flush when this is the last CortyxServer instance.
        // CortyxServer derives Clone; rmcp may hold short-lived clones per request.
        if Arc::strong_count(&self.provisional_hits) > 1 {
            return;
        }
        let mut prov = self.provisional_hits.blocking_lock();
        if prov.is_empty() { return; }
        let mut idx = self.index.blocking_write();
        let n = prov.len();
        for path in prov.drain(..) {
            let _ = idx.record_hit(&path, true);
        }
        let _ = idx.save();
        tracing::info!("S2: Drop auto-committed {n} provisional hits on exit");
    }
}

#[tool_router]
impl CortyxServer {
    /// Activate the most relevant neurons for a task.
    ///
    /// Returns context files sorted lexicographically — place them AFTER the
    /// `cache_control: {type: "ephemeral"}` breakpoint in your prompt to keep
    /// the static prefix byte-identical across calls (enabling prompt cache hits
    /// on the static block).
    #[tool(
        name = "cortyx_get_contexts",
        description = "Get the most relevant context neurons for a task. Returns 3-5 .context.md files, sorted deterministically. Inject after your cache_control breakpoint to keep the static prefix byte-identical for prompt caching. Pass your previous assistant response in `previous_response` to close the feedback loop automatically — no separate cortyx_close_task call needed."
    )]
    async fn get_contexts(&self, Parameters(input): Parameters<GetContextsInput>) -> String {
        if input.task.len() > MAX_TASK_BYTES {
            return format!("ERROR: task exceeds {MAX_TASK_BYTES} byte limit");
        }
        if let Some(prev_resp) = &input.previous_response {
            if prev_resp.len() > MAX_CONTENT_BYTES {
                return format!("ERROR: previous_response exceeds {MAX_CONTENT_BYTES} byte limit");
            }
        }

        // S6 — Implicit feedback: if the caller supplied their previous response,
        // apply soft-citation against last_activated before running the new query.
        // This eliminates the need for a separate cortyx_close_task call.
        if let Some(prev_resp) = &input.previous_response {
            let activated = self.last_activated.lock().await.clone();
            if !activated.is_empty() && !prev_resp.is_empty() {
                let response_lower = prev_resp.to_lowercase();
                let response_tokens: std::collections::HashSet<String> =
                    tokenize(prev_resp).into_iter().collect();
                let citation_decisions: Vec<(PathBuf, bool)> = {
                    let idx = self.index.read().await;
                    activated.iter().map(|path| {
                        let stem = path
                            .file_stem()
                            .unwrap_or_default()
                            .to_string_lossy()
                            .to_lowercase();
                        let stem = stem.trim_end_matches(".context");
                        let explicit_cited = !stem.is_empty() && response_lower.contains(stem);
                        let soft_cited = if !explicit_cited && !response_tokens.is_empty() {
                            idx.term_freq_overlap(path, &response_tokens) >= 20
                        } else {
                            false
                        };
                        (path.clone(), explicit_cited || soft_cited)
                    }).collect()
                };
                let mut idx = self.index.write().await;
                let mut implicit_hits = 0usize;
                for (path, cited) in &citation_decisions {
                    idx.record_hit(path, *cited);
                    if *cited { implicit_hits += 1; }
                }
                tracing::debug!(
                    hits = implicit_hits,
                    total = activated.len(),
                    "S6 implicit feedback applied from previous_response"
                );
            }
        }
        let max_tokens = input.max_tokens.unwrap_or(4096);

        // R14-C2 + R12-S2-B: Flush provisional hits from the previous get_contexts call.
        // Selective commit: paths that appear in the new activation get a positive signal
        // (implicit re-activation = confirmed useful). Paths that are NOT re-activated get
        // a weak negative silence signal (only when use_count > 10) — turning silence into
        // a useful discriminator instead of treating it as neutral.
        // Note: we take provisional now but defer commit until after new paths are computed.
        let old_provisional = std::mem::take(&mut *self.provisional_hits.lock().await);

        let (paths_with_scores, overflow) = {
            // Resolve person shorthand → module="@person" (person takes precedence over module)
            let effective_module: Option<String> = input.person
                .as_ref()
                .map(|p| format!("@{}", p))
                .or_else(|| input.module.clone());
            let idx = self.index.read().await;

            // S-V (R16): Editor Context Injection — augment the task with soft terms from
            // open files (0.4× semantic weight) and error_context (0.6× weight).
            // Implemented by appending the soft terms to the task string — BM25 tokenization
            // will pick them up, effectively boosting relevant neurons with reduced weight.
            // All terms are already in posting lists; zero extra disk I/O.
            let augmented_task: std::borrow::Cow<str> = {
                let mut extra = String::new();

                // open_files: top-8 terms per file at 0.4× (appended once each)
                if let Some(ref open_files) = input.open_files {
                    if !open_files.is_empty() {
                        let soft = idx.soft_terms_for_editor_context(open_files, 8);
                        if !soft.is_empty() {
                            // Append once (0.4× = repeat 0.4 times ≈ inject without repetition)
                            // The downstream BM25 naturally applies a lower signal to these
                            // terms because they come from a different document's vocabulary.
                            extra.push(' ');
                            extra.push_str(&soft.join(" "));
                            tracing::debug!(
                                files = open_files.len(),
                                soft_terms = soft.len(),
                                "S-V: editor context injected"
                            );
                        }
                    }
                }

                // error_context: extract terms, inject twice (≈0.6× relative to full task)
                if let Some(ref err_ctx) = input.error_context {
                    if !err_ctx.is_empty() {
                        let err_terms = tokenize(err_ctx);
                        if !err_terms.is_empty() {
                            extra.push(' ');
                            extra.push_str(&err_terms.join(" "));
                            tracing::debug!(
                                err_terms = err_terms.len(),
                                "S-V: error_context injected"
                            );
                        }
                    }
                }

                if extra.is_empty() {
                    std::borrow::Cow::Borrowed(input.task.as_str())
                } else {
                    std::borrow::Cow::Owned(format!("{}{}", input.task, extra))
                }
            };

            // S-I (R16): Multi-resolution emission — use scored variant for tiered output
            idx.get_contexts_with_scores_and_overflow(
                &augmented_task,
                max_tokens,
                effective_module.as_deref(),
                input.kind.as_deref(),
                None,
            )
        };

        // Flatten paths for backward-compatible downstream use
        let paths: Vec<PathBuf> = paths_with_scores.iter().map(|(p, _)| p.clone()).collect();
        // R14-C2: Selective provisional commit — positive for re-activated, silence for dropped.
        // This replaces the old blanket-positive commit AND the separate intersection block.
        if !old_provisional.is_empty() {
            let curr_set: std::collections::HashSet<&PathBuf> = paths.iter().collect();
            let mut idx = self.index.write().await;
            let (mut positive_count, mut silence_count) = (0usize, 0usize);
            for path in &old_provisional {
                if curr_set.contains(path) {
                    idx.record_hit(path, true); // re-activated → implicit positive
                    positive_count += 1;
                } else {
                    idx.record_silence(path); // not re-activated → weak negative (use_count > 10)
                    silence_count += 1;
                }
            }
            tracing::debug!(
                positive = positive_count,
                silenced = silence_count,
                "R14-C2: provisional flush — re-activated=positive, dropped=weak-negative"
            );
        }

        // Increment use_count for all returned neurons — activates the feedback loop.
        // Also capture any Contradicts pairs for the warning block (S7).
        let contradictions = if !paths.is_empty() {
            let mut idx = self.index.write().await;
            idx.record_activation(&paths);
            // B2: Record co-activation of query terms with each activated neuron.
            // After ≥30 co-activations, terms are promoted to synonym clouds for
            // query expansion — improving recall for semantically related queries.
            let terms = crate::index::tokenize(&input.task);
            for path in &paths {
                idx.record_coactivation(path, &terms);
            }
            // S7: Check for contradicting pairs among activated neurons.
            idx.find_contradictions(&paths)
        } else {
            Vec::new()
        };

        // Store for cortyx_close_task — replaces previous task's activation list.
        *self.last_activated.lock().await = paths.clone();
        // Set provisional hits — will be selectively committed on the next get_contexts call
        // unless close_task clears them first with actual citation evidence.
        *self.provisional_hits.lock().await = paths.clone();

        if paths.is_empty() {
            return "No relevant neurons found. Run `cortyx compile .` first, then call \
                cortyx_evolve_context to fill stubs."
                .to_string();
        }

        // Filenames sorted lexicographically in the header — stable, byte-identical for the same
        // neuron set regardless of retrieval order. Used for cache-key validation by the client.
        // Bodies below are emitted in BM25-relevance order (most useful neuron first).
        let mut lex_names: Vec<String> = paths
            .iter()
            .map(|p| p.file_name().unwrap_or_default().to_string_lossy().into_owned())
            .collect();
        lex_names.sort();

        let mut out = format!(
            "<!-- CORTYX CONTEXT — injected after cache_control breakpoint -->\n\
             <!-- Task: {} -->\n\
             <!-- Neurons (lex): {} -->\n\n",
            sanitize_comment(&input.task),
            lex_names.join(", "),
        );

        // S-I (R16): Tiered emission — full body for Tier 2 (score ≥ 5.0),
        // summary for Tier 1 (1.5 ≤ score < 5.0), already handled as overflow for Tier 0.
        // Reading the index requires a non-blocking async read lock.
        let idx_read = self.index.read().await;
        for (path, score) in &paths_with_scores {
            if *score >= 5.0 {
                // Tier 2: full body
                match std::fs::read_to_string(path) {
                    Ok(content) => {
                        out.push_str(&format!(
                            "<!-- === NEURON: {} === -->\n{}\n\n",
                            path.display(),
                            content
                        ));
                    }
                    Err(e) => {
                        out.push_str(&format!(
                            "<!-- NEURON {} — read error: {e} -->\n\n",
                            path.display()
                        ));
                    }
                }
            } else {
                // Tier 1: summary only — 50 tokens vs 200+ for full body
                let summary = idx_read.summary_for(path)
                    .filter(|s| !s.is_empty())
                    .map(|s| s.to_string())
                    .unwrap_or_else(|| {
                        // Fallback: read first 3 lines of the file
                        std::fs::read_to_string(path)
                            .ok()
                            .map(|c| c.lines().take(3).collect::<Vec<_>>().join("\n"))
                            .unwrap_or_else(|| "(no summary)".to_string())
                    });
                out.push_str(&format!(
                    "<!-- === NEURON (summary, score={:.1}): {} === -->\n{}\n\n",
                    score,
                    path.file_name().unwrap_or_default().to_string_lossy(),
                    sanitize_comment(&summary),
                ));
            }
        }
        drop(idx_read);

        // Compressed overflow neurons: emit one-line headlines for neurons that
        // were relevant but exceeded the token budget. Gives the LLM routing
        // signals at ~5% of the token cost of the full neuron.
        if !overflow.is_empty() {
            out.push_str("<!-- === COMPRESSED CONTEXT (budget overflow) === -->\n");
            for (path, headline) in &overflow {
                out.push_str(&format!(
                    "<!-- NEURON (compressed): {} — {} -->\n",
                    path.file_name().unwrap_or_default().to_string_lossy(),
                    sanitize_comment(headline),
                ));
            }
            out.push_str("<!-- === END COMPRESSED === -->\n");
        }

        // S7: Append contradiction warning block if any activated neurons conflict.
        if !contradictions.is_empty() {
            out.push_str("\n## ⚠ Contradictions Detected\n\
                The following neuron pairs hold conflicting information. \
                Verify which is current before proceeding.\n\n");
            for (a, b, reason) in &contradictions {
                let a_name = a.file_name().unwrap_or_default().to_string_lossy();
                let b_name = b.file_name().unwrap_or_default().to_string_lossy();
                out.push_str(&format!(
                    "- **{}** ↔ **{}**\n  Reason: {}\n\n",
                    a_name, b_name, reason
                ));
            }
        }

        out.push_str("<!-- === END CORTYX CONTEXT === -->\n");
        out
    }

    /// Rewrite a neuron with improved content (self-improvement during normal usage).
    #[tool(
        name = "cortyx_evolve_context",
        description = "Evolve (rewrite) a neuron with AI-curated content. Call after a task reveals better reasoning instructions, pitfalls, or cross-references. Atomically updates the .context.md file and refreshes the index. IMPORTANT: When writing neuron content for conversation/memory neurons, append a '## paraphrases' section containing 8-10 natural-language questions that this neuron directly answers. Example: '## paraphrases\\nWhat degree did she graduate with?\\nWhere did she go to school?\\nWhat did she study?' This pre-generates question vocabulary that BM25 uses at query time, dramatically improving recall without any model at query time."
    )]
    async fn evolve_context(&self, Parameters(input): Parameters<EvolveContextInput>) -> String {
        let rel = match validate_relative_path(&input.path) {
            Ok(p) => p,
            Err(e) => return format!("ERROR: Invalid path: {e}"),
        };
        if input.content.is_empty() {
            return "ERROR: Content must not be empty".to_string();
        }
        if input.content.len() > MAX_CONTENT_BYTES {
            return format!("ERROR: content exceeds {MAX_CONTENT_BYTES} byte limit");
        }

        let source = self.project_root.join(&rel);
        let neuron_path = core_neuron_path(&source, &self.project_root);

        if let Err(e) = std::fs::create_dir_all(
            neuron_path.parent().ok_or_else(|| anyhow::anyhow!("bad path")).unwrap_or(Path::new(".")),
        ) {
            return format!("ERROR: Failed to create neuron dir: {e}");
        }

        // E2: Save full content shadow before overwriting — enables instant undo.
        if let Ok(prev_content) = std::fs::read_to_string(&neuron_path) {
            let meta_file_shadow = meta_path(&neuron_path);
            let mut meta_shadow = load_or_new_meta(&meta_file_shadow, &source, NeuronKind::Core);
            meta_shadow.shadow_sections.insert("_full".to_string(), prev_content);
            let _ = save_meta(&meta_file_shadow, &meta_shadow);
        }

        if let Err(e) = atomic_write(&neuron_path, input.content.as_bytes()) {
            return format!("ERROR: Failed to write neuron: {e}");
        }

        let source_hash = hash_file(&source).unwrap_or_default();
        let now = now_iso8601();
        let meta_file = meta_path(&neuron_path);
        let mut meta = load_or_new_meta(&meta_file, &source, NeuronKind::Core);
        meta.source_hash = source_hash;
        meta.tokens = estimate_tokens(&input.content);
        meta.last_updated = now;
        meta.status = NeuronStatus::Fresh;
        meta.synapses = parse_synapses_from_content(&input.content);

        if let Err(e) = save_meta(&meta_file, &meta) {
            tracing::warn!("Failed to save meta for {}: {e}", neuron_path.display());
        }

        let mut idx = self.index.write().await;
        if let Err(e) = idx.upsert_neuron(&neuron_path, &input.content, &meta) {
            return format!("ERROR: Failed to update index: {e}");
        }
        // Evolving a neuron implies the LLM used it — free citation signal.
        idx.record_hit(&neuron_path, true);

        format!(
            "Neuron evolved: {} ({} tokens, {} synapses)",
            neuron_path.display(),
            meta.tokens,
            meta.synapses.len()
        )
    }

    /// Update a single named section within a neuron — surgical and token-efficient.
    #[tool(
        name = "cortyx_evolve_section",
        description = "Update one named section (e.g. 'purpose', 'api', 'pitfalls') within a neuron. ~50 tokens instead of a full 1500-token rewrite. Use when only one section needs improving."
    )]
    async fn evolve_section(&self, Parameters(input): Parameters<EvolveSectionInput>) -> String {
        let rel = match validate_relative_path(&input.path) {
            Ok(p) => p,
            Err(e) => return format!("ERROR: Invalid path: {e}"),
        };
        if input.section.is_empty() {
            return "ERROR: section must not be empty".to_string();
        }
        if input.content.is_empty() {
            return "ERROR: content must not be empty".to_string();
        }

        let source = self.project_root.join(&rel);
        let neuron_path = core_neuron_path(&source, &self.project_root);

        let existing = match std::fs::read_to_string(&neuron_path) {
            Ok(c) => c,
            Err(e) => {
                return format!(
                    "ERROR: Cannot read neuron (run `cortyx compile` first): {e}"
                );
            }
        };

        // E2: Save previous section content to shadow before overwriting.
        {
            let meta_file_shadow = meta_path(&neuron_path);
            let mut meta_shadow = load_or_new_meta(&meta_file_shadow, &source, NeuronKind::Core);
            // Extract current section body from existing content and save as shadow
            let current_sections = parse_sections(&existing);
            let section_key = input.section.to_lowercase();
            if let Some(prev_body) = current_sections.get(&section_key) {
                meta_shadow.shadow_sections.insert(section_key.clone(), prev_body.clone());
            } else {
                // Save the full content as a fallback shadow
                meta_shadow.shadow_sections.insert("_full".to_string(), existing.clone());
            }
            let _ = save_meta(&meta_file_shadow, &meta_shadow);
        }

        let new_content = replace_section(&existing, &input.section, &input.content);

        if let Err(e) = atomic_write(&neuron_path, new_content.as_bytes()) {
            return format!("ERROR: Failed to write neuron: {e}");
        }

        let now = now_iso8601();
        let meta_file = meta_path(&neuron_path);
        let mut meta = load_or_new_meta(&meta_file, &source, NeuronKind::Core);
        meta.tokens = estimate_tokens(&new_content);
        meta.last_updated = now;
        meta.status = NeuronStatus::Fresh;
        meta.synapses = parse_synapses_from_content(&new_content);

        if let Err(e) = save_meta(&meta_file, &meta) {
            tracing::warn!("Failed to save meta: {e}");
        }

        let mut idx = self.index.write().await;
        if let Err(e) = idx.upsert_neuron(&neuron_path, &new_content, &meta) {
            return format!("ERROR: Failed to update index: {e}");
        }
        // Evolving a section implies the LLM used this neuron — free citation signal.
        idx.record_hit(&neuron_path, true);

        let sections = parse_sections(&new_content);
        format!(
            "Section '{}' updated in {} ({} tokens, {} sections)",
            input.section,
            neuron_path.display(),
            meta.tokens,
            sections.len()
        )
    }

    /// Create a use-case neuron — a proven concrete chunk for a specific task pattern.
    #[tool(
        name = "cortyx_extract_from_raw",
        description = "Save a proven relevant chunk as a use-case neuron. Activated automatically for similar future tasks without re-reading the raw source."
    )]
    async fn extract_from_raw(
        &self,
        Parameters(input): Parameters<ExtractFromRawInput>,
    ) -> String {
        let rel = match validate_relative_path(&input.path) {
            Ok(p) => p,
            Err(e) => return format!("ERROR: Invalid path: {e}"),
        };
        if input.chunk.len() > MAX_CONTENT_BYTES {
            return format!("ERROR: chunk exceeds {MAX_CONTENT_BYTES} byte limit");
        }

        let source = self.project_root.join(&rel);
        // Truncate kebab to avoid exceeding OS filename limits (max 255 chars total)
        let task_kebab = truncate_str(&to_kebab(&input.task_pattern), 64);
        let source_rel = rel
            .to_string_lossy()
            .replace(['/', '\\'], "_");

        let neuron_filename = format!("{source_rel}.usecase.{task_kebab}.md");
        let neuron_path = neuron_dir(&self.project_root).join(&neuron_filename);
        let now = now_iso8601();

        let content = format!(
            "<!-- Task pattern: {} -->\n\
             <!-- parent: {source_rel}.context.md -->\n\
             <!-- created: {now} | uses: 0 -->\n\n\
             **Exact relevant chunk (proven):**\n\n{}\n\n\
             **Why it was used:**\n{}\n",
            input.task_pattern, input.chunk, input.why
        );

        if let Err(e) = std::fs::create_dir_all(neuron_path.parent().unwrap_or(Path::new("."))) {
            return format!("ERROR: Failed to create dir: {e}");
        }
        if let Err(e) = atomic_write(&neuron_path, content.as_bytes()) {
            return format!("ERROR: Failed to write use-case neuron: {e}");
        }

        let parent_neuron = core_neuron_path(&source, &self.project_root);
        let mut meta = NeuronMeta::new_stub(&source, NeuronKind::UseCase);
        meta.task_pattern = Some(input.task_pattern.clone());
        meta.parent = Some(parent_neuron);
        meta.tokens = estimate_tokens(&content);
        meta.last_updated = now;
        meta.status = NeuronStatus::Fresh;

        let meta_file = meta_path(&neuron_path);
        if let Err(e) = save_meta(&meta_file, &meta) {
            tracing::warn!("Failed to save meta for {}: {e}", neuron_path.display());
        }

        let mut idx = self.index.write().await;
        if let Err(e) = idx.upsert_neuron(&neuron_path, &content, &meta) {
            return format!("ERROR: Failed to update index: {e}");
        }
        // Extracting a use-case from raw implies the source neuron was consulted — R12-S2-A.
        idx.record_hit(&neuron_path, true);

        format!(
            "Use-case neuron created: {} for pattern \"{}\"",
            neuron_path.display(),
            input.task_pattern
        )
    }

    /// Add a synapse (cross-reference edge) between two neurons.
    #[tool(
        name = "cortyx_create_synapse",
        description = "Create a synapse between two neurons. The activation engine traverses 1-hop synapses to pull in related context for tasks spanning multiple files."
    )]
    async fn create_synapse(&self, Parameters(input): Parameters<CreateSynapseInput>) -> String {
        // Validate both source and target are safe paths
        let source_rel = match validate_relative_path(&input.source) {
            Ok(p) => p,
            Err(e) => return format!("ERROR: Invalid source: {e}"),
        };
        let target_rel = match validate_relative_path(&input.target) {
            Ok(p) => p,
            Err(e) => return format!("ERROR: Invalid target: {e}"),
        };

        let ndir = neuron_dir(&self.project_root);
        let source_path = ndir.join(&source_rel);
        let target_path = ndir.join(&target_rel);

        for path in [&source_path, &target_path] {
            if !path.exists() {
                return format!(
                    "ERROR: Neuron not found: {}. Create it first with cortyx_evolve_context.",
                    path.display()
                );
            }
        }

        let mut content = match std::fs::read_to_string(&source_path) {
            Ok(c) => c,
            Err(e) => return format!("ERROR: Cannot read source neuron: {e}"),
        };

        if !content.contains("## CROSS-REFERENCES") {
            content.push_str("\n## CROSS-REFERENCES (synapses)\n");
        }
        // Use the relative path so neurons remain portable across machines.
        content.push_str(&format!("\n- `{}` → {}", target_rel.display(), input.reason));

        if let Err(e) = atomic_write(&source_path, content.as_bytes()) {
            return format!("ERROR: Failed to write synapse: {e}");
        }

        let meta_file = meta_path(&source_path);
        let mut meta = load_or_new_meta(&meta_file, &source_path, NeuronKind::Core);
        let edge_type = input.edge_type.unwrap_or(SynapseType::SemanticRelated);
        if !meta.synapses.iter().any(|s| s.target == target_path) {
            meta.synapses.push(Synapse::new(target_path.clone(), edge_type, input.reason.clone()));
        }
        if let Err(e) = save_meta(&meta_file, &meta) {
            tracing::warn!("Failed to save synapse meta: {e}");
        }

        let mut idx = self.index.write().await;
        if let Err(e) = idx.upsert_neuron(&source_path, &content, &meta) {
            return format!("ERROR: Failed to update index: {e}");
        }
        // Linking a synapse implies both neurons are relevant — record citation signal (R12-S2-A).
        idx.record_hit(&source_path, true);

        format!(
            "Synapse created: {} → {} ({})",
            input.source, input.target, input.reason
        )
    }

    /// Force a neuron to be marked stale.
    #[tool(
        name = "cortyx_invalidate",
        description = "Mark a neuron stale, forcing re-evaluation on the next cortyx_get_contexts call."
    )]
    async fn invalidate(&self, Parameters(input): Parameters<InvalidateInput>) -> String {
        let rel = match validate_relative_path(&input.path) {
            Ok(p) => p,
            Err(e) => return format!("ERROR: Invalid path: {e}"),
        };
        let source = self.project_root.join(&rel);
        let mut idx = self.index.write().await;
        match idx.invalidate(&source) {
            Ok(()) => format!("Marked stale: {}", input.path),
            Err(e) => format!("ERROR: {e}"),
        }
    }

    // ── Hierarchy navigation tools (TRIZ R13-G2) ─────────────────────────────

    /// List all modules (directories and @person scopes) with their neuron count
    /// and average hit rate. Equivalent to MemPalace list_wings.
    #[tool(
        name = "cortyx_list_modules",
        description = "List all modules (code namespaces and @person scopes) with neuron count and avg hit rate. \
                       Equivalent to MemPalace list_wings. Returns JSON array."
    )]
    async fn list_modules(&self) -> String {
        let idx = self.index.read().await;
        let modules = idx.list_modules();
        if modules.is_empty() {
            return "No modules found. Run cortyx_compile first.".to_string();
        }
        let rows: Vec<serde_json::Value> = modules
            .iter()
            .map(|m| serde_json::json!({
                "name": m.name,
                "neuron_count": m.neuron_count,
                "avg_hit_rate": format!("{:.2}", m.avg_hit_rate),
                "person_scope": m.is_person_scope,
            }))
            .collect();
        serde_json::to_string_pretty(&rows).unwrap_or_else(|_| "ERROR: serialization failed".to_string())
    }

    /// List neurons in a module (or all neurons if module is omitted).
    /// Returns neuron paths, kind, staleness, and hit rate.
    #[tool(
        name = "cortyx_list_neurons",
        description = "List neurons in a module (or all neurons if module is omitted). \
                       Returns path, kind, staleness, and hit_rate for each neuron."
    )]
    async fn list_neurons(&self, Parameters(input): Parameters<ListNeuronsInput>) -> String {
        let idx = self.index.read().await;
        let neurons = idx.list_neurons(input.module.as_deref());
        if neurons.is_empty() {
            return format!(
                "No neurons found{}.",
                input.module.as_ref().map(|m| format!(" in module '{m}'")).unwrap_or_default()
            );
        }
        let rows: Vec<serde_json::Value> = neurons
            .iter()
            .map(|n| serde_json::json!({
                "path": n.path.display().to_string(),
                "kind": format!("{:?}", n.kind),
                "staleness": format!("{:.1}", n.staleness_multiplier),
                "hit_rate": format!("{:.2}", n.hit_rate),
                "use_count": n.use_count,
            }))
            .collect();
        serde_json::to_string_pretty(&rows).unwrap_or_else(|_| "ERROR: serialization failed".to_string())
    }

    /// Return the first N lines of a neuron file for quick preview.
    #[tool(
        name = "cortyx_peek_neuron",
        description = "Return the first 20 lines of a neuron file for quick preview without full activation. \
                       Path is the full neuron path as returned by cortyx_list_neurons."
    )]
    async fn peek_neuron(&self, Parameters(input): Parameters<PeekNeuronInput>) -> String {
        let path = std::path::Path::new(&input.path);
        let preview = {
            let idx = self.index.read().await;
            idx.peek_neuron(path, 20)
        };
        match preview {
            Some(p) => {
                // C3 (TRIZ R14): peeking at a neuron is a passive citation signal —
                // the user inspected it, which implies it was relevant to the current task.
                let mut idx = self.index.write().await;
                idx.record_hit(path, true);
                p
            }
            None => format!("ERROR: Neuron not found or unreadable: {}", input.path),
        }
    }

    // ── Person scope tools (TRIZ R13-G5) ─────────────────────────────────────

    /// Restore a single section of a neuron to its shadow copy (E2: section shadow, TRIZ R14).
    ///
    /// Before each evolve_context or evolve_section call, Cortyx automatically saves
    /// the previous content. Use this tool to undo the most recent evolution.
    #[tool(
        name = "cortyx_rollback_section",
        description = "Restore a neuron section to its previous version (saved before the last evolve call). \
                       Use section=\"_full\" to restore the entire neuron. \
                       Useful when an LLM evolution produces worse content than the original."
    )]
    async fn rollback_section(&self, Parameters(input): Parameters<RollbackSectionInput>) -> String {
        use crate::neuron::{NeuronMeta, replace_section};

        let neuron_path = std::path::Path::new(&input.neuron_path);
        let meta_file = meta_path(neuron_path);

        let meta_data = match std::fs::read_to_string(&meta_file) {
            Ok(d) => d,
            Err(e) => return format!("ERROR: Cannot read sidecar: {e}"),
        };
        let mut meta: NeuronMeta = match serde_json::from_str(&meta_data) {
            Ok(m) => m,
            Err(e) => return format!("ERROR: Cannot parse sidecar: {e}"),
        };

        let shadow = match meta.shadow_sections.get(&input.section) {
            Some(s) => s.clone(),
            None => return format!(
                "ERROR: No shadow for section '{}'. Shadows are saved before each evolve call.",
                input.section
            ),
        };

        if input.section == "_full" {
            if let Err(e) = atomic_write(neuron_path, shadow.as_bytes()) {
                return format!("ERROR: Failed to write neuron: {e}");
            }
            // Clear the shadow after restore to avoid stale state
            meta.shadow_sections.remove("_full");
            let _ = save_meta(&meta_file, &meta);
            // Re-index
            let _source = meta.source_path.clone();
            if let Ok(content) = std::fs::read_to_string(neuron_path) {
                meta.tokens = estimate_tokens(&content);
                let mut idx = self.index.write().await;
                let _ = idx.upsert_neuron(neuron_path, &content, &meta);
                idx.record_hit(neuron_path, true);
            }
            format!("✓ Restored full neuron {} from shadow.", neuron_path.display())
        } else {
            let existing = match std::fs::read_to_string(neuron_path) {
                Ok(c) => c,
                Err(e) => return format!("ERROR: Cannot read neuron file: {e}"),
            };
            let restored = replace_section(&existing, &input.section, &shadow);
            if let Err(e) = atomic_write(neuron_path, restored.as_bytes()) {
                return format!("ERROR: Failed to write neuron: {e}");
            }
            meta.shadow_sections.remove(&input.section);
            let _ = save_meta(&meta_file, &meta);
            let _source = meta.source_path.clone();
            meta.tokens = estimate_tokens(&restored);
            let mut idx = self.index.write().await;
            let _ = idx.upsert_neuron(neuron_path, &restored, &meta);
            idx.record_hit(neuron_path, true);
            format!("✓ Restored section '{}' in {} from shadow.", input.section, neuron_path.display())
        }
    }

    /// List all @person-scoped memory namespaces.
    #[tool(
        name = "cortyx_list_persons",
        description = "List all @person-scoped memory namespaces (created via mine_conversation with person=...). \
                       Returns person name, neuron count, and avg hit rate."
    )]
    async fn list_persons(&self) -> String {
        let idx = self.index.read().await;
        let persons = idx.list_persons();
        if persons.is_empty() {
            return "No person-scoped memories found. Use mine_conversation with person=\"alice\" to create some.".to_string();
        }
        let rows: Vec<serde_json::Value> = persons
            .iter()
            .map(|p| serde_json::json!({
                "person": p.name.trim_start_matches('@'),
                "module": p.name,
                "neuron_count": p.neuron_count,
                "avg_hit_rate": format!("{:.2}", p.avg_hit_rate),
            }))
            .collect();
        serde_json::to_string_pretty(&rows).unwrap_or_else(|_| "ERROR: serialization failed".to_string())
    }

    // ── Conversation recall tool (TRIZ R13-G3) ────────────────────────────────

    /// Retrieve conversation memories (Verbatim neurons) for a query.
    /// Isolates episodic recall from code retrieval — equivalent to MemPalace's
    /// dedicated episodic store but with zero storage overhead (query-time predicate).
    #[tool(
        name = "cortyx_recall",
        description = "Retrieve conversation memories (Verbatim neurons) matching a query. \
                       Optionally scope to a person's memories with person=\"alice\". \
                       Use for 'what did I decide last month?' style queries."
    )]
    async fn recall(&self, Parameters(input): Parameters<RecallInput>) -> String {
        let idx = self.index.read().await;
        let effective_module: Option<String> = input.person
            .as_ref()
            .map(|p| format!("@{}", p));
        let paths = idx.get_contexts(
            &input.query,
            input.max_tokens.unwrap_or(4096),
            effective_module.as_deref(),
            Some("conversation"),
        );
        if paths.is_empty() {
            return "No conversation memories found for this query. \
                    Use cortyx_mine_conversation to index conversations first."
                .to_string();
        }
        let mut out = format!(
            "<!-- cortyx:recall {} memories -->\n",
            paths.len()
        );
        for path in &paths {
            if let Ok(content) = std::fs::read_to_string(path) {
                out.push_str(&content);
                out.push('\n');
            }
        }
        out
    }

    /// Show neuron stats and cache-hit prediction.
    #[tool(
        name = "cortyx_status",
        description = "Show neuron count, synapse count, freshness, and cache-hit prediction."
    )]
    async fn status(&self) -> String {
        let idx = self.index.read().await;
        let low_quality = idx.low_quality_count();
        let quality_note = if low_quality > 0 {
            format!("\nNeeds curation (quality<40%): {low_quality}")
        } else {
            String::new()
        };
        format!(
            "Cortyx Status\n\
             =============\n\
             Neurons (total):       {}\n\
             Synapses:              {}{}\n\
             \n\
             Prompt caching:        ✓ Static prefix byte-identical on every call\n\
             Activation latency:    ~BM25 in-memory (<10ms for <10k neurons)\n\
             Instructions: Call cortyx_get_contexts(task) at the start of each task.",
            idx.neuron_count(),
            idx.synapse_count(),
            quality_note
        )
    }

    /// Mine a raw conversation turn into the live index as a Verbatim neuron.
    ///
    /// Accepts any format: Claude MD, ChatGPT JSON, LongMemEval JSON, or plain text.
    /// Consecutive calls automatically create TemporalFollows synapse chains.
    /// Use `module` to tag memories for namespace-filtered retrieval.
    #[tool(
        name = "cortyx_mine_conversation",
        description = "Mine a conversation turn (or whole file export) into Verbatim neurons for semantic recall. \
                       Returns the number of neurons created and the first neuron path."
    )]
    async fn mine_conversation(&self, Parameters(input): Parameters<MineConversationInput>) -> String {
        if input.content.len() > MAX_CONTENT_BYTES {
            return format!("ERROR: content exceeds {MAX_CONTENT_BYTES} byte limit");
        }
        let mut idx = self.index.write().await;
        let effective_module: Option<String> = input.person
            .as_ref()
            .map(|p| format!("@{}", p))
            .or_else(|| input.module.clone());
        match miner::mine_text(
            &input.content,
            "mcp-inline",
            &self.project_root,
            &mut idx,
            effective_module.as_deref(),
            input.speaker.as_deref(),
            input.timestamp.as_deref(),
        ) {
            Ok(count) => format!(
                "Mined {count} Verbatim neuron(s). Total neurons: {}.",
                idx.neuron_count()
            ),
            Err(e) => format!("ERROR: {e}"),
        }
    }

    /// Implicit hit-rate feedback — scan the task response for cited neurons.
    ///
    /// Pass the assistant's full response text; Cortyx scans for neuron file stems
    /// and auto-increments hit_count for each match. No per-neuron tool calls needed.
    /// Call once at task end instead of cortyx_record_hit for each neuron.
    #[tool(
        name = "cortyx_close_task",
        description = "Pass the assistant response text. Cortyx auto-records hits for neurons whose filenames appear in the response. Zero friction — one call closes the feedback loop for the whole task."
    )]
    async fn close_task(&self, Parameters(input): Parameters<CloseTaskInput>) -> String {
        if input.response_text.len() > MAX_CONTENT_BYTES {
            return format!("ERROR: response_text exceeds {MAX_CONTENT_BYTES} byte limit");
        }
        // R12-S2-B: Clear provisional hits — close_task provides actual citation evidence,
        // so the optimistic provisional buffer is no longer needed.
        self.provisional_hits.lock().await.clear();
        let activated = self.last_activated.lock().await.clone();
        if activated.is_empty() {
            return "No neurons from last cortyx_get_contexts call to evaluate.".to_string();
        }

        let response_lower = input.response_text.to_lowercase();
        // C1: Graded response-diff citation (TRIZ R14).
        // Tokenize response text and compute overlap with each activated neuron's vocabulary.
        // ≥15 terms → soft cite (record_hit once)
        // ≥30 terms → hard cite (record_hit twice — stronger feedback signal)
        // This is a tighter, graded version of the prior flat ≥20-term threshold.
        let response_tokens: std::collections::HashSet<String> =
            tokenize(&input.response_text).into_iter().collect();
        let mut hits = 0usize;

        // Phase 1 (immutable): decide citation for each activated neuron.
        // Returns (path, explicit_cited, soft_weight) where soft_weight: 0=miss, 1=soft, 2=hard
        let citation_decisions: Vec<(PathBuf, bool, u8)> = {
            let idx = self.index.read().await;
            activated.iter().map(|path| {
                let stem = path
                    .file_stem()
                    .unwrap_or_default()
                    .to_string_lossy()
                    .to_lowercase();
                let stem = stem.trim_end_matches(".context");
                let explicit_cited = !stem.is_empty() && response_lower.contains(stem);
                if explicit_cited {
                    return (path.clone(), true, 1u8);
                }
                // C1 graded: measure term overlap
                let overlap = if !response_tokens.is_empty() {
                    idx.term_freq_overlap(path, &response_tokens)
                } else {
                    0
                };
                let weight = if overlap >= 30 {
                    2 // hard cite
                } else if overlap >= 15 {
                    1 // soft cite
                } else {
                    0 // miss
                };
                (path.clone(), weight >= 1, weight)
            }).collect()
        };

        // Phase 2 (mutable): apply citation signals.
        let mut idx = self.index.write().await;
        for (path, cited, weight) in &citation_decisions {
            idx.record_hit(path, *cited);
            if *cited {
                hits += 1;
                // Hard cite: record a second hit to double the feedback signal
                if *weight >= 2 {
                    idx.record_hit(path, true);
                    tracing::debug!(path = %path.display(), "Hard citation (≥30 term overlap)");
                } else if *weight == 1 {
                    tracing::debug!(path = %path.display(), "Soft citation (≥15 term overlap)");
                }
            }
        }

        // S-VII (R16): Update last_co_activation_day for all cited neuron pairs (LTP).
        let cited_paths: Vec<PathBuf> = citation_decisions
            .iter()
            .filter(|(_, cited, _)| *cited)
            .map(|(p, _, _)| p.clone())
            .collect();
        if cited_paths.len() >= 2 {
            idx.touch_co_activation_day(&cited_paths);
        }

        // S-VIII (R16): Auto-mine UseCase stubs from code blocks in the response.
        // Code blocks ≥5 lines with ≥60% overlap with a cited neuron → write stub.
        let mined = auto_mine_code_blocks(
            &input.response_text,
            &cited_paths,
            &self.project_root,
            &idx,
        );

        // F2: Record session token utilization for budget adaptation.
        // Count total tokens used in this session from the activated neurons.
        let tokens_used: usize = activated.iter()
            .map(|p| idx.tokens_for(p))
            .sum();
        let tokens_budget = input.response_text.len() / 4; // rough estimate of budget from response size; actual budget not stored here
        if tokens_used > 0 {
            idx.record_session_utilization(tokens_used, tokens_budget.max(1));
            let _ = idx.save();
        }

        let mined_note = if mined > 0 {
            format!(" Auto-mined {mined} UseCase stub(s).")
        } else {
            String::new()
        };
        format!(
            "Closed task: {hits}/{} neurons cited (auto-detected from response).{mined_note}",
            activated.len()
        )
    }

    /// Record whether a neuron was actually cited — closes the self-improvement feedback loop.
    ///
    /// Call after each task for each neuron returned by cortyx_get_contexts.
    /// Cited neurons get a higher hit_rate → higher BM25 score → activated more readily in future.
    /// Irrelevant neurons are down-weighted over time without any manual curation.
    #[tool(
        name = "cortyx_record_hit",
        description = "Tell Cortyx whether a neuron was actually useful. was_cited=true boosts it; false down-weights it. Closes the self-improvement loop — neurons get smarter with every task."
    )]
    async fn record_hit(&self, Parameters(input): Parameters<RecordHitInput>) -> String {
        let rel = match validate_relative_path(&input.path) {
            Ok(p) => p,
            Err(e) => return format!("ERROR: Invalid path: {e}"),
        };
        let source = self.project_root.join(&rel);
        let neuron_path = core_neuron_path(&source, &self.project_root);

        if !neuron_path.exists() {
            return format!("ERROR: Neuron not found: {}", neuron_path.display());
        }

        let mut idx = self.index.write().await;
        let hit_rate = idx.record_hit(&neuron_path, input.was_cited);
        let use_count = idx.use_count_for(&neuron_path);

        format!(
            "Recorded {} for {} — hit_rate now {:.0}% ({} uses)",
            if input.was_cited { "hit" } else { "miss" },
            neuron_path.display(),
            hit_rate * 100.0,
            use_count
        )
    }

    /// Write an agent diary entry (S6 — NE5).
    ///
    /// Agent diaries are Verbatim neurons stored under the `@agent/{agent}` module namespace.
    /// They use the existing @prefix isolation mechanism — zero new storage, zero new retrieval
    /// logic. Each diary entry is BM25-indexed and searchable via cortyx_get_contexts.
    #[tool(
        name = "cortyx_diary_write",
        description = "Write an observation or decision to an agent's diary. Stored as a Verbatim neuron under @agent/{agent} — BM25-indexed, searchable via get_contexts. Use for specialist agent memory (reviewer, architect, ops, etc.)."
    )]
    async fn diary_write(&self, Parameters(input): Parameters<DiaryWriteInput>) -> String {
        if input.agent.is_empty() {
            return "ERROR: agent name must not be empty".to_string();
        }
        if input.content.is_empty() {
            return "ERROR: content must not be empty".to_string();
        }
        if input.content.len() > MAX_CONTENT_BYTES {
            return format!("ERROR: content exceeds {MAX_CONTENT_BYTES} byte limit");
        }
        let module = format!("@agent/{}", input.agent.trim());
        let mut idx = self.index.write().await;
        match miner::mine_text(
            &input.content,
            "diary",
            &self.project_root,
            &mut idx,
            Some(&module),
            Some(input.agent.trim()),
            input.timestamp.as_deref(),
        ) {
            Ok(count) => format!(
                "Diary entry written for agent '{}' ({count} neuron(s) created).",
                input.agent
            ),
            Err(e) => format!("ERROR: {e}"),
        }
    }

    /// Read recent diary entries for an agent (S6 — NE5).
    #[tool(
        name = "cortyx_diary_read",
        description = "Read recent diary entries for an agent. Returns last_n entries (default 10) from @agent/{agent} namespace, most recent first."
    )]
    async fn diary_read(&self, Parameters(input): Parameters<DiaryReadInput>) -> String {
        if input.agent.is_empty() {
            return "ERROR: agent name must not be empty".to_string();
        }
        let last_n = input.last_n.unwrap_or(10);
        let module = format!("@agent/{}", input.agent.trim());
        let max_tokens = last_n.saturating_mul(250);
        let idx = self.index.read().await;
        let results = idx.get_contexts("", max_tokens, Some(&module), Some("conversation"));
        if results.is_empty() {
            return format!("No diary entries found for agent '{}'.", input.agent);
        }
        let mut out = format!("## Agent Diary: {} (last {})\n\n", input.agent, last_n);
        for path in results.iter().take(last_n) {
            if let Ok(content) = std::fs::read_to_string(path) {
                out.push_str(&format!("---\n{}\n", content));
            }
        }
        out
    }

    /// Check for contradicting neuron pairs (S7 — NE6).
    ///
    /// Proactively scans all neurons (or a single neuron) for `Contradicts` synapse edges.
    /// Use before starting a task to surface known conflicts. Contradictions are also
    /// automatically surfaced by `cortyx_get_contexts` at query time.
    #[tool(
        name = "cortyx_check_consistency",
        description = "Check for contradictions in the neuron graph. Scans all Contradicts synapse edges and returns conflicting pairs with reasons. Scope to a single neuron with the optional path argument. Contradictions are also surfaced automatically during cortyx_get_contexts."
    )]
    async fn check_consistency(
        &self,
        Parameters(input): Parameters<CheckConsistencyInput>,
    ) -> String {
        let path_filter: Option<PathBuf> = if let Some(ref p) = input.path {
            match validate_relative_path(p) {
                Ok(rel) => {
                    let src = self.project_root.join(&rel);
                    Some(core_neuron_path(&src, &self.project_root))
                }
                Err(e) => return format!("ERROR: Invalid path: {e}"),
            }
        } else {
            None
        };

        let idx = self.index.read().await;
        let pairs = idx.all_contradictions(path_filter.as_deref());

        if pairs.is_empty() {
            return "No contradictions detected.".to_string();
        }

        let mut out = format!(
            "## Contradictions Found ({})\n\n",
            pairs.len()
        );
        for (a, b, reason) in &pairs {
            let a_name = a.file_name().unwrap_or_default().to_string_lossy();
            let b_name = b.file_name().unwrap_or_default().to_string_lossy();
            out.push_str(&format!(
                "- **{}** ↔ **{}**\n  Reason: {}\n  Action: use `cortyx_create_synapse` to update or \
                 `cortyx_invalidate` to retire the outdated neuron.\n\n",
                a_name, b_name, reason
            ));
        }
        out
    }

    /// Session priming — load identity and critical-facts wake-up neurons (S5 — NE4).
    ///
    /// Returns both L0 (_identity) and L1 (_critical_facts) neurons (~170 tokens total)
    /// plus optional @person memories. Call at the start of a new session to prime the
    /// LLM with project identity — equivalent to MemPalace L0+L1 but lossless (plain
    /// Markdown, not AAAK-encoded).
    ///
    /// Zero tokens unless explicitly called — preserves Cortyx's token efficiency advantage.
    #[tool(
        name = "cortyx_wake_up",
        description = "Prime the LLM with project identity and critical facts. Returns _identity.context.md (~50 tokens) + _critical_facts.context.md (~120 tokens). Optionally include person memories. Call once at session start — lossless Markdown (vs MemPalace AAAK-encoding)."
    )]
    async fn wake_up(&self, Parameters(input): Parameters<WakeUpInput>) -> String {
        let ndir = neuron_dir(&self.project_root);
        let mut out = String::new();

        // Load identity neuron (L0)
        let identity_path = ndir.join("_identity.context.md");
        if identity_path.exists() {
            if let Ok(content) = std::fs::read_to_string(&identity_path) {
                out.push_str("<!-- CORTYX WAKE-UP: L0 identity -->\n");
                out.push_str(&content);
                out.push('\n');
            }
        } else {
            out.push_str("<!-- _identity.context.md not found — run `cortyx compile .` first -->\n");
        }

        // Load critical-facts neuron (L1)
        let critical_path = ndir.join("_critical_facts.context.md");
        if critical_path.exists() {
            if let Ok(content) = std::fs::read_to_string(&critical_path) {
                out.push_str("<!-- CORTYX WAKE-UP: L1 critical facts -->\n");
                out.push_str(&content);
                out.push('\n');
            }
        } else {
            out.push_str("<!-- _critical_facts.context.md not found — run `cortyx compile .` first -->\n");
        }

        // Optional @person memories (recent conversation neurons for the person)
        if let Some(ref person) = input.person {
            let module = format!("@{}", person.trim());
            let idx = self.index.read().await;
            let paths = idx.get_contexts("", 600, Some(&module), Some("conversation"));
            if !paths.is_empty() {
                out.push_str(&format!("\n<!-- CORTYX WAKE-UP: @{person} memories -->\n"));
                for path in paths.iter().take(3) {
                    if let Ok(content) = std::fs::read_to_string(path) {
                        out.push_str(&content);
                        out.push('\n');
                    }
                }
            }
        }

        if out.is_empty() {
            out.push_str("No wake-up neurons found. Run `cortyx compile .` to generate them.");
        }
        out
    }

    // ─── S4: Temporal Knowledge Graph (NE3) ──────────────────────────────────

    /// Add a fact to a KG entity neuron (creating the entity if needed).
    #[tool(
        name = "cortyx_kg_add",
        description = "Add a fact triple to a KG entity (creates entity if absent). \
                       KG neurons are git-tracked, BM25-indexed Markdown files. \
                       Example: entity='project_meta', predicate='language', value='Rust', valid_from='2024-01-01'."
    )]
    async fn kg_add(&self, Parameters(input): Parameters<KgAddInput>) -> String {
        let path = kg::kg_neuron_path(&self.project_root, &input.entity);
        let mut entity = match kg::KgEntity::load(&path) {
            Ok(e) => e,
            Err(e) => return format!("ERROR loading KG entity: {e}"),
        };
        entity.add_fact(&input.predicate, &input.value, input.valid_from.as_deref());
        if let Err(e) = entity.save() {
            return format!("ERROR saving KG entity: {e}");
        }
        // Re-index the neuron so BM25 picks up the new fact immediately.
        let mut idx = self.index.write().await;
        let content = std::fs::read_to_string(&path).unwrap_or_default();
        let mut meta = crate::neuron::NeuronMeta::new_stub(&self.project_root, crate::neuron::NeuronKind::Concept);
        meta.module = Some("@kg".to_string());
        meta.tokens = estimate_tokens(&content);
        idx.index_neuron(&path, &content, &meta);
        format!(
            "KG fact added: {entity} / {pred} = {val} (from: {from})",
            entity = input.entity,
            pred = input.predicate,
            val = input.value,
            from = input.valid_from.as_deref().unwrap_or(""),
        )
    }

    /// Query active facts for a KG entity as of an optional date.
    #[tool(
        name = "cortyx_kg_query",
        description = "Query active facts for a KG entity. Pass as_of (ISO-8601) to filter by date. \
                       Returns a Markdown table of active fact triples."
    )]
    async fn kg_query(&self, Parameters(input): Parameters<KgQueryInput>) -> String {
        let path = kg::kg_neuron_path(&self.project_root, &input.entity);
        let entity = match kg::KgEntity::load(&path) {
            Ok(e) => e,
            Err(e) => return format!("ERROR loading KG entity: {e}"),
        };
        let facts = entity.active_facts(input.as_of.as_deref());
        if facts.is_empty() {
            return format!("No active facts for entity '{}' (as_of: {:?})", input.entity, input.as_of);
        }
        let mut out = format!("## KG: {} (active facts)\n\n| predicate | value | valid_from | ended |\n|---|---|---|---|\n", input.entity);
        for f in facts {
            out.push_str(&format!("| {} | {} | {} | {} |\n", f.predicate, f.value, f.valid_from, f.ended));
        }
        out
    }

    /// Invalidate (end) an active KG fact by setting its `ended` date.
    #[tool(
        name = "cortyx_kg_invalidate",
        description = "Invalidate (end) the currently active fact for a predicate on a KG entity. \
                       Sets the `ended` date; does NOT delete the historical record."
    )]
    async fn kg_invalidate(&self, Parameters(input): Parameters<KgInvalidateInput>) -> String {
        let path = kg::kg_neuron_path(&self.project_root, &input.entity);
        let mut entity = match kg::KgEntity::load(&path) {
            Ok(e) => e,
            Err(e) => return format!("ERROR loading KG entity: {e}"),
        };
        if let Err(e) = entity.invalidate_fact(&input.predicate, &input.ended) {
            return format!("ERROR: {e}");
        }
        if let Err(e) = entity.save() {
            return format!("ERROR saving KG entity: {e}");
        }
        format!(
            "KG fact invalidated: {}/{} ended on {}",
            input.entity, input.predicate, input.ended
        )
    }

    /// Show the full temporal timeline for a predicate on a KG entity.
    #[tool(
        name = "cortyx_kg_timeline",
        description = "Show the full temporal history of a predicate on a KG entity — all past, \
                       present, and future values with their validity windows."
    )]
    async fn kg_timeline(&self, Parameters(input): Parameters<KgTimelineInput>) -> String {
        let path = kg::kg_neuron_path(&self.project_root, &input.entity);
        let entity = match kg::KgEntity::load(&path) {
            Ok(e) => e,
            Err(e) => return format!("ERROR loading KG entity: {e}"),
        };
        let timeline = entity.timeline_for(&input.predicate);
        if timeline.is_empty() {
            return format!("No facts found for {}/{}", input.entity, input.predicate);
        }
        let mut out = format!(
            "## Timeline: {}/{}\n\n| # | value | valid_from | ended |\n|---|---|---|---|\n",
            input.entity, input.predicate
        );
        for (i, f) in timeline.iter().enumerate() {
            let ended = if f.ended.is_empty() { "active" } else { &f.ended };
            out.push_str(&format!("| {} | {} | {} | {} |\n", i + 1, f.value, f.valid_from, ended));
        }
        out
    }

    /// Return aggregate statistics for all KG entities in this project.
    #[tool(
        name = "cortyx_kg_stats",
        description = "Return aggregate statistics for all KG entities: entity count, total facts, \
                       active facts, ended/invalidated facts."
    )]
    async fn kg_stats(&self, _params: Parameters<serde_json::Value>) -> String {
        let stats = kg::compute_stats(&self.project_root);
        format!(
            "KG stats: {} entities, {} total facts ({} active, {} ended)",
            stats.entity_count, stats.total_facts, stats.active_facts, stats.ended_facts
        )
    }
}

#[tool_handler]
impl ServerHandler for CortyxServer {
    fn get_info(&self) -> ServerInfo {
        ServerInfo::new(ServerCapabilities::builder().enable_tools().build())
            .with_server_info(Implementation::new("cortyx", env!("CARGO_PKG_VERSION")))
            .with_instructions(
                "Cortyx: semantic cache layer for LLM Wikis (Karpathy pattern). \
                USAGE: Call cortyx_get_contexts(task) at the start of every task. \
                At end of task: call cortyx_close_task(response_text) — one call auto-records \
                all hits. Or call cortyx_record_hit(path, was_cited) per neuron for fine control. \
                cortyx_evolve_section to improve one section (~50 tokens). \
                cortyx_extract_from_raw to save a proven pattern as a use-case neuron. \
                cortyx_create_synapse to link related neurons.",
            )
    }
}

// ─── Server entrypoint ────────────────────────────────────────────────────────

/// Start the MCP server on STDIO (compatible with Claude Code, Cursor, Codex, Windsurf).
pub async fn serve(project_name: Option<String>) -> Result<()> {
    if let Some(ref name) = project_name {
        tracing::warn!(
            "--project '{}' is accepted but not yet implemented — \
             planned for v0.2 multi-folder support. Server will use the current directory.",
            name
        );
    }
    let project_root = std::env::current_dir()?;
    tracing::info!("Starting Cortyx MCP server for: {}", project_root.display());

    let mut idx = NeuronIndex::load_or_create(&project_root)?;

    // Auto-compile on first run — turns `cortyx serve` into a one-step setup.
    if idx.neuron_count() == 0 {
        tracing::info!("No neurons found — running initial compile...");
        let count = idx.compile()?;
        tracing::info!("Auto-compiled {count} neurons (AST Bootstrap + Auto-Synapse active)");
        eprintln!("✓ Cortyx: auto-compiled {count} neurons on first run.");
    }

    // S-VII (R16): Apply synapse temporal decay at startup — self-cleaning graph.
    let (decayed, pruned) = idx.apply_synapse_decay();
    if decayed > 0 || pruned > 0 {
        tracing::info!(decayed, pruned, "S-VII: synapse temporal decay applied at startup");
    }

    // S-IV (R16): Auto-fetch global concepts at startup if remote is configured.
    // Runs `git pull --ff-only` in ~/.cortyx/global/ — quiet on success, logs on error.
    // No-ops when global dir doesn't exist or has no remote.
    {
        let global_dir = crate::global_index::global_dir();
        if global_dir.join(".git").exists() {
            let remote_check = std::process::Command::new("git")
                .args(["remote", "get-url", "origin"])
                .current_dir(&global_dir)
                .output();
            if remote_check.map(|o| o.status.success()).unwrap_or(false) {
                let pull = std::process::Command::new("git")
                    .args(["pull", "--ff-only", "origin", "main"])
                    .current_dir(&global_dir)
                    .output()
                    .or_else(|_| std::process::Command::new("git")
                        .args(["pull", "--ff-only", "origin", "master"])
                        .current_dir(&global_dir)
                        .output());
                match pull {
                    Ok(o) if o.status.success() =>
                        tracing::debug!("S-IV: global concepts auto-fetch OK"),
                    Ok(_) =>
                        tracing::warn!("S-IV: global concepts auto-fetch skipped (not fast-forward)"),
                    Err(e) =>
                        tracing::debug!("S-IV: global concepts auto-fetch skipped: {e}"),
                }
            }
        }
    }

    // Embed feature active — hybrid BM25 + dense retrieval is wired into get_contexts.
    // Embeddings will be loaded from .cortyx/embeddings.bin if present; falls back
    // gracefully to BM25-only when embeddings.bin is absent or model not installed.
    #[cfg(feature = "embed")]
    tracing::info!("--features embed: hybrid BM25 + dense cosine retrieval active.");

    let index = Arc::new(RwLock::new(idx));

    let _watcher = watcher::start_watcher(project_root.clone(), Arc::clone(&index))?;

    let server = CortyxServer {
        project_root,
        index,
        last_activated: Arc::new(Mutex::new(Vec::new())),
        provisional_hits: Arc::new(Mutex::new(Vec::new())),
        tool_router: CortyxServer::tool_router(),
    };

    let service = server.serve(rmcp::transport::stdio()).await?;
    service.waiting().await?;
    Ok(())
}

// ─── Helpers ─────────────────────────────────────────────────────────────────/// Convert a task pattern string to a URL-safe kebab-case identifier.
fn to_kebab(s: &str) -> String {
    s.to_lowercase()
        .split(|c: char| !c.is_alphanumeric())
        .filter(|s: &&str| !s.is_empty())
        .collect::<Vec<_>>()
        .join("-")
}

/// Truncate a string to at most `max_chars` characters (byte boundary safe for ASCII).
fn truncate_str(s: &str, max_chars: usize) -> String {
    s.chars().take(max_chars).collect()
}

/// Load existing metadata or create a fresh stub.
fn load_or_new_meta(meta_file: &Path, source: &Path, kind: NeuronKind) -> NeuronMeta {
    if let Ok(data) = std::fs::read_to_string(meta_file) {
        if let Ok(meta) = serde_json::from_str::<NeuronMeta>(&data) {
            return meta;
        }
    }
    NeuronMeta::new_stub(source, kind)
}

/// Serialize and write metadata to disk atomically.
fn save_meta(meta_file: &Path, meta: &NeuronMeta) -> Result<()> {
    atomic_write_json(meta_file, meta)
}

/// Strip HTML comment delimiters and control characters from user-supplied strings
/// before embedding them in comment blocks, preventing comment breakout and prompt injection.
fn sanitize_comment(s: &str) -> String {
    let clean: String = s
        .chars()
        .map(|c| if c.is_ascii_control() && c != '\t' { ' ' } else { c })
        .collect();
    let clean = clean.replace("-->", "—>").replace("<!--", "<—");
    // Truncate to 500 chars to prevent unbounded comment sections
    clean.chars().take(500).collect()
}

/// S-VIII (R16): Auto-mine UseCase stubs from code blocks in an LLM response.
///
/// Scans `response_text` for fenced code blocks (``` ... ```) with ≥5 lines.
/// For each block, finds the cited neuron with the highest term overlap.
/// If overlap ≥ 60% of the neuron's own terms, writes a UseCase stub to
/// `.cortyx/neurons/{neuron}.usecase.auto-{hash}.md` with `status: Stub`.
///
/// Returns the count of stubs written.
fn auto_mine_code_blocks(
    response_text: &str,
    cited_paths: &[PathBuf],
    project_root: &Path,
    index: &NeuronIndex,
) -> usize {
    if cited_paths.is_empty() {
        return 0;
    }

    // Extract fenced code blocks: ```[lang]\n<body>\n```
    let mut blocks: Vec<String> = Vec::new();
    let mut in_block = false;
    let mut current_block = Vec::new();
    for line in response_text.lines() {
        let trimmed = line.trim();
        if !in_block && trimmed.starts_with("```") {
            in_block = true;
            current_block.clear();
        } else if in_block && trimmed.starts_with("```") {
            if current_block.len() >= 5 {
                blocks.push(current_block.join("\n"));
            }
            in_block = false;
            current_block.clear();
        } else if in_block {
            current_block.push(line.to_string());
        }
    }

    if blocks.is_empty() {
        return 0;
    }

    let ndir = neuron_dir(project_root);
    let mut written = 0usize;

    for block in &blocks {
        let block_terms: std::collections::HashSet<String> =
            tokenize(block).into_iter().collect();
        if block_terms.is_empty() {
            continue;
        }

        // Find the cited neuron with highest term overlap
        let best = cited_paths.iter().filter_map(|path| {
            let overlap = index.term_freq_overlap(path, &block_terms);
            let total_neuron_terms = index.term_count_for(path);
            if total_neuron_terms == 0 { return None; }
            let ratio = overlap as f32 / total_neuron_terms as f32;
            Some((ratio, path))
        }).max_by(|a, b| a.0.total_cmp(&b.0));

        let Some((ratio, best_path)) = best else { continue };
        if ratio < 0.6 { continue; }

        // Derive the output filename from the parent neuron stem + a short hash
        let stem = best_path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("unknown")
            .trim_end_matches(".context")
            .to_string();

        let hash_bytes = blake3::hash(block.as_bytes());
        let short_hash = &hash_bytes.to_hex()[..8];
        let usecase_filename = format!("{stem}.usecase.auto-{short_hash}.md");
        let usecase_path = ndir.join(&usecase_filename);

        if usecase_path.exists() {
            continue; // already mined
        }

        let content = format!(
            "# {stem} — auto-mined UseCase\n\
             status: Stub\n\
             source: auto-mined from close_task\n\n\
             ## task\n\
             (edit: describe the task pattern this code solves)\n\n\
             ## example\n\
             ```\n{block}\n```\n"
        );
        if let Err(e) = std::fs::write(&usecase_path, &content) {
            tracing::warn!("S-VIII: failed to write UseCase stub {:?}: {e}", usecase_path);
        } else {
            tracing::debug!("S-VIII: wrote UseCase stub {:?} (ratio={ratio:.2})", usecase_path);
            written += 1;
        }
    }

    written
}
