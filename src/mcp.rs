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
use tokio::sync::RwLock;

use crate::index::NeuronIndex;
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
    // Kept for the rmcp macro-generated dispatch table; not called directly.
    #[allow(dead_code)]
    tool_router: ToolRouter<Self>,
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
        description = "Get the most relevant context neurons for a task. Returns 3-5 .context.md files, sorted deterministically. Inject after your cache_control breakpoint to keep the static prefix byte-identical for prompt caching."
    )]
    async fn get_contexts(&self, Parameters(input): Parameters<GetContextsInput>) -> String {
        if input.task.len() > MAX_TASK_BYTES {
            return format!("ERROR: task exceeds {MAX_TASK_BYTES} byte limit");
        }
        let max_tokens = input.max_tokens.unwrap_or(4096);
        let paths = {
            let idx = self.index.read().await;
            idx.get_contexts(&input.task, max_tokens, input.module.as_deref())
        };

        // Increment use_count for all returned neurons — activates the feedback loop.
        if !paths.is_empty() {
            let mut idx = self.index.write().await;
            idx.record_activation(&paths);
        }

        if paths.is_empty() {
            return "No relevant neurons found. Run `cortyx compile .` first, then call \
                cortyx_evolve_context to fill stubs."
                .to_string();
        }

        let mut out = format!(
            "<!-- CORTYX CONTEXT — injected after cache_control breakpoint -->\n\
             <!-- Task: {} -->\n\
             <!-- Neurons activated: {} -->\n\n",
            sanitize_comment(&input.task),
            paths.len()
        );

        for path in &paths {
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
        }
        out.push_str("<!-- === END CORTYX CONTEXT === -->\n");
        out
    }

    /// Rewrite a neuron with improved content (self-improvement during normal usage).
    #[tool(
        name = "cortyx_evolve_context",
        description = "Evolve (rewrite) a neuron with AI-curated content. Call after a task reveals better reasoning instructions, pitfalls, or cross-references. Atomically updates the .context.md file and refreshes the index."
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

    /// Show neuron stats and cache-hit prediction.
    #[tool(
        name = "cortyx_status",
        description = "Show neuron count, synapse count, freshness, and cache-hit prediction."
    )]
    async fn status(&self) -> String {
        let idx = self.index.read().await;
        format!(
            "Cortyx Status\n\
             =============\n\
             Neurons (total):       {}\n\
             Synapses:              {}\n\
             \n\
             Prompt caching:        ✓ Static prefix byte-identical on every call\n\
             Activation latency:    ~BM25 in-memory (<10ms for <10k neurons)\n\
             Instructions: Call cortyx_get_contexts(task) at the start of each task.",
            idx.neuron_count(),
            idx.synapse_count()
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
        match miner::mine_text(
            &input.content,
            "mcp-inline",
            &self.project_root,
            &mut idx,
            input.module.as_deref(),
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
}

#[tool_handler]
impl ServerHandler for CortyxServer {
    fn get_info(&self) -> ServerInfo {
        ServerInfo::new(ServerCapabilities::builder().enable_tools().build())
            .with_server_info(Implementation::new("cortyx", env!("CARGO_PKG_VERSION")))
            .with_instructions(
                "Cortyx: semantic cache layer for LLM Wikis (Karpathy pattern). \
                USAGE: Call cortyx_get_contexts(task) at the start of every task. \
                After the task: call cortyx_record_hit(path, was_cited) for each neuron \
                to close the self-improvement loop. \
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

    // E-2: Warn users who compiled with --features embed that dense retrieval
    // is not yet wired into get_contexts (BM25-only path is always used).
    #[cfg(feature = "embed")]
    {
        tracing::warn!(
            "--features embed: dense retrieval is experimental and not yet wired into \
             get_contexts. BM25-only retrieval is active. Remove --features embed to \
             skip the 80MB model download until v0.2 wires it."
        );
        eprintln!(
            "⚠ Cortyx: --features embed is experimental — dense retrieval not yet active. \
             BM25-only mode. See docs for v0.2 roadmap."
        );
    }

    let index = Arc::new(RwLock::new(idx));

    let _watcher = watcher::start_watcher(project_root.clone(), Arc::clone(&index))?;

    let server = CortyxServer {
        project_root,
        index,
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
