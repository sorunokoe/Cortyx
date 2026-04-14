use clap::{Parser, Subcommand, ValueEnum};
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "cortyx", version, about = "MCP-native semantic cache layer for LLM Wikis")]
pub struct Cli {
    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Copy, Clone, PartialEq, Eq, ValueEnum, Debug)]
pub enum Provider {
    Anthropic,
    Openai,
}

#[derive(Subcommand)]
pub enum Commands {
    /// Start the MCP server (STDIO transport — works with Claude Code, Cursor, Codex)
    Serve {
        /// Optional project name for multi-folder context sharing
        #[arg(short, long)]
        project: Option<String>,
    },
    /// Scan a folder and create neuron stubs (.context.md files)
    Compile {
        /// Path to scan (defaults to current directory)
        path: Option<PathBuf>,
        /// Only re-process files listed in .cortyx/dirty.json (written by the watcher).
        /// Much faster on large repos when only a few files changed.
        #[arg(long)]
        incremental: bool,
    },
    /// Show neuron status, token estimates, and cache-hit prediction
    Status {
        /// Path to inspect (defaults to current directory)
        path: Option<PathBuf>,
    },
    /// Force a neuron to be marked stale so it gets re-evaluated on next use
    Invalidate {
        /// Source file whose neuron to invalidate
        file: PathBuf,
    },
    /// Export a ready-to-paste prompt JSON with cache_control breakpoint
    Export {
        /// Target LLM provider format
        #[arg(long, value_enum, default_value = "anthropic")]
        provider: Provider,
        /// Output file (defaults to stdout)
        #[arg(short, long)]
        output: Option<PathBuf>,
        /// Project root (defaults to current directory)
        path: Option<PathBuf>,
    },
    /// Mine conversation files into Verbatim neurons
    Mine {
        /// File or directory to mine (JSON/MD conversation exports)
        path: PathBuf,
        /// Tag all mined neurons with a module name for filtered queries
        #[arg(long)]
        module: Option<String>,
    },
    /// Run the file watcher daemon (keeps neurons fresh as sources change)
    Watch {
        /// Path to watch (defaults to current directory)
        path: Option<PathBuf>,
    },
    /// Diagnose the Cortyx installation and index health
    Doctor {
        /// Project root to inspect (defaults to current directory)
        path: Option<PathBuf>,
        /// Output machine-readable JSON (for CI integration)
        #[arg(long)]
        json: bool,
    },
    /// Remove unused or outdated neurons from the index to keep it lean
    Prune {
        /// Project root (defaults to current directory)
        path: Option<PathBuf>,
        /// Remove neurons activated fewer than N times (default: 0 = never activated)
        #[arg(long, default_value = "0")]
        min_use: u32,
        /// Remove neurons whose neuron file is older than N days (e.g. 90)
        #[arg(long)]
        older_than: Option<u64>,
        /// Dry run — list what would be removed without deleting anything
        #[arg(long)]
        dry_run: bool,
    },
    /// Retrieve the top neurons for a task query and print their content to stdout.
    /// Useful for scripting, CI benchmarks, and debugging retrieval quality.
    GetContexts {
        /// The task query to retrieve neurons for
        #[arg(long)]
        task: String,
        /// Maximum token budget for the retrieved context
        #[arg(long, default_value = "4000")]
        max_tokens: usize,
        /// Optional module filter (e.g. "auth", "@alice")
        #[arg(long)]
        module: Option<String>,
        /// Kind filter: "code", "conversation", or "all" (default: "all")
        #[arg(long)]
        kind: Option<String>,
        /// Minimum BM25 confidence score to return results (abstention threshold).
        /// When set, returns "(no neurons matched — confidence below threshold)" if the
        /// top retrieval score is below this value. Recommended for LongMemEval "absent"
        /// questions: use 0.5. Disabled by default (normal behaviour).
        #[arg(long)]
        min_confidence: Option<f64>,
        /// Enable 2-hop retrieval: after Phase 1, expand the query using the top
        /// result's concept cloud and vocabulary, then run a second retrieval pass
        /// for indirectly-related neurons. Targets LME-500 multi-session questions.
        /// Increases latency by ~2× for conversational queries. Default: off.
        #[arg(long, default_value = "false")]
        multi_hop: bool,
        /// Project root (defaults to current directory)
        path: Option<PathBuf>,
    },
    /// Restore a neuron file to its previous git version (E1: git version store).
    ///
    /// Runs `git checkout HEAD~1 -- <neuron_path>` for the given neuron.
    /// Requires the project to be a git repository with the neurons directory tracked.
    Rollback {
        /// Neuron file path to restore (e.g. ".cortyx/neurons/src/engine_rs.context.md")
        neuron: PathBuf,
    },
    /// Restore a single section of a neuron to its shadow copy (E2: section shadow).
    ///
    /// Before each `cortyx_evolve_context` or `cortyx_evolve_section` call, Cortyx
    /// saves the previous content in a shadow field inside the sidecar JSON.
    /// This command restores that shadow for a named section — instant undo.
    RollbackSection {
        /// Neuron file path (e.g. ".cortyx/neurons/src/engine_rs.context.md")
        neuron: PathBuf,
        /// Section to restore: "purpose", "api", "pitfalls", etc., or "_full" for the whole neuron
        section: String,
    },
    /// Publish a neuron to the global concept library (~/.cortyx/global/).
    ///
    /// Published concepts are project-agnostic neurons describing universal patterns
    /// (JWT auth, BM25 retrieval, event sourcing, etc.). They are injected in Phase 3
    /// of get_contexts when the local index has fewer than 3 high-confidence results.
    /// Published neurons are read-only from the global layer's perspective.
    PublishConcept {
        /// Path to the neuron file to publish
        neuron: PathBuf,
    },
    /// List all neurons published in the global concept library.
    ListConcepts,
    /// Auto-configure all detected LLM clients with the Cortyx MCP server entry.
    ///
    /// Detects Claude Code, Cursor, Windsurf, and Codex configs in standard paths.
    /// Writes `cortyx serve` MCP entry to each. Also writes Claude Code hook scripts
    /// for auto-save (Stop + PreCompact events). Idempotent — safe to run multiple times.
    Install {
        /// Register system-wide (searches HOME). Pass --global to force detection even
        /// if no client configs are found.
        #[arg(long)]
        global: bool,
    },
    /// Hook called by the Claude Code Stop event to commit pending feedback (S3 — NE2).
    ///
    /// Reads the latest BM25 index and commits any provisional hits that accumulated
    /// during the session. Called via the hook script written by `cortyx install`.
    /// Safe to call manually: no-ops if no provisional hits are pending.
    #[command(name = "close-task-hook")]
    CloseTaskHook {
        /// Project root (defaults to current directory)
        #[arg(long)]
        project: Option<PathBuf>,
    },
    /// Manage the git-federated global concept library (S-IV, TRIZ R16).
    ///
    /// The concept library lives at `~/.cortyx/global/` and is a plain git repository.
    /// Teams share universal concept neurons (JWT, BM25, event-sourcing, etc.) via git —
    /// zero server, works offline. Deduplication prevents duplicate concepts.
    #[command(subcommand)]
    Concepts(ConceptsCommand),
}

/// Sub-commands for `cortyx concepts`
#[derive(Subcommand)]
pub enum ConceptsCommand {
    /// Initialize the global concept library and optionally add a git remote.
    ///
    /// Creates `~/.cortyx/global/` as a git repo (if not already). When `--remote`
    /// is provided, adds it as the `origin` remote for push/pull.
    Init {
        /// Git remote URL (optional, e.g. git@github.com:org/cortyx-concepts.git)
        #[arg(long)]
        remote: Option<String>,
    },
    /// Pull latest concepts from the remote.
    ///
    /// Runs `git fetch && git merge --ff-only` in `~/.cortyx/global/`.
    /// Fast-forward only to prevent accidental conflicts in the shared library.
    Pull,
    /// Push local concepts to the remote.
    ///
    /// Runs `git push origin main` (or `master`) in `~/.cortyx/global/`.
    Push,
    /// Show the status of the global concept library (remote, commit, neuron count).
    Status,
}
