//! CLI argument parsing and command definitions.
//!
//! Defines the command-line interface for Cortyx using clap.

use clap::{Parser, Subcommand, ValueEnum};
use std::path::PathBuf;

/// Cortyx CLI root command.
#[derive(Parser)]
#[command(
    name = "cortyx",
    version,
    about = "MCP-native context delivery engine for coding agents and long-lived memory"
)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Copy, Clone, PartialEq, Eq, ValueEnum, Debug)]
pub enum Provider {
    Anthropic,
    Openai,
}

#[derive(Copy, Clone, PartialEq, Eq, ValueEnum, Debug)]
pub enum RouteIntent {
    Auto,
    Context,
    Answer,
    WakeUp,
    AgentStatus,
    Consistency,
    Capabilities,
}

#[derive(Subcommand)]
pub enum Commands {
    /// Start the MCP server (STDIO transport — works with Claude Code, Cursor, Codex)
    Serve {
        /// Path to the project root (defaults to current directory)
        #[arg(short, long)]
        project: Option<PathBuf>,
        /// Disable feedback writes for reproducible retrieval.
        #[arg(long)]
        frozen: bool,
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
    /// Show neuron status, token estimates, cache-hit prediction, or collaboration summaries
    Status {
        /// Path to inspect (defaults to current directory)
        path: Option<PathBuf>,
        /// Show collaboration-kernel status instead of raw index counters.
        #[arg(long, default_value = "false")]
        collaboration: bool,
        /// Filter collaboration status to one agent.
        #[arg(long)]
        agent: Option<String>,
        /// Filter collaboration status to one shared module.
        #[arg(long)]
        module: Option<String>,
        /// Include recent collaboration timeline events.
        #[arg(long, default_value = "false")]
        include_timeline: bool,
    },
    /// Force a neuron to be marked stale so it gets re-evaluated on next use
    Invalidate {
        /// Source file whose neuron to invalidate
        file: PathBuf,
    },
    /// Export a ready-to-paste prompt JSON with cache_control breakpoint.
    ///
    /// Includes `_cortyx_meta.quickstart` and `_cortyx_meta.ux_proof` so
    /// onboarding, recovery, and one-entrypoint coverage stay reproducible.
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
    /// Evaluate and route a single tool observation to Verbatim, diary, or discard.
    MineObservation {
        /// Tool name (e.g. "Edit", "Bash", "Write")
        #[arg(long)]
        tool: String,
        /// Raw content/output to evaluate (read from stdin if not provided)
        #[arg(long)]
        content: Option<String>,
        /// Optional session identifier for grouping diary entries
        #[arg(long)]
        session: Option<String>,
        /// Project root (defaults to current directory)
        #[arg(long = "project")]
        path: Option<PathBuf>,
    },
    /// Write an agent diary entry as a Verbatim neuron under `@agent/{agent}`.
    DiaryWrite {
        /// Agent identifier used for the diary namespace.
        #[arg(long)]
        agent: String,
        /// Free-form action text, optionally followed by `key: value` metadata lines.
        #[arg(long)]
        content: String,
        /// Optional structured title override.
        #[arg(long)]
        title: Option<String>,
        /// Optional structured status override.
        #[arg(long)]
        status: Option<String>,
        /// Optional structured goal override.
        #[arg(long)]
        goal: Option<String>,
        /// Optional structured next-step override.
        #[arg(long)]
        next_step: Option<String>,
        /// Optional structured blocker override.
        #[arg(long)]
        blocker: Option<String>,
        /// Optional structured outcome override.
        #[arg(long)]
        outcome: Option<String>,
        /// Optional comma-delimited related entities.
        #[arg(long, value_delimiter = ',')]
        entities: Vec<String>,
        /// Optional comma-delimited dependencies.
        #[arg(long, value_delimiter = ',')]
        depends_on: Vec<String>,
        /// Optional ISO-8601 timestamp for the diary entry.
        #[arg(long)]
        timestamp: Option<String>,
        /// Project root (defaults to current directory)
        path: Option<PathBuf>,
    },
    /// Read recent diary entries for an agent.
    DiaryRead {
        /// Agent identifier used for the diary namespace.
        #[arg(long)]
        agent: String,
        /// Number of recent entries to show.
        #[arg(long, default_value_t = 10)]
        last_n: usize,
        /// Project root (defaults to current directory)
        path: Option<PathBuf>,
    },
    /// Show a chronological timeline of recent session activity.
    Timeline {
        /// Look-back duration: "2h", "1d", "3d", "1w" (default: "1d")
        #[arg(long, default_value = "1d")]
        since: String,
        /// Optional agent name filter
        #[arg(long)]
        agent: Option<String>,
        /// Max items to show (default: 20)
        #[arg(long, default_value = "20")]
        limit: usize,
        /// Project root (defaults to current directory)
        path: Option<PathBuf>,
    },
    /// Snapshot activated session context before Claude Code compacts the conversation.
    /// Called automatically by the PreCompact hook installed by `cortyx install`.
    #[command(name = "precompact-snapshot")]
    Precompact {
        /// Path to the project root (defaults to current directory)
        #[arg(long, short = 'p')]
        project: Option<PathBuf>,
    },
    /// Show neuron health report: top activated, stalest, and module stats.
    Insights {
        /// Only include neurons active within this window (e.g. "2d", "1w", "1h")
        #[arg(long)]
        since: Option<String>,
        /// Number of entries per section (default: 10)
        #[arg(long, default_value = "10")]
        top: usize,
        /// Path to the project root
        #[arg(long, short = 'p')]
        project: Option<PathBuf>,
    },
    /// Promote frequently-referenced diary entries to permanent Verbatim neurons.
    Consolidate {
        /// Minimum reference count required for promotion (default: 3)
        #[arg(long, default_value = "3")]
        min_refs: u32,
        /// Preview what would be promoted without writing
        #[arg(long)]
        dry_run: bool,
        /// Path to the project root
        #[arg(long, short = 'p')]
        project: Option<PathBuf>,
    },
    /// Keep the local index fresh as files change.
    ///
    /// Auto-bootstraps a missing index on first run, then keeps dirty-file hot
    /// patches flowing until you stop the process. The startup banner includes
    /// a stable `ux-proof` JSON line for TTFC/bootstrap benchmarking.
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
    /// Retrieve the top local/project neurons for a task query and print their content to stdout.
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
        /// Return a concise answer-oriented output derived from the selected
        /// contexts instead of printing the full neuron bodies.
        /// Keeps the retrieval path unchanged; this is an optional output layer.
        #[arg(long, default_value = "false")]
        answer_mode: bool,
        /// Minimum answer confidence required when --answer-mode is enabled.
        /// Low-support heuristic snippet guesses abstain below this threshold.
        #[arg(long)]
        min_answer_confidence: Option<f64>,
        /// Include lightweight provenance/explanation metadata in the output.
        /// In context mode this prepends a provenance block; in answer mode it
        /// appends the supporting sources and summaries after the answer.
        #[arg(long, default_value = "false")]
        provenance: bool,
        /// Project root (defaults to current directory)
        path: Option<PathBuf>,
    },
    /// Default terminal entrypoint — route a task through the best matching Cortyx flow.
    ///
    /// This is the CLI counterpart to the universal `cortyx` MCP tool.
    /// It auto-routes across the current local-first Cortyx surface: context
    /// retrieval, answer mode, wake-up priming, agent status, consistency
    /// checks, or a capability summary. The CLI prints readiness guidance on
    /// stderr and falls back to the current capability summary when invoked
    /// without task/agent/path inputs. The stderr banner includes a stable
    /// `ux-proof` JSON line for TTFC + latency benchmarking.
    Route {
        /// High-level intent. `auto` infers the best route from the inputs.
        #[arg(long, value_enum, default_value = "auto")]
        intent: RouteIntent,
        /// Task or question to route. Required for `context` and `answer`.
        #[arg(long)]
        task: Option<String>,
        /// Optional agent identifier for agent-status or wake-up flows.
        #[arg(long)]
        agent: Option<String>,
        /// Optional person scope for wake-up or retrieval flows.
        #[arg(long)]
        person: Option<String>,
        /// Optional module filter for retrieval flows.
        #[arg(long)]
        module: Option<String>,
        /// Optional kind filter: "code", "conversation", or "all".
        #[arg(long)]
        kind: Option<String>,
        /// Optional path filter for consistency checks.
        #[arg(long)]
        scope_path: Option<String>,
        /// Maximum token budget for routed retrieval flows.
        #[arg(long, default_value = "4000")]
        max_tokens: usize,
        /// Minimum BM25 confidence for routed retrieval flows.
        #[arg(long)]
        min_confidence: Option<f64>,
        /// Enable 2-hop retrieval for routed retrieval flows.
        #[arg(long, default_value = "false")]
        multi_hop: bool,
        /// Enable stable capsules for routed retrieval flows.
        #[arg(long, default_value = "false")]
        capsule_mode: bool,
        /// Minimum answer confidence for routed answer-mode flows.
        #[arg(long)]
        min_answer_confidence: Option<f64>,
        /// Enable delta-mode context emission for routed retrieval flows.
        #[arg(long, default_value = "false")]
        delta_mode: bool,
        /// Optional delta-mode context handle from a previous routed call.
        #[arg(long)]
        context_handle: Option<String>,
        /// Include provenance metadata where supported.
        #[arg(long, default_value = "false")]
        provenance: bool,
        /// Include timelines for agent-status routes.
        #[arg(long, default_value = "false")]
        include_timeline: bool,
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
    /// This command restores one saved step for a named section — instant rollback.
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
    /// Detects Claude Code, Cursor, Windsurf, Codex, VS Code, and Zed configs
    /// in standard paths. Writes `cortyx serve` MCP entry to each, adds Claude
    /// Code hook scripts for auto-priming, auto-capture, and auto-save
    /// (SessionStart + PostToolUse + Stop + PreCompact events), and prints
    /// both terminal and in-tool quickstart guidance. `--global` scaffolds the
    /// standard config files even when they do not exist yet. The summary ends
    /// with a stable `ux-proof` JSON line for onboarding benchmarks. Idempotent
    /// — safe to run multiple times.
    Install {
        /// Scaffold the standard per-client config files under HOME even if none exist yet.
        #[arg(long)]
        global: bool,
    },
    /// Hook-safe index readability check for external clients.
    ///
    /// This command does **not** commit MCP feedback. `cortyx_close_task` feedback is
    /// in-process/session-scoped, so a standalone CLI hook cannot flush it honestly.
    /// Kept as a hook-friendly health check and backward-compatible alias.
    #[command(name = "hook-check", alias = "close-task-hook")]
    HookCheck {
        /// Project root (defaults to current directory)
        #[arg(long)]
        project: Option<PathBuf>,
    },
    /// Print a versioned proof certificate of Cortyx's measured capabilities.
    ProofCertificate {
        /// Fail if any headline proof metric still comes from a hardcoded fallback.
        #[arg(long)]
        validate: bool,
    },
    /// Manage the local fleet of registered Cortyx projects.
    ///
    /// Fleet enables cross-project context sharing: when local confidence is low,
    /// Cortyx automatically queries registered peer projects for supplementary context.
    /// Zero server required — all coordination is local-first via ~/.cortyx/fleet/nodes.json.
    #[command(subcommand)]
    Fleet(FleetCommand),
    /// Manage the git-federated global concept library (S-IV, TRIZ R16).
    ///
    /// The concept library lives at `~/.cortyx/global/` and is a plain git repository.
    /// Teams share universal concept neurons (JWT, BM25, event-sourcing, etc.) via git —
    /// zero server, works offline. Deduplication prevents duplicate concepts.
    #[command(subcommand)]
    Concepts(ConceptsCommand),
    /// Manage the user-extensible evidence pattern registry (TRIZ C2 resolution).
    ///
    /// The pattern registry lets you extend evidence extraction with domain-specific
    /// patterns without touching Cortyx source code. Patterns live in `.cortyx/patterns/*.toml`.
    #[command(subcommand)]
    Patterns(PatternsCommand),
}

/// Sub-commands for `cortyx patterns`
#[derive(Subcommand)]
pub enum PatternsCommand {
    /// List all loaded evidence patterns (built-in + user-defined).
    ///
    /// Shows pattern name, family, confidence, and whether it is built-in or user-defined.
    List,
    /// Scaffold a new TOML pattern file in `.cortyx/patterns/`.
    ///
    /// Creates `.cortyx/patterns/<name>.toml` with a template pattern entry.
    Add {
        /// Filename stem for the new pattern file (e.g. "my_domain").
        /// Creates `.cortyx/patterns/<name>.toml`.
        name: String,
    },
}

/// Sub-commands for `cortyx fleet`
#[derive(Subcommand)]
pub enum FleetCommand {
    /// Register a project directory or git-backed corpus as a fleet node.
    ///
    /// For a local project: scans the project's Cortyx index to extract its module
    /// manifest, then records the node at ~/.cortyx/fleet/nodes.json.
    ///
    /// For a shared git corpus: clones the repo to ~/.cortyx/fleet/{alias}/ and
    /// registers it as a fleet node. Run `cortyx fleet sync` to pull future updates.
    ///
    /// Examples:
    ///   cortyx fleet register /path/to/project
    ///   cortyx fleet register --git-url git@github.com:org/neurons.git --alias team
    Register {
        /// Path to the Cortyx project to register (defaults to current directory).
        /// Mutually exclusive with --git-url.
        path: Option<PathBuf>,
        /// Human-readable alias for this node (defaults to directory name).
        #[arg(long)]
        alias: Option<String>,
        /// Git URL of a shared corpus to clone and register.
        /// Accepted: https://github.com/, https://gitlab.com/, git@github.com:, git@gitlab.com:
        /// Requires --alias.
        #[arg(long)]
        git_url: Option<String>,
    },
    /// Pull the latest commits for all git-backed fleet nodes (or a specific one).
    ///
    /// Runs `git fetch --ff-only` in each git-backed node's local clone.
    /// Failures are non-fatal — the cached clone continues to be used offline.
    Sync {
        /// Alias of a specific node to sync (syncs all git-backed nodes when omitted).
        alias: Option<String>,
    },
    /// Remove a fleet node by alias or path.
    Deregister {
        /// Alias or path of the node to remove.
        target: String,
    },
    /// List all registered fleet nodes.
    List,
    /// Show fleet health: node count, module totals, and last registration times.
    Status,
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
    /// List neurons in the current project that are ready to be shared.
    ///
    /// Uses the same evidence Cortyx already tracks locally: use_count, hit_rate,
    /// and self-quality score. Already-published fingerprints are filtered out.
    Ready {
        /// Project root to scan (defaults to current directory).
        #[arg(long)]
        project: Option<PathBuf>,
        /// Maximum number of candidates to show (0 = no limit).
        #[arg(long, default_value_t = 10)]
        limit: usize,
        /// Minimum use_count required to consider a neuron share-ready.
        #[arg(long, default_value_t = 10)]
        min_use: u32,
        /// Minimum hit_rate required to consider a neuron share-ready.
        #[arg(long, default_value_t = 0.5)]
        min_hit_rate: f32,
        /// Minimum quality_score required to consider a neuron share-ready.
        #[arg(long, default_value_t = 0.6)]
        min_quality: f32,
    },
    /// Publish all share-ready neurons from the current project into the global library.
    ///
    /// This batch version applies the same quality gates as `concepts ready` and
    /// auto-commits the global concept repo when it is git-backed.
    PublishReady {
        /// Project root to scan (defaults to current directory).
        #[arg(long)]
        project: Option<PathBuf>,
        /// Maximum number of candidates to publish (0 = no limit).
        #[arg(long, default_value_t = 10)]
        limit: usize,
        /// Minimum use_count required to consider a neuron share-ready.
        #[arg(long, default_value_t = 10)]
        min_use: u32,
        /// Minimum hit_rate required to consider a neuron share-ready.
        #[arg(long, default_value_t = 0.5)]
        min_hit_rate: f32,
        /// Minimum quality_score required to consider a neuron share-ready.
        #[arg(long, default_value_t = 0.6)]
        min_quality: f32,
    },
    /// Show the status of the global concept library (remote, commit, neuron count).
    Status,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn status_command_parses_collaboration_flags() {
        let cli = Cli::try_parse_from([
            "cortyx",
            "status",
            "--collaboration",
            "--agent",
            "reviewer",
            "--include-timeline",
        ])
        .expect("status command should parse");

        match cli.command {
            Commands::Status {
                collaboration,
                agent,
                module,
                include_timeline,
                ..
            } => {
                assert!(collaboration);
                assert_eq!(agent.as_deref(), Some("reviewer"));
                assert!(module.is_none());
                assert!(include_timeline);
            },
            _ => panic!("unexpected command"),
        }
    }

    #[test]
    fn route_command_parses_capabilities_intent() {
        let cli = Cli::try_parse_from(["cortyx", "route", "--intent", "capabilities"])
            .expect("route command should parse capability intent");

        match cli.command {
            Commands::Route { intent, .. } => assert_eq!(intent, RouteIntent::Capabilities),
            _ => panic!("unexpected command"),
        }
    }

    #[test]
    fn diary_write_command_parses_inline_content_flags() {
        let cli = Cli::try_parse_from([
            "cortyx",
            "diary-write",
            "--agent",
            "reviewer",
            "--content",
            "Investigate auth\nstatus: blocked",
        ])
        .expect("diary-write command should parse");

        match cli.command {
            Commands::DiaryWrite { agent, content, .. } => {
                assert_eq!(agent, "reviewer");
                assert!(content.contains("status: blocked"));
            },
            _ => panic!("unexpected command"),
        }
    }

    #[test]
    fn insights_command_parses_since_top_and_project_flags() {
        let cli = Cli::try_parse_from([
            "cortyx",
            "insights",
            "--since",
            "2d",
            "--top",
            "7",
            "--project",
            "/repo",
        ])
        .expect("insights command should parse");

        match cli.command {
            Commands::Insights {
                since,
                top,
                project,
            } => {
                assert_eq!(since.as_deref(), Some("2d"));
                assert_eq!(top, 7);
                assert_eq!(project, Some(PathBuf::from("/repo")));
            },
            _ => panic!("unexpected command"),
        }
    }

    #[test]
    fn mine_observation_command_parses_project_flag() {
        let cli = Cli::try_parse_from([
            "cortyx",
            "mine-observation",
            "--tool",
            "Edit",
            "--project",
            "/repo",
        ])
        .expect("mine-observation command should parse");

        match cli.command {
            Commands::MineObservation { tool, path, .. } => {
                assert_eq!(tool, "Edit");
                assert_eq!(path, Some(PathBuf::from("/repo")));
            },
            _ => panic!("unexpected command"),
        }
    }

    #[test]
    fn precompact_snapshot_command_parses_project_flag() {
        let cli = Cli::try_parse_from(["cortyx", "precompact-snapshot", "--project", "/repo"])
            .expect("precompact-snapshot command should parse");

        match cli.command {
            Commands::Precompact { project } => {
                assert_eq!(project, Some(PathBuf::from("/repo")));
            },
            _ => panic!("unexpected command"),
        }
    }

    #[test]
    fn consolidate_command_parses_project_flag() {
        let cli = Cli::try_parse_from([
            "cortyx",
            "consolidate",
            "--min-refs",
            "5",
            "--dry-run",
            "--project",
            "/repo",
        ])
        .expect("consolidate command should parse");

        match cli.command {
            Commands::Consolidate {
                min_refs,
                dry_run,
                project,
            } => {
                assert_eq!(min_refs, 5);
                assert!(dry_run);
                assert_eq!(project, Some(PathBuf::from("/repo")));
            },
            _ => panic!("unexpected command"),
        }
    }
}
