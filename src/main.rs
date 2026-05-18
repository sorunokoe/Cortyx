use anyhow::Result;
use clap::Parser;
use cortyx::cli::{Cli, Commands, RouteIntent};
use cortyx::{
    agent_memory, answer_plane, commands, export, global_index, index, installer, mcp, miner,
    neuron, watcher,
};
use serde_json::json;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};
use tracing_subscriber::{fmt, EnvFilter};

/// Resolve an optional path argument to a canonical project root.
///
/// Falls back to `.` when no path is given; silently uses the non-canonical
/// path if `canonicalize` fails (e.g. directory does not yet exist).
fn project_root(path: Option<PathBuf>) -> PathBuf {
    let p = path.unwrap_or_else(|| PathBuf::from("."));
    p.canonicalize().unwrap_or(p)
}

const TERMINAL_ROUTE_EXAMPLE: &str = r#"cortyx route --task "trace the auth flow""#;
const MCP_CAPABILITY_EXAMPLE: &str = r#"cortyx()"#;
const MCP_TASK_EXAMPLE: &str = r#"cortyx(task="trace the auth flow")"#;
const WATCH_EXAMPLE: &str = "cortyx watch";
const DOCTOR_EXAMPLE: &str = "cortyx doctor";
const INCREMENTAL_COMPILE_EXAMPLE: &str = "cortyx compile --incremental";

#[derive(Default)]
struct CliDiaryContent {
    action: String,
    title: Option<String>,
    status: Option<String>,
    goal: Option<String>,
    next_step: Option<String>,
    blocker: Option<String>,
    outcome: Option<String>,
    entities: Vec<String>,
    depends_on: Vec<String>,
}

fn normalize_cli_value(value: &str) -> Option<String> {
    let value = value.trim();
    if value.is_empty() {
        None
    } else {
        Some(value.split_whitespace().collect::<Vec<_>>().join(" "))
    }
}

fn normalize_cli_list(value: &str) -> Vec<String> {
    value.split(',').filter_map(normalize_cli_value).collect()
}

fn parse_cli_diary_content(content: &str) -> CliDiaryContent {
    if let Some(entry) = agent_memory::parse_structured_diary_entry(content) {
        return CliDiaryContent {
            action: entry.action.unwrap_or_default(),
            title: entry.title,
            status: entry.status,
            goal: entry.goal,
            next_step: entry.next_step,
            blocker: entry.blocker,
            outcome: entry.outcome,
            entities: entry.entities,
            depends_on: entry.depends_on,
        };
    }

    let mut parsed = CliDiaryContent::default();
    let mut action_lines = Vec::new();
    for raw_line in content.lines() {
        let line = raw_line.trim();
        if line.is_empty() {
            continue;
        }
        let candidate = line.strip_prefix("- ").unwrap_or(line);
        if let Some((label, value)) = candidate.split_once(':') {
            match label.trim().to_ascii_lowercase().as_str() {
                "title" => parsed.title = normalize_cli_value(value),
                "status" => parsed.status = normalize_cli_value(value),
                "goal" => parsed.goal = normalize_cli_value(value),
                "next_step" | "next-step" | "next step" => {
                    parsed.next_step = normalize_cli_value(value)
                },
                "blocker" => parsed.blocker = normalize_cli_value(value),
                "outcome" => parsed.outcome = normalize_cli_value(value),
                "entities" => parsed.entities = normalize_cli_list(value),
                "depends_on" | "depends-on" | "depends on" => {
                    parsed.depends_on = normalize_cli_list(value)
                },
                _ => action_lines.push(line.to_string()),
            }
        } else {
            action_lines.push(line.to_string());
        }
    }
    parsed.action = action_lines.join("\n");
    parsed
}

fn recent_agent_diary_paths(index: &index::NeuronIndex, agent: &str, limit: usize) -> Vec<PathBuf> {
    if limit == 0 {
        return Vec::new();
    }
    let module = format!("@agent/{}", agent.trim());
    let mut items: Vec<(i64, PathBuf)> = index
        .list_neurons(Some(&module))
        .into_iter()
        .filter(|summary| summary.kind == neuron::NeuronKind::Verbatim)
        .map(|summary| {
            let timestamp = index
                .context_metadata_for(&summary.path)
                .and_then(|metadata| metadata.timestamp_secs)
                .unwrap_or(i64::MIN);
            (timestamp, summary.path)
        })
        .collect();
    items.sort_by(|a, b| b.0.cmp(&a.0).then_with(|| a.1.cmp(&b.1)));
    items
        .into_iter()
        .take(limit)
        .map(|(_, path)| path)
        .collect()
}

fn git_repo_root_and_relative_path(path: &Path) -> Result<(PathBuf, PathBuf)> {
    let cwd = std::env::current_dir()?;
    let resolved = if path.is_absolute() {
        path.to_path_buf()
    } else {
        cwd.join(path)
    };
    let resolved = resolved.canonicalize().unwrap_or(resolved);

    for ancestor in resolved.ancestors() {
        if ancestor.join(".git").exists() {
            let rel = resolved.strip_prefix(ancestor).map_err(|_| {
                anyhow::anyhow!(
                    "Path {} is not inside git repository {}",
                    resolved.display(),
                    ancestor.display()
                )
            })?;
            if rel.as_os_str().is_empty() {
                anyhow::bail!("Path must point to a file inside the git repository");
            }
            return Ok((ancestor.to_path_buf(), rel.to_path_buf()));
        }
    }

    anyhow::bail!("No git repository found for {}", path.display())
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct IndexReadiness {
    neurons: usize,
    synapses: usize,
    fresh: usize,
    stale: usize,
    stubs: usize,
}

impl IndexReadiness {
    fn from_index(idx: &index::NeuronIndex) -> Self {
        let (fresh, stale, stubs) = idx.status_counts();
        Self {
            neurons: idx.neuron_count(),
            synapses: idx.synapse_count(),
            fresh,
            stale,
            stubs,
        }
    }

    fn summary_line(&self) -> String {
        format!(
            "{} fresh, {} stale, {} stubs ({} neurons, {} synapses)",
            self.fresh, self.stale, self.stubs, self.neurons, self.synapses
        )
    }

    fn note(&self) -> Option<&'static str> {
        if self.fresh == 0 && self.stubs > 0 {
            Some(
                "index is stub-heavy, so results may stay placeholder-rich until your AI tool runs `cortyx_evolve_context`",
            )
        } else if self.stale > 0 {
            Some("stale neurons detected — `cortyx watch` or `cortyx compile --incremental` will refresh them")
        } else {
            None
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CliRouteMode {
    Context,
    Answer,
    WakeUp,
    AgentStatus,
    Consistency,
    Capabilities,
}

impl CliRouteMode {
    fn label(self) -> &'static str {
        match self {
            Self::Context => "context",
            Self::Answer => "answer",
            Self::WakeUp => "wake_up",
            Self::AgentStatus => "agent_status",
            Self::Consistency => "consistency",
            Self::Capabilities => "capabilities",
        }
    }
}

fn format_duration(duration: Duration) -> String {
    if duration.as_secs_f64() >= 1.0 {
        format!("{:.2}s", duration.as_secs_f64())
    } else {
        format!("{}ms", duration.as_millis())
    }
}

fn duration_ms(duration: Duration) -> u64 {
    duration.as_millis().min(u64::MAX as u128) as u64
}

fn render_route_ux_proof(
    mode: CliRouteMode,
    readiness: &IndexReadiness,
    bootstrap: Option<(usize, Duration)>,
    route_elapsed: Duration,
) -> String {
    json!({
        "mode": mode.label(),
        "ttfc": {
            "triggered": bootstrap.is_some(),
            "compiled_neurons": bootstrap.map(|(compiled, _)| compiled).unwrap_or(0),
            "elapsed_ms": bootstrap
                .map(|(_, elapsed)| duration_ms(elapsed))
                .unwrap_or(0),
        },
        "route_latency_ms": duration_ms(route_elapsed),
        "index": {
            "neurons": readiness.neurons,
            "synapses": readiness.synapses,
            "fresh": readiness.fresh,
            "stale": readiness.stale,
            "stubs": readiness.stubs,
        },
        "recovery": {
            "watch": WATCH_EXAMPLE,
            "doctor": DOCTOR_EXAMPLE,
        },
        "entrypoint": {
            "terminal_route": TERMINAL_ROUTE_EXAMPLE,
            "mcp_summary": MCP_CAPABILITY_EXAMPLE,
            "mcp_task": MCP_TASK_EXAMPLE,
        },
    })
    .to_string()
}

fn render_watch_ux_proof(
    readiness: &IndexReadiness,
    bootstrap: Option<(usize, Duration)>,
) -> String {
    json!({
        "bootstrap": {
            "triggered": bootstrap.is_some(),
            "compiled_neurons": bootstrap.map(|(compiled, _)| compiled).unwrap_or(0),
            "elapsed_ms": bootstrap
                .map(|(_, elapsed)| duration_ms(elapsed))
                .unwrap_or(0),
        },
        "index": {
            "neurons": readiness.neurons,
            "synapses": readiness.synapses,
            "fresh": readiness.fresh,
            "stale": readiness.stale,
            "stubs": readiness.stubs,
        },
        "live_freshness": watcher::HOT_PATCH_WATCH_SUMMARY,
        "recovery": {
            "doctor": DOCTOR_EXAMPLE,
            "incremental_compile": INCREMENTAL_COMPILE_EXAMPLE,
        },
    })
    .to_string()
}

fn looks_like_question(task: &str) -> bool {
    let trimmed = task.trim();
    if trimmed.ends_with('?') {
        return true;
    }
    matches!(
        trimmed
            .split_whitespace()
            .next()
            .unwrap_or_default()
            .to_ascii_lowercase()
            .as_str(),
        "what"
            | "who"
            | "when"
            | "where"
            | "why"
            | "how"
            | "which"
            | "is"
            | "are"
            | "did"
            | "does"
            | "do"
            | "can"
            | "could"
            | "should"
            | "will"
    )
}

fn looks_like_wake_up_request(task: &str) -> bool {
    let lower = task.to_ascii_lowercase();
    lower.contains("wake up")
        || lower.contains("wake-up")
        || lower.contains("prime session")
        || lower.contains("prime the session")
        || lower.contains("prime context")
}

fn looks_like_consistency_request(task: &str) -> bool {
    let lower = task.to_ascii_lowercase();
    lower.contains("consistency") || lower.contains("contradiction") || lower.contains("conflict")
}

fn infer_cli_route_mode(
    intent: RouteIntent,
    task: Option<&str>,
    agent: Option<&str>,
    person: Option<&str>,
    scope_path: Option<&str>,
) -> CliRouteMode {
    match intent {
        RouteIntent::Context => CliRouteMode::Context,
        RouteIntent::Answer => CliRouteMode::Answer,
        RouteIntent::WakeUp => CliRouteMode::WakeUp,
        RouteIntent::AgentStatus => CliRouteMode::AgentStatus,
        RouteIntent::Consistency => CliRouteMode::Consistency,
        RouteIntent::Capabilities => CliRouteMode::Capabilities,
        RouteIntent::Auto => {
            if task.is_none() {
                if agent.is_some() {
                    CliRouteMode::AgentStatus
                } else if person.is_some() {
                    CliRouteMode::WakeUp
                } else if scope_path.is_some() {
                    CliRouteMode::Consistency
                } else {
                    CliRouteMode::Capabilities
                }
            } else {
                let task = task.unwrap_or_default();
                if looks_like_wake_up_request(task) || person.is_some() {
                    CliRouteMode::WakeUp
                } else if agent.is_some() && looks_like_question(task) {
                    CliRouteMode::Answer
                } else if agent.is_some() {
                    CliRouteMode::Context
                } else if looks_like_consistency_request(task) || scope_path.is_some() {
                    CliRouteMode::Consistency
                } else if looks_like_question(task) {
                    CliRouteMode::Answer
                } else {
                    CliRouteMode::Context
                }
            }
        },
    }
}

fn render_route_banner(
    root: &Path,
    mode: CliRouteMode,
    readiness: &IndexReadiness,
    bootstrap: Option<(usize, Duration)>,
    route_elapsed: Duration,
) -> String {
    let mut lines = vec![
        "Cortyx CLI route".to_string(),
        format!("- mode: {}", mode.label()),
        format!("- project: {}", root.display()),
    ];

    if let Some((compiled, elapsed)) = bootstrap {
        lines.push(format!(
            "- time-to-first-context: compiled {compiled} neurons in {}",
            format_duration(elapsed)
        ));
    }
    lines.push(format!(
        "- route latency: {}",
        format_duration(route_elapsed)
    ));
    lines.push(format!("- index: {}", readiness.summary_line()));
    lines.push(format!(
        "- terminal loop: `{WATCH_EXAMPLE}` keeps changes hot; `{DOCTOR_EXAMPLE}` diagnoses drift"
    ));
    if mode == CliRouteMode::Capabilities {
        lines.push(format!("- terminal quickstart: `{TERMINAL_ROUTE_EXAMPLE}`"));
    } else {
        lines.push(format!(
            "- AI tool mirror: `{MCP_CAPABILITY_EXAMPLE}` for summary, `{MCP_TASK_EXAMPLE}` for task start"
        ));
    }
    if let Some(note) = readiness.note() {
        lines.push(format!("- note: {note}"));
    }
    lines.push(format!(
        "- ux-proof: {}",
        render_route_ux_proof(mode, readiness, bootstrap, route_elapsed)
    ));

    lines.join("\n")
}

fn render_watch_banner(
    root: &Path,
    readiness: &IndexReadiness,
    bootstrap: Option<(usize, Duration)>,
) -> String {
    let mut lines = vec![
        format!("✓ Watch loop ready for {}", root.display()),
        format!("  index: {}", readiness.summary_line()),
    ];
    if let Some((compiled, elapsed)) = bootstrap {
        lines.insert(
            1,
            format!(
                "  bootstrap: compiled {compiled} neurons in {}",
                format_duration(elapsed)
            ),
        );
    }
    lines.push(format!(
        "  live freshness: {}",
        watcher::HOT_PATCH_WATCH_SUMMARY
    ));
    lines.push(format!(
        "  recovery: `{DOCTOR_EXAMPLE}` diagnoses drift; `{INCREMENTAL_COMPILE_EXAMPLE}` refreshes dirty files on demand"
    ));
    if let Some(note) = readiness.note() {
        lines.push(format!("  note: {note}"));
    }
    lines.push(format!(
        "  ux-proof: {}",
        render_watch_ux_proof(readiness, bootstrap)
    ));
    lines.push("  Press Ctrl+C to stop.".to_string());
    lines.join("\n")
}

#[tokio::main]
async fn main() -> Result<()> {
    fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .with_target(false)
        .init();

    let cli = Cli::parse();

    match cli.command {
        Commands::Serve { project } => {
            mcp::serve(project).await?;
        },
        Commands::Compile { path, incremental } => {
            let root = project_root(path);
            let mut idx = index::NeuronIndex::load_or_create(&root)?;
            if incremental {
                let n = idx.compile_dirty()?;
                println!(
                    "✓ Incremental compile: {n} neurons updated in {}",
                    root.display()
                );
            } else {
                let n = idx.compile()?;
                println!("✓ Compiled {n} neurons in {}", root.display());
            }
            // B2: auto-trigger embedding pass when embed feature is present.
            // Falls back silently if CORTYX_NO_DOWNLOAD is set or model isn't available.
            #[cfg(feature = "embed")]
            {
                let paths: Vec<_> = idx
                    .neuron_paths_and_use_counts()
                    .into_iter()
                    .map(|(p, _)| p)
                    .collect();
                if !paths.is_empty() {
                    println!(
                        "  Embedding {} neurons (all-MiniLM-L6-v2; ~80MB one-time model download)…",
                        paths.len()
                    );
                    miner::embed_all(&paths, &root);
                    println!("  ✓ Embeddings saved — hybrid BM25 + dense retrieval active.");
                }
            }
            println!("  Next: call cortyx_evolve_context to fill stubs, or `cortyx serve` to start the MCP server.");
        },
        Commands::Status {
            path,
            collaboration,
            agent,
            module,
            include_timeline,
        } => {
            let root = project_root(path);
            let idx = index::NeuronIndex::load_or_create(&root)?;
            if collaboration || agent.is_some() || module.is_some() || include_timeline {
                let server = mcp::CortyxServer::for_benchmark(root, idx);
                let output = server
                    .benchmark_collaboration_status(mcp::CollaborationStatusInput {
                        agent,
                        module,
                        include_timeline: Some(include_timeline),
                    })
                    .await;
                print!("{output}");
            } else {
                idx.print_status();
            }
        },
        Commands::Invalidate { file } => {
            let root = project_root(None);
            let mut idx = index::NeuronIndex::load_or_create(&root)?;
            idx.invalidate(&file)?;
            println!("✓ Marked {} as stale", file.display());
        },
        Commands::Export {
            provider,
            output,
            path,
        } => {
            let root = project_root(path);
            let idx = index::NeuronIndex::load_or_create(&root)?;
            let json = export::build_prompt_json(&root, &idx, provider)?;
            match output {
                Some(out) => {
                    std::fs::write(&out, &json)?;
                    println!("✓ Prompt JSON written to {}", out.display());
                },
                None => println!("{json}"),
            }
        },
        Commands::Mine { path, module } => {
            let root = project_root(None);
            let mut idx = index::NeuronIndex::load_or_create(&root)?;
            let count = miner::mine_path(&path, &root, &mut idx, module.as_deref())?;
            println!("✓ Mined {count} Verbatim neurons from {}", path.display());
        },
        Commands::DiaryWrite {
            agent,
            content,
            title,
            status,
            goal,
            next_step,
            blocker,
            outcome,
            entities,
            depends_on,
            timestamp,
            path,
        } => {
            if agent.trim().is_empty() {
                anyhow::bail!("agent name must not be empty");
            }
            let root = project_root(path);
            let mut idx = index::NeuronIndex::load_or_create(&root)?;
            let parsed = parse_cli_diary_content(&content);
            let action = parsed.action;
            let title = title.or(parsed.title);
            let status = status.or(parsed.status);
            let goal = goal.or(parsed.goal);
            let next_step = next_step.or(parsed.next_step);
            let blocker = blocker.or(parsed.blocker);
            let outcome = outcome.or(parsed.outcome);
            let entities = if entities.is_empty() {
                parsed.entities
            } else {
                entities
            };
            let depends_on = if depends_on.is_empty() {
                parsed.depends_on
            } else {
                depends_on
            };
            let structured = agent_memory::has_structured_diary_fields(
                title.as_deref(),
                status.as_deref(),
                goal.as_deref(),
                next_step.as_deref(),
                blocker.as_deref(),
                outcome.as_deref(),
                &entities,
                &depends_on,
            );
            let body = if structured {
                agent_memory::render_structured_diary_entry(
                    agent.trim(),
                    &action,
                    title.as_deref(),
                    status.as_deref(),
                    goal.as_deref(),
                    next_step.as_deref(),
                    blocker.as_deref(),
                    outcome.as_deref(),
                    &entities,
                    &depends_on,
                )
            } else {
                action.trim().to_string()
            };
            if body.is_empty() {
                anyhow::bail!(
                    "content must not be empty unless structured diary fields are supplied"
                );
            }
            let effective_timestamp = timestamp.unwrap_or_else(neuron::now_iso8601);
            let module = format!("@agent/{}", agent.trim());
            let count = miner::mine_text(
                &body,
                "diary",
                &root,
                &mut idx,
                Some(&module),
                Some(agent.trim()),
                Some(effective_timestamp.as_str()),
            )?;
            println!(
                "✓ Diary entry written for agent '{}' ({count} neuron(s) created).",
                agent.trim()
            );
        },
        Commands::DiaryRead {
            agent,
            last_n,
            path,
        } => {
            if agent.trim().is_empty() {
                anyhow::bail!("agent name must not be empty");
            }
            let root = project_root(path);
            let idx = index::NeuronIndex::load_or_create(&root)?;
            let paths = recent_agent_diary_paths(&idx, &agent, last_n);
            if paths.is_empty() {
                println!("No diary entries found for agent '{}'.", agent.trim());
            } else {
                let mut out = format!("## Agent Diary: {} (last {})\n\n", agent.trim(), last_n);
                for path in paths {
                    let timestamp_secs = idx
                        .context_metadata_for(&path)
                        .and_then(|metadata| metadata.timestamp_secs);
                    match std::fs::read_to_string(&path) {
                        Ok(content) => {
                            if let Some(entry) =
                                agent_memory::parse_structured_diary_entry(&content)
                            {
                                out.push_str(&agent_memory::render_structured_diary_history_entry(
                                    &entry,
                                    timestamp_secs,
                                ));
                            } else {
                                out.push_str(&format!("---\n{}\n", content));
                            }
                        },
                        Err(err) => {
                            out.push_str(&format!("- {} — read error: {}\n", path.display(), err));
                        },
                    }
                }
                print!("{out}");
            }
        },
        Commands::Watch { path } => {
            let root = project_root(path);
            let mut idx = index::NeuronIndex::load_or_create(&root)?;
            let bootstrap = if idx.neuron_count() == 0 {
                let started = Instant::now();
                let count = idx.compile()?;
                Some((count, started.elapsed()))
            } else {
                None
            };
            let readiness = IndexReadiness::from_index(&idx);
            let index = std::sync::Arc::new(tokio::sync::RwLock::new(idx));
            let dirty_handle = index.read().await.dirty_set_handle();
            let _w =
                watcher::start_watcher(root.clone(), std::sync::Arc::clone(&index), dirty_handle)?;
            println!("{}", render_watch_banner(&root, &readiness, bootstrap));
            tokio::signal::ctrl_c().await?;
        },
        Commands::Doctor { path, json } => {
            let root = project_root(path);
            let code = commands::doctor::run(&root, json);
            std::process::exit(code);
        },
        Commands::Prune {
            path,
            min_use,
            older_than,
            dry_run,
        } => {
            let root = project_root(path);
            let removed = commands::prune::run(&root, min_use, older_than, dry_run)?;
            if dry_run {
                println!(
                    "Dry run — {} neuron(s) would be removed (re-run without --dry-run to delete)",
                    removed
                );
            } else {
                println!("✓ Pruned {removed} neuron(s) from {}", root.display());
            }
        },
        Commands::GetContexts {
            task,
            max_tokens,
            module,
            kind,
            min_confidence,
            multi_hop,
            answer_mode,
            min_answer_confidence,
            provenance,
            path,
        } => {
            let root = project_root(path);
            let idx = index::NeuronIndex::load_or_create(&root)?;
            let min_conf = min_confidence.map(|v| v as f32);
            let (included, overflow) = idx.get_contexts_with_scores_and_overflow(
                &task,
                max_tokens,
                module.as_deref(),
                kind.as_deref(),
                min_conf,
                multi_hop,
            );
            if answer_mode {
                match answer_plane::render_answer_output_decision(
                    &idx,
                    &task,
                    &included,
                    provenance,
                    min_answer_confidence.map(|value| value as f32),
                ) {
                    Ok(answer) => print!("{answer}"),
                    Err(answer_plane::AnswerAbstentionReason::LowFormConfidence)
                        if min_answer_confidence.is_some() =>
                    {
                        println!("(no confident answer — answer confidence below threshold)");
                    },
                    Err(answer_plane::AnswerAbstentionReason::LowFormConfidence)
                    | Err(answer_plane::AnswerAbstentionReason::Unsupported) => {},
                }
                return Ok(());
            }
            if provenance {
                if let Some(block) = answer_plane::render_provenance_output(&idx, &included) {
                    print!("{block}");
                }
            }
            for (neuron_path, _) in &included {
                if let Ok(content) = std::fs::read_to_string(neuron_path) {
                    println!("=== {} ===", neuron_path.display());
                    println!("{content}");
                }
            }
            if !overflow.is_empty() {
                println!("=== PROVENANCE OVERFLOW HINTS ===");
                for (path, headline) in &overflow {
                    println!("{} — {}", path.display(), headline);
                }
            }
            if included.is_empty() {
                if min_confidence.is_some() {
                    println!("(no neurons matched — confidence below threshold)");
                } else {
                    println!("(no neurons matched)");
                }
            }
        },
        Commands::Route {
            intent,
            task,
            agent,
            person,
            module,
            kind,
            scope_path,
            max_tokens,
            min_confidence,
            multi_hop,
            capsule_mode,
            min_answer_confidence,
            delta_mode,
            context_handle,
            provenance,
            include_timeline,
            path,
        } => {
            let root = project_root(path);
            let mut idx = index::NeuronIndex::load_or_create(&root)?;
            let bootstrap = if idx.neuron_count() == 0 {
                let started = Instant::now();
                let count = idx.compile()?;
                Some((count, started.elapsed()))
            } else {
                None
            };
            let readiness = IndexReadiness::from_index(&idx);
            let route_mode = infer_cli_route_mode(
                intent,
                task.as_deref(),
                agent.as_deref(),
                person.as_deref(),
                scope_path.as_deref(),
            );
            let route_root = root.clone();
            let server = mcp::CortyxServer::for_benchmark(root, idx);
            let intent = match intent {
                RouteIntent::Auto => None,
                RouteIntent::Context => Some("context".to_string()),
                RouteIntent::Answer => Some("answer".to_string()),
                RouteIntent::WakeUp => Some("wake_up".to_string()),
                RouteIntent::AgentStatus => Some("agent_status".to_string()),
                RouteIntent::Consistency => Some("consistency".to_string()),
                RouteIntent::Capabilities => Some("capabilities".to_string()),
            };
            let route_started = Instant::now();
            let output = server
                .benchmark_cortyx(mcp::CortyxInput {
                    intent,
                    task,
                    agent,
                    person,
                    module,
                    kind,
                    path: scope_path,
                    max_tokens: Some(max_tokens),
                    min_confidence,
                    multi_hop: Some(multi_hop),
                    previous_response: None,
                    delta_mode: Some(delta_mode),
                    context_handle,
                    capsule_mode: Some(capsule_mode),
                    min_answer_confidence,
                    provenance_mode: Some(provenance),
                    include_timeline: Some(include_timeline),
                })
                .await;
            eprintln!(
                "{}",
                render_route_banner(
                    &route_root,
                    route_mode,
                    &readiness,
                    bootstrap,
                    route_started.elapsed()
                )
            );
            print!("{output}");
        },
        Commands::Rollback { neuron } => {
            // E1: Git-based neuron versioning — restore previous commit
            let (repo_root, rel_neuron) = git_repo_root_and_relative_path(&neuron)?;
            let output = std::process::Command::new("git")
                .current_dir(&repo_root)
                .args(["checkout", "HEAD~1", "--", &rel_neuron.to_string_lossy()])
                .output()?;
            if output.status.success() {
                println!("✓ Rolled back {} to HEAD~1", neuron.display());
            } else {
                let stderr = String::from_utf8_lossy(&output.stderr);
                anyhow::bail!("git rollback failed: {stderr}");
            }
        },
        Commands::RollbackSection { neuron, section } => {
            // E2: Section shadow copy — restore from sidecar shadow_sections
            commands::rollback::run_section(&neuron, &section)?;
        },
        Commands::PublishConcept { neuron } => {
            // D1: Global concept layer — publish neuron to ~/.cortyx/global/
            let root = project_root(None);
            let global_dir = global_index::global_dir();
            let mut idx = global_index::GlobalIndex::load();
            match idx.publish(&neuron, &root) {
                Ok(dest) => {
                    println!("✓ Published concept to {}", dest.display());
                    let concept_name = neuron
                        .file_name()
                        .and_then(|name| name.to_str())
                        .unwrap_or("concept");
                    match commands::concepts::auto_commit_global_concepts(
                        &global_dir,
                        &format!("cortyx: publish concept {concept_name}"),
                    ) {
                        Ok(true) => println!("✓ Committed global concept library update"),
                        Ok(false) => {},
                        Err(err) => anyhow::bail!(
                            "concept published to {}, but failed to commit global library: {err}",
                            dest.display()
                        ),
                    }
                },
                Err(e) => anyhow::bail!("publish-concept failed: {e}"),
            }
        },
        Commands::ListConcepts => {
            // D1: List all global concepts
            let concepts = global_index::list_global_concepts();
            if concepts.is_empty() {
                println!("No global concepts published yet. Use `cortyx publish-concept <neuron>` to add one.");
            } else {
                println!("Global concepts ({} total):", concepts.len());
                for (path, project) in &concepts {
                    println!("  {} [from {}]", path.display(), project);
                }
            }
        },
        Commands::Install { global } => {
            // S1+S3: Auto-configure LLM clients + write hook scripts.
            match installer::run_install(global) {
                Ok(summary) => println!("{summary}"),
                Err(e) => {
                    eprintln!("cortyx install failed: {e}");
                    std::process::exit(1);
                },
            }
        },
        Commands::HookCheck { project } => {
            let root = project_root(project);
            index::NeuronIndex::load_or_create(&root)
                .map(|_| ())
                .map_err(|e| anyhow::anyhow!("hook-check: could not load index: {e}"))?;
        },
        Commands::Fleet(sub) => {
            commands::fleet::run(sub)?;
        },
        Commands::Concepts(sub) => {
            commands::concepts::run(sub)?;
        },
        Commands::Patterns(sub) => {
            let root = project_root(None);
            commands::patterns::run(sub, &root)?;
        },
    }

    Ok(())
}
