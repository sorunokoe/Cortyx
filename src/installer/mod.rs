//! S1 (R15 NE1): cortyx install — auto-configure all detected LLM clients.
//!
//! Detects config files for Claude Code, Cursor, Windsurf, Codex, VS Code, and Zed.
//! Writes the `cortyx serve` MCP entry to each found config and optionally
//! writes Claude Code hook scripts for auto-priming, auto-capture, and auto-save (S3).
//!
//! Design (TRIZ P10 + P25 + P6):
//! - P10: Preliminary action — all registration happens at install time.
//! - P25: Self-service — uses `std::env::current_exe()` for the absolute binary path;
//!   no marketplace or external plugin infrastructure needed.
//! - P6: Universality — one command registers Cortyx with every detected client.
//!
//! Result: `cargo install cortyx && cortyx install` — matches MemPalace 1-cmd setup.

mod client;
mod detection;
mod hooks;
mod registration;
mod utils;

use crate::error::Result;
use detection::detect_clients;
use hooks::write_hook_scripts;
use registration::register_mcp_server;
use std::path::PathBuf;
use utils::dirs_home;

const TERMINAL_ROUTE_EXAMPLE: &str = "cortyx route --task \"trace the auth flow\"";
const WATCH_EXAMPLE: &str = "cortyx watch";
const DOCTOR_EXAMPLE: &str = "cortyx doctor";
const MCP_CAPABILITY_EXAMPLE: &str = "cortyx()";
const MCP_TASK_EXAMPLE: &str = "cortyx(task=\"trace the auth flow\")";

fn install_ux_proof(registered: usize, already: usize, created: usize) -> String {
    serde_json::json!({
        "registered": registered,
        "already_configured": already,
        "created": created,
        "counts": {
            "terminal_steps": 3,
            "in_tool_steps": 2,
        },
        "terminal_quickstart": [
            TERMINAL_ROUTE_EXAMPLE,
            WATCH_EXAMPLE,
            DOCTOR_EXAMPLE,
        ],
        "in_tool_quickstart": [
            MCP_CAPABILITY_EXAMPLE,
            MCP_TASK_EXAMPLE,
        ],
        "recovery": {
            "watch": WATCH_EXAMPLE,
            "doctor": DOCTOR_EXAMPLE,
        },
    })
    .to_string()
}

/// Run `cortyx install [--global]`.
///
/// Writes the MCP server entry for every detected client config and optionally
/// writes Claude Code hook scripts (S3).
///
/// Returns a human-readable summary of all actions taken.
///
/// # Errors
///
/// Returns an error if the underlying operation fails.
pub fn run_install(global: bool) -> Result<String> {
    let exe = std::env::current_exe().unwrap_or_else(|_| PathBuf::from("cortyx"));

    let home = dirs_home();
    let (clients, scaffolded_count) = detect_clients(&home, global);

    if clients.is_empty() {
        return Ok(format!(
            "No LLM client configs detected.\n\
             Checked: ~/.claude/, ~/.cursor/, ~/.codeium/windsurf/, ~/.codex/, ~/.vscode/, ~/.config/zed/\n\
             Tip: rerun with `cortyx install --global` to scaffold the standard config files and Claude hooks.\n\n\
             Terminal quickstart:\n\
             - `{TERMINAL_ROUTE_EXAMPLE}` — verify local context without editor setup yet.\n\
             - `{WATCH_EXAMPLE}` — keep the index fresh during daily use.\n\
             - `{DOCTOR_EXAMPLE}` — diagnose install or index drift.\n\n\
             In your AI tool after install:\n\
             - `{MCP_CAPABILITY_EXAMPLE}` — capability summary.\n\
             - `{MCP_TASK_EXAMPLE}` — one-entrypoint task start.\n\n\
             ux-proof: {}",
            install_ux_proof(0, 0, 0)
        ));
    }

    let mut registered_count = 0;
    let mut already_count = 0;
    let mut hook_created_count = 0;

    for client in &clients {
        match register_mcp_server(client, &exe) {
            Ok(true) => {
                registered_count += 1;
            },
            Ok(false) => {
                already_count += 1;
            },
            Err(err) => {
                eprintln!(
                    "Warning: failed to register {} MCP server: {err}",
                    client.name
                );
            },
        }
    }

    // Write Claude Code hooks if ~/.claude/ was detected
    let claude_dir = home.join(".claude");
    if claude_dir.exists() {
        let hooks_dir = claude_dir.join("hooks");
        match write_hook_scripts(&hooks_dir, &exe) {
            Ok(true) => {
                hook_created_count = 4; // session-start + close + precompact + post-tool-use
            },
            Ok(false) => {
                // Scripts already exist
            },
            Err(err) => {
                eprintln!("Warning: failed to write Claude Code hooks: {err}");
            },
        }
    }

    let summary = format!(
        "Cortyx MCP server registration complete!\n\
         - Registered: {registered_count} client(s)\n\
         - Already configured: {already_count} client(s)\n\
         - Claude hooks created: {hook_created_count} script(s)\n\n\
         Terminal quickstart:\n\
         - `{TERMINAL_ROUTE_EXAMPLE}` — route queries from the command line.\n\
         - `{WATCH_EXAMPLE}` — auto-rebuild the index on file changes.\n\
         - `{DOCTOR_EXAMPLE}` — diagnose any install or index issues.\n\n\
         In your AI tool (after restart):\n\
         - `{MCP_CAPABILITY_EXAMPLE}` — list available Cortyx capabilities.\n\
         - `{MCP_TASK_EXAMPLE}` — start a context-aware task.\n\n\
         ux-proof: {}",
        install_ux_proof(registered_count, already_count, scaffolded_count)
    );

    Ok(summary)
}
