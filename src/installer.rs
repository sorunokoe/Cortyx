/// S1 (R15 NE1): cortyx install — auto-configure all detected LLM clients.
///
/// Detects config files for Claude Code, Cursor, Windsurf, and Codex.
/// Writes the `cortyx serve` MCP entry to each found config and optionally
/// writes Claude Code hook scripts for auto-save (S3).
///
/// Design (TRIZ P10 + P25 + P6):
/// - P10: Preliminary action — all registration happens at install time.
/// - P25: Self-service — uses `std::env::current_exe()` for the absolute binary path;
///        no marketplace or external plugin infrastructure needed.
/// - P6: Universality — one command registers Cortyx with every detected client.
///
/// Result: `cargo install cortyx && cortyx install` — matches MemPalace 1-cmd setup.
use anyhow::Result;
use std::path::PathBuf;

/// A detected LLM client config location.
struct ClientConfig {
    name: &'static str,
    config_path: PathBuf,
    kind: ConfigKind,
}

enum ConfigKind {
    /// JSON object with "mcpServers" key (Claude Code, Cursor, Windsurf, Codex)
    McpServersJson,
}

/// Run `cortyx install [--global]`.
///
/// Writes the MCP server entry for every detected client config and optionally
/// writes Claude Code hook scripts (S3).
///
/// Returns a human-readable summary of all actions taken.
pub fn run_install(global: bool) -> Result<String> {
    let exe = std::env::current_exe()
        .unwrap_or_else(|_| PathBuf::from("cortyx"));

    let home = dirs_home();
    let clients = detect_clients(&home);

    if clients.is_empty() {
        return Ok(
            "No LLM client configs detected.\n\
             Checked: ~/.claude/, ~/.cursor/, ~/.codeium/windsurf/, ~/.codex/\n\
             Add your config path manually or ensure the client is installed."
                .to_string(),
        );
    }

    let mut actions: Vec<String> = Vec::new();
    let mut registered = 0usize;
    let mut already = 0usize;

    for client in &clients {
        match register_mcp_server(client, &exe) {
            Ok(true) => {
                actions.push(format!("✓ Registered with {} ({})", client.name, client.config_path.display()));
                registered += 1;
            }
            Ok(false) => {
                actions.push(format!("  Already configured: {}", client.name));
                already += 1;
            }
            Err(e) => {
                actions.push(format!("✗ Failed to configure {}: {e}", client.name));
            }
        }
    }

    // S3: Write Claude Code hook scripts if Claude config was detected.
    let claude_hooks_dir = home.join(".claude").join("hooks");
    if claude_hooks_dir.parent().map(|p| p.exists()).unwrap_or(false) || global {
        match write_hook_scripts(&claude_hooks_dir, &exe) {
            Ok(true) => actions.push(format!("✓ Hook scripts written to {}", claude_hooks_dir.display())),
            Ok(false) => actions.push(format!("  Hook scripts already present: {}", claude_hooks_dir.display())),
            Err(e) => actions.push(format!("✗ Hook scripts failed: {e}")),
        }
    }

    let summary = format!(
        "cortyx install complete.\n\
         Registered: {registered} client(s). Already configured: {already}.\n\n\
         {}\n\n\
         Restart your AI tool to activate Cortyx.",
        actions.join("\n")
    );
    Ok(summary)
}

/// Detect all LLM client config paths that exist on this machine.
fn detect_clients(home: &PathBuf) -> Vec<ClientConfig> {
    let candidates: &[(&str, &[&str])] = &[
        ("Claude Code", &[".claude/settings.json", ".claude/claude_desktop_config.json"]),
        ("Cursor", &[".cursor/mcp.json"]),
        ("Windsurf", &[".codeium/windsurf/mcp_config.json"]),
        ("Codex CLI", &[".codex/config.json"]),
    ];

    let mut found = Vec::new();
    for (name, paths) in candidates {
        for rel in *paths {
            let full = home.join(rel);
            if full.exists() {
                found.push(ClientConfig {
                    name,
                    config_path: full,
                    kind: ConfigKind::McpServersJson,
                });
                break; // only first match per client
            }
        }
    }
    found
}

/// Write the MCP server entry to a client config.
/// Returns `Ok(true)` if written, `Ok(false)` if already present, `Err` on failure.
fn register_mcp_server(client: &ClientConfig, exe: &PathBuf) -> Result<bool> {
    let ConfigKind::McpServersJson = client.kind;

    let content = std::fs::read_to_string(&client.config_path).unwrap_or_else(|_| "{}".to_string());
    let mut json: serde_json::Value = serde_json::from_str(&content).unwrap_or(serde_json::json!({}));

    // Check if cortyx is already registered
    if let Some(servers) = json.get("mcpServers") {
        if servers.get("cortyx").is_some() {
            return Ok(false);
        }
    }

    // Add the entry
    let entry = serde_json::json!({
        "command": exe.to_string_lossy().as_ref(),
        "args": ["serve"]
    });
    json["mcpServers"]["cortyx"] = entry;

    let out = serde_json::to_string_pretty(&json)?;
    std::fs::write(&client.config_path, out.as_bytes())?;
    Ok(true)
}

/// S3: Write Claude Code hook scripts for auto-save (NE2 full close).
///
/// - `cortyx-close-hook.sh`: called on Claude Code Stop event → commits pending feedback
/// - `cortyx-precompact-hook.sh`: called on PreCompact event → incremental compile
///
/// Returns `Ok(true)` if written, `Ok(false)` if already present.
fn write_hook_scripts(hooks_dir: &PathBuf, exe: &PathBuf) -> Result<bool> {
    std::fs::create_dir_all(hooks_dir)?;

    let close_hook = hooks_dir.join("cortyx-close-hook.sh");
    let precompact_hook = hooks_dir.join("cortyx-precompact-hook.sh");

    if close_hook.exists() && precompact_hook.exists() {
        return Ok(false);
    }

    let exe_str = exe.to_string_lossy();

    let close_content = format!(
        "#!/usr/bin/env bash\n\
         # Cortyx auto-save hook (S3 — NE2)\n\
         # Called by Claude Code on session Stop — commits pending neuron feedback.\n\
         # Written by cortyx install. Do not edit manually.\n\
         set -euo pipefail\n\
         PROJECT=\"${{CLAUDE_WORKING_DIR:-$(pwd)}}\"\n\
         \"{exe_str}\" close-task-hook --project \"$PROJECT\" 2>/dev/null || true\n"
    );
    let precompact_content = format!(
        "#!/usr/bin/env bash\n\
         # Cortyx PreCompact hook (S3 — NE2)\n\
         # Called by Claude Code before context window compaction — re-indexes changed files.\n\
         # Written by cortyx install. Do not edit manually.\n\
         set -euo pipefail\n\
         PROJECT=\"${{CLAUDE_WORKING_DIR:-$(pwd)}}\"\n\
         \"{exe_str}\" compile \"$PROJECT\" --incremental 2>/dev/null || true\n"
    );

    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::write(&close_hook, close_content.as_bytes())?;
        std::fs::set_permissions(&close_hook, std::fs::Permissions::from_mode(0o755))?;
        std::fs::write(&precompact_hook, precompact_content.as_bytes())?;
        std::fs::set_permissions(&precompact_hook, std::fs::Permissions::from_mode(0o755))?;
    }
    #[cfg(not(unix))]
    {
        std::fs::write(&close_hook, close_content.as_bytes())?;
        std::fs::write(&precompact_hook, precompact_content.as_bytes())?;
    }

    // Attempt to register with Claude Code hook arrays in settings.json
    let _ = register_claude_hooks(hooks_dir, &close_hook, &precompact_hook);

    Ok(true)
}

/// Attempt to register the hook scripts in Claude Code's hooks arrays.
fn register_claude_hooks(
    _hooks_dir: &PathBuf,
    close_hook: &PathBuf,
    precompact_hook: &PathBuf,
) -> Result<()> {
    let home = dirs_home();
    let settings = home.join(".claude").join("settings.json");
    if !settings.exists() { return Ok(()); }

    let content = std::fs::read_to_string(&settings)?;
    let mut json: serde_json::Value = serde_json::from_str(&content)?;

    let close_cmd = close_hook.to_string_lossy().to_string();
    let precompact_cmd = precompact_hook.to_string_lossy().to_string();

    // Add to hooks.Stop array if not present
    let stop_hooks = json["hooks"]["Stop"]
        .as_array_mut()
        .map(|a| a.iter().any(|h| h.as_str() == Some(&close_cmd)))
        .unwrap_or(false);
    if !stop_hooks {
        json["hooks"]["Stop"]
            .as_array_mut()
            .get_or_insert(&mut vec![])
            .push(serde_json::json!(close_cmd));
        // If hooks.Stop didn't exist as array, create it
        if !json["hooks"]["Stop"].is_array() {
            json["hooks"]["Stop"] = serde_json::json!([close_cmd]);
        }
    }

    // Add to hooks.PreCompact array if not present
    let precompact_present = json["hooks"]["PreCompact"]
        .as_array()
        .map(|a| a.iter().any(|h| h.as_str() == Some(&precompact_cmd)))
        .unwrap_or(false);
    if !precompact_present {
        if !json["hooks"]["PreCompact"].is_array() {
            json["hooks"]["PreCompact"] = serde_json::json!([precompact_cmd]);
        } else if let Some(arr) = json["hooks"]["PreCompact"].as_array_mut() {
            arr.push(serde_json::json!(precompact_cmd));
        }
    }

    let out = serde_json::to_string_pretty(&json)?;
    std::fs::write(&settings, out.as_bytes())?;
    Ok(())
}

/// Return the home directory path.
fn dirs_home() -> PathBuf {
    std::env::var("HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("."))
}
