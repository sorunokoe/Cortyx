//! Client detection logic.

use crate::installer::client::{ClientConfig, ConfigKind};
use std::path::Path;

/// Detect all LLM client configs on this machine.
///
/// If `global` is true, scaffolds a minimal `{}` config for every client that
/// isn't already present — making `cortyx install --global` a single-command
/// bootstrap for any supported AI tool.
///
/// Returns `(clients, scaffolded_count)`.
pub(super) fn detect_clients(home: &Path, global: bool) -> (Vec<ClientConfig>, usize) {
    let mut clients = Vec::new();
    let mut scaffolded = 0;

    let scaffold = |dir: &std::path::Path, path: &std::path::Path| {
        std::fs::create_dir_all(dir).ok();
        std::fs::write(path, b"{}").ok();
    };

    // 1. Claude Code: ~/.claude/config.json
    let claude_dir = home.join(".claude");
    let claude_config = claude_dir.join("config.json");
    if !claude_config.exists() && global {
        scaffold(&claude_dir, &claude_config);
        std::fs::create_dir_all(claude_dir.join("hooks")).ok();
        scaffolded += 1;
    }
    if claude_config.exists() {
        clients.push(ClientConfig {
            name: "Claude Code",
            config_path: claude_config,
            kind: ConfigKind::McpServersJson,
        });
    }

    // 2. Cursor: ~/.cursor/config.json
    let cursor_dir = home.join(".cursor");
    let cursor_config = cursor_dir.join("config.json");
    if !cursor_config.exists() && global {
        scaffold(&cursor_dir, &cursor_config);
        scaffolded += 1;
    }
    if cursor_config.exists() {
        clients.push(ClientConfig {
            name: "Cursor",
            config_path: cursor_config,
            kind: ConfigKind::McpServersJson,
        });
    }

    // 3. Windsurf: ~/.codeium/windsurf/User/settings.json
    let windsurf_dir = home.join(".codeium/windsurf/User");
    let windsurf_config = windsurf_dir.join("settings.json");
    if !windsurf_config.exists() && global {
        scaffold(&windsurf_dir, &windsurf_config);
        scaffolded += 1;
    }
    if windsurf_config.exists() {
        clients.push(ClientConfig {
            name: "Windsurf",
            config_path: windsurf_config,
            kind: ConfigKind::McpServersJson,
        });
    }

    // 4. Codex: ~/.codex/config.json
    let codex_dir = home.join(".codex");
    let codex_config = codex_dir.join("config.json");
    if !codex_config.exists() && global {
        scaffold(&codex_dir, &codex_config);
        scaffolded += 1;
    }
    if codex_config.exists() {
        clients.push(ClientConfig {
            name: "Codex",
            config_path: codex_config,
            kind: ConfigKind::McpServersJson,
        });
    }

    // 5. VS Code: ~/.vscode/settings.json (primary) or macOS app-data path
    let vscode_config1 = home.join(".vscode/settings.json");
    let vscode_config2 = home.join("Library/Application Support/Code/User/settings.json");
    if !vscode_config1.exists() && !vscode_config2.exists() && global {
        scaffold(&home.join(".vscode"), &vscode_config1);
        scaffolded += 1;
    }
    if vscode_config1.exists() {
        clients.push(ClientConfig {
            name: "VS Code",
            config_path: vscode_config1,
            kind: ConfigKind::McpServersJson,
        });
    } else if vscode_config2.exists() {
        clients.push(ClientConfig {
            name: "VS Code",
            config_path: vscode_config2,
            kind: ConfigKind::McpServersJson,
        });
    }

    // 6. Zed: ~/.config/zed/settings.json
    let zed_dir = home.join(".config/zed");
    let zed_config = zed_dir.join("settings.json");
    if !zed_config.exists() && global {
        scaffold(&zed_dir, &zed_config);
        scaffolded += 1;
    }
    if zed_config.exists() {
        clients.push(ClientConfig {
            name: "Zed",
            config_path: zed_config,
            kind: ConfigKind::McpServersJson,
        });
    }

    (clients, scaffolded)
}
