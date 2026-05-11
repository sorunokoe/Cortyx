//! Claude Code hook script generation and registration.

use crate::error::Result;
use crate::installer::{
    registration::{ensure_parent_dir, load_json_object_or_default},
    utils::dirs_home,
};
use std::path::Path;

/// Write Claude Code hook scripts for auto-save and hook-side health checks.
///
/// - `cortyx-close-hook.sh`: called on Claude Code Stop event → validates the index is readable
/// - `cortyx-precompact-hook.sh`: called on PreCompact event → incremental compile
///
/// Returns `Ok(true)` if written, `Ok(false)` if already present.
pub(super) fn write_hook_scripts(hooks_dir: &Path, exe: &Path) -> Result<bool> {
    std::fs::create_dir_all(hooks_dir)?;

    let close_hook = hooks_dir.join("cortyx-close-hook.sh");
    let precompact_hook = hooks_dir.join("cortyx-precompact-hook.sh");
    let scripts_written = !(close_hook.exists() && precompact_hook.exists());

    let exe_str = exe.to_string_lossy();
    // Shell-safe single-quote escape: replace ' with '\'' so the path is safe
    // even if it contains double-quotes, spaces, or other special characters.
    let exe_safe = exe_str.replace('\'', "'\\''");

    let close_content = format!(
        "#!/usr/bin/env bash\n\
         # Cortyx Stop hook (S3 — NE2)\n\
         # Called by Claude Code on session Stop — validates the project index is readable.\n\
         # Written by cortyx install. Do not edit manually.\n\
         set -euo pipefail\n\
         PROJECT=\"${{CLAUDE_WORKING_DIR:-$(pwd)}}\"\n\
         '{exe_safe}' hook-check --project \"$PROJECT\"\n"
    );
    let precompact_content = format!(
        "#!/usr/bin/env bash\n\
         # Cortyx PreCompact hook (S3 — NE2)\n\
         # Called by Claude Code before context window compaction — re-indexes changed files.\n\
         # Written by cortyx install. Do not edit manually.\n\
         set -euo pipefail\n\
         PROJECT=\"${{CLAUDE_WORKING_DIR:-$(pwd)}}\"\n\
         '{exe_safe}' compile \"$PROJECT\" --incremental\n"
    );

    if scripts_written {
        crate::neuron::atomic_write(&close_hook, close_content.as_bytes())?;
        crate::neuron::atomic_write(&precompact_hook, precompact_content.as_bytes())?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&close_hook, std::fs::Permissions::from_mode(0o755))?;
            std::fs::set_permissions(&precompact_hook, std::fs::Permissions::from_mode(0o755))?;
        }
    }

    register_claude_hooks(&close_hook, &precompact_hook)?;

    Ok(scripts_written)
}

/// Register hook scripts in Claude Code settings.json.
fn register_claude_hooks(close_hook: &Path, precompact_hook: &Path) -> Result<()> {
    let home = dirs_home();
    let settings = home.join(".claude").join("settings.json");
    register_claude_hooks_in_settings(&settings, close_hook, precompact_hook)
}

/// Register hook scripts in a specific settings file.
fn register_claude_hooks_in_settings(
    settings: &Path,
    close_hook: &Path,
    precompact_hook: &Path,
) -> Result<()> {
    let mut json = load_json_object_or_default(settings, "Claude Code settings")?;
    let root = json.as_object_mut().ok_or_else(|| {
        crate::cortyx_err!(
            "Claude Code settings {} must contain a top-level JSON object",
            settings.display()
        )
    })?;
    if !root.contains_key("hooks") {
        root.insert(
            "hooks".to_string(),
            serde_json::Value::Object(serde_json::Map::new()),
        );
    }
    let hooks = root
        .get_mut("hooks")
        .and_then(serde_json::Value::as_object_mut)
        .ok_or_else(|| {
            crate::cortyx_err!(
                "Claude Code settings {} has non-object hooks",
                settings.display()
            )
        })?;

    let close_cmd = close_hook.to_string_lossy().to_string();
    let precompact_cmd = precompact_hook.to_string_lossy().to_string();
    append_hook_command(hooks, "Stop", &close_cmd)?;
    append_hook_command(hooks, "PreCompact", &precompact_cmd)?;

    ensure_parent_dir(settings)?;
    crate::neuron::atomic_write_json(settings, &json)?;
    Ok(())
}

/// Append a command to a hook array if not already present.
fn append_hook_command(
    hooks: &mut serde_json::Map<String, serde_json::Value>,
    event: &str,
    command: &str,
) -> Result<()> {
    if !hooks.contains_key(event) {
        hooks.insert(event.to_string(), serde_json::Value::Array(Vec::new()));
    }
    let commands = hooks
        .get_mut(event)
        .and_then(serde_json::Value::as_array_mut)
        .ok_or_else(|| crate::cortyx_err!("Claude Code hooks.{event} must be an array"))?;
    if !commands.iter().any(|value| value.as_str() == Some(command)) {
        commands.push(serde_json::Value::String(command.to_string()));
    }
    Ok(())
}
