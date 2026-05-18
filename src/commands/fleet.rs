//! CLI handler for `cortyx fleet` subcommands.

use std::path::PathBuf;

use crate::cli::FleetCommand;
use crate::error::Result;
use crate::fleet::{
    deregister_node, fleet_registry_path, load_registry, register_git_node, register_node,
    sync_fleet_node,
};

pub fn run(sub: FleetCommand) -> Result<()> {
    match sub {
        FleetCommand::Register {
            path,
            alias,
            git_url,
        } => {
            if let Some(url) = git_url {
                let alias = alias.ok_or_else(|| {
                    crate::cortyx_err!(
                        "--alias is required when registering a git-backed fleet node"
                    )
                })?;
                let node = register_git_node(&url, &alias)?;
                println!(
                    "✓ Registered git fleet node '{}' ({})\n  URL: {}\n  Path: {}\n  Modules: {}",
                    node.alias,
                    node.id,
                    url,
                    node.path.display(),
                    if node.modules.is_empty() {
                        "(none — run cortyx fleet sync to populate)".to_string()
                    } else {
                        node.modules.join(", ")
                    },
                );
            } else {
                let project_path = path.unwrap_or_else(|| {
                    std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."))
                });
                let node = register_node(&project_path, alias)?;
                println!(
                    "✓ Registered fleet node '{}' ({})\n  Path: {}\n  Modules: {}",
                    node.alias,
                    node.id,
                    node.path.display(),
                    if node.modules.is_empty() {
                        "(none)".to_string()
                    } else {
                        node.modules.join(", ")
                    },
                );
            }
            Ok(())
        },
        FleetCommand::Sync { alias } => {
            let mut registry = load_registry()?;
            if registry.nodes.is_empty() {
                println!("No fleet nodes registered.");
                return Ok(());
            }
            let nodes: Vec<_> = registry
                .nodes
                .iter()
                .filter(|n| n.git_url.is_some())
                .filter(|n| alias.as_deref().is_none_or(|a| n.alias == a))
                .cloned()
                .collect();
            if nodes.is_empty() {
                println!("No git-backed fleet nodes to sync.");
                return Ok(());
            }
            for mut node in nodes {
                print!("Syncing '{}'…", node.alias);
                match sync_fleet_node(&node) {
                    Ok(()) => {
                        use crate::fleet::sync::update_last_fetched;
                        update_last_fetched(&mut node);
                        if let Some(n) = registry.nodes.iter_mut().find(|n| n.alias == node.alias) {
                            n.last_fetched = node.last_fetched;
                        }
                        println!(" ✓");
                    },
                    Err(e) => println!(" ✗ {e}"),
                }
            }
            crate::fleet::save_registry(&registry)?;
            Ok(())
        },
        FleetCommand::Deregister { target } => {
            let removed = deregister_node(&target)?;
            if removed {
                println!("✓ Deregistered fleet node '{target}'");
            } else {
                println!("No fleet node found matching '{target}'");
            }
            Ok(())
        },
        FleetCommand::List => {
            let registry = load_registry()?;
            if registry.nodes.is_empty() {
                println!("No fleet nodes registered. Run: cortyx fleet register <path>");
                return Ok(());
            }
            println!("Fleet nodes ({}):", registry.nodes.len());
            for node in &registry.nodes {
                let kind = if node.git_url.is_some() {
                    "git"
                } else {
                    "local"
                };
                let sync_info = node
                    .last_fetched
                    .as_deref()
                    .map(|t| format!(", synced {t}"))
                    .unwrap_or_default();
                println!(
                    "  [{kind}] {} — {} ({} module(s), registered{}{})",
                    node.alias,
                    node.path.display(),
                    node.modules.len(),
                    node.last_registered,
                    sync_info,
                );
            }
            Ok(())
        },
        FleetCommand::Status => {
            let registry_path = fleet_registry_path()?;
            if !registry_path.exists() {
                println!("Fleet not configured. Run: cortyx fleet register <path>");
                return Ok(());
            }
            let registry = load_registry()?;
            let total_modules: usize = registry.nodes.iter().map(|n| n.modules.len()).sum();
            let git_count = registry
                .nodes
                .iter()
                .filter(|n| n.git_url.is_some())
                .count();
            println!(
                "Fleet status: {} node(s) ({} git-backed), {} total module(s)\nRegistry: {}",
                registry.nodes.len(),
                git_count,
                total_modules,
                registry_path.display(),
            );
            for node in &registry.nodes {
                let kind = if node.git_url.is_some() {
                    "git"
                } else {
                    "local"
                };
                println!(
                    "  ✓ [{}] {} — {} module(s), last registered {}",
                    kind,
                    node.alias,
                    node.modules.len(),
                    node.last_registered,
                );
            }
            Ok(())
        },
    }
}
