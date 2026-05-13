//! CLI handler for `cortyx fleet` subcommands.

use std::path::PathBuf;

use crate::cli::FleetCommand;
use crate::error::Result;
use crate::fleet::{deregister_node, fleet_registry_path, load_registry, register_node};

pub fn run(sub: FleetCommand) -> Result<()> {
    match sub {
        FleetCommand::Register { path, alias } => {
            let project_path = path
                .unwrap_or_else(|| std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")));
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
                println!(
                    "  {} — {} ({} module(s), registered {})",
                    node.alias,
                    node.path.display(),
                    node.modules.len(),
                    node.last_registered,
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
            println!(
                "Fleet status: {} node(s), {} total module(s)\nRegistry: {}",
                registry.nodes.len(),
                total_modules,
                registry_path.display(),
            );
            for node in &registry.nodes {
                println!(
                    "  ✓ {} — {} module(s), last registered {}",
                    node.alias,
                    node.modules.len(),
                    node.last_registered,
                );
            }
            Ok(())
        },
    }
}
