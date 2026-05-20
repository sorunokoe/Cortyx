//! CLI handler for `cortyx patterns` subcommands.

use crate::cli::PatternsCommand;
use crate::error::Result;
use crate::miner::pattern_registry::{PatternRegistry, PATTERN_TOML_TEMPLATE};

/// # Errors
///
/// Returns an error if the underlying operation fails.
pub fn run(sub: PatternsCommand, project_root: &std::path::Path) -> Result<()> {
    match sub {
        PatternsCommand::List => {
            let registry = PatternRegistry::load(project_root);
            let (builtin, user) = registry.stats();
            println!(
                "Evidence pattern registry: {} built-in, {} user-defined\n",
                builtin, user
            );
            println!("{:<30} {:<22} {:<6} SOURCE", "NAME", "FAMILY", "CONF");
            println!("{}", "-".repeat(80));
            for p in &registry.patterns {
                let family = format!("{:?}", p.family);
                let source = if p.builtin { "built-in" } else { "user" };
                println!(
                    "{:<30} {:<22} {:<6.2} {}",
                    p.name, family, p.confidence, source
                );
                if let Some(desc) = &p.description {
                    println!("  ↳ {desc}");
                }
            }
            if user == 0 {
                println!(
                    "\nAdd domain-specific patterns with: cortyx patterns add <name>\n\
                     Pattern files live in: {}",
                    project_root.join(".cortyx").join("patterns").display()
                );
            }
            Ok(())
        },
        PatternsCommand::Add { name } => {
            let pattern_dir = project_root.join(".cortyx").join("patterns");
            std::fs::create_dir_all(&pattern_dir)?;
            let file_path = pattern_dir.join(format!("{name}.toml"));
            if file_path.exists() {
                println!(
                    "Pattern file already exists: {}\nEdit it directly to add patterns.",
                    file_path.display()
                );
                return Ok(());
            }
            std::fs::write(&file_path, PATTERN_TOML_TEMPLATE)?;
            println!(
                "✓ Created pattern file: {}\n\
                 Edit it to add your domain-specific patterns.\n\
                 Run `cortyx patterns list` to verify they load correctly.",
                file_path.display()
            );
            Ok(())
        },
    }
}
