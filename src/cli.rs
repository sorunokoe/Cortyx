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
    },
}
