//! Local-first shared-sync transport/layout helpers.

use crate::error::Result;
use serde::de::DeserializeOwned;
use std::fs;
use std::io::ErrorKind;
use std::path::{Path, PathBuf};

mod envelope;
mod repository;
mod status;
#[cfg(test)]
mod tests;
mod types;

pub use envelope::*;
pub use repository::*;
pub use status::*;
pub use types::*;

pub const SYNC_TRANSPORT_VERSION: u32 = 1;

pub fn sync_transport_dir(project_root: &Path) -> PathBuf {
    project_root.join(".cortyx").join("sync")
}

fn read_json<T: DeserializeOwned>(path: &Path) -> Result<Option<T>> {
    if !path.exists() {
        return Ok(None);
    }
    let data = fs::read_to_string(path)?;
    Ok(Some(serde_json::from_str(&data)?))
}

fn clear_optional_file(path: &Path) -> Result<()> {
    match fs::remove_file(path) {
        Ok(()) => Ok(()),
        Err(err) if err.kind() == ErrorKind::NotFound => Ok(()),
        Err(err) => Err(err.into()),
    }
}

fn ensure_parent_dir(path: &Path) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    Ok(())
}
