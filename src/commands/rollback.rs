//! Rollback command - restore neuron sections from shadow copies.

use crate::error::Result;
use crate::neuron::{
    atomic_write, atomic_write_json, latest_shadow, meta_path, pop_shadow, replace_section,
    NeuronMeta,
};
use std::path::Path;

/// E2 (TRIZ R14): Restore a single neuron section from its shadow copy in the sidecar JSON.
///
/// Before each evolve_context or evolve_section call, Cortyx saves the previous content
/// to `meta.shadow_sections[key]`. This function restores the latest saved step and
/// leaves older shadows available for repeated rollback calls.
///
/// # Errors
///
/// Returns an error if the underlying operation fails.
pub fn run_section(neuron: &Path, section: &str) -> Result<()> {
    let meta_p = meta_path(neuron);
    let data = std::fs::read_to_string(&meta_p)
        .map_err(|e| crate::cortyx_err!("Cannot read sidecar for {}: {e}", neuron.display()))?;
    let mut meta: NeuronMeta =
        serde_json::from_str(&data).map_err(|e| crate::cortyx_err!("Cannot parse sidecar: {e}"))?;

    let shadow = latest_shadow(&meta.shadow_sections, section)
        .ok_or_else(|| {
            crate::cortyx_err!(
                "No shadow for section '{}' in {}. Shadows are saved before each evolve call.",
                section,
                neuron.display()
            )
        })?
        .to_string();

    if section == "_full" {
        atomic_write(neuron, shadow.as_bytes())?;
        pop_shadow(&mut meta.shadow_sections, "_full");
        atomic_write_json(&meta_p, &meta)?;
        println!("✓ Restored full neuron {} from shadow.", neuron.display());
    } else {
        let current = std::fs::read_to_string(neuron)
            .map_err(|e| crate::cortyx_err!("Cannot read neuron file: {e}"))?;
        let restored = replace_section(&current, section, &shadow);
        atomic_write(neuron, restored.as_bytes())?;
        pop_shadow(&mut meta.shadow_sections, section);
        atomic_write_json(&meta_p, &meta)?;
        println!("✓ Restored section '{}' in {}.", section, neuron.display());
    }

    Ok(())
}
