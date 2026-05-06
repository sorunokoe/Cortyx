use std::path::{Path, PathBuf};

use crate::error::Result;

use super::super::io::{atomic_write_json, sidecar_path};
use super::super::meta::NeuronMeta;
use super::chain::{NeuronProvenance, PROVENANCE_VERSION};
use super::edit::ProvenanceEdit;

/// Map a `.context.md` path to its provenance sidecar.
pub fn provenance_path(neuron_md: &Path) -> PathBuf {
    sidecar_path(neuron_md, ".provenance.json")
}

/// Load provenance if the additive sidecar exists.
pub fn load_provenance(neuron_md: &Path) -> Result<Option<NeuronProvenance>> {
    let path = provenance_path(neuron_md);
    if !path.exists() {
        return Ok(None);
    }
    let data = std::fs::read_to_string(&path)?;
    let mut provenance: NeuronProvenance = serde_json::from_str(&data)?;
    if provenance.version == 0 {
        provenance.version = PROVENANCE_VERSION;
    }
    Ok(Some(provenance))
}

/// Persist a provenance sidecar next to the neuron markdown file.
pub fn save_provenance(neuron_md: &Path, provenance: &NeuronProvenance) -> Result<()> {
    let path = provenance_path(neuron_md);
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    atomic_write_json(&path, provenance)
}

/// Create the sidecar if needed and sync the stored identity fields from `NeuronMeta`.
pub fn ensure_provenance(neuron_md: &Path, meta: &NeuronMeta) -> Result<NeuronProvenance> {
    let mut provenance =
        load_provenance(neuron_md)?.unwrap_or_else(|| NeuronProvenance::from_meta(meta));
    provenance.sync_from_meta(meta);
    save_provenance(neuron_md, &provenance)?;
    Ok(provenance)
}

/// Append and persist a new provenance edit entry.
pub fn record_provenance_edit(
    neuron_md: &Path,
    meta: &NeuronMeta,
    edit: ProvenanceEdit,
) -> Result<NeuronProvenance> {
    let mut provenance =
        load_provenance(neuron_md)?.unwrap_or_else(|| NeuronProvenance::from_meta(meta));
    provenance.sync_from_meta(meta);
    provenance.append_edit(edit);
    save_provenance(neuron_md, &provenance)?;
    Ok(provenance)
}

/// Stable content hash for neuron edits.
///
/// Uses the same normalization as `sync::hash_sync_body` (CRLF→LF, strip trailing newlines,
/// BLAKE3) so provenance and sync hashes remain interchangeable.
pub fn provenance_content_hash(content: &str) -> String {
    let normalized = content.replace("\r\n", "\n").replace('\r', "\n");
    let normalized = normalized.trim_end_matches('\n');
    blake3::hash(normalized.as_bytes()).to_hex()[..32].to_string()
}

/// Append a provenance edit for the current neuron content, defaulting the content hash.
pub fn record_content_provenance_edit(
    neuron_md: &Path,
    meta: &NeuronMeta,
    content: &str,
    mut edit: ProvenanceEdit,
) -> Result<NeuronProvenance> {
    if edit.content_hash.is_none() {
        edit.content_hash = Some(provenance_content_hash(content));
    }
    record_provenance_edit(neuron_md, meta, edit)
}

#[cfg(test)]
mod tests {
    use super::super::super::kind::NeuronKind;
    use super::super::edit::{ProvenanceAuthor, ProvenanceOperation};
    use super::*;
    use std::fs;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU64, Ordering};

    static TEST_DIR_COUNTER: AtomicU64 = AtomicU64::new(0);

    struct TestDir {
        path: PathBuf,
    }

    impl TestDir {
        fn new(name: &str) -> Self {
            let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("target")
                .join("test-artifacts")
                .join(format!(
                    "{name}-{}-{}",
                    std::process::id(),
                    TEST_DIR_COUNTER.fetch_add(1, Ordering::Relaxed)
                ));
            fs::create_dir_all(&path).unwrap();
            Self { path }
        }
    }

    impl Drop for TestDir {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.path);
        }
    }

    fn test_meta(source_path: &str) -> NeuronMeta {
        let mut meta = NeuronMeta::new_stub(Path::new(source_path), NeuronKind::Core);
        meta.uuid = Some("uuid-1234".to_string());
        meta
    }

    #[test]
    fn provenance_path_uses_distinct_sidecar_suffix() {
        let path = Path::new("/project/.cortyx/neurons/src/engine_rs.context.md");
        assert_eq!(
            provenance_path(path),
            PathBuf::from("/project/.cortyx/neurons/src/engine_rs.context.provenance.json")
        );
    }

    #[test]
    fn ensure_provenance_backfills_identity_without_history() {
        let dir = TestDir::new("ensure-provenance");
        let neuron_path = dir.path.join("engine_rs.context.md");
        let meta = test_meta("src/engine.rs");

        let provenance = ensure_provenance(&neuron_path, &meta).unwrap();

        assert!(provenance_path(&neuron_path).exists());
        assert_eq!(provenance.version, PROVENANCE_VERSION);
        assert_eq!(provenance.neuron_uuid, meta.uuid);
        assert!(provenance.edit_history.is_empty());
    }

    #[test]
    fn record_provenance_edit_persists_revision_chain() {
        let dir = TestDir::new("record-provenance");
        let neuron_path = dir.path.join("engine_rs.context.md");
        let meta = test_meta("src/engine.rs");
        let author = ProvenanceAuthor {
            author_id: "local:alice@macbook".to_string(),
            display_name: None,
            device_id: None,
        };

        let created = record_provenance_edit(
            &neuron_path,
            &meta,
            ProvenanceEdit {
                operation: ProvenanceOperation::Create,
                author: Some(author.clone()),
                content_hash: Some("ctx-1".to_string()),
                summary: Some("bootstrap neuron".to_string()),
                edited_at: Some("2026-01-02T03:04:05Z".to_string()),
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(created.edit_history.len(), 1);

        let updated = record_provenance_edit(
            &neuron_path,
            &meta,
            ProvenanceEdit {
                operation: ProvenanceOperation::SectionUpdate,
                author: Some(author),
                section: Some("purpose".to_string()),
                content_hash: Some("ctx-2".to_string()),
                summary: Some("refine purpose".to_string()),
                edited_at: Some("2026-01-02T03:04:06Z".to_string()),
                ..Default::default()
            },
        )
        .unwrap();

        assert_eq!(updated.edit_history.len(), 2);
        assert_eq!(
            updated.edit_history[1].parent_edit_id.as_deref(),
            Some(updated.edit_history[0].edit_id.as_str())
        );

        let loaded = load_provenance(&neuron_path).unwrap().unwrap();
        assert_eq!(loaded, updated);
    }

    #[test]
    fn record_content_provenance_edit_defaults_to_sync_hash() {
        let dir = TestDir::new("record-content-provenance");
        let neuron_path = dir.path.join("engine_rs.context.md");
        let meta = test_meta("src/engine.rs");

        let provenance = record_content_provenance_edit(
            &neuron_path,
            &meta,
            "line one\r\nline two\r\n",
            ProvenanceEdit {
                operation: ProvenanceOperation::Update,
                summary: Some("refresh content".to_string()),
                ..Default::default()
            },
        )
        .unwrap();

        let expected_hash = provenance_content_hash("line one\nline two\n");
        assert_eq!(
            provenance.edit_history[0].content_hash.as_deref(),
            Some(expected_hash.as_str())
        );
    }
}
