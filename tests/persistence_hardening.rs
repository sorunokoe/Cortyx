#![allow(missing_docs)]

use cortyx::index::NeuronIndex;
use cortyx::neuron::{NeuronKind, NeuronMeta};
use serde_json::json;
use std::fs;
use std::path::{Path, PathBuf};
use tempfile::TempDir;

fn cortyx_dir(root: &Path) -> PathBuf {
    root.join(".cortyx")
}

fn neurons_dir(root: &Path) -> PathBuf {
    cortyx_dir(root).join("neurons")
}

fn index_path(root: &Path) -> PathBuf {
    cortyx_dir(root).join("index.json")
}

fn checksum_path(root: &Path) -> PathBuf {
    cortyx_dir(root).join("index.checksum")
}

fn wal_path(root: &Path) -> PathBuf {
    cortyx_dir(root).join("index.wal")
}

fn activation_cache_path(root: &Path) -> PathBuf {
    cortyx_dir(root).join("index.fast.bin")
}

fn sidecar_path(neuron_path: &Path) -> PathBuf {
    let file_name = neuron_path
        .file_name()
        .and_then(|name| name.to_str())
        .expect("neuron file name");
    let sidecar_name = file_name
        .strip_suffix(".md")
        .map(|name| format!("{name}.json"))
        .expect("markdown neuron file");
    neuron_path
        .parent()
        .expect("neuron parent")
        .join(sidecar_name)
}

fn write_neuron(root: &Path, name: &str, content: &str) -> (PathBuf, NeuronMeta) {
    let neuron_path = neurons_dir(root).join(name);
    fs::create_dir_all(neuron_path.parent().expect("neuron parent")).expect("create neuron dir");
    fs::write(&neuron_path, content).expect("write neuron markdown");

    let source_name = name.replace(['/', '.'], "_");
    let meta = NeuronMeta::new_stub(
        &root.join("src").join(format!("{source_name}.rs")),
        NeuronKind::Core,
    );
    fs::write(
        sidecar_path(&neuron_path),
        serde_json::to_string_pretty(&meta).expect("serialize neuron meta"),
    )
    .expect("write neuron meta");
    (neuron_path, meta)
}

fn stage_neuron(idx: &mut NeuronIndex, root: &Path, name: &str, content: &str) -> PathBuf {
    let (neuron_path, meta) = write_neuron(root, name, content);
    idx.stage(&neuron_path, content, &meta);
    neuron_path
}

fn build_base_index(root: &Path, base_count: usize) -> (NeuronIndex, Vec<PathBuf>) {
    let mut idx = NeuronIndex::load_or_create(root).expect("load empty index");
    let paths = (0..base_count)
        .map(|i| {
            stage_neuron(
                &mut idx,
                root,
                &format!("base-{i}.context.md"),
                &format!("base neuron {i} authentication token"),
            )
        })
        .collect::<Vec<_>>();
    idx.commit().expect("commit base index");
    (idx, paths)
}

fn list_paths(idx: &NeuronIndex) -> Vec<PathBuf> {
    idx.list_neurons(None)
        .into_iter()
        .map(|neuron| neuron.path)
        .collect()
}

fn extract_new_wal_entry_payloads(root: &Path) -> Vec<String> {
    fs::read_to_string(wal_path(root))
        .expect("read wal")
        .lines()
        .skip(1)
        .filter(|line| !line.is_empty())
        .map(|line| {
            line.split_once('\t')
                .expect("wal entry delimiter")
                .1
                .to_string()
        })
        .collect()
}

#[test]
fn wal_with_corrupted_middle_entry_replays_only_clean_entries() {
    let dir = TempDir::new().expect("tempdir");
    let root = dir.path();
    let (mut idx, _base_paths) = build_base_index(root, 12);

    let delta_paths = [
        stage_neuron(
            &mut idx,
            root,
            "delta-0.context.md",
            "delta zero cache strategy",
        ),
        stage_neuron(
            &mut idx,
            root,
            "delta-1.context.md",
            "delta one cache invalidation",
        ),
        stage_neuron(
            &mut idx,
            root,
            "delta-2.context.md",
            "delta two cache warming",
        ),
    ];
    idx.commit().expect("commit wal append");
    assert!(
        wal_path(root).exists(),
        "wal should exist after append save"
    );

    let wal_file = wal_path(root);
    let original = fs::read_to_string(&wal_file).expect("read wal text");
    let had_trailing_newline = original.ends_with('\n');
    let mut lines = original
        .lines()
        .map(ToOwned::to_owned)
        .collect::<Vec<String>>();
    let crc = lines[2].split_once('\t').expect("crc delimiter").0;
    let replacement = if crc.starts_with('0') { '1' } else { '0' };
    lines[2].replace_range(0..1, &replacement.to_string());
    let mut corrupted = lines.join("\n");
    if had_trailing_newline {
        corrupted.push('\n');
    }
    fs::write(&wal_file, corrupted).expect("write corrupted wal");
    let _ = fs::remove_file(activation_cache_path(root));

    let reloaded = NeuronIndex::load_or_create(root).expect("reload with corrupted wal");
    let paths = list_paths(&reloaded);
    assert_eq!(
        paths.len(),
        14,
        "base entries plus two clean WAL entries should load"
    );
    assert!(paths.contains(&delta_paths[0]));
    assert!(!paths.contains(&delta_paths[1]));
    assert!(paths.contains(&delta_paths[2]));
}

#[test]
fn corrupt_activation_cache_is_ignored_and_rebuilt() {
    let dir = TempDir::new().expect("tempdir");
    let root = dir.path();
    let (_idx, base_paths) = build_base_index(root, 12);

    let cache_path = activation_cache_path(root);
    assert!(
        cache_path.exists(),
        "activation cache should exist before corruption"
    );
    let mut bytes = fs::read(&cache_path).expect("read activation cache");
    let last = bytes.len() - 1;
    bytes[last] ^= 0xFF;
    fs::write(&cache_path, bytes).expect("corrupt activation cache");

    let reloaded = NeuronIndex::load_or_create(root).expect("reload with corrupt activation cache");
    let paths = list_paths(&reloaded);
    assert_eq!(paths.len(), 12);
    for path in base_paths {
        assert!(paths.contains(&path));
    }
}

#[test]
fn legacy_wal_format_replays_cleanly() {
    let dir = TempDir::new().expect("tempdir");
    let root = dir.path();
    let (mut idx, _base_paths) = build_base_index(root, 6);

    let legacy_entry_path = stage_neuron(
        &mut idx,
        root,
        "legacy.context.md",
        "legacy wal replay path",
    );
    idx.commit().expect("commit wal append");
    let entry_payloads = extract_new_wal_entry_payloads(root);
    assert_eq!(entry_payloads.len(), 1, "expected one WAL payload");
    let entry_json: serde_json::Value =
        serde_json::from_str(&entry_payloads[0]).expect("parse wal entry payload");

    let payload = serde_json::to_vec(&json!({
        "base_count": 6,
        "entries": [entry_json],
    }))
    .expect("serialize legacy wal payload");
    let mut legacy_wal = blake3::hash(&payload).as_bytes().to_vec();
    legacy_wal.extend_from_slice(&payload);
    fs::write(wal_path(root), legacy_wal).expect("write legacy wal");
    let _ = fs::remove_file(activation_cache_path(root));

    let reloaded = NeuronIndex::load_or_create(root).expect("reload with legacy wal");
    let paths = list_paths(&reloaded);
    assert_eq!(paths.len(), 7);
    assert!(paths.contains(&legacy_entry_path));
}

#[test]
fn checksum_mismatch_rebuilds_index_from_neuron_files() {
    let dir = TempDir::new().expect("tempdir");
    let root = dir.path();
    let (_idx, base_paths) = build_base_index(root, 3);

    let mut bytes = fs::read(checksum_path(root)).expect("read checksum file");
    bytes[0] ^= 0xFF;
    fs::write(checksum_path(root), bytes).expect("corrupt checksum file");

    let reloaded = NeuronIndex::load_or_create(root).expect("reload after checksum mismatch");
    let paths = list_paths(&reloaded);
    assert_eq!(
        paths.len(),
        3,
        "index should rebuild from persisted neurons"
    );
    for path in base_paths {
        assert!(paths.contains(&path));
    }
    assert!(index_path(root).exists(), "rebuilt index.json should exist");
    assert!(
        checksum_path(root).exists(),
        "rebuilt checksum should exist"
    );
}

#[test]
fn index_without_checksum_loads_cleanly_and_save_restores_checksum() {
    let dir = TempDir::new().expect("tempdir");
    let root = dir.path();
    let (_idx, base_paths) = build_base_index(root, 2);

    fs::remove_file(checksum_path(root)).expect("remove checksum file");
    let _ = fs::remove_file(activation_cache_path(root));

    let reloaded = NeuronIndex::load_or_create(root).expect("reload without checksum file");
    let paths = list_paths(&reloaded);
    assert_eq!(paths.len(), 2);
    for path in base_paths {
        assert!(paths.contains(&path));
    }
    reloaded.save().expect("save reloaded index");
    assert!(
        checksum_path(root).exists(),
        "save should restore checksum sidecar"
    );
}
