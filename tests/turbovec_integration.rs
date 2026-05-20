//! Integration tests for the TurboVec-backed embedding store.

#![cfg(feature = "embed")]

#[cfg(target_os = "macos")]
extern crate blas_src;

use std::collections::HashSet;
use std::path::{Path, PathBuf};

use cortyx::embedder::{
    load_embeddings, save_embeddings, unit_norm, EmbeddingBackend, EmbeddingStore,
};
use cortyx::index::NeuronIndex;
use cortyx::neuron::{NeuronKind, NeuronMeta};
use tempfile::TempDir;

fn embeddings_index_path(root: &Path) -> PathBuf {
    root.join(".cortyx").join("embeddings.tvim")
}

fn embeddings_raw_path(root: &Path) -> PathBuf {
    root.join(".cortyx").join("embeddings.bin")
}

fn one_hot(dim: usize, idx: usize) -> Vec<f32> {
    let mut vec = vec![0.0; dim];
    vec[idx] = 1.0;
    vec
}

fn orthogonal_unit(query_vec: &[f32]) -> Vec<f32> {
    let basis_idx = query_vec
        .iter()
        .enumerate()
        .min_by(|a, b| a.1.abs().total_cmp(&b.1.abs()))
        .map(|(idx, _)| idx)
        .unwrap_or(0);
    let mut basis = vec![0.0; query_vec.len()];
    basis[basis_idx] = 1.0;
    let projection: f32 = basis.iter().zip(query_vec.iter()).map(|(a, b)| a * b).sum();
    for (value, query) in basis.iter_mut().zip(query_vec.iter()) {
        *value -= projection * query;
    }
    unit_norm(basis)
}

fn index_core_neuron(
    idx: &mut NeuronIndex,
    root: &Path,
    file_name: &str,
    module: Option<&str>,
    content: &str,
) -> PathBuf {
    let path = root.join(".cortyx").join("neurons").join(file_name);
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    std::fs::write(&path, content).unwrap();
    let mut meta = NeuronMeta::new_stub(root, NeuronKind::Core);
    meta.module = module.map(str::to_string);
    idx.index_neuron(&path, content, &meta);
    path
}

#[test]
fn migration_from_legacy_raw_store_rebuilds_turbovec_index() {
    let dir = TempDir::new().unwrap();
    let path = PathBuf::from(".cortyx/neurons/legacy.context.md");
    let mut store = EmbeddingStore::new();
    store.insert(path.clone(), one_hot(384, 0));
    save_embeddings(dir.path(), &store).unwrap();
    std::fs::remove_file(embeddings_index_path(dir.path())).unwrap();

    let loaded = load_embeddings(dir.path());

    assert!(embeddings_raw_path(dir.path()).exists());
    assert!(embeddings_index_path(dir.path()).exists());
    let results = loaded.search(&one_hot(384, 0), 1);
    assert_eq!(results[0].1, path);
}

#[test]
fn round_trip_insert_and_search_returns_top_match() {
    let dir = TempDir::new().unwrap();
    let p1 = PathBuf::from(".cortyx/neurons/a.context.md");
    let p2 = PathBuf::from(".cortyx/neurons/b.context.md");
    let p3 = PathBuf::from(".cortyx/neurons/c.context.md");

    let mut store = EmbeddingStore::new();
    store.insert(p1.clone(), one_hot(384, 0));
    store.insert(p2.clone(), one_hot(384, 1));
    store.insert(p3.clone(), one_hot(384, 2));
    save_embeddings(dir.path(), &store).unwrap();

    let loaded = load_embeddings(dir.path());
    let results = loaded.search(&one_hot(384, 1), 3);

    assert_eq!(results.first().map(|(_, path)| path), Some(&p2));
}

#[test]
fn module_masked_search_only_returns_allowed_paths() {
    let p1 = PathBuf::from(".cortyx/neurons/auth.context.md");
    let p2 = PathBuf::from(".cortyx/neurons/ui.context.md");
    let p3 = PathBuf::from(".cortyx/neurons/db.context.md");

    let mut store = EmbeddingStore::new();
    store.insert(p1.clone(), one_hot(384, 0));
    store.insert(p2.clone(), one_hot(384, 1));
    store.insert(p3.clone(), one_hot(384, 2));

    let allow_paths = HashSet::from([p1.clone(), p3.clone()]);
    let results = store.search_filtered(&one_hot(384, 1), 3, &allow_paths);

    assert!(!results.is_empty());
    assert!(results.iter().all(|(_, path)| allow_paths.contains(path)));
    assert!(!results.iter().any(|(_, path)| path == &p2));
}

#[test]
fn prepare_is_idempotent_and_stable() {
    let p1 = PathBuf::from(".cortyx/neurons/a.context.md");
    let p2 = PathBuf::from(".cortyx/neurons/b.context.md");

    let mut store = EmbeddingStore::new();
    store.insert(p1.clone(), one_hot(384, 0));
    store.insert(p2.clone(), one_hot(384, 1));

    store.prepare();
    let before = store.search(&one_hot(384, 0), 2);
    store.prepare();
    let after = store.search(&one_hot(384, 0), 2);

    assert_eq!(before, after);
}

#[test]
fn len_is_empty_and_update_semantics_are_correct() {
    let path = PathBuf::from(".cortyx/neurons/update.context.md");
    let mut store = EmbeddingStore::new();

    assert!(store.is_empty());
    assert_eq!(store.len(), 0);

    store.insert(path.clone(), one_hot(384, 0));
    assert!(!store.is_empty());
    assert_eq!(store.len(), 1);
    assert!(store.contains(path.as_path()));

    store.insert(path.clone(), one_hot(384, 1));
    assert_eq!(store.len(), 1);
    assert_eq!(store.search(&one_hot(384, 1), 1)[0].1, path);
}

#[test]
fn neuron_index_dense_search_can_promote_ann_match() {
    let dir = TempDir::new().unwrap();
    let mut idx = NeuronIndex::load_or_create(dir.path()).unwrap();
    let query = "oauth authorization callback";

    let lexical_path = index_core_neuron(
        &mut idx,
        dir.path(),
        "lexical.context.md",
        Some("auth"),
        "oauth authorization callback oauth authorization callback oauth authorization callback",
    );
    let semantic_path = index_core_neuron(
        &mut idx,
        dir.path(),
        "semantic.context.md",
        Some("auth"),
        "identity provider browser redirect sign in flow",
    );
    idx.rebuild_derived_pub();

    let baseline = idx.get_contexts(query, 4096, None, None);
    assert_eq!(baseline.first(), Some(&lexical_path));

    if std::env::var_os("CORTYX_RUN_EMBED_MODEL_TESTS").is_none() {
        return;
    }

    let backend = match EmbeddingBackend::new() {
        Ok(backend) => backend,
        Err(_) => return,
    };
    let query_vec = unit_norm(backend.embed_query(query).unwrap());
    let orthogonal = orthogonal_unit(&query_vec);

    let mut store = EmbeddingStore::new();
    store.insert(lexical_path.clone(), orthogonal);
    store.insert(semantic_path.clone(), query_vec.clone());
    save_embeddings(dir.path(), &store).unwrap();
    idx.reload_embeddings();

    let fused = idx.get_contexts(query, 4096, None, None);
    assert_eq!(fused.first(), Some(&semantic_path));
}
