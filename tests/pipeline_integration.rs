use std::fs;
use std::path::PathBuf;

use cortyx::index::NeuronIndex;
use cortyx::neuron::{NeuronKind, NeuronMeta, Synapse, SynapseType};
use tempfile::TempDir;

struct CorpusPaths {
    recent: PathBuf,
    old: PathBuf,
    anchor: PathBuf,
    neighbor: PathBuf,
    session_anchor: PathBuf,
    session_sibling_a: PathBuf,
    session_sibling_b: PathBuf,
}

fn build_pipeline_corpus() -> (TempDir, NeuronIndex, CorpusPaths) {
    let dir = TempDir::new().unwrap();
    let mut idx = NeuronIndex::load_or_create(dir.path()).unwrap();
    let neuron_dir = dir.path().join(".cortyx").join("neurons");
    fs::create_dir_all(&neuron_dir).unwrap();

    let recent = neuron_dir.join("lme_2000_0_user.verbatim.md");
    let old = neuron_dir.join("lme_1000_0_user.verbatim.md");
    let anchor = neuron_dir.join("router.context.md");
    let neighbor = neuron_dir.join("oauth_helper.context.md");
    let session_anchor = neuron_dir.join("lme_3000_0_user.verbatim.md");
    let session_sibling_a = neuron_dir.join("lme_3000_1_assistant.verbatim.md");
    let session_sibling_b = neuron_dir.join("lme_3000_2_user.verbatim.md");

    write_neuron(
        &mut idx,
        &recent,
        "project timeline milestone recap latest status",
        {
            let mut meta = NeuronMeta::new_stub(dir.path(), NeuronKind::Verbatim);
            meta.timestamp = Some("2024-06-01T00:00:00Z".into());
            meta
        },
    );
    write_neuron(
        &mut idx,
        &old,
        "project timeline milestone recap original status",
        {
            let mut meta = NeuronMeta::new_stub(dir.path(), NeuronKind::Verbatim);
            meta.timestamp = Some("2023-01-01T00:00:00Z".into());
            meta
        },
    );
    write_neuron(&mut idx, &anchor, "router auth dispatch context bridge", {
        let mut meta = NeuronMeta::new_stub(dir.path(), NeuronKind::Core);
        meta.synapses.push(Synapse::new(
            neighbor.clone(),
            SynapseType::ConceptExpands,
            "integration graph".into(),
        ));
        meta
    });
    write_neuron(
        &mut idx,
        &neighbor,
        "redirect exchange callback helper",
        NeuronMeta::new_stub(dir.path(), NeuronKind::Core),
    );
    write_neuron(
        &mut idx,
        &session_anchor,
        "cabin itinerary lakeside vacation checklist",
        {
            let mut meta = NeuronMeta::new_stub(dir.path(), NeuronKind::Verbatim);
            meta.timestamp = Some("2024-05-10T00:00:00Z".into());
            meta
        },
    );
    write_neuron(
        &mut idx,
        &session_sibling_a,
        "sunrise photos and coffee by the dock",
        {
            let mut meta = NeuronMeta::new_stub(dir.path(), NeuronKind::Verbatim);
            meta.timestamp = Some("2024-05-10T00:05:00Z".into());
            meta
        },
    );
    write_neuron(
        &mut idx,
        &session_sibling_b,
        "campfire stories before heading home",
        {
            let mut meta = NeuronMeta::new_stub(dir.path(), NeuronKind::Verbatim);
            meta.timestamp = Some("2024-05-10T00:10:00Z".into());
            meta
        },
    );

    idx.rebuild_derived_pub();

    (
        dir,
        idx,
        CorpusPaths {
            recent,
            old,
            anchor,
            neighbor,
            session_anchor,
            session_sibling_a,
            session_sibling_b,
        },
    )
}

fn write_neuron(idx: &mut NeuronIndex, path: &PathBuf, content: &str, meta: NeuronMeta) {
    fs::write(path, content).unwrap();
    idx.index_neuron(path, content, &meta);
}

#[test]
fn recent_neuron_ranks_above_older_equivalent_match() {
    let (_dir, idx, paths) = build_pipeline_corpus();

    let results = idx.get_contexts("project timeline recap", 4096, None, None);
    let recent_pos = results
        .iter()
        .position(|path| path == &paths.recent)
        .unwrap();
    let old_pos = results.iter().position(|path| path == &paths.old).unwrap();

    assert!(
        recent_pos < old_pos,
        "recent neuron should outrank old one: {results:?}"
    );
}

#[test]
fn synapse_neighbor_surfaces_from_matching_anchor() {
    let (_dir, idx, paths) = build_pipeline_corpus();

    let results = idx.get_contexts("router auth bridge", 4096, None, None);

    assert!(
        results.contains(&paths.anchor),
        "anchor should remain present: {results:?}"
    );
    assert!(
        results.contains(&paths.neighbor),
        "neighbor should surface via traversal: {results:?}"
    );
}

#[test]
fn session_cluster_injects_top_session_siblings() {
    let (_dir, idx, paths) = build_pipeline_corpus();

    let results = idx.get_contexts("cabin itinerary vacation", 4096, None, None);

    assert!(
        results.contains(&paths.session_anchor),
        "anchor chunk should match directly: {results:?}"
    );
    assert!(
        results.contains(&paths.session_sibling_a),
        "first sibling should be injected: {results:?}"
    );
    assert!(
        results.contains(&paths.session_sibling_b),
        "second sibling should be injected: {results:?}"
    );
}
