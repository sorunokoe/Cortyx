use std::fs;
use std::path::{Path, PathBuf};

use cortyx::agent_memory::{parse_structured_diary_entry, refine_entry};

mod common;
use common::run;

fn latest_diary_path(root: &Path) -> PathBuf {
    let neurons_dir = root.join(".cortyx").join("neurons");
    let mut diary_paths: Vec<_> = fs::read_dir(&neurons_dir)
        .unwrap()
        .filter_map(|entry| entry.ok())
        .map(|entry| entry.path())
        .filter(|path| {
            path.file_name()
                .map(|name| {
                    let name = name.to_string_lossy();
                    name.starts_with("diary_") && name.ends_with(".verbatim.md")
                })
                .unwrap_or(false)
        })
        .collect();
    diary_paths.sort();
    diary_paths.pop().expect("expected a diary neuron file")
}

#[test]
fn diary_refinement_populates_plan_for_vague_blocker() {
    let dir = tempfile::tempdir().unwrap();

    fs::write(
        dir.path().join("auth_notes.md"),
        "## Note\nAuthentication currently relies on a shared token validator.\n",
    )
    .unwrap();

    let mine = run(&["mine", "auth_notes.md"], dir.path());
    assert!(
        mine.status.success(),
        "mine failed: {}",
        String::from_utf8_lossy(&mine.stderr)
    );

    let content = "I need to fix authentication but I'm not sure about the approach\ntitle: auth-fix\nstatus: blocked\nblocker: unclear what the right design is";
    let write = run(
        &["diary-write", "--agent", "reviewer", "--content", content],
        dir.path(),
    );
    assert!(
        write.status.success(),
        "diary-write failed: {}",
        String::from_utf8_lossy(&write.stderr)
    );

    let read = run(&["diary-read", "--agent", "reviewer"], dir.path());
    assert!(
        read.status.success(),
        "diary-read failed: {}",
        String::from_utf8_lossy(&read.stderr)
    );
    let stdout = String::from_utf8_lossy(&read.stdout);
    assert!(
        stdout.contains("auth-fix"),
        "expected title in diary-read: {stdout}"
    );
    assert!(
        stdout.contains("blocked"),
        "expected status in diary-read: {stdout}"
    );

    let diary_path = latest_diary_path(dir.path());
    let stored = fs::read_to_string(diary_path).unwrap();
    let mut entry = parse_structured_diary_entry(&stored).expect("expected structured diary entry");

    assert!(entry.refined_plan.is_none());
    assert!(
        refine_entry(&mut entry),
        "expected refine_entry to populate a refined plan"
    );

    let refined_plan = entry.refined_plan.expect("expected refined plan");
    assert!(
        refined_plan.contains("sub-tasks") || refined_plan.contains("sub-"),
        "expected sub-task heuristic guidance, got: {refined_plan}"
    );
}
