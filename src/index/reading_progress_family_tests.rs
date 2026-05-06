use super::reading_progress_extractors::extract_just_finished_page_count;
use super::*;
use crate::neuron::{NeuronKind, NeuronMeta};
use tempfile::TempDir;

fn make_index(dir: &TempDir) -> NeuronIndex {
    NeuronIndex::load_or_create(dir.path()).unwrap()
}

fn index_verbatim_neuron(idx: &mut NeuronIndex, dir: &TempDir, file_name: &str, content: &str) {
    let path = dir.path().join(".cortyx").join("neurons").join(file_name);
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    std::fs::write(&path, content).unwrap();
    let meta = NeuronMeta::new_stub(dir.path(), NeuronKind::Verbatim);
    idx.index_neuron(&path, content, &meta);
    idx.rebuild_derived();
}

fn read_answer(path: PathBuf) -> String {
    std::fs::read_to_string(path).unwrap()
}

#[test]
fn finished_page_count_extractor_ignores_prior_books_in_same_line() {
    assert_eq!(
        extract_just_finished_page_count(
            "User: I just finished a 416-page novel, but before that, I read \"The Power\" in December, which had 341 pages and took me around 5 weeks to finish."
        ),
        Some(416)
    );
    assert_eq!(
        extract_just_finished_page_count(
            "User: I just finished a historical fiction novel, \"The Nightingale\" by Kristin Hannah, which had 440 pages and took me around 3 weeks to complete."
        ),
        Some(440)
    );
}

#[test]
fn synthetic_pages_left_supports_unquoted_title_queries() {
    let dir = TempDir::new().unwrap();
    let mut idx = make_index(&dir);
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "reading.conv.md",
        "User: I just finished a historical fiction novel, The Nightingale by Kristin Hannah, which had 440 pages and took me around 3 weeks to complete.\n\
         User: I'm currently on page 120 of The Nightingale and trying to finish it this month.\n",
    );

    let task = "How many pages do I have left in The Nightingale?";
    let answer = read_answer(
        idx.synthetic_reading_progress_answer(task, &task.to_ascii_lowercase())
            .expect("expected reading-progress answer"),
    );
    assert!(answer.contains("Answer: 320"), "{answer}");
}

#[test]
fn synthetic_pages_read_uses_current_page_for_quoted_title() {
    let dir = TempDir::new().unwrap();
    let mut idx = make_index(&dir);
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "reading.conv.md",
        "User: I'm currently on page 120 of \"The Nightingale\" and trying to finish it this month.\n",
    );

    let task = "How many pages have I read so far in \"The Nightingale\"?";
    let answer = read_answer(
        idx.synthetic_reading_progress_answer(task, &task.to_ascii_lowercase())
            .expect("expected reading-progress answer"),
    );
    assert!(answer.contains("Answer: 120"), "{answer}");
}

#[test]
fn synthetic_pages_left_abstains_when_only_reading_pace_is_known() {
    let dir = TempDir::new().unwrap();
    let mut idx = make_index(&dir);
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "reading.conv.md",
        "User: I've been reading \"Sapiens\" at a pace of 10-20 pages a week.\n\
         Assistant: Since you've been reading \"Sapiens\" at a pace of 10-20 pages a week, let's assume you read 15 pages per week.\n",
    );

    let task = "How many pages do I have left to read in \"Sapiens\"?";
    assert!(idx
        .synthetic_reading_progress_answer(task, &task.to_ascii_lowercase())
        .is_none());
}

#[test]
fn synthetic_page_count_of_two_novels_sums_finished_books() {
    let dir = TempDir::new().unwrap();
    let mut idx = make_index(&dir);
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "novels.conv.md",
        "User: I just finished a 416-page novel over the weekend.\n\
         User: I just finished a historical fiction novel, The Nightingale by Kristin Hannah, which had 440 pages and took me around 3 weeks to complete.\n",
    );

    let task = "What is the page count of the two novels I just finished?";
    let answer = read_answer(
        idx.synthetic_reading_progress_answer(task, &task.to_ascii_lowercase())
            .expect("expected novel page total"),
    );
    assert!(answer.contains("Answer: 856"), "{answer}");
}
