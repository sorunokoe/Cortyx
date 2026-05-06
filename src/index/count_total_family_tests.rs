use super::count_total_extractors::{parse_count_total_query, CountTotalQuery};
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

fn read_answer_text(idx: &NeuronIndex, task: &str) -> String {
    let path = idx
        .derived_answer_path_for_task(task)
        .expect("expected synthetic answer");
    std::fs::read_to_string(path).unwrap()
}

#[test]
fn parse_count_total_query_understands_metric_bundles() {
    let query = parse_count_total_query(
        "What is the total number of goals and assists I have in the recreational indoor soccer league?",
        "what is the total number of goals and assists i have in the recreational indoor soccer league?",
    )
    .unwrap();
    assert!(matches!(query, CountTotalQuery::MetricBundle(_)));
}

#[test]
fn synthetic_count_total_answers_goals_and_assists_bundle() {
    let dir = TempDir::new().unwrap();
    let mut idx = make_index(&dir);
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "soccer.conv.md",
        "User: I'm also playing in a recreational indoor soccer league, and I've scored 3 goals so far.\n\
         User: I've been playing indoor soccer with my colleagues from work and I've had two assists in the league so far.\n",
    );
    let answer = read_answer_text(
        &idx,
        "What is the total number of goals and assists I have in the recreational indoor soccer league?",
    );
    assert!(answer.contains("Answer: 5"), "{answer}");
}

#[test]
fn synthetic_count_total_answers_lunch_meal_bundle() {
    let dir = TempDir::new().unwrap();
    let mut idx = make_index(&dir);
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "meals.conv.md",
        "User: I just had the best lunch today. This is the third meal I got from my chicken fajitas.\n\
         User: I just made a big batch of lentil soup that lasted me for 5 lunches.\n",
    );
    let answer = read_answer_text(
        &idx,
        "What is the total number of lunch meals I got from the chicken fajitas and lentil soup?",
    );
    assert!(answer.contains("Answer: 8 meals"), "{answer}");
}

#[test]
fn synthetic_count_total_answers_online_course_totals() {
    let dir = TempDir::new().unwrap();
    let mut idx = make_index(&dir);
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "courses.conv.md",
        "User: Since I've already completed 12 courses on Coursera, I'm confident that I have a solid foundation in data analysis.\n\
         User: I'm glad I already have a solid foundation in data analysis from my previous 8 edX courses, so I'm confident that I can focus on machine learning concepts in this course.\n\
         User: I started an online course with weekly video lectures in May.\n",
    );
    let answer = read_answer_text(
        &idx,
        "What is the total number of online courses I've completed?",
    );
    assert!(answer.contains("Answer: 20"), "{answer}");
}

#[test]
fn online_course_total_surfaces_keep_generic_how_many_query_on_legacy_route() {
    let dir = TempDir::new().unwrap();
    let mut idx = make_index(&dir);
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "courses-high.conv.md",
        "User: Since I've already completed 12 courses on Coursera, I'm confident that I have a solid foundation in data analysis.\n\
         User: I'm glad I already have a solid foundation in data analysis from my previous 8 edX courses.\n",
    );
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "courses-generic.conv.md",
        "User: I've already completed 3 courses on Coursera, so I have a good foundation to build upon.\n\
         User: Since I've completed two courses on edX, I'm familiar with the online learning format.\n",
    );

    let answer = read_answer_text(&idx, "How many online courses have I completed in total?");
    assert!(answer.contains("Answer: 5"), "{answer}");
}
