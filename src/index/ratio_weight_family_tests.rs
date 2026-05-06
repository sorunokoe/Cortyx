use super::ratio_weight_extractors::{
    extract_percentage_part_fact_from_line, extract_percentage_whole_fact_from_line,
    parse_ratio_weight_query, RatioWeightQuery,
};
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
fn synthetic_ratio_weight_answers_feed_weight_total() {
    let dir = TempDir::new().unwrap();
    let mut idx = make_index(&dir);
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "feed.conv.md",
        "User: I got a 50-pound batch of layer feed, and I'm trying to track my expenses for the farm.\n\
         User: I also bought 20 pounds of organic scratch grains for my chickens recently.\n",
    );
    let answer = read_answer_text(
        &idx,
        "What is the total weight of the new feed I purchased in the past two months?",
    );
    assert!(answer.contains("Answer: 70 pounds"), "{answer}");
}

#[test]
fn synthetic_ratio_weight_answers_leadership_percentage() {
    let dir = TempDir::new().unwrap();
    let mut idx = make_index(&dir);
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "leadership.conv.md",
        "User: We have a total of 100 leadership positions across the company.\n\
         User: I recently attended a workshop on gender equality and was impressed to learn that women occupy 20 of the leadership positions in our company.\n",
    );
    let answer = read_answer_text(
        &idx,
        "What percentage of leadership positions do women hold in the my company?",
    );
    assert!(answer.contains("Answer: 20%"), "{answer}");
}

#[test]
fn synthetic_ratio_weight_answers_renovation_price_percentage() {
    let dir = TempDir::new().unwrap();
    let mut idx = make_index(&dir);
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "property.conv.md",
        "User: I'm looking at a countryside property. It's listed at $200,000, which seems like a good deal.\n\
         User: My renovations, which I estimate will cost around $20,000, include adding a deck and a patio.\n",
    );
    let answer = read_answer_text(
        &idx,
        "What percentage of the countryside property's price is the cost of the renovations I plan to do on my current house?",
    );
    assert!(answer.contains("Answer: 10%"), "{answer}");
}

#[test]
fn ratio_weight_money_percentage_extractors_parse_grounded_pair() {
    let RatioWeightQuery::Percentage(query) = parse_ratio_weight_query(
        "What percentage of the countryside property's price is the cost of the renovations I plan to do on my current house?",
        "what percentage of the countryside property's price is the cost of the renovations i plan to do on my current house?",
    )
    .expect("expected percentage query")
    else {
        panic!("expected percentage query");
    };

    let whole = extract_percentage_whole_fact_from_line(
        "User: I'm looking at a countryside property. It's listed at $200,000, which seems like a good deal.",
        "user: i'm looking at a countryside property. it's listed at $200,000, which seems like a good deal.",
        &query,
    )
    .expect("expected whole fact");
    assert_eq!(whole.value, 20_000_000);

    let part = extract_percentage_part_fact_from_line(
        "User: My renovations, which I estimate will cost around $20,000, include adding a deck and a patio.",
        "user: my renovations, which i estimate will cost around $20,000, include adding a deck and a patio.",
        &query,
    )
    .expect("expected part fact");
    assert_eq!(part.value, 2_000_000);
}
