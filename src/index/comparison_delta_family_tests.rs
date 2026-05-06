use super::comparison_delta_extractors::{parse_comparison_delta_query, ComparisonDeltaQuery};
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
fn synthetic_age_average_delta_prefers_grouped_raw_conversation() {
    let dir = TempDir::new().unwrap();
    let mut idx = make_index(&dir);
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "salary.conv.md",
        "User: I'm a digital marketing manager and the salary range is $80,000 to $110,000.\n",
    );

    let raw_chunk_zero = dir
        .path()
        .join(".cortyx")
        .join("neurons")
        .join("age_conv_0000_chunk.verbatim.md");
    let raw_chunk_one = dir
        .path()
        .join(".cortyx")
        .join("neurons")
        .join("age_conv_0001_chunk.verbatim.md");
    std::fs::create_dir_all(raw_chunk_zero.parent().unwrap()).unwrap();
    std::fs::write(
        &raw_chunk_zero,
        "User: Considering the average age of employees in my department is 29.5 years old, I think I'm not too far off from that demographic.\n",
    )
    .unwrap();
    std::fs::write(
        &raw_chunk_one,
        "User: By the way, I'm currently 32 years old, so I want to make sure I'm using products that are suitable for my skin at this stage.\n",
    )
    .unwrap();

    let answer = read_answer_text(
        &idx,
        "How much older am I than the average age of employees in my department?",
    );
    assert!(answer.contains("Answer: 2.5 years"), "{answer}");
}

#[test]
fn parses_discount_comparison_query() {
    let query = parse_comparison_delta_query(
        "did i receive a higher percentage discount on my first order from hellofresh, compared to my first ubereats order?",
    )
    .expect("expected comparison query");
    let ComparisonDeltaQuery::DiscountComparison(query) = query else {
        panic!("expected discount comparison query");
    };
    assert_eq!(
        query.required_terms,
        vec![
            "discount".to_string(),
            "hellofresh".to_string(),
            "ubereats".to_string(),
        ]
    );
}

#[test]
fn synthetic_discount_comparison_answers_yes_for_hellofresh_over_ubereats() {
    let dir = TempDir::new().unwrap();
    let mut idx = make_index(&dir);
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "discounts.conv.md",
        "User: I'm thinking of trying out meal kit delivery services and was wondering if you have any recommendations or promotions available. By the way, I recently tried HelloFresh and got a 40% discount on my first order, which was a great deal!\n\
         User: I'm planning to order food from UberEats again this week and I was wondering if you could help me find some good deals or promo codes. By the way, last week I got 20% off my UberEats order, which was awesome!\n",
    );

    let answer = read_answer_text(
        &idx,
        "Did I receive a higher percentage discount on my first order from HelloFresh, compared to my first UberEats order?",
    );
    assert!(answer.contains("Answer: yes"), "{answer}");
}

#[test]
fn synthetic_savings_money_delta_answers_train_over_taxi() {
    let dir = TempDir::new().unwrap();
    let idx = make_index(&dir);
    let raw_chunk_zero = dir
        .path()
        .join(".cortyx")
        .join("neurons")
        .join("tokyo_conv_0000_chunk.verbatim.md");
    let raw_chunk_one = dir
        .path()
        .join(".cortyx")
        .join("neurons")
        .join("tokyo_conv_0001_chunk.verbatim.md");
    std::fs::create_dir_all(raw_chunk_zero.parent().unwrap()).unwrap();
    std::fs::write(
        &raw_chunk_zero,
        "User: I think I got the price from my friend wrong, yeah it's actually $10 to get to my hotel from the airport by train.\n",
    )
    .unwrap();
    std::fs::write(
        &raw_chunk_one,
        "User: By the way, I was told that taking a taxi from the airport to my hotel would cost around $60, which is a bit pricey for me.\n",
    )
    .unwrap();

    let answer = read_answer_text(
        &idx,
        "How much will I save by taking the train from the airport to my hotel instead of a taxi?",
    );
    assert!(answer.contains("Answer: $50"), "{answer}");
}
