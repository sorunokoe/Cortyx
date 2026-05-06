use super::quantity_total_extractors::{parse_quantity_total_query, QuantityTotalQuery};
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
fn parse_quantity_total_query_handles_place_bundle_days() {
    let query = parse_quantity_total_query(
        "what is the total number of days i spent in japan and chicago?",
    )
    .unwrap();
    let QuantityTotalQuery::StayDays(query) = query else {
        panic!("expected stay-days query");
    };
    assert_eq!(
        query.places,
        vec!["japan".to_string(), "chicago".to_string()]
    );
}

#[test]
fn synthetic_quantity_total_answers_four_road_trip_distance() {
    let dir = TempDir::new().unwrap();
    let mut idx = make_index(&dir);
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "road_trip_bundle.conv.md",
        "User: I'm planning another road trip and I'd like to know the best route from Denver to Mount Rushmore. By the way, speaking of road trips, I just got back from an amazing 4-day trip to Yellowstone National Park with my family last month, where we covered a total of 1,200 miles.\n\
         User: I'm glad I could get some helpful information about the shuttle service. By the way, I was thinking about our Yellowstone trip last month, and I realized that we drove around 300 miles on the first day to reach Jackson, Wyoming.\n\
         User: I'm glad I could fit in Maroon Lake. Since I've covered a total of 1,800 miles on my recent three road trips, including a solo trip to Durango, a weekend trip to Breckenridge, and a family trip to Santa Fe, I'm comfortable with the drive and exploring new scenic spots.\n",
    );

    let answer = read_answer_text(
        &idx,
        "What is the total distance I covered in my four road trips?",
    );
    assert!(answer.contains("Answer: 3,000 miles"), "{answer}");
}

#[test]
fn synthetic_quantity_total_answers_consecutive_weekend_hikes() {
    let dir = TempDir::new().unwrap();
    let mut idx = make_index(&dir);
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "weekend_hikes.conv.md",
        "User: I'm looking for some yoga classes near my new apartment. By the way, I've been enjoying the outdoors a lot lately, just did a 3-mile loop trail at Valley of Fire State Park last weekend.\n\
         User: I'm planning a road trip to the Grand Canyon in January. Oh, and by the way, I just got back from an amazing 5-mile hike at Red Rock Canyon two weekends ago - the views from the top of the ridge were incredible!\n",
    );

    let answer = read_answer_text(
        &idx,
        "What is the total distance of the hikes I did on two consecutive weekends?",
    );
    assert!(answer.contains("Answer: 8 miles"), "{answer}");
}

#[test]
fn synthetic_quantity_total_answers_japan_and_chicago_days() {
    let dir = TempDir::new().unwrap();
    let mut idx = make_index(&dir);
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "travel_days.conv.md",
        "User: I'm actually looking for some good Italian restaurants, I had some great Italian food during my last 4-day trip to Chicago.\n\
         User: I'm planning a trip to Asia and I was wondering if you could recommend some must-visit places in Tokyo. I went to Japan before from April 15th to 22nd, and I fell in love with the city.\n",
    );

    let answer = read_answer_text(
        &idx,
        "What is the total number of days I spent in Japan and Chicago?",
    );
    assert!(answer.contains("Answer: 11 days"), "{answer}");
    assert!(answer.contains("12 days"), "{answer}");
    assert!(answer.contains("April 15th to 22nd"), "{answer}");
}
