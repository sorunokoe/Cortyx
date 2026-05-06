use super::temporal_anchor_extractors::{
    parse_temporal_anchor_query, RelativeTemporalRecallAnswerKind, RelativeTemporalRecallQuery,
    TemporalAnchorQuery, TemporalElapsedGapQuery, TemporalIntervalQuery,
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

fn assert_no_answer(idx: &NeuronIndex, task: &str) {
    assert!(
        idx.derived_answer_path_for_task(task).is_none(),
        "unexpected synthetic answer for {task}"
    );
}

#[test]
fn synthetic_temporal_anchor_answers_elapsed_before_event() {
    let dir = TempDir::new().unwrap();
    let mut idx = make_index(&dir);
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "binoculars.conv.md",
        "User: I did manage to sneak in some birding time a week ago when I took a walk around my neighborhood after dinner. I did notice that the American goldfinches seem to be returning to the area.\n\
         User: Speaking of my new binoculars, I remember that I got them exactly three weeks ago, after months of waiting.\n",
    );
    let answer = read_answer_text(
        &idx,
        "How long did I use my new binoculars before I saw the American goldfinches returning to the area?",
    );
    assert!(answer.contains("Answer: two weeks"), "{answer}");
}

#[test]
fn synthetic_temporal_anchor_answers_days_before_event_with_inclusive_option() {
    let dir = TempDir::new().unwrap();
    let mut idx = make_index(&dir);
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "birthday.conv.md",
        "User: I ordered her gift, a customized photo album, on the 15th of April for my best friend's birthday and it turned out amazing!\n\
         User: I had a great time celebrating my best friend's 30th birthday party recently, it was on the 22nd of April.\n",
    );
    let answer = read_answer_text(
        &idx,
        "How many days before my best friend's birthday party did I order her gift?",
    );
    assert!(answer.contains("Answer: 7 days"), "{answer}");
    assert!(answer.contains("8 days"), "{answer}");
}

#[test]
fn synthetic_temporal_anchor_answers_days_after_event_with_inclusive_option() {
    let dir = TempDir::new().unwrap();
    let mut idx = make_index(&dir);
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "shipping.conv.md",
        "User: I ordered a new remote shutter release online on February 5th after I realized I lost my old one during a photo walk.\n\
         User: I just got a new remote shutter release that arrived on February 10th, and it's been a game-changer.\n",
    );
    let answer = read_answer_text(
        &idx,
        "How many days did it take for me to receive the new remote shutter release after I ordered it?",
    );
    assert!(answer.contains("Answer: 5 days"), "{answer}");
    assert!(answer.contains("6 days"), "{answer}");
}

#[test]
fn temporal_anchor_query_parses_between_event_intervals() {
    let parsed = parse_temporal_anchor_query(
        "how many days had passed between the sunday mass at st. mary's church and the ash wednesday service at the cathedral?",
    );
    assert_eq!(
        parsed,
        Some(TemporalAnchorQuery::Interval(TemporalIntervalQuery {
            start_phrase: "the sunday mass at st. mary's church".to_string(),
            end_phrase: "the ash wednesday service at the cathedral".to_string(),
            required_terms: vec![
                "ash".to_string(),
                "cathedral".to_string(),
                "church".to_string(),
                "mary".to_string(),
                "mass".to_string(),
                "service".to_string(),
                "st".to_string(),
                "sunday".to_string(),
                "wednesday".to_string(),
            ],
        }))
    );

    let parsed_without_had = parse_temporal_anchor_query(
        "how many days passed between my visit to the museum of modern art (moma) and the ancient civilizations exhibit at the metropolitan museum of art?",
    );
    assert!(matches!(
        parsed_without_had,
        Some(TemporalAnchorQuery::Interval(_))
    ));
}

#[test]
fn synthetic_temporal_anchor_answers_days_between_events_with_inclusive_option() {
    let dir = TempDir::new().unwrap();
    let mut idx = make_index(&dir);
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "church.conv.md",
        "User: I went to the Sunday mass at St. Mary's Church on February 11th and it was such a peaceful service.\n\
         User: I attended the Ash Wednesday service at the cathedral on March 13th, and the reflection really stayed with me.\n",
    );
    let answer = read_answer_text(
        &idx,
        "How many days had passed between the Sunday mass at St. Mary's Church and the Ash Wednesday service at the cathedral?",
    );
    assert!(answer.contains("Answer: 30 days"), "{answer}");
    assert!(answer.contains("31 days"), "{answer}");
}

#[test]
fn temporal_anchor_query_parses_elapsed_gap_since_when() {
    let parsed = parse_temporal_anchor_query(
        "how many days had passed since i finished reading 'the seven husbands of evelyn hugo' when i attended the book reading event at the local library?",
    );
    match parsed {
        Some(TemporalAnchorQuery::ElapsedGap(TemporalElapsedGapQuery {
            start_phrase,
            end_phrase,
            unit,
            required_terms,
        })) => {
            assert_eq!(
                start_phrase,
                "i finished reading 'the seven husbands of evelyn hugo'"
            );
            assert_eq!(
                end_phrase,
                "i attended the book reading event at the local library"
            );
            assert_eq!(unit, "day");
            assert!(required_terms.contains(&"reading".to_string()));
            assert!(required_terms.contains(&"library".to_string()));
            assert!(required_terms.contains(&"evelyn".to_string()));
        },
        other => panic!("unexpected parse result: {other:?}"),
    }
}

#[test]
fn synthetic_temporal_anchor_answers_days_passed_since_event_when_event() {
    let dir = TempDir::new().unwrap();
    let mut idx = make_index(&dir);
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "reading-gap.conv.md",
        "[Session 1 - 5 February, 2023]\n\
         User: I finished reading 'The Seven Husbands of Evelyn Hugo' today and I can't stop thinking about it.\n\
         [Session 2 - 23 February, 2023]\n\
         User: I attended the book reading event at the local library today, where the author of 'The Silent Patient' was discussing a new thriller.\n",
    );
    let answer = read_answer_text(
        &idx,
        "How many days had passed since I finished reading 'The Seven Husbands of Evelyn Hugo' when I attended the book reading event at the local library?",
    );
    assert!(answer.contains("Answer: 18 days"), "{answer}");
}

#[test]
fn synthetic_temporal_anchor_answers_days_it_took_after_start_event() {
    let dir = TempDir::new().unwrap();
    let mut idx = make_index(&dir);
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "housing-gap.conv.md",
        "[Session 1 - 1 March, 2023]\n\
         User: I started working with Rachel today to find a house I really love.\n\
         [Session 2 - 15 March, 2023]\n\
         User: I finally found a house I loved today and I'm getting ready to make an offer.\n",
    );
    let answer = read_answer_text(
        &idx,
        "How many days did it take for me to find a house I loved after starting to work with Rachel?",
    );
    assert!(answer.contains("Answer: 14 days"), "{answer}");
}

#[test]
fn synthetic_temporal_anchor_abstains_on_device_identity_mismatch() {
    let dir = TempDir::new().unwrap();
    let mut idx = make_index(&dir);
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "holiday.conv.md",
        "User: I attended the annual Holiday Market at the local mall a week before Black Friday.\n\
         User: I got my iPhone 13 Pro at a discounted price from Best Buy on Black Friday.\n",
    );
    assert_no_answer(
        &idx,
        "How many days before I bought my iPad did I attend the Holiday Market?",
    );
}

#[test]
fn synthetic_temporal_anchor_abstains_without_object_specific_purchase() {
    let dir = TempDir::new().unwrap();
    let mut idx = make_index(&dir);
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "backpack.conv.md",
        "User: I just started using my new laptop backpack and it's been great. It arrived on 1/20.\n\
         User: I'm thinking of getting a new wireless mouse, but I haven't bought it yet.\n",
    );
    assert_no_answer(
        &idx,
        "How many days did it take for my iPad case to arrive after I bought it?",
    );
}

#[test]
fn synthetic_temporal_anchor_abstains_on_case_type_collision() {
    let dir = TempDir::new().unwrap();
    let mut idx = make_index(&dir);
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "shipping-collision.conv.md",
        "[Session 1 - 15 January, 2023]\n\
         User: I bought a new laptop backpack today because my old one finally tore.\n\
         [Session 2 - 20 January, 2023]\n\
         User: My new laptop backpack arrived today and it fits everything perfectly.\n\
         [Session 3 - 22 January, 2023]\n\
         User: I'm still deciding whether to buy an iPad case.\n\
         [Session 4 - 24 January, 2023]\n\
         User: I ended up getting a phone wallet case today after comparing a few styles.\n\
         User: The phone wallet case arrived two days later and it's been handy.\n",
    );
    assert_no_answer(
        &idx,
        "How many days did it take for my iPad case to arrive after I bought it?",
    );
}

#[test]
fn temporal_anchor_query_parses_anchored_relative_recall() {
    let parsed =
        parse_temporal_anchor_query("as of 7 february, 2023, which book did i finish a week ago?");
    match parsed {
        Some(TemporalAnchorQuery::RelativeRecall(RelativeTemporalRecallQuery {
            target_day,
            prompt_body,
            focus_terms,
            answer_kind,
        })) => {
            assert_eq!(target_day, ymd_to_days(2023, 1, 31));
            assert_eq!(prompt_body, "which book did i finish a week ago");
            assert_eq!(answer_kind, RelativeTemporalRecallAnswerKind::BookTitle);
            assert!(focus_terms.contains(&"book".to_string()));
            assert!(focus_terms.contains(&"finish".to_string()));
        },
        other => panic!("unexpected parse result: {other:?}"),
    }
}

#[test]
fn synthetic_temporal_anchor_answers_anchored_relative_book_recall() {
    let dir = TempDir::new().unwrap();
    let mut idx = make_index(&dir);
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "books.conv.md",
        "[Session 1 - 31 January, 2023]\n\
         User: I just finished a historical fiction novel, \"The Nightingale\" by Kristin Hannah, today and I loved it.\n\
         User: I also picked up a copy of \"The Alice Network\" for next month.\n",
    );
    let answer = read_answer_text(
        &idx,
        "As of 7 February, 2023, Which book did I finish a week ago?",
    );
    assert!(
        answer.contains("Answer: 'The Nightingale' by Kristin Hannah"),
        "{answer}"
    );
}

#[test]
fn synthetic_temporal_anchor_answers_anchored_relative_event_clause() {
    let dir = TempDir::new().unwrap();
    let mut idx = make_index(&dir);
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "business.conv.md",
        "[Session 2 - 1 March, 2023]\n\
         User: I signed a contract with my first client today, and I'm still buzzing about finally landing that milestone.\n",
    );
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "noise.conv.md",
        "[Session 1 - 28 February, 2023]\n\
         User: I recently had a skin tag removed from my neck and I'm still taking antibiotics for pneumonia today. It feels like a milestone just to be getting back to normal.\n",
    );
    let answer = read_answer_text(
        &idx,
        "As of 28 March, 2023, What was the significant buisiness milestone I mentioned four weeks ago?",
    );
    assert!(
        answer.contains("Answer: I signed a contract with my first client."),
        "{answer}"
    );
}

#[test]
fn synthetic_temporal_anchor_answers_anchored_relative_partner_activity() {
    let dir = TempDir::new().unwrap();
    let mut idx = make_index(&dir);
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "rachel.conv.md",
        "[Session 1 - 1 February, 2023]\n\
         User: I'm still not sure about the keyboards, but I was thinking about my ukulele lessons. I just started taking ukulele lessons with my friend Rachel today and it's been really fun so far. Can you give me some tips on how to practice effectively and improve my chord changes?\n",
    );
    let answer = read_answer_text(
        &idx,
        "As of 1 April, 2023, What did I do with Rachel on the Wednesday two months ago?",
    );
    assert!(
        answer.contains("Answer: I started taking ukulele lessons with Rachel."),
        "{answer}"
    );
}

#[test]
fn synthetic_temporal_anchor_answers_anchored_relative_source_person() {
    let dir = TempDir::new().unwrap();
    let mut idx = make_index(&dir);
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "gift.conv.md",
        "[Session 1 - 4 March, 2023]\n\
         User: I recently acquired a beautiful vintage armchair from the 1950s and I want to make sure I'm taking good care of it. By the way, I also got a stunning crystal chandelier from my aunt today, which used to belong to my great-grandmother.\n",
    );
    let answer = read_answer_text(
        &idx,
        "As of 9 March, 2023, I received a piece of jewelry last Saturday from whom?",
    );
    assert!(answer.contains("Answer: my aunt"), "{answer}");
}

#[test]
fn synthetic_temporal_anchor_answers_anchored_relative_direct_object() {
    let dir = TempDir::new().unwrap();
    let mut idx = make_index(&dir);
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "cake.conv.md",
        "[Session 1 - 10 April, 2022]\n\
         User: I'm excited to try making croissants again, and I think I'll also make some banana bread for the dinner party. I recently made a batch with walnuts, and it turned out amazing. By the way, I just baked a chocolate cake for my friend's birthday party last weekend that turned out amazing. It was a new recipe I found online that used espresso powder to intensify the chocolate flavor.\n",
    );
    let answer = read_answer_text(
        &idx,
        "As of 12 April, 2022, I mentioned cooking something for my friend a couple of days ago. What was it?",
    );
    assert!(answer.contains("Answer: a chocolate cake"), "{answer}");
}
