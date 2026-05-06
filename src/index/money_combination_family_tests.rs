use super::money_combination_extractors::extract_spend_focus_fact_from_line;
use super::money_queries::parse_money_query;
use super::money_support::MoneyQuery;
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
fn synthetic_sale_minimum_sums_floor_values() {
    let dir = TempDir::new().unwrap();
    let mut idx = make_index(&dir);
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "heirlooms.conv.md",
        "User: I'm also thinking of selling my vintage diamond necklace, which is worth $5,000.\n\
         User: By following these tips, you should be able to sell your restored vanity online for at least $150, and possibly more.\n",
    );

    let answer = read_answer_text(
        &idx,
        "What is the minimum amount I could get if I sold the vintage diamond necklace and the antique vanity?",
    );
    assert!(answer.contains("Answer: $5,150"), "{answer}");
}

#[test]
fn synthetic_spend_sum_handles_recipients_and_ignores_broader_totals() {
    let dir = TempDir::new().unwrap();
    let mut idx = make_index(&dir);
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "gifts.conv.md",
        "User: I know I spent a total of $500 on gifts recently, but I'm having trouble breaking it down.\n\
         User: I did get my brother a really nice graduation gift in May - a $100 gift card to his favorite electronics store.\n\
         User: I purchased a set of adorable baby clothes and toys from Buy Buy Baby for my coworker's baby shower, totaling $100.\n",
    );

    let direct_path = idx
        .synthetic_money_combination_answer(
            "What is the total amount I spent on gifts for my coworker and brother?",
            "what is the total amount i spent on gifts for my coworker and brother?",
        )
        .expect("expected direct money-combination answer");
    let direct_answer = std::fs::read_to_string(direct_path).unwrap();
    assert!(direct_answer.contains("Answer: $200"), "{direct_answer}");

    let answer = read_answer_text(
        &idx,
        "What is the total amount I spent on gifts for my coworker and brother?",
    );
    assert!(answer.contains("Answer: $200"), "{answer}");
}

#[test]
fn parse_spend_sum_query_builds_gift_recipient_focuses() {
    let query = parse_money_query(
        "What is the total amount I spent on gifts for my coworker and brother?",
        "what is the total amount i spent on gifts for my coworker and brother?",
    )
    .unwrap();
    let MoneyQuery::SpendSum(query) = query else {
        panic!("expected spend-sum query");
    };
    assert_eq!(query.focuses.len(), 2);
    assert_eq!(
        query.focuses[0].required_terms,
        vec!["coworker".to_string(), "gift".to_string()]
    );
    assert_eq!(
        query.focuses[1].required_terms,
        vec!["brother".to_string(), "gift".to_string()]
    );
    assert!(extract_spend_focus_fact_from_line(
        "User: I did get my brother a really nice graduation gift in May - a $100 gift card to his favorite electronics store.",
        "user: i did get my brother a really nice graduation gift in may - a $100 gift card to his favorite electronics store.",
        &query.focuses[1],
    )
    .is_some());
    assert!(extract_spend_focus_fact_from_line(
        "user: I'm trying to get a better grip on my finances, especially when it comes to gift-giving. I've been tracking my expenses and noticed I've been spending a lot on gifts lately. Can you help me come up with a budgeting plan to stick to, so I don't overspend? By the way, speaking of gifts, I once spent $60 on some coffee mugs for my coworkers, and it was a bit of a splurge, but they loved them.",
        "user: i'm trying to get a better grip on my finances, especially when it comes to gift-giving. i've been tracking my expenses and noticed i've been spending a lot on gifts lately. can you help me come up with a budgeting plan to stick to, so i don't overspend? by the way, speaking of gifts, i once spent $60 on some coffee mugs for my coworkers, and it was a bit of a splurge, but they loved them.",
        &query.focuses[0],
    )
    .is_none());
}

#[test]
fn parse_spend_sum_query_builds_single_gift_recipient_focus() {
    let query = parse_money_query(
        "How much did I spend on gifts for my sister?",
        "how much did i spend on gifts for my sister?",
    )
    .unwrap();
    let MoneyQuery::RecipientGiftTotal(query) = query else {
        panic!("expected recipient-gift-total query");
    };
    assert_eq!(query.focus.display, "gift for sister");
    assert_eq!(
        query.focus.required_terms,
        vec!["gift".to_string(), "sister".to_string()]
    );
}

#[test]
fn parse_spend_sum_query_understands_total_cost_item_bundles() {
    let query = parse_money_query(
        "What is the total cost of the new food bowl, measuring cup, dental chews, and flea and tick collar I got for Max?",
        "what is the total cost of the new food bowl, measuring cup, dental chews, and flea and tick collar i got for max?",
    )
    .unwrap();
    let MoneyQuery::SpendSum(query) = query else {
        panic!("expected spend-sum query");
    };
    assert_eq!(query.focuses.len(), 4);
    assert_eq!(query.focuses[0].display, "the new food bowl");
    assert_eq!(query.focuses[1].display, "measuring cup");
    assert_eq!(query.focuses[2].display, "dental chews");
    assert_eq!(query.focuses[3].display, "flea and tick collar");
}

#[test]
fn synthetic_spend_sum_handles_single_recipient_totals() {
    let dir = TempDir::new().unwrap();
    let mut idx = make_index(&dir);
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "sister-gifts.conv.md",
        "User: I'm trying to get some ideas for my sister's birthday gift next year. Last time I got her a gift card to her favorite spa for $100 and she loved it.\n\
         User: I recently got my sister a silver necklace with a small pendant from Tiffany's that cost around $200, and she loved it.\n",
    );

    let direct_path = idx
        .synthetic_money_combination_answer(
            "How much did I spend on gifts for my sister?",
            "how much did i spend on gifts for my sister?",
        )
        .expect("expected direct single-recipient gift answer");
    let direct_answer = std::fs::read_to_string(direct_path).unwrap();
    assert!(direct_answer.contains("Answer: $300"), "{direct_answer}");

    let answer = read_answer_text(&idx, "How much did I spend on gifts for my sister?");
    assert!(answer.contains("Answer: $300"), "{answer}");
}

#[test]
fn synthetic_spend_sum_drops_luxury_descriptors_for_item_matching() {
    let dir = TempDir::new().unwrap();
    let mut idx = make_index(&dir);
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "luxury.conv.md",
        "User: I've recently invested $500 in some high-end skincare products during the Nordstrom anniversary sale.\n\
         User: I recently treated myself to a Coach handbag, which costed $800.\n",
    );

    let answer = read_answer_text(
        &idx,
        "What is the total amount I spent on the designer handbag and high-end skincare products?",
    );
    assert!(answer.contains("Answer: $1,300"), "{answer}");
}

#[test]
fn synthetic_spend_sum_answers_total_cost_of_purchased_items() {
    let dir = TempDir::new().unwrap();
    let mut idx = make_index(&dir);
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "car-care.conv.md",
        "User: I think I'll go ahead and get the dash cam. One more thing, do you know if I can claim the cost of the waterproof car cover on my car insurance? I remember it cost me $120, and it's been working great in protecting my car's paint from the elements.\n\
         User: I'm also thinking of getting a detailing spray to keep my car's exterior clean. I've had good experiences with detailing sprays in the past, like the one I got from Amazon for $20 that removed tar and bug stains from my car's paint.\n",
    );

    let answer = read_answer_text(
        &idx,
        "What is the total cost of the car cover and detailing spray I purchased?",
    );
    assert!(answer.contains("Answer: $140"), "{answer}");
}

#[test]
fn synthetic_spend_sum_handles_oxford_comma_item_bundle() {
    let dir = TempDir::new().unwrap();
    let mut idx = make_index(&dir);
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "max-supplies.conv.md",
        "User: I've got a pretty good record of what I've been spending on Max. Let me think for a sec... So, there's the grain-free kibble, which I buy every month, and then there are the occasional expenses like the dental chews - I started using a new one to help with his teeth, and the chews are $10 a pack.\n\
         User: I think I forgot to mention that I also got a flea and tick collar for Max recently, which was $20, but that's also a one-time expense.\n\
         User: I'm thinking of getting Max a new toy to add to his collection. I just got him a new stainless steel food bowl from Amazon for $15, and a measuring cup from the pet store down the street for $5, which has been working out great for his new grain-free kibble.\n",
    );

    let answer = read_answer_text(
        &idx,
        "What is the total cost of the new food bowl, measuring cup, dental chews, and flea and tick collar I got for Max?",
    );
    assert!(answer.contains("Answer: $50"), "{answer}");
}

#[test]
fn synthetic_spend_sum_carries_recipient_context_to_followup_amounts() {
    let dir = TempDir::new().unwrap();
    let mut idx = make_index(&dir);
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "gifts-benchmark.conv.md",
        "User: I'm trying to stay on top of my finances and I was wondering if you could help me track my spending on gifts over the past few months. I know I spent a total of $500 on gifts recently, but I'm having trouble breaking it down. By the way, I did get my brother a really nice graduation gift in May - a $100 gift card to his favorite electronics store.\n\
         User: I remember buying a birthday present for my sister last month, a pair of earrings from that new jewelry store downtown, and it cost $75.\n\
         User: I also got my best friend a funny meme-themed mug from Amazon for her housewarming party, which was $20.\n\
         User: I'm still trying to remember if I got anything for my coworker's baby shower last month.\n\
         User: I'm pretty sure I got something for my coworker's baby shower...\n\
         User: I think it was a set of baby clothes and toys from Buy Buy Baby, and it cost around $100.\n",
    );

    let answer = read_answer_text(
        &idx,
        "What is the total amount I spent on gifts for my coworker and brother?",
    );
    assert!(answer.contains("Answer: $200"), "{answer}");
}
