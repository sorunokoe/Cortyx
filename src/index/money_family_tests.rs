use super::money_extractors::{
    extract_cashback_purchase_fact_from_line, extract_cashback_rate_fact_from_line,
    extract_paid_price_fact_from_line,
};
use super::money_queries::parse_money_query;
use super::money_support::{format_money_cents, MoneyQuery};
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
fn format_money_cents_preserves_decimal_precision() {
    assert_eq!(format_money_cents(75), "$0.75");
    assert_eq!(format_money_cents(30_000), "$300");
    assert_eq!(format_money_cents(375_000), "$3,750");
}

#[test]
fn parse_money_query_splits_product_and_store_terms() {
    let query = parse_money_query(
        "How much did I save on the designer handbag at TK Maxx?",
        "how much did i save on the designer handbag at tk maxx?",
    )
    .unwrap();
    match query {
        MoneyQuery::Savings(query) => {
            assert_eq!(
                query.product_terms,
                vec!["designer".to_string(), "handbag".to_string()]
            );
            assert_eq!(
                query.context_terms,
                vec!["maxx".to_string(), "tk".to_string()]
            );
        },
        other => panic!("unexpected query variant: {other:?}"),
    }
}

#[test]
fn parse_money_query_recognizes_discount_percent_queries() {
    let query = parse_money_query(
        "What percentage discount did I get on the book from my favorite author?",
        "what percentage discount did i get on the book from my favorite author?",
    )
    .unwrap();
    match query {
        MoneyQuery::DiscountPercent(query) => {
            assert!(query.product_terms.contains(&"book".to_string()));
            assert!(query.product_terms.contains(&"author".to_string()));
        },
        other => panic!("unexpected query variant: {other:?}"),
    }
}

#[test]
fn synthetic_cashback_earned_combines_purchase_and_rate() {
    let dir = TempDir::new().unwrap();
    let mut idx = make_index(&dir);
    let query = parse_money_query(
        "How much cashback did I earn at SaveMart last Thursday?",
        "how much cashback did i earn at savemart last thursday?",
    )
    .unwrap();
    let MoneyQuery::Cashback(query) = query else {
        panic!("expected cashback query");
    };
    assert!(extract_cashback_purchase_fact_from_line(
        "User: I spent $75 on groceries at SaveMart last Thursday.",
        "user: i spent $75 on groceries at savemart last thursday.",
        &query,
    )
    .is_some());
    assert!(extract_cashback_rate_fact_from_line(
        "User: I have a membership at SaveMart and can earn 1% cashback on all purchases.",
        "user: i have a membership at savemart and can earn 1% cashback on all purchases.",
        &query,
    )
    .is_some());
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "savemart.conv.md",
        "User: I spent $75 on groceries at SaveMart last Thursday.\n\
         User: I have a membership at SaveMart and can earn 1% cashback on all purchases.\n",
    );

    let answer = read_answer_text(
        &idx,
        "How much cashback did I earn at SaveMart last Thursday?",
    );
    assert!(answer.contains("Answer: $0.75"));
}

#[test]
fn synthetic_savings_delta_handles_contextless_original_price() {
    let dir = TempDir::new().unwrap();
    let mut idx = make_index(&dir);
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "handbag_deal.conv.md",
        "User: I've had luck finding great deals at TK Maxx before, like that designer handbag I got for $200.\n\
         User: I recently got a designer handbag and it was originally $500.\n",
    );

    let answer = read_answer_text(
        &idx,
        "How much did I save on the designer handbag at TK Maxx?",
    );
    assert!(answer.contains("Answer: $300"));
}

#[test]
fn synthetic_discount_percent_computes_book_sale_percentage() {
    let dir = TempDir::new().unwrap();
    let mut idx = make_index(&dir);
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "book_sale.conv.md",
        "User: It's actually the new release from my favorite author, which was originally priced at $30.\n\
         User: I got the book for $24 after a discount during the sale at my favorite bookstore.\n",
    );

    let answer = read_answer_text(
        &idx,
        "What percentage discount did I get on the book from my favorite author?",
    );
    assert!(answer.contains("Answer: 20%"), "{answer}");
}

#[test]
fn discount_percent_prefers_discounted_price_when_line_contains_other_amounts() {
    let query = parse_money_query(
        "What percentage discount did I get on the book from my favorite author?",
        "what percentage discount did i get on the book from my favorite author?",
    )
    .unwrap();
    let MoneyQuery::DiscountPercent(query) = query else {
        panic!("expected discount-percent query");
    };

    let fact = extract_paid_price_fact_from_line(
        "User: I found a similar necklace that I got for my sister's birthday, which was $75. I'm hoping to find something around that price range or a bit higher. By the way, I've been good about budgeting for gifts lately, except for one impulse buy last week when I saw a sale at my favorite bookstore - I got the book for $24 after a discount.",
        "user: i found a similar necklace that i got for my sister's birthday, which was $75. i'm hoping to find something around that price range or a bit higher. by the way, i've been good about budgeting for gifts lately, except for one impulse buy last week when i saw a sale at my favorite bookstore - i got the book for $24 after a discount.",
        &query,
    )
    .expect("expected paid-price fact");

    assert_eq!(fact.amount_cents, 2_400);
}

#[test]
fn synthetic_spend_sum_totals_multiple_focus_items() {
    let dir = TempDir::new().unwrap();
    let mut idx = make_index(&dir);
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "car_budget.conv.md",
        "User: I also got a parking ticket on January 5th near my work for $50.\n\
         User: I've been spending a bit on maintenance lately, like a car wash on February 3rd that cost $15.\n",
    );

    let answer = read_answer_text(&idx, "How much did I spend on car wash and parking ticket?");
    assert!(answer.contains("Answer: $65"));
}

#[test]
fn synthetic_raised_total_dedupes_repeated_charity_events() {
    let dir = TempDir::new().unwrap();
    let mut idx = make_index(&dir);
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "charity.conv.md",
        "User: I helped raise $2,000 for a local animal shelter on January 20th.\n\
         User: I helped raise over $2,000 for a local animal shelter on January 20th.\n\
         User: I just ran 5 kilometers in the \"Run for Hunger\" charity event on March 12th and raised $250 for a local food bank.\n\
         User: I recently volunteered at a charity bake sale and it was amazing to see how much of an impact we can make - we raised $1,000 for the local children's hospital!\n\
         User: I recently completed a charity fitness challenge in February and managed to raise $500 for the American Cancer Society.\n",
    );

    let answer = read_answer_text(&idx, "How much money did I raise for charity in total?");
    assert!(answer.contains("Answer: $3,750"), "{answer}");
}

#[test]
fn synthetic_revenue_answer_multiplies_quantity_and_unit_price() {
    let dir = TempDir::new().unwrap();
    let mut idx = make_index(&dir);
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "egg_sales.conv.md",
        "User: I've had a great month for egg production - I've sold a total of 40 dozen eggs so far.\n\
         User: I've been selling the eggs to my neighbor for $3 a dozen.\n",
    );

    let answer = read_answer_text(&idx, "How much have I made from selling eggs this month?");
    assert!(answer.contains("Answer: $120"));
}

#[test]
fn synthetic_raised_total_event_scope_prefers_explicit_charity_events() {
    let dir = TempDir::new().unwrap();
    let mut idx = make_index(&dir);
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "charity_events.conv.md",
        "User: I recently participated in a charity walk and managed to raise $250 through sponsors.\n\
         User: I recently participated in a Bike-a-Thon for Cancer Research and my team managed to raise $5,000!\n\
         User: I just helped organize a charity yoga event that raised $600 for a local animal shelter.\n",
    );

    let answer = read_answer_text(
        &idx,
        "How much money did I raise in total through all the charity events I participated in?",
    );
    assert!(answer.contains("Answer: $5,850"), "{answer}");
}

#[test]
fn synthetic_raised_total_event_scope_beats_generic_charity_session() {
    let dir = TempDir::new().unwrap();
    let mut idx = make_index(&dir);
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "charity_events.conv.md",
        "User: I recently participated in a charity walk and managed to raise $250 through sponsors.\n\
         User: I recently participated in a Bike-a-Thon for Cancer Research and my team managed to raise $5,000!\n\
         User: I just helped organize a charity yoga event that raised $600 for a local animal shelter.\n",
    );
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "charity_general.conv.md",
        "User: I helped raise $2,000 for a local animal shelter on January 20th.\n\
         User: I just ran 5 kilometers in the \"Run for Hunger\" charity event on March 12th and raised $250 for a local food bank.\n\
         User: I recently volunteered at a charity bake sale and it was amazing to see how much of an impact we can make - we raised $1,000 for the local children's hospital!\n\
         User: I recently completed a charity fitness challenge in February and managed to raise $500 for the American Cancer Society.\n",
    );

    let event_scoped = read_answer_text(
        &idx,
        "How much money did I raise in total through all the charity events I participated in?",
    );
    assert!(event_scoped.contains("Answer: $5,850"), "{event_scoped}");

    let generic = read_answer_text(&idx, "How much money did I raise for charity in total?");
    assert!(generic.contains("Answer: $3,750"), "{generic}");
}

#[test]
fn synthetic_raised_total_event_scope_falls_back_to_raw_conversation_scan() {
    let dir = TempDir::new().unwrap();
    let mut idx = make_index(&dir);
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "charity_general.conv.md",
        "User: I helped raise $2,000 for a local animal shelter on January 20th.\n\
         User: I just ran 5 kilometers in the \"Run for Hunger\" charity event on March 12th and raised $250 for a local food bank.\n\
         User: I recently volunteered at a charity bake sale and it was amazing to see how much of an impact we can make - we raised $1,000 for the local children's hospital!\n\
         User: I recently completed a charity fitness challenge in February and managed to raise $500 for the American Cancer Society.\n",
    );

    let raw_only_path = dir
        .path()
        .join(".cortyx")
        .join("neurons")
        .join("charity_events_conv_0000_chunk.verbatim.md");
    std::fs::create_dir_all(raw_only_path.parent().unwrap()).unwrap();
    std::fs::write(
        &raw_only_path,
        "User: I'm really interested in the \"big four\" you mentioned, especially reducing plastic bag usage. I've realized how often I've been using them for grocery shopping, and it's definitely an area I can improve on. By the way, I recently participated in a charity walk and managed to raise $250 through sponsors, which got me thinking about the impact of my daily habits on the community and environment.\n\
         User: Thanks for the resources! I'll definitely check them out. By the way, speaking of charity events, I recently participated in a Bike-a-Thon for Cancer Research and my team managed to raise $5,000! It was an amazing experience. Do you have any tips on how to stay motivated to continue volunteering and making a difference in the community?\n\
         User: I just helped organize a charity yoga event that raised $600 for a local animal shelter.\n",
    )
    .unwrap();

    let answer = read_answer_text(
        &idx,
        "How much money did I raise in total through all the charity events I participated in?",
    );
    assert!(answer.contains("Answer: $5,850"), "{answer}");
}

#[test]
fn synthetic_raised_total_event_scope_dedupes_summary_and_raw_event_keys() {
    let dir = TempDir::new().unwrap();
    let mut idx = make_index(&dir);
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "charity_events.conv.md",
        "User: I'm really interested in the \"big four\" you mentioned, especially reducing plastic bag usage. I've realized how often I've been using them for grocery shopping, and it's definitely an area I can improve on. By the way, I recently participated in a charity walk and managed to raise $250 through sponsors, which got me thinking about the impact of my daily habits on the community and environment.\n\
         User: Thanks for the resources! I'll definitely check them out. By the way, speaking of charity events, I recently participated in a Bike-a-Thon for Cancer Research and my team managed to raise $5,000! It was an amazing experience. Do you have any tips on how to stay motivated to continue volunteering and making a difference in the community?\n\
         User: I'm looking for some tips on zero-waste living. I recently met someone who's really into it, and I'm curious to learn more. By the way, I just helped organize a charity yoga event that raised $600 for a local animal shelter.\n",
    );
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "charity_general.conv.md",
        "User: I helped raise $2,000 for a local animal shelter on January 20th.\n\
         User: I just ran 5 kilometers in the \"Run for Hunger\" charity event on March 12th and raised $250 for a local food bank.\n\
         User: I recently volunteered at a charity bake sale and it was amazing to see how much of an impact we can make - we raised $1,000 for the local children's hospital!\n\
         User: I recently completed a charity fitness challenge in February and managed to raise $500 for the American Cancer Society.\n",
    );

    let answer = read_answer_text(
        &idx,
        "How much money did I raise in total through all the charity events I participated in?",
    );
    assert!(answer.contains("Answer: $5,850"), "{answer}");
}
