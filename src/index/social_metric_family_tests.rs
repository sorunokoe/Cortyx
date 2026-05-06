use super::social_metric_extractors::{parse_social_metric_query, SocialMetricQuery};
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
fn sums_social_reach_across_ad_and_influencer_channels() {
    let dir = TempDir::new().unwrap();
    let mut idx = make_index(&dir);
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "reach.conv.md",
        "User: I'm trying to improve my Instagram engagement, so I collaborated with an influencer who promoted my product to her 10,000 followers.\n\
         User: I also ran a Facebook ad campaign that reached around 2,000 people.\n",
    );
    let answer = read_answer_text(
        &idx,
        "What was the total number of people reached by my Facebook ad campaign and Instagram influencer collaboration?",
    );
    assert!(answer.contains("Answer: 12,000"), "{answer}");
}

#[test]
fn ranks_platform_with_largest_recent_growth() {
    let dir = TempDir::new().unwrap();
    let mut idx = make_index(&dir);
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "growth.conv.md",
        "User: My Facebook page has stayed around 800 followers lately.\n\
         User: My Twitter follower count has jumped from 420 to 540 over the past month.\n\
         User: TikTok gained around 200 followers over the past three weeks.\n",
    );
    let answer = read_answer_text(
        &idx,
        "Which social media platform did I gain the most followers on over the past month?",
    );
    assert!(answer.contains("Answer: TikTok"), "{answer}");
}

#[test]
fn parses_social_comment_total_query() {
    let query = parse_social_metric_query(
        "what is the total number of comments on my recent facebook live session and my most popular youtube video?",
    )
    .expect("expected social metric query");
    let SocialMetricQuery::CommentTotal(query) = query else {
        panic!("expected comment total query");
    };
    assert_eq!(
        query.required_terms,
        vec![
            "comments".to_string(),
            "facebook".to_string(),
            "live".to_string(),
            "most".to_string(),
            "popular".to_string(),
            "video".to_string(),
            "youtube".to_string(),
        ]
    );
}

#[test]
fn sums_comments_across_facebook_live_and_popular_youtube_video() {
    let dir = TempDir::new().unwrap();
    let mut idx = make_index(&dir);
    index_verbatim_neuron(
        &mut idx,
        &dir,
        "comments.conv.md",
        "User: I'm looking for some tips on increasing engagement on my social media platforms. I've been experimenting with different content types, like my recent Facebook Live session about cooking vegan recipes, which got 12 comments.\n\
         User: I'm trying to improve my social media strategy and was wondering if you could help me brainstorm some new content ideas for my YouTube channel. My most popular video on social media analytics has quite a few comments, so I think I'm on the right track.\n\
         User: I like the idea of creating a tutorial on Google Analytics. Since I've got experience with it and it's a popular tool, it'll be easy for me to create a comprehensive tutorial. My most popular video has 21 comments, and I wish to do better than that.\n\
         Assistant: Encourage viewers to share their own experiences in the comments.\n",
    );
    let answer = read_answer_text(
        &idx,
        "What is the total number of comments on my recent Facebook Live session and my most popular YouTube video?",
    );
    assert!(answer.contains("Answer: 33"), "{answer}");
}
