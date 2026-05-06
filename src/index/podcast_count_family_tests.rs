use super::podcast_count_extractors::parse_podcast_episode_total_query;
use super::*;
use tempfile::TempDir;

fn make_index(dir: &TempDir) -> NeuronIndex {
    NeuronIndex::load_or_create(dir.path()).unwrap()
}

fn read_answer_text(idx: &NeuronIndex, task: &str) -> String {
    let path = idx
        .derived_answer_path_for_task(task)
        .expect("expected synthetic answer");
    std::fs::read_to_string(path).unwrap()
}

#[test]
fn parses_podcast_episode_total_query_from_quoted_titles() {
    let query = parse_podcast_episode_total_query(
        "What is the total number of episodes I've listened to from 'How I Built This' and 'My Favorite Murder'?",
        "what is the total number of episodes i've listened to from 'how i built this' and 'my favorite murder'?",
    )
    .expect("expected podcast episode query");
    assert_eq!(query.titles.len(), 2);
    assert_eq!(query.titles[0].display, "How I Built This");
    assert_eq!(query.titles[1].display, "My Favorite Murder");
}

#[test]
fn synthetic_podcast_episode_total_answers_grouped_chunked_conversation() {
    let dir = TempDir::new().unwrap();
    let idx = make_index(&dir);
    let chunk_zero = dir
        .path()
        .join(".cortyx")
        .join("neurons")
        .join("podcasts_conv_0000_chunk.verbatim.md");
    let chunk_one = dir
        .path()
        .join(".cortyx")
        .join("neurons")
        .join("podcasts_conv_0001_chunk.verbatim.md");
    std::fs::create_dir_all(chunk_zero.parent().unwrap()).unwrap();
    std::fs::write(
        &chunk_zero,
        "User: I'll definitely check some of these out. I've been listening to podcasts during my daily commute. Are there any podcasts on this list that are more focused on the stories behind the companies, similar to \"How I Built This\"? I've finished around 15 episodes so far and I really enjoy hearing about the founders' journeys.\n",
    )
    .unwrap();
    std::fs::write(
        &chunk_one,
        "User: I'm looking for some new podcast recommendations. I'm really into true crime and inspiring stories, so if you have any suggestions, let me know. By the way, I just finished episode 12 of the \"My Favorite Murder\" podcast, and I try to listen to at least one episode a week.\n",
    )
    .unwrap();
    let task =
        "What is the total number of episodes I've listened to from 'How I Built This' and 'My Favorite Murder'?";
    idx.synthetic_podcast_episode_total_answer(task, &task.to_ascii_lowercase())
        .expect("expected podcast total answer");
    let answer = read_answer_text(&idx, task);
    assert!(answer.contains("Answer: 27"), "{answer}");
}
