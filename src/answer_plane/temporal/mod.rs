//! Temporal reasoning: query processing, gap/duration answering, and calendar-grounded ranking.

pub mod candidates;
pub mod comparison_order;
pub mod duration_count;
pub mod duration_events;
pub mod duration_extraction;
pub mod duration_selection;
pub mod gap_parser;
pub mod kg_temporal;
pub mod query_classify;
pub mod ranking;
pub mod selectors;
pub mod state_dialogue;

// Re-export public items from submodules
pub(crate) use self::candidates::*;
pub(crate) use self::comparison_order::*;
pub(crate) use self::duration_count::*;
pub(crate) use self::duration_events::*;
pub(crate) use self::duration_extraction::*;
pub(crate) use self::duration_selection::*;
pub(crate) use self::gap_parser::*;
pub(crate) use self::kg_temporal::*;
pub(crate) use self::query_classify::*;
pub(crate) use self::ranking::*;
pub(crate) use self::state_dialogue::*;

// Re-export items from the parent answer_plane module so all submodules can access via `use super::*`
// Helper functions are pub(super) in answer_plane so they stay at pub(super) here.
pub(super) use super::{
    answer_candidate_lines, answer_items_overlap, candidate_weight, collapse_inline_whitespace,
    compact_answer, contains_standalone_token, dialogue_focus_terms, dialogue_match_score,
    extract_explicit_date, extract_explicit_date_match, extract_explicit_date_range,
    extract_relative_date, extract_session_base_date, extract_subject_hints, extract_temporal_rank,
    extract_trailing_count, normalized_answer_key, parse_binary_choice, parse_count_token,
    parse_dialogue_turns, read_context_text, salient_query_terms, sanitize_answer_text,
    sanitize_inline, speaker_match_bonus, split_candidate_fragments,
    strip_temporal_discourse_prefix, summarize_turn_text, task_overlap_count,
    term_list_overlap_count, trim_answer_tail, turn_matches_subject, update_best_answer,
    ymd_to_days, DialogueTurn, ExplicitDateMatch, GENERIC_ANCHOR_TERMS, QUESTION_STOPWORDS,
};
// Types are pub(crate) in types.rs so they can be re-exported at pub(crate) level.
pub(crate) use super::types::{
    CalendarGroundedRank, ChoiceOption, EvidenceItem, TemporalAnswerPoint, TemporalCandidate,
    TemporalDirection, TemporalGapAnswerStyle, TemporalGapEndpoint, TemporalGapQuery,
    TemporalStateQuery,
};
