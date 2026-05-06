mod conversation;
mod patterns;
mod query;

pub(super) use conversation::{
    generate_cross_chunk_dialogue_answer_surface_rows, generate_dialogue_answer_surface_rows,
    generate_dialogue_bridge_surface_rows, generate_session_bridge_surface_rows,
    generate_temporal_turn_answer_surface_rows, is_dialogue_speaker,
    normalize_dialogue_speaker_label, parse_embedded_dialogue_line,
    parse_embedded_session_timestamp,
};
pub(super) use patterns::{
    append_answer_surface_section, extract_fact_after_any, extract_fitness_record_surface_value,
    extract_issue_surface_value, extract_korean_restaurant_count_surface_value,
    extract_largemouth_bass_count_surface_value, extract_national_geographic_count_surface_value,
    normalize_dialogue_reason_phrase,
};
pub(super) use query::{fact_alias_lines, generate_query_surface};
