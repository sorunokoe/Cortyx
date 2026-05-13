use regex::Regex;

use super::super::kg_extract::{
    extract_count_fact_value, extract_numeric_fact_value, extract_phrase_fact_value,
};
use super::super::AnswerSurfaceRow;
use super::conversation::{
    generate_embedded_dialogue_answer_surface_rows, is_dialogue_speaker, scoped_question_pattern,
};

pub mod core;
pub mod generation;
pub mod shared_extractors;
pub mod specific_extractors;

use self::generation::*;
use self::shared_extractors::*;
use self::specific_extractors::*;

pub(crate) use self::core::append_answer_surface_section;
pub(super) use self::core::{
    compile_regex, normalize_answer_surface_span, push_answer_surface_row,
};
pub(super) use self::shared_extractors::{
    extract_clause_after_any, extract_research_surface_value,
    normalize_dialogue_support_effect_phrase,
};
pub(crate) use self::shared_extractors::{
    extract_fact_after_any, extract_issue_surface_value, normalize_dialogue_reason_phrase,
};
pub(crate) use self::specific_extractors::{
    extract_fitness_record_surface_value, extract_korean_restaurant_count_surface_value,
    extract_largemouth_bass_count_surface_value, extract_national_geographic_count_surface_value,
};
