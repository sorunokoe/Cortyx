use std::collections::HashMap;

use crate::answer_plane::{mine_dialogue_answer_surface_span, mine_dialogue_question_pattern};

use super::super::{AnswerSurfaceRow, Turn};
use super::patterns::{
    compile_regex, extract_clause_after_any, extract_fact_after_any,
    extract_research_surface_value, normalize_answer_surface_span,
    normalize_dialogue_reason_phrase, normalize_dialogue_support_effect_phrase,
    push_answer_surface_row,
};

pub mod answer;
pub mod bridge;
pub mod embedded;
pub mod preferences;
pub mod profile;
pub mod temporal;

use self::bridge::*;
use self::embedded::*;
use self::preferences::*;
use self::profile::*;
use self::temporal::*;

pub(super) use self::answer::scoped_question_pattern;
pub(crate) use self::answer::{
    generate_cross_chunk_dialogue_answer_surface_rows, generate_dialogue_answer_surface_rows,
};
pub(crate) use self::bridge::{
    generate_dialogue_bridge_surface_rows, generate_session_bridge_surface_rows,
};
pub(super) use self::embedded::generate_embedded_dialogue_answer_surface_rows;
pub(crate) use self::embedded::{
    is_dialogue_speaker, normalize_dialogue_speaker_label, parse_embedded_dialogue_line,
    parse_embedded_session_timestamp,
};
pub(crate) use self::temporal::generate_temporal_turn_answer_surface_rows;
