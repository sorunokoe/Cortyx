//! Temporal anchor query extraction and parsing.

mod parsers;
mod relative_recall;
mod types;

pub(super) use types::*;

pub(super) fn parse_temporal_anchor_query(task_lower: &str) -> Option<TemporalAnchorQuery> {
    parsers::parse_elapsed_before_event_query(task_lower)
        .map(TemporalAnchorQuery::ElapsedBeforeEvent)
        .or_else(|| {
            parsers::parse_temporal_elapsed_gap_query(task_lower)
                .map(TemporalAnchorQuery::ElapsedGap)
        })
        .or_else(|| {
            parsers::parse_temporal_interval_query(task_lower).map(TemporalAnchorQuery::Interval)
        })
        .or_else(|| {
            relative_recall::parse_relative_temporal_recall_query(task_lower)
                .map(TemporalAnchorQuery::RelativeRecall)
        })
}
