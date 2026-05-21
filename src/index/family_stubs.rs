use crate::index::NeuronIndex;
use std::path::PathBuf;

#[cfg(not(feature = "personal-families"))]
impl NeuronIndex {
    pub(super) fn synthetic_instagram_delta_answer(
        &self,
        _task: &str,
        _task_lower: &str,
    ) -> Option<PathBuf> {
        None
    }

    pub(super) fn synthetic_travel_packing_answer(
        &self,
        _task: &str,
        _task_lower: &str,
    ) -> Option<PathBuf> {
        None
    }

    pub(super) fn synthetic_podcast_episode_total_answer(
        &self,
        _task: &str,
        _task_lower: &str,
    ) -> Option<PathBuf> {
        None
    }

    pub(super) fn synthetic_paper_submission_answer(
        &self,
        _task: &str,
        _task_lower: &str,
    ) -> Option<PathBuf> {
        None
    }
}
