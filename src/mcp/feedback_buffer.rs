use std::path::PathBuf;
use tokio::sync::Mutex;

/// Feedback state: last retrieved paths and provisional hits for close-task recording.
///
/// Cleared on next `get_contexts` or `close_task` to prevent control-plane leakage
/// into training signals.
#[derive(Default)]
pub struct FeedbackBuffer {
    /// Paths returned by the most recent cortyx_get_contexts call.
    pub last_activated: Mutex<Vec<PathBuf>>,
    /// Carry-over from last response; cleared on next get_contexts or close_task.
    pub provisional_hits: Mutex<Vec<PathBuf>>,
}
