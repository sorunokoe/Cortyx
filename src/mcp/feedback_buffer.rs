use crate::index::NeuronIndex;
use std::path::PathBuf;
use tokio::sync::Mutex;

/// Feedback state: last retrieved paths and provisional hits for close-task recording.
#[derive(Default)]
pub struct FeedbackBuffer {
    /// Paths returned by the most recent cortyx_get_contexts call.
    pub last_activated: Mutex<Vec<PathBuf>>,
    /// Activated paths that have not yet been resolved into explicit feedback.
    pub provisional_hits: Mutex<Vec<PathBuf>>,
}

impl FeedbackBuffer {
    pub async fn on_session_end(&self, memory: &mut NeuronIndex) {
        let hits = {
            let mut lock = self.provisional_hits.lock().await;
            std::mem::take(&mut *lock)
        };
        for path in hits {
            memory.record_hit(&path, false);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::neuron::{meta_path, NeuronKind, NeuronMeta};

    #[tokio::test]
    async fn on_session_end_drains_provisional_hits() {
        let dir = tempfile::tempdir().unwrap();
        let neuron_path = dir.path().join("example.context.md");
        std::fs::write(&neuron_path, "example").unwrap();
        let meta = NeuronMeta::new_stub(&neuron_path, NeuronKind::Core);
        std::fs::write(
            meta_path(&neuron_path),
            serde_json::to_string(&meta).unwrap(),
        )
        .unwrap();

        let mut idx = NeuronIndex::default();
        idx.index_neuron(&neuron_path, "example context body", &meta);
        let buffer = FeedbackBuffer::default();
        *buffer.provisional_hits.lock().await = vec![neuron_path.clone()];

        buffer.on_session_end(&mut idx).await;

        assert!(buffer.provisional_hits.lock().await.is_empty());
        let metadata = idx.context_metadata_for(&neuron_path).unwrap();
        assert_eq!(metadata.use_count, 1);
        assert_eq!(metadata.hit_count, 0);
    }
}
