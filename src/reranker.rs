/// ONNX cross-encoder reranker — `--features rerank` (TRIZ R13-G4).
///
/// Uses ms-marco-MiniLM-L-2-v2 quantized INT8 ONNX model (~7 MB) to rerank
/// BM25 candidates. Activated only when:
///   - the `rerank` Cargo feature is enabled
///   - `.cortyx/reranker.onnx` exists on disk
///
/// The score is blended with the hit_rate prior:
///   final = cross_encoder_score × (0.8 + 0.2 × hit_rate)
///
/// # Model download
///   python3 scripts/download_reranker.py
///
/// # Latency budget
/// Cross-encoder inference: < 1 ms per (query, passage) pair for INT8 on CPU.
/// Applied to top-10 BM25 candidates: < 10 ms total.
#[cfg(feature = "rerank")]
pub mod inner {
    use anyhow::Result;
    use std::path::Path;
    use std::sync::{Mutex, OnceLock};

    use ort::{
        session::{builder::GraphOptimizationLevel, Session},
        value::Tensor,
    };
    use tokenizers::Tokenizer;

    /// Model file name within the project's `.cortyx/` directory.
    const MODEL_FILE: &str = "reranker.onnx";
    /// Tokenizer JSON file name within the project's `.cortyx/` directory.
    const TOKENIZER_FILE: &str = "tokenizer.json";
    /// Maximum tokens per (query, passage) pair — MiniLM-L-2-v2 has a 512-token window.
    const MAX_TOKENS: usize = 512;

    /// Loaded reranker state — ONNX session + tokenizer.
    pub struct Reranker {
        /// `Session::run` takes `&mut self`, so mutex-guarded.
        session: Mutex<Session>,
        tokenizer: Tokenizer,
    }

    impl Reranker {
        pub fn load(project_root: &Path) -> Result<Self> {
            let model_path = project_root.join(".cortyx").join(MODEL_FILE);
            if !model_path.exists() {
                anyhow::bail!(
                    "Reranker model not found at {}. \
                     Run: python3 scripts/download_reranker.py",
                    model_path.display()
                );
            }

            let session = Session::builder()?
                .with_optimization_level(GraphOptimizationLevel::Level3)?
                .commit_from_file(&model_path)?;

            let tokenizer_path = project_root.join(".cortyx").join(TOKENIZER_FILE);
            if !tokenizer_path.exists() {
                anyhow::bail!(
                    "Reranker tokenizer not found at {}. \
                     Run: python3 scripts/download_reranker.py",
                    tokenizer_path.display()
                );
            }
            let tokenizer = Tokenizer::from_file(&tokenizer_path)
                .map_err(|e| anyhow::anyhow!("Tokenizer load failed: {e}"))?;

            Ok(Self {
                session: Mutex::new(session),
                tokenizer,
            })
        }

        /// Score a (query, passage) pair. Returns a relevance score in [0, 1].
        ///
        /// The cross-encoder encodes both query and passage jointly, capturing
        /// semantic alignment that bag-of-words BM25 misses.
        pub fn score_pair(&self, query: &str, passage: &str) -> f32 {
            let passage_cap: String = passage.chars().take(MAX_TOKENS * 4).collect();
            let encoding = match self.tokenizer.encode((query, passage_cap.as_str()), true) {
                Ok(e) => e,
                Err(_) => return 0.0,
            };

            let len = encoding.get_ids().len().min(MAX_TOKENS);
            let shape = [1usize, len];

            let input_ids: Vec<i64> = encoding.get_ids()[..len]
                .iter()
                .map(|&x| x as i64)
                .collect();
            let attention_mask: Vec<i64> = encoding.get_attention_mask()[..len]
                .iter()
                .map(|&x| x as i64)
                .collect();
            let token_type_ids: Vec<i64> = encoding.get_type_ids()[..len]
                .iter()
                .map(|&x| x as i64)
                .collect();

            let ids_t = match Tensor::<i64>::from_array((shape, input_ids)) {
                Ok(t) => t,
                Err(_) => return 0.0,
            };
            let mask_t = match Tensor::<i64>::from_array((shape, attention_mask)) {
                Ok(t) => t,
                Err(_) => return 0.0,
            };
            let type_t = match Tensor::<i64>::from_array((shape, token_type_ids)) {
                Ok(t) => t,
                Err(_) => return 0.0,
            };

            let mut sess = match self.session.lock() {
                Ok(s) => s,
                Err(_) => return 0.0,
            };

            let outputs = match sess.run(ort::inputs![ids_t, mask_t, type_t]) {
                Ok(o) => o,
                Err(_) => return 0.0,
            };

            // MiniLM-L-2-v2 outputs a single logit; sigmoid maps it to [0, 1].
            let logit: f32 = outputs[0]
                .try_extract_tensor::<f32>()
                .ok()
                .and_then(|(_, data)| data.first().copied())
                .unwrap_or(0.0);

            sigmoid(logit)
        }
    }

    fn sigmoid(x: f32) -> f32 {
        1.0 / (1.0 + (-x).exp())
    }

    static RERANKER: OnceLock<Option<Reranker>> = OnceLock::new();

    pub fn global_reranker(project_root: &Path) -> Option<&'static Reranker> {
        RERANKER
            .get_or_init(|| match Reranker::load(project_root) {
                Ok(r) => {
                    tracing::info!("Cross-encoder reranker loaded from {}", MODEL_FILE);
                    Some(r)
                },
                Err(e) => {
                    tracing::debug!("Reranker not available ({}); BM25-only mode active.", e);
                    None
                },
            })
            .as_ref()
    }
}

/// Public interface — always present, conditionally functional.
///
/// When `--features rerank` is not enabled, this module is empty and the
/// `#[cfg(feature = "rerank")]` guards in index.rs produce a no-op.
#[cfg(not(feature = "rerank"))]
pub mod inner {}
