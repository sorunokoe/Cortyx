/// ONNX cross-encoder reranker — `--features rerank` (TRIZ R13-G4).
///
/// Uses ms-marco-MiniLM-L-2-v2 quantized INT8 ONNX model (~7 MB) to rerank
/// BM25 candidates for low-confidence queries. Activated only when:
///   - the `rerank` Cargo feature is enabled
///   - `top_bm25_score < LOW_CONFIDENCE_THRESHOLD` (already detected in index.rs)
///   - `.cortyx/reranker.onnx` exists on disk
///
/// The score is blended with the hit_rate prior:
///   final = cross_encoder_score × (0.8 + 0.2 × hit_rate)
///
/// This gives battle-tested neurons a mild boost even for ambiguous queries,
/// without any new signal beyond what is already in the feedback loop.
///
/// # Model download
/// Download the INT8 quantized model from HuggingFace:
///   huggingface-cli download cross-encoder/ms-marco-MiniLM-L-2-v2 \
///     --local-dir .cortyx/ --include "*.onnx"
/// Rename to `.cortyx/reranker.onnx`.
///
/// # Latency budget
/// Cross-encoder inference: < 1 ms per (query, passage) pair for INT8 on CPU.
/// Applied to top-10 BM25 candidates: < 10 ms total.
/// Far under the 50 ms p95 activation budget.
#[cfg(feature = "rerank")]
pub mod inner {
    use anyhow::Result;
    use std::path::Path;
    use std::sync::{Arc, Mutex, OnceLock};

    use ort::{Environment, GraphOptimizationLevel, Session, SessionBuilder, Value};
    use tokenizers::Tokenizer;

    /// Model file name within the project's `.cortyx/` directory.
    const MODEL_FILE: &str = "reranker.onnx";
    /// Tokenizer name (downloaded automatically by the `tokenizers` crate from HuggingFace).
    const TOKENIZER_NAME: &str = "cross-encoder/ms-marco-MiniLM-L-2-v2";
    /// Maximum tokens per (query, passage) pair — MiniLM-L-2-v2 has a 512 token window.
    const MAX_TOKENS: usize = 512;

    /// Loaded reranker state — session + tokenizer.
    pub struct Reranker {
        session: Arc<Mutex<Session>>,
        tokenizer: Arc<Tokenizer>,
    }

    impl Reranker {
        /// Load the ONNX session and tokenizer.
        ///
        /// Returns an error if the model file is absent or the tokenizer cannot be fetched.
        /// Callers should handle the error gracefully (fall back to BM25-only).
        pub fn load(project_root: &Path) -> Result<Self> {
            let model_path = project_root.join(".cortyx").join(MODEL_FILE);
            if !model_path.exists() {
                anyhow::bail!(
                    "Reranker model not found at {}. \
                     Download with: huggingface-cli download cross-encoder/ms-marco-MiniLM-L-2-v2 \
                     --local-dir .cortyx/ --include '*.onnx' && mv .cortyx/model.onnx .cortyx/reranker.onnx",
                    model_path.display()
                );
            }

            let environment = Arc::new(
                Environment::builder()
                    .with_name("cortyx_reranker")
                    .build()?
            );
            let session = SessionBuilder::new(&environment)?
                .with_optimization_level(GraphOptimizationLevel::Level3)?
                .with_model_from_file(&model_path)?;

            let tokenizer = Tokenizer::from_pretrained(TOKENIZER_NAME, None)
                .map_err(|e| anyhow::anyhow!("Tokenizer load failed: {e}"))?;

            Ok(Self {
                session: Arc::new(Mutex::new(session)),
                tokenizer: Arc::new(tokenizer),
            })
        }

        /// Score a (query, passage) pair. Returns a relevance score in [0, 1].
        ///
        /// The cross-encoder encodes both query and passage jointly, capturing
        /// semantic alignment that bag-of-words BM25 misses.
        pub fn score_pair(&self, query: &str, passage: &str) -> f32 {
            let encoding = match self.tokenizer.encode_pair(
                query.to_string(),
                passage.chars().take(MAX_TOKENS * 4).collect::<String>(), // rough char cap
                true,
            ) {
                Ok(e) => e,
                Err(_) => return 0.0,
            };

            let input_ids: Vec<i64> = encoding.get_ids().iter().map(|&x| x as i64).collect();
            let attention_mask: Vec<i64> =
                encoding.get_attention_mask().iter().map(|&x| x as i64).collect();
            let token_type_ids: Vec<i64> =
                encoding.get_type_ids().iter().map(|&x| x as i64).collect();

            let len = input_ids.len().min(MAX_TOKENS);
            let shape = [1usize, len];

            let session = match self.session.lock() {
                Ok(s) => s,
                Err(_) => return 0.0,
            };

            let allocator = session.allocator();

            let ids_tensor = Value::from_array(
                allocator,
                &ndarray::Array2::from_shape_vec(shape, input_ids[..len].to_vec())
                    .unwrap_or_default(),
            );
            let mask_tensor = Value::from_array(
                allocator,
                &ndarray::Array2::from_shape_vec(shape, attention_mask[..len].to_vec())
                    .unwrap_or_default(),
            );
            let type_tensor = Value::from_array(
                allocator,
                &ndarray::Array2::from_shape_vec(shape, token_type_ids[..len].to_vec())
                    .unwrap_or_default(),
            );

            let (ids_ok, mask_ok, type_ok) = match (ids_tensor, mask_tensor, type_tensor) {
                (Ok(a), Ok(b), Ok(c)) => (a, b, c),
                _ => return 0.0,
            };

            let outputs = match session.run(vec![ids_ok, mask_ok, type_ok]) {
                Ok(o) => o,
                Err(_) => return 0.0,
            };

            // MiniLM-L-2-v2 outputs a single logit; sigmoid maps it to [0, 1].
            let logit = outputs
                .first()
                .and_then(|t| t.try_extract::<f32>().ok())
                .and_then(|v| v.view().iter().next().copied());

            match logit {
                Some(l) => sigmoid(l),
                None => 0.0,
            }
        }
    }

    fn sigmoid(x: f32) -> f32 {
        1.0 / (1.0 + (-x).exp())
    }

    /// Global lazily-initialized reranker per project root.
    ///
    /// Stored as `Option<Reranker>` — `None` means model absent or load failed.
    /// Once loaded (or determined absent), the result is cached for the process lifetime.
    static RERANKER: OnceLock<Option<Reranker>> = OnceLock::new();

    /// Get the global reranker, initializing it on first call.
    ///
    /// Returns `None` if the model is absent or fails to load (graceful fallback).
    pub fn global_reranker(project_root: &Path) -> Option<&'static Reranker> {
        RERANKER
            .get_or_init(|| {
                match Reranker::load(project_root) {
                    Ok(r) => {
                        tracing::info!("Cross-encoder reranker loaded from {}", MODEL_FILE);
                        Some(r)
                    }
                    Err(e) => {
                        tracing::debug!("Reranker not available ({}); BM25-only mode active.", e);
                        None
                    }
                }
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
