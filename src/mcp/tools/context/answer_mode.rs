use super::*;

pub(super) async fn dispatch_answer_mode(
    server: &CortyxServer,
    answer_mode: bool,
    task: &str,
    paths_with_scores: &[(PathBuf, f32)],
    provenance_mode: bool,
    min_answer_confidence: Option<f32>,
) -> Option<String> {
    if !answer_mode {
        return None;
    }

    let idx_read = server.index.read().await;
    Some(
        match answer_plane::render_answer_output_decision(
            &idx_read,
            task,
            paths_with_scores,
            provenance_mode,
            min_answer_confidence,
        ) {
            Ok(answer) => {
                // A7: ECS filter — abstain if the generated answer is likely hallucinated.
                let verdict = verify_gate::check(&answer);
                if verdict.risk_score > 0.50 {
                    if provenance_mode {
                        return Some(format!(
                            "(answer abstained — ECS={}/100, risk={:.2}: {})",
                            verdict.ecs_score(),
                            verdict.risk_score,
                            verdict.summary.as_deref().unwrap_or("high risk")
                        ));
                    }
                    return Some(String::new());
                }
                // Append ECS score to provenance output when available.
                if provenance_mode {
                    format!("{answer}\n\n<!-- ECS: {}/100 -->", verdict.ecs_score())
                } else {
                    answer
                }
            },
            Err(answer_plane::AnswerAbstentionReason::LowFormConfidence)
                if min_answer_confidence.is_some() =>
            {
                "(no confident answer — answer confidence below threshold)".to_string()
            },
            Err(answer_plane::AnswerAbstentionReason::LowFormConfidence)
            | Err(answer_plane::AnswerAbstentionReason::Unsupported) => String::new(),
        },
    )
}
