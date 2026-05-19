use super::*;
use crate::mcp::InflightGuard;

pub(super) fn estimate_inflight_bytes(input: &GetContextsInput) -> usize {
    input.task.len() + input.previous_response.as_deref().map_or(0, str::len)
}

pub(super) fn try_acquire_inflight_guard<'a>(
    inflight_bytes: &'a std::sync::atomic::AtomicUsize,
    estimated: usize,
) -> std::result::Result<InflightGuard<'a>, crate::error::CortyxError> {
    let result = inflight_bytes.fetch_update(
        std::sync::atomic::Ordering::AcqRel,
        std::sync::atomic::Ordering::Relaxed,
        |current| {
            current
                .checked_add(estimated)
                .filter(|&new| new <= MAX_INFLIGHT_BYTES)
        },
    );
    if result.is_err() {
        return Err(crate::error::CortyxError::Security(
            crate::error::SecurityError::SizeExceeded {
                limit: MAX_INFLIGHT_BYTES,
                context: "concurrent in-flight bytes limit".to_string(),
            },
        ));
    }

    // RAII decrement: use a guard so the counter is released even on early returns.
    Ok(InflightGuard(inflight_bytes, estimated))
}
