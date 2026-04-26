use super::*;
use crate::GcConfig;

#[test]
fn test_concurrent_barrier_satb_capture_when_marking() {
    let heap = GcHeap::new(GcConfig::default());
    let state = SatbMarkingState::new();
    let mut buffer = SatbBuffer::new();
    let queue = SatbQueue::new();

    // Activate marking
    state.start_marking();

    // Use a fake old pointer — the SATB barrier should capture it
    let fake_old_ptr = 0x12345678 as *const ();
    write_barrier_concurrent_ptr(
        &heap,
        std::ptr::null(),
        fake_old_ptr,
        std::ptr::null(),
        &state,
        &mut buffer,
        &queue,
    );

    // Buffer should have captured the old pointer
    assert_eq!(buffer.len(), 1);
}

#[test]
fn test_concurrent_barrier_no_capture_when_idle() {
    let heap = GcHeap::new(GcConfig::default());
    let state = SatbMarkingState::new();
    let mut buffer = SatbBuffer::new();
    let queue = SatbQueue::new();

    // State is Idle — SATB should NOT capture
    let fake_old_ptr = 0x12345678 as *const ();
    write_barrier_concurrent_ptr(
        &heap,
        std::ptr::null(),
        fake_old_ptr,
        std::ptr::null(),
        &state,
        &mut buffer,
        &queue,
    );

    assert_eq!(buffer.len(), 0);
}
