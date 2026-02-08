//! Write barriers for generational GC.
//!
//! Write barriers track old→young references to enable efficient
//! minor collection. Without barriers, we'd have to scan the entire
//! old generation to find references into the nursery.
//!
//! During concurrent marking, barriers also capture old pointer values
//! via the SATB (Snapshot-At-The-Beginning) mechanism to prevent the
//! concurrent marker from missing live objects.

mod card_table;
mod remembered_set;
pub mod satb_buffer;

#[cfg(test)]
mod satb_tests;

pub use card_table::CardTable;
pub use remembered_set::{RememberedEntry, RememberedSet};
pub use satb_buffer::{
    satb_capture_unconditional, satb_write_barrier, MarkingPhase, SatbBuffer, SatbMarkingState,
    SatbQueue,
};

use crate::heap::GcHeap;
use prism_core::Value;

// =============================================================================
// Generational Write Barriers
// =============================================================================

/// Write barrier for pointer stores.
///
/// Call this after storing a reference into a heap object.
/// The barrier tracks old→young references in a card table.
///
/// # Arguments
///
/// * `heap` - The GC heap
/// * `holder` - Pointer to the object containing the field
/// * `new_value` - The value being stored
///
/// # Performance
///
/// This is called on every pointer store, so it must be fast.
/// The fast path is a single comparison + conditional store.
///
/// # Example
///
/// ```ignore
/// // When storing a reference:
/// obj.field = new_value;
/// write_barrier(heap, obj as *const (), new_value);
/// ```
#[inline(always)]
pub fn write_barrier(heap: &GcHeap, holder: *const (), new_value: Value) {
    // Only check object pointers
    if let Some(new_ptr) = new_value.as_object_ptr() {
        write_barrier_ptr(heap, holder, new_ptr);
    }
}

/// Write barrier for raw pointer stores.
#[inline(always)]
pub fn write_barrier_ptr(heap: &GcHeap, holder: *const (), new_ptr: *const ()) {
    // Fast path: if holder is young, no barrier needed
    if heap.is_young(holder) {
        return;
    }

    // Check if new_ptr points to young generation
    if heap.is_young(new_ptr) {
        // Old→Young reference: record in remembered set
        heap.remembered_set().insert(holder);
    }
}

/// Unconditional write barrier that always marks the card.
///
/// Used when we can't easily determine the value's generation.
#[inline(always)]
pub fn write_barrier_unconditional(holder: *const (), card_table: &CardTable) {
    card_table.mark(holder);
}

// =============================================================================
// Concurrent Write Barriers (Generational + SATB)
// =============================================================================

/// Combined write barrier for concurrent marking phases.
///
/// Performs BOTH the generational barrier (old→young tracking) AND
/// the SATB barrier (old-value capture for concurrent marking) in
/// a single call.
///
/// # Fast Path
///
/// When concurrent marking is not active (`SatbMarkingState::Idle`),
/// the SATB portion reduces to a single atomic load + branch,
/// adding negligible overhead to the generational barrier.
///
/// # Arguments
///
/// * `heap` - The GC heap
/// * `holder` - Pointer to the object containing the field
/// * `old_value` - The old value being overwritten (for SATB capture)
/// * `new_value` - The new value being stored
/// * `marking_state` - Global SATB marking state
/// * `satb_buffer` - Thread-local SATB buffer
/// * `satb_queue` - Global SATB queue (for flushing)
///
/// # Usage
///
/// ```ignore
/// // Before + after overwriting a field:
/// let old = obj.field;
/// write_barrier_concurrent(
///     heap, obj_ptr, old, new_value,
///     &marking_state, &mut thread_buffer, &global_queue,
/// );
/// obj.field = new_value;
/// ```
#[inline(always)]
pub fn write_barrier_concurrent(
    heap: &GcHeap,
    holder: *const (),
    old_value: Value,
    new_value: Value,
    marking_state: &SatbMarkingState,
    satb_buffer: &mut SatbBuffer,
    satb_queue: &SatbQueue,
) {
    // 1. Generational barrier for new value
    if let Some(new_ptr) = new_value.as_object_ptr() {
        write_barrier_ptr(heap, holder, new_ptr);
    }

    // 2. SATB barrier for old value (captures pre-write reference)
    if let Some(old_ptr) = old_value.as_object_ptr() {
        satb_write_barrier(old_ptr, marking_state, satb_buffer, satb_queue);
    }
}

/// Combined write barrier for concurrent marking with raw pointers.
///
/// Like `write_barrier_concurrent` but operates on raw pointers instead
/// of `Value`s. Used when the caller already has extracted pointers.
///
/// # Performance
///
/// This avoids the `Value::as_object_ptr()` extraction overhead when
/// the caller already knows the pointer values.
#[inline(always)]
pub fn write_barrier_concurrent_ptr(
    heap: &GcHeap,
    holder: *const (),
    old_ptr: *const (),
    new_ptr: *const (),
    marking_state: &SatbMarkingState,
    satb_buffer: &mut SatbBuffer,
    satb_queue: &SatbQueue,
) {
    // 1. Generational barrier for new pointer
    write_barrier_ptr(heap, holder, new_ptr);

    // 2. SATB barrier for old pointer
    satb_write_barrier(old_ptr, marking_state, satb_buffer, satb_queue);
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::GcConfig;

    #[test]
    fn test_write_barrier_no_panic() {
        let heap = GcHeap::new(GcConfig::default());

        // Should not panic with null pointers
        write_barrier(&heap, std::ptr::null(), Value::none());
        write_barrier(&heap, std::ptr::null(), Value::int(42).unwrap());
    }

    #[test]
    fn test_write_barrier_concurrent_no_panic() {
        let heap = GcHeap::new(GcConfig::default());
        let state = SatbMarkingState::new();
        let mut buffer = SatbBuffer::new();
        let queue = SatbQueue::new();

        // Should not panic with null/none values
        write_barrier_concurrent(
            &heap,
            std::ptr::null(),
            Value::none(),
            Value::none(),
            &state,
            &mut buffer,
            &queue,
        );

        // Should not panic with integer values (non-object)
        write_barrier_concurrent(
            &heap,
            std::ptr::null(),
            Value::int(1).unwrap(),
            Value::int(2).unwrap(),
            &state,
            &mut buffer,
            &queue,
        );
    }

    #[test]
    fn test_write_barrier_concurrent_ptr_no_panic() {
        let heap = GcHeap::new(GcConfig::default());
        let state = SatbMarkingState::new();
        let mut buffer = SatbBuffer::new();
        let queue = SatbQueue::new();

        write_barrier_concurrent_ptr(
            &heap,
            std::ptr::null(),
            std::ptr::null(),
            std::ptr::null(),
            &state,
            &mut buffer,
            &queue,
        );
    }

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
}
