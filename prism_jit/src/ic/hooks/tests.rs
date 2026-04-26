use super::*;
use std::sync::Arc;
use std::thread;

// -------------------------------------------------------------------------
// ShapeHookStats Tests
// -------------------------------------------------------------------------

#[test]
fn test_hook_stats_new() {
    let stats = ShapeHookStats::new();
    assert_eq!(stats.transition_count(), 0);
    assert_eq!(stats.deletion_count(), 0);
    assert_eq!(stats.prototype_change_count(), 0);
    assert_eq!(stats.accessor_install_count(), 0);
    assert_eq!(stats.total_events(), 0);
}

#[test]
fn test_hook_stats_record_transition() {
    let stats = ShapeHookStats::new();
    stats.record_transition();
    stats.record_transition();
    assert_eq!(stats.transition_count(), 2);
    assert_eq!(stats.total_events(), 2);
}

#[test]
fn test_hook_stats_record_deletion() {
    let stats = ShapeHookStats::new();
    stats.record_deletion();
    assert_eq!(stats.deletion_count(), 1);
    assert_eq!(stats.total_events(), 1);
}

#[test]
fn test_hook_stats_record_prototype_change() {
    let stats = ShapeHookStats::new();
    stats.record_prototype_change();
    assert_eq!(stats.prototype_change_count(), 1);
}

#[test]
fn test_hook_stats_record_accessor_install() {
    let stats = ShapeHookStats::new();
    stats.record_accessor_install();
    assert_eq!(stats.accessor_install_count(), 1);
}

#[test]
fn test_hook_stats_snapshot() {
    let stats = ShapeHookStats::new();
    stats.record_transition();
    stats.record_deletion();
    stats.record_prototype_change();
    stats.record_accessor_install();
    stats.record_batch_transitions(5);

    let snap = stats.snapshot();
    assert_eq!(snap.transitions, 1);
    assert_eq!(snap.deletions, 1);
    assert_eq!(snap.prototype_changes, 1);
    assert_eq!(snap.accessor_installs, 1);
    assert_eq!(snap.batch_transitions, 5);
}

#[test]
fn test_hook_stats_concurrent() {
    let stats = Arc::new(ShapeHookStats::new());
    let mut handles = vec![];

    for _ in 0..10 {
        let s = Arc::clone(&stats);
        handles.push(thread::spawn(move || {
            for _ in 0..100 {
                s.record_transition();
            }
        }));
    }

    for h in handles {
        h.join().unwrap();
    }

    assert_eq!(stats.transition_count(), 1000);
}

// -------------------------------------------------------------------------
// RecordingListener Tests
// -------------------------------------------------------------------------

#[test]
fn test_recording_listener_new() {
    let listener = RecordingListener::new();
    assert_eq!(listener.event_count(), 0);
    assert!(listener.events().is_empty());
}

#[test]
fn test_recording_listener_transition() {
    let listener = RecordingListener::new();
    listener.on_transition(ShapeId(1), ShapeId(2));

    let events = listener.events();
    assert_eq!(events.len(), 1);
    assert_eq!(
        events[0],
        RecordedEvent::Transition {
            old: ShapeId(1),
            new: ShapeId(2)
        }
    );
}

#[test]
fn test_recording_listener_property_delete() {
    let listener = RecordingListener::new();
    listener.on_property_delete(ShapeId(1), "foo");

    let events = listener.events();
    assert_eq!(events.len(), 1);
    assert_eq!(
        events[0],
        RecordedEvent::PropertyDelete {
            shape: ShapeId(1),
            property: "foo".to_string()
        }
    );
}

#[test]
fn test_recording_listener_prototype_change() {
    let listener = RecordingListener::new();
    listener.on_prototype_change(ShapeId(1));

    let events = listener.events();
    assert_eq!(events.len(), 1);
    assert_eq!(
        events[0],
        RecordedEvent::PrototypeChange { shape: ShapeId(1) }
    );
}

#[test]
fn test_recording_listener_accessor_installed() {
    let listener = RecordingListener::new();
    listener.on_accessor_installed(ShapeId(1), "bar");

    let events = listener.events();
    assert_eq!(events.len(), 1);
    assert_eq!(
        events[0],
        RecordedEvent::AccessorInstalled {
            shape: ShapeId(1),
            property: "bar".to_string()
        }
    );
}

#[test]
fn test_recording_listener_clear() {
    let listener = RecordingListener::new();
    listener.on_transition(ShapeId(1), ShapeId(2));
    listener.on_transition(ShapeId(2), ShapeId(3));
    assert_eq!(listener.event_count(), 2);

    listener.clear();
    assert_eq!(listener.event_count(), 0);
}

#[test]
fn test_recording_listener_multiple_events() {
    let listener = RecordingListener::new();
    listener.on_transition(ShapeId(1), ShapeId(2));
    listener.on_property_delete(ShapeId(2), "x");
    listener.on_prototype_change(ShapeId(2));
    listener.on_accessor_installed(ShapeId(2), "y");

    assert_eq!(listener.event_count(), 4);

    let events = listener.events();
    assert!(matches!(&events[0], RecordedEvent::Transition { .. }));
    assert!(matches!(&events[1], RecordedEvent::PropertyDelete { .. }));
    assert!(matches!(&events[2], RecordedEvent::PrototypeChange { .. }));
    assert!(matches!(
        &events[3],
        RecordedEvent::AccessorInstalled { .. }
    ));
}

// -------------------------------------------------------------------------
// CountingListener Tests
// -------------------------------------------------------------------------

#[test]
fn test_counting_listener_new() {
    let listener = CountingListener::new();
    assert_eq!(listener.transition_count(), 0);
    assert_eq!(listener.deletion_count(), 0);
    assert_eq!(listener.prototype_change_count(), 0);
    assert_eq!(listener.accessor_install_count(), 0);
    assert_eq!(listener.total(), 0);
}

#[test]
fn test_counting_listener_counts_transitions() {
    let listener = CountingListener::new();
    listener.on_transition(ShapeId(1), ShapeId(2));
    listener.on_transition(ShapeId(2), ShapeId(3));
    assert_eq!(listener.transition_count(), 2);
}

#[test]
fn test_counting_listener_counts_deletions() {
    let listener = CountingListener::new();
    listener.on_property_delete(ShapeId(1), "a");
    listener.on_property_delete(ShapeId(1), "b");
    assert_eq!(listener.deletion_count(), 2);
}

#[test]
fn test_counting_listener_counts_prototype_changes() {
    let listener = CountingListener::new();
    listener.on_prototype_change(ShapeId(1));
    listener.on_prototype_change(ShapeId(2));
    listener.on_prototype_change(ShapeId(3));
    assert_eq!(listener.prototype_change_count(), 3);
}

#[test]
fn test_counting_listener_counts_accessor_installs() {
    let listener = CountingListener::new();
    listener.on_accessor_installed(ShapeId(1), "x");
    assert_eq!(listener.accessor_install_count(), 1);
}

#[test]
fn test_counting_listener_total() {
    let listener = CountingListener::new();
    listener.on_transition(ShapeId(1), ShapeId(2));
    listener.on_property_delete(ShapeId(1), "a");
    listener.on_prototype_change(ShapeId(1));
    listener.on_accessor_installed(ShapeId(1), "x");
    assert_eq!(listener.total(), 4);
}

#[test]
fn test_counting_listener_concurrent() {
    let listener = Arc::new(CountingListener::new());
    let mut handles = vec![];

    for _ in 0..10 {
        let l = Arc::clone(&listener);
        handles.push(thread::spawn(move || {
            for i in 0..100u32 {
                l.on_transition(ShapeId(i), ShapeId(i + 1));
            }
        }));
    }

    for h in handles {
        h.join().unwrap();
    }

    assert_eq!(listener.transition_count(), 1000);
}

// -------------------------------------------------------------------------
// Batch Transition Tests
// -------------------------------------------------------------------------

#[test]
fn test_batch_transition_default_impl() {
    let listener = CountingListener::new();
    let transitions = vec![
        (ShapeId(1), ShapeId(2)),
        (ShapeId(2), ShapeId(3)),
        (ShapeId(3), ShapeId(4)),
    ];

    let count = listener.on_batch_transition(&transitions);
    assert_eq!(count, 3);
    assert_eq!(listener.transition_count(), 3);
}

#[test]
fn test_batch_transition_empty() {
    let listener = CountingListener::new();
    let count = listener.on_batch_transition(&[]);
    assert_eq!(count, 0);
    assert_eq!(listener.transition_count(), 0);
}

// -------------------------------------------------------------------------
// Edge Case Tests
// -------------------------------------------------------------------------

#[test]
fn test_empty_property_name() {
    let listener = RecordingListener::new();
    listener.on_property_delete(ShapeId(1), "");
    listener.on_accessor_installed(ShapeId(1), "");

    let events = listener.events();
    assert_eq!(events.len(), 2);

    if let RecordedEvent::PropertyDelete { property, .. } = &events[0] {
        assert!(property.is_empty());
    }
}

#[test]
fn test_unicode_property_name() {
    let listener = RecordingListener::new();
    listener.on_property_delete(ShapeId(1), "日本語");
    listener.on_accessor_installed(ShapeId(1), "𝓤𝓷𝓲𝓬𝓸𝓭𝓮");

    let events = listener.events();
    assert_eq!(events.len(), 2);

    if let RecordedEvent::PropertyDelete { property, .. } = &events[0] {
        assert_eq!(property, "日本語");
    }
}

#[test]
fn test_same_shape_transition() {
    let listener = RecordingListener::new();
    // Transition to same shape (edge case)
    listener.on_transition(ShapeId(1), ShapeId(1));

    let events = listener.events();
    assert_eq!(events.len(), 1);
    assert_eq!(
        events[0],
        RecordedEvent::Transition {
            old: ShapeId(1),
            new: ShapeId(1)
        }
    );
}

#[test]
fn test_max_shape_id() {
    let listener = RecordingListener::new();
    listener.on_transition(ShapeId(u32::MAX), ShapeId(0));
    listener.on_property_delete(ShapeId(u32::MAX), "prop");

    let events = listener.events();
    assert_eq!(events.len(), 2);
}

// -------------------------------------------------------------------------
// Global Stats Tests
// -------------------------------------------------------------------------
