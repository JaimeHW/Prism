use super::*;

// =========================================================================
// CollectorConfig Tests
// =========================================================================

#[test]
fn test_config_default() {
    let config = CollectorConfig::default();
    assert_eq!(config.hot_method_threshold, 1000);
    assert_eq!(config.sampling_rate, 1);
    assert!(config.track_calls);
    assert!(config.track_types);
}

#[test]
fn test_config_production() {
    let config = CollectorConfig::production();
    assert_eq!(config.hot_method_threshold, 5000);
    assert_eq!(config.sampling_rate, 8);
}

#[test]
fn test_config_detailed() {
    let config = CollectorConfig::detailed();
    assert_eq!(config.hot_method_threshold, 100);
    assert_eq!(config.sampling_rate, 1);
}

#[test]
fn test_config_testing() {
    let config = CollectorConfig::for_testing();
    assert_eq!(config.hot_method_threshold, 10);
    assert_eq!(config.max_tracked_units, 256);
}

// =========================================================================
// CodeUnitCounters Tests
// =========================================================================

#[test]
fn test_counters_new() {
    let c = CodeUnitCounters::new(42);
    assert_eq!(c.code_id, 42);
    assert_eq!(c.invocation_count(), 0);
}

#[test]
fn test_counters_record_invocation() {
    let c = CodeUnitCounters::new(1);
    assert_eq!(c.record_invocation(), 1);
    assert_eq!(c.record_invocation(), 2);
    assert_eq!(c.record_invocation(), 3);
    assert_eq!(c.invocation_count(), 3);
}

#[test]
fn test_counters_record_branch() {
    let c = CodeUnitCounters::new(1);
    c.record_branch(10, true);
    c.record_branch(10, true);
    c.record_branch(10, false);
    let snapshot = c.snapshot();
    let bp = snapshot.branch_at(10).unwrap();
    assert_eq!(bp.taken, 2);
    assert_eq!(bp.not_taken, 1);
}

#[test]
fn test_counters_record_loop() {
    let c = CodeUnitCounters::new(1);
    c.record_loop(50);
    c.record_loop(50);
    c.record_loop(50);
    let snapshot = c.snapshot();
    assert_eq!(snapshot.loop_count(50), 3);
}

#[test]
fn test_counters_snapshot() {
    let c = CodeUnitCounters::new(7);
    c.record_invocation();
    c.record_invocation();
    c.record_branch(10, true);
    c.record_loop(20);

    let snapshot = c.snapshot();
    assert_eq!(snapshot.code_id(), 7);
    assert_eq!(snapshot.execution_count(), 2);
    assert!(snapshot.branch_at(10).is_some());
    assert_eq!(snapshot.loop_count(20), 1);
}

#[test]
fn test_counters_reset() {
    let c = CodeUnitCounters::new(1);
    c.record_invocation();
    c.record_branch(10, true);
    c.record_loop(20);
    c.reset();

    assert_eq!(c.invocation_count(), 0);
    let snapshot = c.snapshot();
    assert_eq!(snapshot.execution_count(), 0);
}

#[test]
fn test_counters_multiple_branch_offsets() {
    let c = CodeUnitCounters::new(1);
    for i in 0..20u32 {
        c.record_branch(i * 10, i % 2 == 0);
    }
    let snapshot = c.snapshot();
    assert_eq!(snapshot.branch_count(), 20);
}

#[test]
fn test_counters_concurrent_invocations() {
    use std::sync::Arc;
    use std::thread;

    let c = Arc::new(CodeUnitCounters::new(1));
    let mut handles = vec![];

    for _ in 0..4 {
        let c = Arc::clone(&c);
        handles.push(thread::spawn(move || {
            for _ in 0..1000 {
                c.record_invocation();
            }
        }));
    }

    for h in handles {
        h.join().unwrap();
    }

    assert_eq!(c.invocation_count(), 4000);
}

// =========================================================================
// HotMethodDetector Tests
// =========================================================================

#[test]
fn test_detector_new() {
    let d = HotMethodDetector::new(100, 1000);
    assert_eq!(d.hot_method_count(), 0);
    assert_eq!(d.hot_loop_count(), 0);
}

#[test]
fn test_detector_method_not_hot_below_threshold() {
    let d = HotMethodDetector::new(100, 1000);
    assert!(!d.check_method_hot(1, 99));
    assert_eq!(d.hot_method_count(), 0);
}

#[test]
fn test_detector_method_becomes_hot() {
    let d = HotMethodDetector::new(100, 1000);
    assert!(d.check_method_hot(1, 100));
    assert_eq!(d.hot_method_count(), 1);
}

#[test]
fn test_detector_method_fires_once() {
    let d = HotMethodDetector::new(100, 1000);
    assert!(d.check_method_hot(1, 100));
    assert!(!d.check_method_hot(1, 200)); // Already known
    assert_eq!(d.hot_method_count(), 1);
}

#[test]
fn test_detector_multiple_methods() {
    let d = HotMethodDetector::new(100, 1000);
    assert!(d.check_method_hot(1, 100));
    assert!(d.check_method_hot(2, 100));
    assert!(d.check_method_hot(3, 100));
    assert_eq!(d.hot_method_count(), 3);
    let ids = d.hot_method_ids();
    assert!(ids.contains(&1));
    assert!(ids.contains(&2));
    assert!(ids.contains(&3));
}

#[test]
fn test_detector_loop_not_hot() {
    let d = HotMethodDetector::new(100, 1000);
    assert!(!d.check_loop_hot(1, 50, 999));
}

#[test]
fn test_detector_loop_becomes_hot() {
    let d = HotMethodDetector::new(100, 1000);
    assert!(d.check_loop_hot(1, 50, 1000));
    assert_eq!(d.hot_loop_count(), 1);
}

#[test]
fn test_detector_loop_fires_once() {
    let d = HotMethodDetector::new(100, 1000);
    assert!(d.check_loop_hot(1, 50, 1000));
    assert!(!d.check_loop_hot(1, 50, 2000));
}

#[test]
fn test_detector_reset() {
    let d = HotMethodDetector::new(100, 1000);
    d.check_method_hot(1, 100);
    d.check_loop_hot(1, 50, 1000);
    d.reset();
    assert_eq!(d.hot_method_count(), 0);
    assert_eq!(d.hot_loop_count(), 0);
    // Should fire again after reset
    assert!(d.check_method_hot(1, 100));
}

// =========================================================================
// ProfileCollector Tests
// =========================================================================

#[test]
fn test_collector_new() {
    let c = ProfileCollector::new();
    assert_eq!(c.total_events(), 0);
    assert_eq!(c.tracked_unit_count(), 0);
}

#[test]
fn test_collector_default() {
    let c = ProfileCollector::default();
    assert_eq!(c.total_events(), 0);
}

#[test]
fn test_collector_record_invocation() {
    let c = ProfileCollector::with_config(CollectorConfig::for_testing());
    let hot = c.record_invocation(1);
    assert!(!hot); // Not yet at threshold
    assert_eq!(c.total_events(), 1);
    assert_eq!(c.tracked_unit_count(), 1);
}

#[test]
fn test_collector_hot_method_detection() {
    let c = ProfileCollector::with_config(CollectorConfig::for_testing());
    // Threshold is 10
    for _ in 0..9 {
        assert!(!c.record_invocation(1));
    }
    assert!(c.record_invocation(1)); // 10th invocation
    assert!(!c.record_invocation(1)); // Already hot
}

#[test]
fn test_collector_record_branch() {
    let c = ProfileCollector::with_config(CollectorConfig::for_testing());
    c.record_branch(1, 10, true);
    c.record_branch(1, 10, false);
    let snapshot = c.snapshot(1).unwrap();
    let bp = snapshot.branch_at(10).unwrap();
    assert_eq!(bp.taken, 1);
    assert_eq!(bp.not_taken, 1);
}

#[test]
fn test_collector_record_loop() {
    let config = CollectorConfig {
        hot_loop_threshold: 5,
        ..CollectorConfig::for_testing()
    };
    let c = ProfileCollector::with_config(config);
    for _ in 0..4 {
        assert!(!c.record_loop(1, 50));
    }
    assert!(c.record_loop(1, 50)); // 5th iteration
    assert!(!c.record_loop(1, 50)); // Already hot
}

#[test]
fn test_collector_snapshot_missing() {
    let c = ProfileCollector::new();
    assert!(c.snapshot(999).is_none());
}

#[test]
fn test_collector_snapshot_all() {
    let c = ProfileCollector::with_config(CollectorConfig::for_testing());
    c.record_invocation(1);
    c.record_invocation(2);
    c.record_invocation(3);
    let all = c.snapshot_all();
    assert_eq!(all.len(), 3);
}

#[test]
fn test_collector_sampling() {
    let config = CollectorConfig {
        sampling_rate: 4,
        ..CollectorConfig::for_testing()
    };
    let c = ProfileCollector::with_config(config);
    // Only every 4th event should be recorded
    for _ in 0..16 {
        c.record_invocation(1);
    }
    // 4 of 16 should have been recorded
    assert_eq!(c.total_events(), 4);
}

#[test]
fn test_collector_reset() {
    let c = ProfileCollector::with_config(CollectorConfig::for_testing());
    for _ in 0..20 {
        c.record_invocation(1);
    }
    c.record_branch(1, 10, true);
    c.reset();
    assert_eq!(c.total_events(), 0);
    assert_eq!(c.detector().hot_method_count(), 0);
}

#[test]
fn test_collector_multiple_units() {
    let c = ProfileCollector::with_config(CollectorConfig::for_testing());
    for code_id in 0..50u32 {
        c.record_invocation(code_id);
    }
    assert_eq!(c.tracked_unit_count(), 50);
}

#[test]
fn test_collector_concurrent() {
    use std::sync::Arc;
    use std::thread;

    let c = Arc::new(ProfileCollector::with_config(CollectorConfig::for_testing()));
    let mut handles = vec![];

    for tid in 0..4u32 {
        let c = Arc::clone(&c);
        handles.push(thread::spawn(move || {
            for _ in 0..100 {
                c.record_invocation(tid);
                c.record_branch(tid, 10, true);
            }
        }));
    }

    for h in handles {
        h.join().unwrap();
    }

    assert_eq!(c.tracked_unit_count(), 4);
    assert_eq!(c.total_events(), 800); // 4 threads × 200 events
}

#[test]
fn test_collector_get_or_create_idempotent() {
    let c = ProfileCollector::new();
    let c1 = c.get_or_create_counters(1);
    let c2 = c.get_or_create_counters(1);
    // Should be the same Arc
    assert!(Arc::ptr_eq(&c1, &c2));
}

#[test]
fn test_collector_snapshot_roundtrip() {
    let c = ProfileCollector::with_config(CollectorConfig::for_testing());
    for _ in 0..50 {
        c.record_invocation(1);
    }
    c.record_branch(1, 10, true);
    c.record_branch(1, 10, false);

    let snapshot = c.snapshot(1).unwrap();
    let bytes = snapshot.serialize();
    let restored = ProfileData::deserialize(&bytes).unwrap();
    assert_eq!(restored.execution_count(), 50);
    let bp = restored.branch_at(10).unwrap();
    assert_eq!(bp.taken, 1);
    assert_eq!(bp.not_taken, 1);
}
