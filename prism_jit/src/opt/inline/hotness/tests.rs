use super::*;
use std::thread;

// =========================================================================
// HotnessLevel Tests
// =========================================================================

#[test]
fn test_hotness_level_ordering() {
    assert!(HotnessLevel::Cold < HotnessLevel::Warm);
    assert!(HotnessLevel::Warm < HotnessLevel::Hot);
    assert!(HotnessLevel::Hot < HotnessLevel::VeryHot);
}

#[test]
fn test_hotness_level_priority_multiplier() {
    assert_eq!(HotnessLevel::Cold.priority_multiplier(), 0.5);
    assert_eq!(HotnessLevel::Warm.priority_multiplier(), 1.0);
    assert_eq!(HotnessLevel::Hot.priority_multiplier(), 2.0);
    assert_eq!(HotnessLevel::VeryHot.priority_multiplier(), 4.0);
}

#[test]
fn test_hotness_level_is_hot() {
    assert!(!HotnessLevel::Cold.is_hot());
    assert!(!HotnessLevel::Warm.is_hot());
    assert!(HotnessLevel::Hot.is_hot());
    assert!(HotnessLevel::VeryHot.is_hot());
}

#[test]
fn test_hotness_level_is_cold() {
    assert!(HotnessLevel::Cold.is_cold());
    assert!(!HotnessLevel::Warm.is_cold());
    assert!(!HotnessLevel::Hot.is_cold());
    assert!(!HotnessLevel::VeryHot.is_cold());
}

// =========================================================================
// HotnessConfig Tests
// =========================================================================

#[test]
fn test_config_default() {
    let config = HotnessConfig::default();
    assert_eq!(config.cold_threshold, 10);
    assert_eq!(config.hot_threshold, 100);
    assert_eq!(config.very_hot_threshold, 1000);
    assert!((config.decay_factor - 0.9).abs() < 0.01);
}

#[test]
fn test_config_aggressive() {
    let config = HotnessConfig::aggressive();
    assert!(config.hot_threshold < HotnessConfig::default().hot_threshold);
    assert!(config.cold_threshold < HotnessConfig::default().cold_threshold);
}

#[test]
fn test_config_conservative() {
    let config = HotnessConfig::conservative();
    assert!(config.hot_threshold > HotnessConfig::default().hot_threshold);
    assert!(config.cold_threshold > HotnessConfig::default().cold_threshold);
}

#[test]
fn test_config_tier1() {
    let config = HotnessConfig::tier1();
    assert!(config.hot_threshold < HotnessConfig::default().hot_threshold);
}

#[test]
fn test_config_tier2() {
    let config = HotnessConfig::tier2();
    assert_eq!(config.hot_threshold, HotnessConfig::default().hot_threshold);
}

// =========================================================================
// HotnessTracker Basic Tests
// =========================================================================

#[test]
fn test_tracker_new() {
    let tracker = HotnessTracker::new();
    assert!(tracker.is_empty());
    assert_eq!(tracker.len(), 0);
}

#[test]
fn test_tracker_record_single_call() {
    let tracker = HotnessTracker::new();
    tracker.record_call(1);
    assert_eq!(tracker.get_call_count(1), 1);
    assert_eq!(tracker.len(), 1);
}

#[test]
fn test_tracker_record_multiple_calls() {
    let tracker = HotnessTracker::new();
    for _ in 0..100 {
        tracker.record_call(1);
    }
    assert_eq!(tracker.get_call_count(1), 100);
}

#[test]
fn test_tracker_record_calls_batch() {
    let tracker = HotnessTracker::new();
    tracker.record_calls(1, 50);
    assert_eq!(tracker.get_call_count(1), 50);
}

#[test]
fn test_tracker_multiple_functions() {
    let tracker = HotnessTracker::new();
    tracker.record_calls(1, 10);
    tracker.record_calls(2, 100);
    tracker.record_calls(3, 1000);

    assert_eq!(tracker.get_call_count(1), 10);
    assert_eq!(tracker.get_call_count(2), 100);
    assert_eq!(tracker.get_call_count(3), 1000);
    assert_eq!(tracker.len(), 3);
}

#[test]
fn test_tracker_unknown_function() {
    let tracker = HotnessTracker::new();
    assert_eq!(tracker.get_call_count(999), 0);
    assert_eq!(tracker.get_hotness(999), HotnessLevel::Cold);
}

// =========================================================================
// Hotness Classification Tests
// =========================================================================

#[test]
fn test_classification_cold() {
    let tracker = HotnessTracker::new();
    tracker.record_calls(1, 5); // Below cold_threshold (10)
    assert_eq!(tracker.get_hotness(1), HotnessLevel::Cold);
}

#[test]
fn test_classification_warm() {
    let tracker = HotnessTracker::new();
    tracker.record_calls(1, 50); // Between cold and hot
    assert_eq!(tracker.get_hotness(1), HotnessLevel::Warm);
}

#[test]
fn test_classification_hot() {
    let tracker = HotnessTracker::new();
    tracker.record_calls(1, 500); // At or above hot_threshold
    assert_eq!(tracker.get_hotness(1), HotnessLevel::Hot);
}

#[test]
fn test_classification_very_hot() {
    let tracker = HotnessTracker::new();
    tracker.record_calls(1, 5000); // At or above very_hot_threshold
    assert_eq!(tracker.get_hotness(1), HotnessLevel::VeryHot);
}

#[test]
fn test_classification_at_thresholds() {
    let tracker = HotnessTracker::new();

    tracker.record_calls(1, 10); // Exactly at cold_threshold
    assert_eq!(tracker.get_hotness(1), HotnessLevel::Warm);

    tracker.record_calls(2, 100); // Exactly at hot_threshold
    assert_eq!(tracker.get_hotness(2), HotnessLevel::Hot);

    tracker.record_calls(3, 1000); // Exactly at very_hot_threshold
    assert_eq!(tracker.get_hotness(3), HotnessLevel::VeryHot);
}

// =========================================================================
// Inline Priority Tests
// =========================================================================

#[test]
fn test_priority_zero_calls() {
    let tracker = HotnessTracker::new();
    assert_eq!(tracker.get_inline_priority(1), 0.0);
}

#[test]
fn test_priority_increases_with_calls() {
    let tracker = HotnessTracker::new();
    tracker.record_calls(1, 10);
    let p1 = tracker.get_inline_priority(1);

    tracker.record_calls(2, 100);
    let p2 = tracker.get_inline_priority(2);

    tracker.record_calls(3, 1000);
    let p3 = tracker.get_inline_priority(3);

    assert!(p1 < p2);
    assert!(p2 < p3);
}

#[test]
fn test_priority_bounded() {
    let tracker = HotnessTracker::new();
    tracker.record_calls(1, 1_000_000);
    let priority = tracker.get_inline_priority(1);
    assert!(priority <= 1.0);
    assert!(priority >= 0.0);
}

// =========================================================================
// Decay Tests
// =========================================================================

#[test]
fn test_decay_reduces_counts() {
    let tracker = HotnessTracker::new();
    tracker.record_calls(1, 1000);

    let before = tracker.get_call_count(1);
    tracker.apply_decay();
    let after = tracker.get_call_count(1);

    assert!(after < before);
}

#[test]
fn test_decay_multiple_applications() {
    let tracker = HotnessTracker::new();
    tracker.record_calls(1, 1000);

    for _ in 0..10 {
        tracker.apply_decay();
    }

    let count = tracker.get_call_count(1);
    assert!(count < 500); // Significant reduction after multiple decays
}

#[test]
fn test_decay_generation_increments() {
    let tracker = HotnessTracker::new();
    assert_eq!(tracker.decay_generation(), 0);

    tracker.apply_decay();
    assert_eq!(tracker.decay_generation(), 1);

    tracker.apply_decay();
    assert_eq!(tracker.decay_generation(), 2);
}

// =========================================================================
// Reset Tests
// =========================================================================

#[test]
fn test_reset_single() {
    let tracker = HotnessTracker::new();
    tracker.record_calls(1, 100);
    tracker.reset(1);
    assert_eq!(tracker.get_call_count(1), 0);
}

#[test]
fn test_reset_all() {
    let tracker = HotnessTracker::new();
    tracker.record_calls(1, 100);
    tracker.record_calls(2, 200);
    tracker.reset_all();
    assert_eq!(tracker.get_call_count(1), 0);
    assert_eq!(tracker.get_call_count(2), 0);
}

#[test]
fn test_remove_function() {
    let tracker = HotnessTracker::new();
    tracker.record_calls(1, 100);
    assert_eq!(tracker.len(), 1);

    let removed = tracker.remove(1);
    assert!(removed);
    assert_eq!(tracker.len(), 0);
}

#[test]
fn test_remove_nonexistent() {
    let tracker = HotnessTracker::new();
    let removed = tracker.remove(999);
    assert!(!removed);
}

// =========================================================================
// Hot Function Query Tests
// =========================================================================

#[test]
fn test_get_hot_functions() {
    let tracker = HotnessTracker::new();
    tracker.record_calls(1, 50); // Warm
    tracker.record_calls(2, 200); // Hot
    tracker.record_calls(3, 2000); // Very hot

    let hot = tracker.get_hot_functions();
    assert_eq!(hot.len(), 2);
    assert!(hot.contains(&2));
    assert!(hot.contains(&3));
}

#[test]
fn test_get_very_hot_functions() {
    let tracker = HotnessTracker::new();
    tracker.record_calls(1, 200); // Hot
    tracker.record_calls(2, 2000); // Very hot

    let very_hot = tracker.get_very_hot_functions();
    assert_eq!(very_hot.len(), 1);
    assert!(very_hot.contains(&2));
}

#[test]
fn test_get_sorted_by_hotness() {
    let tracker = HotnessTracker::new();
    tracker.record_calls(1, 10);
    tracker.record_calls(2, 1000);
    tracker.record_calls(3, 100);

    let sorted = tracker.get_sorted_by_hotness();
    assert_eq!(sorted.len(), 3);
    assert_eq!(sorted[0].0, 2); // Highest
    assert_eq!(sorted[1].0, 3);
    assert_eq!(sorted[2].0, 1); // Lowest
}

// =========================================================================
// Snapshot and Merge Tests
// =========================================================================

#[test]
fn test_snapshot() {
    let tracker = HotnessTracker::new();
    tracker.record_calls(1, 5);
    tracker.record_calls(2, 500);

    let snapshot = tracker.snapshot();
    assert_eq!(snapshot.len(), 2);
}

#[test]
fn test_merge() {
    let tracker1 = HotnessTracker::new();
    tracker1.record_calls(1, 50);

    let tracker2 = HotnessTracker::new();
    tracker2.record_calls(1, 50);
    tracker2.record_calls(2, 100);

    tracker1.merge(&tracker2);
    assert_eq!(tracker1.get_call_count(1), 100);
    assert_eq!(tracker1.get_call_count(2), 100);
}

// =========================================================================
// Clone Tests
// =========================================================================

#[test]
fn test_clone() {
    let tracker = HotnessTracker::new();
    tracker.record_calls(1, 100);

    let cloned = tracker.clone();
    assert_eq!(cloned.get_call_count(1), 100);

    // Modifications to original don't affect clone
    tracker.record_calls(1, 100);
    assert_eq!(cloned.get_call_count(1), 100);
    assert_eq!(tracker.get_call_count(1), 200);
}

// =========================================================================
// Saturation Tests
// =========================================================================

#[test]
fn test_saturation() {
    let config = HotnessConfig {
        max_count: 100,
        ..Default::default()
    };
    let tracker = HotnessTracker::with_config(config);

    tracker.record_calls(1, 200);
    assert_eq!(tracker.get_call_count(1), 100); // Saturated
}

#[test]
fn test_saturation_incremental() {
    let config = HotnessConfig {
        max_count: 100,
        ..Default::default()
    };
    let tracker = HotnessTracker::with_config(config);

    for _ in 0..200 {
        tracker.record_call(1);
    }
    assert_eq!(tracker.get_call_count(1), 100); // Saturated
}

// =========================================================================
// Thread Safety Tests
// =========================================================================

#[test]
fn test_concurrent_recording() {
    let tracker = std::sync::Arc::new(HotnessTracker::new());
    let mut handles = vec![];

    for _ in 0..8 {
        let t = tracker.clone();
        handles.push(thread::spawn(move || {
            for _ in 0..1000 {
                t.record_call(1);
            }
        }));
    }

    for h in handles {
        h.join().unwrap();
    }

    assert_eq!(tracker.get_call_count(1), 8000);
}

#[test]
fn test_concurrent_multiple_functions() {
    let tracker = std::sync::Arc::new(HotnessTracker::new());
    let mut handles = vec![];

    for i in 0..8 {
        let t = tracker.clone();
        handles.push(thread::spawn(move || {
            for _ in 0..100 {
                t.record_call(i as u64);
            }
        }));
    }

    for h in handles {
        h.join().unwrap();
    }

    assert_eq!(tracker.len(), 8);
    for i in 0..8 {
        assert_eq!(tracker.get_call_count(i as u64), 100);
    }
}

#[test]
fn test_concurrent_decay() {
    let tracker = std::sync::Arc::new(HotnessTracker::new());
    tracker.record_calls(1, 10000);

    let mut handles = vec![];

    // Concurrent recording
    let t1 = tracker.clone();
    handles.push(thread::spawn(move || {
        for _ in 0..1000 {
            t1.record_call(1);
        }
    }));

    // Concurrent decay
    let t2 = tracker.clone();
    handles.push(thread::spawn(move || {
        for _ in 0..10 {
            t2.apply_decay();
        }
    }));

    for h in handles {
        h.join().unwrap();
    }

    // Should complete without panicking - exact count depends on interleaving
    let _ = tracker.get_call_count(1);
}

// =========================================================================
// Config Access Tests
// =========================================================================

#[test]
fn test_config_access() {
    let config = HotnessConfig::aggressive();
    let tracker = HotnessTracker::with_config(config.clone());
    assert_eq!(tracker.config().hot_threshold, config.hot_threshold);
}

// =========================================================================
// Custom Configuration Tests
// =========================================================================

#[test]
fn test_custom_thresholds() {
    let config = HotnessConfig {
        cold_threshold: 1,
        hot_threshold: 5,
        very_hot_threshold: 10,
        ..Default::default()
    };
    let tracker = HotnessTracker::with_config(config);

    tracker.record_calls(1, 0);
    assert_eq!(tracker.get_hotness(1), HotnessLevel::Cold);

    tracker.record_calls(2, 3);
    assert_eq!(tracker.get_hotness(2), HotnessLevel::Warm);

    tracker.record_calls(3, 5);
    assert_eq!(tracker.get_hotness(3), HotnessLevel::Hot);

    tracker.record_calls(4, 15);
    assert_eq!(tracker.get_hotness(4), HotnessLevel::VeryHot);
}

#[test]
fn test_aggressive_decay() {
    let config = HotnessConfig {
        decay_factor: 0.5, // Very aggressive
        ..Default::default()
    };
    let tracker = HotnessTracker::with_config(config);

    tracker.record_calls(1, 1000);
    tracker.apply_decay();

    let count = tracker.get_call_count(1);
    assert!(count < 600); // At least 40% reduction
}

#[test]
fn test_no_decay() {
    let config = HotnessConfig {
        decay_factor: 1.0, // No decay
        ..Default::default()
    };
    let tracker = HotnessTracker::with_config(config);

    tracker.record_calls(1, 1000);
    tracker.apply_decay();

    assert_eq!(tracker.get_call_count(1), 1000);
}
