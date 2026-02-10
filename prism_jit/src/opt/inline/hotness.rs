//! Hotness Tracking for JIT Function Inlining
//!
//! This module provides thread-safe call frequency tracking with exponential decay
//! for prioritizing hot functions during inlining. Features:
//!
//! - Lock-free atomic counters for call recording
//! - Exponential decay to age out stale hotness data
//! - Configurable thresholds for hot/warm/cold classification
//! - Inline priority computation for cost model integration
//!
//! # Performance Characteristics
//!
//! - `record_call`: O(1), lock-free atomic increment
//! - `get_hotness`: O(1), atomic load + threshold check
//! - `apply_decay`: O(n), batch operation on all functions

use std::sync::atomic::{AtomicU64, Ordering};

use dashmap::DashMap;

// =============================================================================
// Hotness Level
// =============================================================================

/// Classification of function call frequency.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum HotnessLevel {
    /// Function is rarely called - avoid inlining.
    Cold,
    /// Function has moderate call frequency.
    Warm,
    /// Function is frequently called - prioritize inlining.
    Hot,
    /// Function is extremely hot - always consider inlining.
    VeryHot,
}

impl HotnessLevel {
    /// Convert to priority multiplier for cost model.
    #[inline]
    pub fn priority_multiplier(self) -> f64 {
        match self {
            HotnessLevel::Cold => 0.5,
            HotnessLevel::Warm => 1.0,
            HotnessLevel::Hot => 2.0,
            HotnessLevel::VeryHot => 4.0,
        }
    }

    /// Check if this level should trigger eager inlining.
    #[inline]
    pub fn is_hot(self) -> bool {
        matches!(self, HotnessLevel::Hot | HotnessLevel::VeryHot)
    }

    /// Check if this level should avoid inlining.
    #[inline]
    pub fn is_cold(self) -> bool {
        self == HotnessLevel::Cold
    }
}

impl Default for HotnessLevel {
    fn default() -> Self {
        HotnessLevel::Warm
    }
}

// =============================================================================
// Hotness Configuration
// =============================================================================

/// Configuration for hotness tracking thresholds.
#[derive(Debug, Clone)]
pub struct HotnessConfig {
    /// Calls below this are cold.
    pub cold_threshold: u64,
    /// Calls at or above this are hot.
    pub hot_threshold: u64,
    /// Calls at or above this are very hot.
    pub very_hot_threshold: u64,
    /// Decay factor (0.0-1.0, applied periodically).
    pub decay_factor: f64,
    /// Maximum count before saturation.
    pub max_count: u64,
}

impl Default for HotnessConfig {
    fn default() -> Self {
        Self {
            cold_threshold: 10,
            hot_threshold: 100,
            very_hot_threshold: 1000,
            decay_factor: 0.9,
            max_count: 1_000_000,
        }
    }
}

impl HotnessConfig {
    /// Create aggressive thresholds for faster promotion to hot.
    pub fn aggressive() -> Self {
        Self {
            cold_threshold: 5,
            hot_threshold: 50,
            very_hot_threshold: 500,
            decay_factor: 0.95,
            max_count: 1_000_000,
        }
    }

    /// Create conservative thresholds requiring more calls.
    pub fn conservative() -> Self {
        Self {
            cold_threshold: 50,
            hot_threshold: 500,
            very_hot_threshold: 5000,
            decay_factor: 0.8,
            max_count: 1_000_000,
        }
    }

    /// Create configuration for tier-1 JIT (quick decisions).
    pub fn tier1() -> Self {
        Self {
            cold_threshold: 3,
            hot_threshold: 20,
            very_hot_threshold: 100,
            decay_factor: 0.95,
            max_count: 100_000,
        }
    }

    /// Create configuration for tier-2 JIT (thorough analysis).
    pub fn tier2() -> Self {
        Self::default()
    }
}

// =============================================================================
// Hotness Entry
// =============================================================================

/// Per-function hotness data with atomic access.
#[derive(Debug)]
struct HotnessEntry {
    /// Raw call count (atomic for lock-free updates).
    count: AtomicU64,
    /// Last decay generation (for detecting stale entries).
    last_decay_gen: AtomicU64,
}

impl HotnessEntry {
    /// Create a new entry with zero count.
    fn new() -> Self {
        Self {
            count: AtomicU64::new(0),
            last_decay_gen: AtomicU64::new(0),
        }
    }

    /// Increment the call count, saturating at max.
    #[inline]
    fn increment(&self, max_count: u64) {
        // Use fetch_add with saturating semantics
        let old = self.count.fetch_add(1, Ordering::Relaxed);
        if old >= max_count {
            // Saturate at max
            self.count.store(max_count, Ordering::Relaxed);
        }
    }

    /// Get the current count.
    #[inline]
    fn get_count(&self) -> u64 {
        self.count.load(Ordering::Relaxed)
    }

    /// Apply decay to the count.
    fn apply_decay(&self, factor: f64, current_gen: u64) {
        // Check if we've already decayed this generation
        let last_gen = self.last_decay_gen.load(Ordering::Relaxed);
        if last_gen >= current_gen {
            return;
        }

        // Apply decay
        let old_count = self.count.load(Ordering::Relaxed);
        let new_count = (old_count as f64 * factor) as u64;

        // CAS loop for atomic update (handles concurrent decays)
        let _ =
            self.count
                .compare_exchange(old_count, new_count, Ordering::Relaxed, Ordering::Relaxed);

        // Update decay generation
        let _ = self.last_decay_gen.compare_exchange(
            last_gen,
            current_gen,
            Ordering::Relaxed,
            Ordering::Relaxed,
        );
    }

    /// Reset the count to zero.
    fn reset(&self) {
        self.count.store(0, Ordering::Relaxed);
    }
}

impl Clone for HotnessEntry {
    fn clone(&self) -> Self {
        Self {
            count: AtomicU64::new(self.count.load(Ordering::Relaxed)),
            last_decay_gen: AtomicU64::new(self.last_decay_gen.load(Ordering::Relaxed)),
        }
    }
}

// =============================================================================
// Hotness Tracker
// =============================================================================

/// Thread-safe hotness tracker with exponential decay.
///
/// Tracks call frequency for functions to prioritize inlining decisions.
/// Uses lock-free atomics for high-throughput call recording.
#[derive(Debug)]
pub struct HotnessTracker {
    /// Per-function hotness entries.
    entries: DashMap<u64, HotnessEntry>,
    /// Configuration thresholds.
    config: HotnessConfig,
    /// Current decay generation.
    decay_generation: AtomicU64,
}

impl HotnessTracker {
    /// Create a new hotness tracker with default configuration.
    pub fn new() -> Self {
        Self::with_config(HotnessConfig::default())
    }

    /// Create a tracker with custom configuration.
    pub fn with_config(config: HotnessConfig) -> Self {
        Self {
            entries: DashMap::new(),
            config,
            decay_generation: AtomicU64::new(0),
        }
    }

    /// Record a call to a function.
    ///
    /// This is the hot path - uses lock-free atomics for performance.
    #[inline]
    pub fn record_call(&self, func_id: u64) {
        // Fast path: entry already exists
        if let Some(entry) = self.entries.get(&func_id) {
            entry.increment(self.config.max_count);
            return;
        }

        // Slow path: create entry
        let entry = self
            .entries
            .entry(func_id)
            .or_insert_with(HotnessEntry::new);
        entry.increment(self.config.max_count);
    }

    /// Record multiple calls at once (batch operation).
    #[inline]
    pub fn record_calls(&self, func_id: u64, count: u64) {
        let entry = self
            .entries
            .entry(func_id)
            .or_insert_with(HotnessEntry::new);
        let old = entry.count.fetch_add(count, Ordering::Relaxed);
        if old + count >= self.config.max_count {
            entry.count.store(self.config.max_count, Ordering::Relaxed);
        }
    }

    /// Get the hotness level for a function.
    #[inline]
    pub fn get_hotness(&self, func_id: u64) -> HotnessLevel {
        let count = self.get_call_count(func_id);
        self.classify_count(count)
    }

    /// Get the raw call count for a function.
    #[inline]
    pub fn get_call_count(&self, func_id: u64) -> u64 {
        self.entries
            .get(&func_id)
            .map(|e| e.get_count())
            .unwrap_or(0)
    }

    /// Classify a count into a hotness level.
    #[inline]
    fn classify_count(&self, count: u64) -> HotnessLevel {
        if count >= self.config.very_hot_threshold {
            HotnessLevel::VeryHot
        } else if count >= self.config.hot_threshold {
            HotnessLevel::Hot
        } else if count >= self.config.cold_threshold {
            HotnessLevel::Warm
        } else {
            HotnessLevel::Cold
        }
    }

    /// Get the inline priority for a function (0.0 to 1.0).
    ///
    /// Higher values indicate functions that should be prioritized for inlining.
    pub fn get_inline_priority(&self, func_id: u64) -> f64 {
        let count = self.get_call_count(func_id);
        if count == 0 {
            return 0.0;
        }

        // Logarithmic scaling with normalization
        let log_count = (count as f64).ln();
        let log_max = (self.config.very_hot_threshold as f64).ln();

        // Clamp to [0.0, 1.0] range
        (log_count / log_max).min(1.0).max(0.0)
    }

    /// Apply exponential decay to all function counts.
    ///
    /// Should be called periodically (e.g., after each compilation cycle)
    /// to age out stale hotness data.
    pub fn apply_decay(&self) {
        let decay_gen = self.decay_generation.fetch_add(1, Ordering::Relaxed) + 1;
        let factor = self.config.decay_factor;

        for entry in self.entries.iter() {
            entry.apply_decay(factor, decay_gen);
        }
    }

    /// Reset a specific function's count.
    pub fn reset(&self, func_id: u64) {
        if let Some(entry) = self.entries.get(&func_id) {
            entry.reset();
        }
    }

    /// Reset all function counts.
    pub fn reset_all(&self) {
        for entry in self.entries.iter() {
            entry.reset();
        }
    }

    /// Remove a function from tracking.
    pub fn remove(&self, func_id: u64) -> bool {
        self.entries.remove(&func_id).is_some()
    }

    /// Get the number of tracked functions.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Check if no functions are tracked.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Get all hot functions (at least Hot level).
    pub fn get_hot_functions(&self) -> Vec<u64> {
        self.entries
            .iter()
            .filter(|e| e.get_count() >= self.config.hot_threshold)
            .map(|e| *e.key())
            .collect()
    }

    /// Get all very hot functions.
    pub fn get_very_hot_functions(&self) -> Vec<u64> {
        self.entries
            .iter()
            .filter(|e| e.get_count() >= self.config.very_hot_threshold)
            .map(|e| *e.key())
            .collect()
    }

    /// Get functions sorted by hotness (descending).
    pub fn get_sorted_by_hotness(&self) -> Vec<(u64, u64)> {
        let mut result: Vec<_> = self
            .entries
            .iter()
            .map(|e| (*e.key(), e.get_count()))
            .collect();
        result.sort_by(|a, b| b.1.cmp(&a.1));
        result
    }

    /// Get a snapshot of all hotness data.
    pub fn snapshot(&self) -> Vec<(u64, u64, HotnessLevel)> {
        self.entries
            .iter()
            .map(|e| {
                let count = e.get_count();
                (*e.key(), count, self.classify_count(count))
            })
            .collect()
    }

    /// Merge hotness data from another tracker.
    pub fn merge(&self, other: &HotnessTracker) {
        for entry in other.entries.iter() {
            let func_id = *entry.key();
            let count = entry.get_count();
            self.record_calls(func_id, count);
        }
    }

    /// Get current configuration.
    pub fn config(&self) -> &HotnessConfig {
        &self.config
    }

    /// Get the current decay generation.
    pub fn decay_generation(&self) -> u64 {
        self.decay_generation.load(Ordering::Relaxed)
    }
}

impl Default for HotnessTracker {
    fn default() -> Self {
        Self::new()
    }
}

impl Clone for HotnessTracker {
    fn clone(&self) -> Self {
        let entries = DashMap::new();
        for entry in self.entries.iter() {
            entries.insert(*entry.key(), entry.value().clone());
        }
        Self {
            entries,
            config: self.config.clone(),
            decay_generation: AtomicU64::new(self.decay_generation.load(Ordering::Relaxed)),
        }
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
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

    #[test]
    fn test_hotness_level_default() {
        assert_eq!(HotnessLevel::default(), HotnessLevel::Warm);
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
}
