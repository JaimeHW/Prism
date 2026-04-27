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
