//! Profile collector for gathering runtime feedback.
//!
//! Provides lock-free, low-overhead profile collection during interpretation.
//! Supports atomic branch/call/type counters, hot method detection, and
//! optional sampling mode for production use.
//!
//! # Architecture
//!
//! ```text
//! ┌──────────────┐     ┌──────────────────┐     ┌──────────────┐
//! │  Interpreter │────▶│ ProfileCollector  │────▶│ ProfileData  │
//! │  (hot path)  │     │ (atomic counters) │     │ (snapshot)   │
//! └──────────────┘     └──────────────────┘     └──────────────┘
//! ```

use super::profile_data::{AtomicBranchCounter, ProfileData};
use rustc_hash::FxHashMap;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, RwLock};

// =============================================================================
// Configuration
// =============================================================================

/// Configuration for profile collection.
#[derive(Debug, Clone)]
pub struct CollectorConfig {
    /// Invocation count threshold for hot method detection.
    pub hot_method_threshold: u64,
    /// Loop iteration threshold for hot loop detection.
    pub hot_loop_threshold: u64,
    /// Sampling rate (1 = every execution, N = every Nth execution).
    /// Higher values reduce overhead at the cost of precision.
    pub sampling_rate: u32,
    /// Maximum number of code units to track.
    pub max_tracked_units: usize,
    /// Enable call target tracking.
    pub track_calls: bool,
    /// Enable type feedback tracking.
    pub track_types: bool,
}

impl Default for CollectorConfig {
    fn default() -> Self {
        Self {
            hot_method_threshold: 1000,
            hot_loop_threshold: 10_000,
            sampling_rate: 1,
            max_tracked_units: 4096,
            track_calls: true,
            track_types: true,
        }
    }
}

impl CollectorConfig {
    /// Low-overhead production configuration.
    pub fn production() -> Self {
        Self {
            hot_method_threshold: 5000,
            hot_loop_threshold: 50_000,
            sampling_rate: 8,
            max_tracked_units: 8192,
            track_calls: true,
            track_types: true,
        }
    }

    /// High-detail profiling configuration.
    pub fn detailed() -> Self {
        Self {
            hot_method_threshold: 100,
            hot_loop_threshold: 1000,
            sampling_rate: 1,
            max_tracked_units: 16384,
            track_calls: true,
            track_types: true,
        }
    }

    /// Minimal configuration for testing.
    pub fn for_testing() -> Self {
        Self {
            hot_method_threshold: 10,
            hot_loop_threshold: 100,
            sampling_rate: 1,
            max_tracked_units: 256,
            track_calls: true,
            track_types: true,
        }
    }
}

// =============================================================================
// Per-Code-Unit Counters (lock-free)
// =============================================================================

/// Lock-free counters for a single code unit.
///
/// All counters use atomic operations for concurrent access from
/// multiple interpreter threads.
#[derive(Debug)]
pub struct CodeUnitCounters {
    /// Code unit identifier.
    code_id: u32,
    /// Total invocation count.
    invocation_count: AtomicU64,
    /// Branch counters indexed by bytecode offset.
    /// Protected by RwLock for rare insertions, reads are lock-free via snapshot.
    branches: RwLock<FxHashMap<u32, AtomicBranchCounter>>,
    /// Loop iteration counters indexed by header offset.
    loops: RwLock<FxHashMap<u32, AtomicU64>>,
}

impl CodeUnitCounters {
    /// Create new counters for a code unit.
    pub fn new(code_id: u32) -> Self {
        Self {
            code_id,
            invocation_count: AtomicU64::new(0),
            branches: RwLock::new(FxHashMap::default()),
            loops: RwLock::new(FxHashMap::default()),
        }
    }

    /// Record an invocation (lock-free).
    #[inline]
    pub fn record_invocation(&self) -> u64 {
        self.invocation_count.fetch_add(1, Ordering::Relaxed) + 1
    }

    /// Get current invocation count.
    #[inline]
    pub fn invocation_count(&self) -> u64 {
        self.invocation_count.load(Ordering::Relaxed)
    }

    /// Record a branch outcome.
    pub fn record_branch(&self, offset: u32, taken: bool) {
        // Fast path: try read lock first
        {
            let branches = self.branches.read().unwrap();
            if let Some(counter) = branches.get(&offset) {
                if taken {
                    counter.record_taken();
                } else {
                    counter.record_not_taken();
                }
                return;
            }
        }
        // Slow path: insert new counter
        let mut branches = self.branches.write().unwrap();
        let counter = branches
            .entry(offset)
            .or_insert_with(AtomicBranchCounter::new);
        if taken {
            counter.record_taken();
        } else {
            counter.record_not_taken();
        }
    }

    /// Record a loop iteration.
    pub fn record_loop(&self, header_offset: u32) {
        {
            let loops = self.loops.read().unwrap();
            if let Some(counter) = loops.get(&header_offset) {
                counter.fetch_add(1, Ordering::Relaxed);
                return;
            }
        }
        let mut loops = self.loops.write().unwrap();
        loops
            .entry(header_offset)
            .or_insert_with(|| AtomicU64::new(0))
            .fetch_add(1, Ordering::Relaxed);
    }

    /// Snapshot all counters into a ProfileData.
    pub fn snapshot(&self) -> ProfileData {
        let mut data = ProfileData::new(self.code_id);

        // Execution count
        let exec_count = self.invocation_count.load(Ordering::Relaxed);
        for _ in 0..exec_count {
            data.record_execution();
        }

        // Branch profiles
        let branches = self.branches.read().unwrap();
        for (&offset, counter) in branches.iter() {
            let bp = counter.snapshot();
            for _ in 0..bp.taken {
                data.record_branch(offset, true);
            }
            for _ in 0..bp.not_taken {
                data.record_branch(offset, false);
            }
        }

        // Loop counts
        let loops = self.loops.read().unwrap();
        for (&offset, counter) in loops.iter() {
            let count = counter.load(Ordering::Relaxed);
            for _ in 0..count {
                data.record_loop_iteration(offset);
            }
        }

        data
    }

    /// Reset all counters.
    pub fn reset(&self) {
        self.invocation_count.store(0, Ordering::Relaxed);
        let branches = self.branches.read().unwrap();
        for counter in branches.values() {
            counter.reset();
        }
        let loops = self.loops.read().unwrap();
        for counter in loops.values() {
            counter.store(0, Ordering::Relaxed);
        }
    }
}

// =============================================================================
// Hot Method Detector
// =============================================================================

/// Detects hot methods and loops based on execution frequency.
#[derive(Debug)]
pub struct HotMethodDetector {
    /// Method invocation threshold.
    method_threshold: u64,
    /// Loop iteration threshold.
    loop_threshold: u64,
    /// Set of code IDs already identified as hot (avoid re-signaling).
    hot_methods: RwLock<rustc_hash::FxHashSet<u32>>,
    /// Set of (code_id, loop_header) already identified as hot.
    hot_loops: RwLock<rustc_hash::FxHashSet<(u32, u32)>>,
}

impl HotMethodDetector {
    /// Create a new detector with given thresholds.
    pub fn new(method_threshold: u64, loop_threshold: u64) -> Self {
        Self {
            method_threshold,
            loop_threshold,
            hot_methods: RwLock::new(rustc_hash::FxHashSet::default()),
            hot_loops: RwLock::new(rustc_hash::FxHashSet::default()),
        }
    }

    /// Check if a method just became hot. Returns true exactly once per method.
    pub fn check_method_hot(&self, code_id: u32, invocation_count: u64) -> bool {
        if invocation_count < self.method_threshold {
            return false;
        }
        // Check if already known
        {
            let hot = self.hot_methods.read().unwrap();
            if hot.contains(&code_id) {
                return false;
            }
        }
        // Mark as hot
        let mut hot = self.hot_methods.write().unwrap();
        hot.insert(code_id)
    }

    /// Check if a loop just became hot. Returns true exactly once per loop.
    pub fn check_loop_hot(&self, code_id: u32, header_offset: u32, iteration_count: u64) -> bool {
        if iteration_count < self.loop_threshold {
            return false;
        }
        let key = (code_id, header_offset);
        {
            let hot = self.hot_loops.read().unwrap();
            if hot.contains(&key) {
                return false;
            }
        }
        let mut hot = self.hot_loops.write().unwrap();
        hot.insert(key)
    }

    /// Get all hot method IDs.
    pub fn hot_method_ids(&self) -> Vec<u32> {
        self.hot_methods.read().unwrap().iter().copied().collect()
    }

    /// Get all hot loop keys.
    pub fn hot_loop_keys(&self) -> Vec<(u32, u32)> {
        self.hot_loops.read().unwrap().iter().copied().collect()
    }

    /// Number of hot methods detected.
    pub fn hot_method_count(&self) -> usize {
        self.hot_methods.read().unwrap().len()
    }

    /// Number of hot loops detected.
    pub fn hot_loop_count(&self) -> usize {
        self.hot_loops.read().unwrap().len()
    }

    /// Reset all detection state.
    pub fn reset(&self) {
        self.hot_methods.write().unwrap().clear();
        self.hot_loops.write().unwrap().clear();
    }
}

// =============================================================================
// Profile Collector (top-level)
// =============================================================================

/// Central profile collector coordinating all runtime feedback.
///
/// Thread-safe. Designed for concurrent access from multiple interpreter
/// threads with minimal lock contention.
#[derive(Debug)]
pub struct ProfileCollector {
    /// Configuration.
    config: CollectorConfig,
    /// Per-code-unit counters.
    units: RwLock<FxHashMap<u32, Arc<CodeUnitCounters>>>,
    /// Hot method/loop detector.
    detector: HotMethodDetector,
    /// Global sample counter for sampling mode.
    sample_counter: AtomicU32,
    /// Total events recorded.
    total_events: AtomicU64,
}

impl ProfileCollector {
    /// Create a new collector with default configuration.
    pub fn new() -> Self {
        Self::with_config(CollectorConfig::default())
    }

    /// Create with custom configuration.
    pub fn with_config(config: CollectorConfig) -> Self {
        let detector =
            HotMethodDetector::new(config.hot_method_threshold, config.hot_loop_threshold);
        Self {
            config,
            units: RwLock::new(FxHashMap::default()),
            detector,
            sample_counter: AtomicU32::new(0),
            total_events: AtomicU64::new(0),
        }
    }

    /// Get or create counters for a code unit.
    pub fn get_or_create_counters(&self, code_id: u32) -> Arc<CodeUnitCounters> {
        // Fast path: read lock
        {
            let units = self.units.read().unwrap();
            if let Some(counters) = units.get(&code_id) {
                return Arc::clone(counters);
            }
        }
        // Slow path: write lock
        let mut units = self.units.write().unwrap();
        Arc::clone(
            units
                .entry(code_id)
                .or_insert_with(|| Arc::new(CodeUnitCounters::new(code_id))),
        )
    }

    /// Record a method invocation. Returns true if method just became hot.
    pub fn record_invocation(&self, code_id: u32) -> bool {
        if !self.should_sample() {
            return false;
        }
        self.total_events.fetch_add(1, Ordering::Relaxed);
        let counters = self.get_or_create_counters(code_id);
        let count = counters.record_invocation();
        self.detector.check_method_hot(code_id, count)
    }

    /// Record a branch outcome.
    pub fn record_branch(&self, code_id: u32, offset: u32, taken: bool) {
        if !self.should_sample() {
            return;
        }
        self.total_events.fetch_add(1, Ordering::Relaxed);
        let counters = self.get_or_create_counters(code_id);
        counters.record_branch(offset, taken);
    }

    /// Record a loop iteration. Returns true if loop just became hot.
    pub fn record_loop(&self, code_id: u32, header_offset: u32) -> bool {
        if !self.should_sample() {
            return false;
        }
        self.total_events.fetch_add(1, Ordering::Relaxed);
        let counters = self.get_or_create_counters(code_id);
        counters.record_loop(header_offset);
        let loops = counters.loops.read().unwrap();
        let count = loops
            .get(&header_offset)
            .map_or(0, |c| c.load(Ordering::Relaxed));
        self.detector.check_loop_hot(code_id, header_offset, count)
    }

    /// Snapshot profile data for a specific code unit.
    pub fn snapshot(&self, code_id: u32) -> Option<ProfileData> {
        let units = self.units.read().unwrap();
        units.get(&code_id).map(|c| c.snapshot())
    }

    /// Snapshot all profile data.
    pub fn snapshot_all(&self) -> Vec<ProfileData> {
        let units = self.units.read().unwrap();
        units.values().map(|c| c.snapshot()).collect()
    }

    /// Get the hot method detector.
    pub fn detector(&self) -> &HotMethodDetector {
        &self.detector
    }

    /// Total events recorded.
    pub fn total_events(&self) -> u64 {
        self.total_events.load(Ordering::Relaxed)
    }

    /// Number of tracked code units.
    pub fn tracked_unit_count(&self) -> usize {
        self.units.read().unwrap().len()
    }

    /// Reset all collected data.
    pub fn reset(&self) {
        let units = self.units.read().unwrap();
        for counters in units.values() {
            counters.reset();
        }
        self.detector.reset();
        self.total_events.store(0, Ordering::Relaxed);
        self.sample_counter.store(0, Ordering::Relaxed);
    }

    /// Whether to record this event (sampling support).
    #[inline]
    fn should_sample(&self) -> bool {
        if self.config.sampling_rate <= 1 {
            return true;
        }
        let count = self.sample_counter.fetch_add(1, Ordering::Relaxed);
        count % self.config.sampling_rate == 0
    }
}

impl Default for ProfileCollector {
    fn default() -> Self {
        Self::new()
    }
}
