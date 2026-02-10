//! Graph Invalidation Protocol for JIT Inlining
//!
//! This module handles invalidation of callee graphs when functions are
//! recompiled or deoptimized. Features:
//!
//! - Version-based staleness detection
//! - Dependency tracking (who inlined whom)
//! - Cascade invalidation for transitive dependencies
//! - Thread-safe concurrent access
//!
//! # Invalidation Protocol
//!
//! When a function is recompiled:
//! 1. Increment the function's version
//! 2. Mark the function's cached graph as stale
//! 3. Find all callers that inlined this function
//! 4. Recursively mark those callers for recompilation
//!
//! # Performance Characteristics
//!
//! - `record_inline`: O(1), insert into dependency map
//! - `invalidate`: O(d), where d = number of dependents
//! - `is_stale`: O(1), version comparison

use std::collections::HashSet;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};

use dashmap::DashMap;
use rustc_hash::FxHashSet;

// =============================================================================
// Invalidation Reason
// =============================================================================

/// Reason for invalidating a function's graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum InvalidationReason {
    /// Function was recompiled at a higher tier.
    TierUpgrade,
    /// Function was deoptimized due to speculation failure.
    Deoptimization,
    /// Function source code changed (future: hot reload).
    SourceChanged,
    /// Callee was invalidated, cascading to caller.
    CalleeCascade,
    /// Manual invalidation request.
    Manual,
    /// Memory pressure eviction.
    MemoryPressure,
}

impl InvalidationReason {
    /// Check if this invalidation should cascade to callers.
    pub fn should_cascade(self) -> bool {
        match self {
            InvalidationReason::TierUpgrade => true,
            InvalidationReason::Deoptimization => true,
            InvalidationReason::SourceChanged => true,
            InvalidationReason::CalleeCascade => true,
            InvalidationReason::Manual => true,
            InvalidationReason::MemoryPressure => false, // Don't cascade evictions
        }
    }

    /// Check if this is a serious problem requiring attention.
    pub fn is_error(self) -> bool {
        self == InvalidationReason::Deoptimization
    }
}

// =============================================================================
// Invalidation Event
// =============================================================================

/// An invalidation event with metadata.
#[derive(Debug, Clone)]
pub struct InvalidationEvent {
    /// The function that was invalidated.
    pub func_id: u64,
    /// Reason for invalidation.
    pub reason: InvalidationReason,
    /// Version before invalidation.
    pub old_version: u32,
    /// Version after invalidation.
    pub new_version: u32,
    /// Timestamp (monotonic counter).
    pub timestamp: u64,
}

// =============================================================================
// Version Info
// =============================================================================

/// Per-function version information.
#[derive(Debug)]
struct VersionInfo {
    /// Current version number.
    version: AtomicU32,
    /// Last invalidation reason.
    last_reason: std::sync::RwLock<Option<InvalidationReason>>,
}

impl VersionInfo {
    fn new() -> Self {
        Self {
            version: AtomicU32::new(0),
            last_reason: std::sync::RwLock::new(None),
        }
    }

    fn get_version(&self) -> u32 {
        self.version.load(Ordering::Acquire)
    }

    fn increment(&self, reason: InvalidationReason) -> (u32, u32) {
        let old = self.version.fetch_add(1, Ordering::AcqRel);
        if let Ok(mut guard) = self.last_reason.write() {
            *guard = Some(reason);
        }
        (old, old + 1)
    }
}

impl Clone for VersionInfo {
    fn clone(&self) -> Self {
        let reason = self.last_reason.read().ok().and_then(|g| *g);
        Self {
            version: AtomicU32::new(self.version.load(Ordering::Relaxed)),
            last_reason: std::sync::RwLock::new(reason),
        }
    }
}

// =============================================================================
// Dependency Info
// =============================================================================

/// Tracks which functions inlined this function.
#[derive(Debug, Default, Clone)]
struct DependencyInfo {
    /// Set of callers that inlined this function.
    callers: FxHashSet<u64>,
    /// Set of callees that this function inlined.
    callees: FxHashSet<u64>,
}

// =============================================================================
// Invalidation Registry
// =============================================================================

/// Thread-safe registry for tracking function versions and dependencies.
///
/// The registry supports:
/// - Version tracking for staleness detection
/// - Dependency tracking for cascade invalidation
/// - Concurrent access from multiple compilation threads
#[derive(Debug)]
pub struct InvalidationRegistry {
    /// Per-function version info.
    versions: DashMap<u64, VersionInfo>,
    /// Per-function dependency info.
    dependencies: DashMap<u64, DependencyInfo>,
    /// Global event counter for timestamps.
    event_counter: AtomicU64,
    /// History of recent invalidation events.
    event_history: std::sync::RwLock<Vec<InvalidationEvent>>,
    /// Maximum history size.
    max_history: usize,
}

impl InvalidationRegistry {
    /// Create a new invalidation registry.
    pub fn new() -> Self {
        Self::with_history_size(1000)
    }

    /// Create with custom history size.
    pub fn with_history_size(max_history: usize) -> Self {
        Self {
            versions: DashMap::new(),
            dependencies: DashMap::new(),
            event_counter: AtomicU64::new(0),
            event_history: std::sync::RwLock::new(Vec::new()),
            max_history,
        }
    }

    // =========================================================================
    // Version Management
    // =========================================================================

    /// Get the current version of a function.
    pub fn get_version(&self, func_id: u64) -> u32 {
        self.versions
            .get(&func_id)
            .map(|v| v.get_version())
            .unwrap_or(0)
    }

    /// Check if a cached version is stale.
    #[inline]
    pub fn is_stale(&self, func_id: u64, cached_version: u32) -> bool {
        self.get_version(func_id) > cached_version
    }

    /// Register a function with initial version 0.
    pub fn register(&self, func_id: u64) {
        self.versions
            .entry(func_id)
            .or_insert_with(VersionInfo::new);
        self.dependencies.entry(func_id).or_default();
    }

    // =========================================================================
    // Dependency Tracking
    // =========================================================================

    /// Record that a caller inlined a callee.
    ///
    /// This establishes a dependency: if the callee is invalidated,
    /// the caller should also be invalidated.
    pub fn record_inline(&self, caller: u64, callee: u64) {
        // Ensure both functions are registered
        self.register(caller);
        self.register(callee);

        // Add caller to callee's dependents
        self.dependencies
            .entry(callee)
            .or_default()
            .callers
            .insert(caller);

        // Add callee to caller's callees
        self.dependencies
            .entry(caller)
            .or_default()
            .callees
            .insert(callee);
    }

    /// Remove an inlining dependency.
    pub fn remove_inline(&self, caller: u64, callee: u64) {
        if let Some(mut dep) = self.dependencies.get_mut(&callee) {
            dep.callers.remove(&caller);
        }
        if let Some(mut dep) = self.dependencies.get_mut(&caller) {
            dep.callees.remove(&callee);
        }
    }

    /// Get all callers that inlined a function.
    pub fn get_dependents(&self, func_id: u64) -> Vec<u64> {
        self.dependencies
            .get(&func_id)
            .map(|d| d.callers.iter().copied().collect())
            .unwrap_or_default()
    }

    /// Get all callees that a function inlined.
    pub fn get_callees(&self, func_id: u64) -> Vec<u64> {
        self.dependencies
            .get(&func_id)
            .map(|d| d.callees.iter().copied().collect())
            .unwrap_or_default()
    }

    /// Clear all dependencies for a function (before recompilation).
    pub fn clear_dependencies(&self, func_id: u64) {
        // Remove from all callees' caller lists
        if let Some(dep) = self.dependencies.get(&func_id) {
            for callee in dep.callees.iter() {
                if let Some(mut callee_dep) = self.dependencies.get_mut(callee) {
                    callee_dep.callers.remove(&func_id);
                }
            }
        }

        // Clear this function's callees
        if let Some(mut dep) = self.dependencies.get_mut(&func_id) {
            dep.callees.clear();
        }
    }

    // =========================================================================
    // Invalidation
    // =========================================================================

    /// Invalidate a function and return all transitively affected functions.
    ///
    /// This increments the function's version and, if cascading is enabled,
    /// recursively invalidates all callers.
    pub fn invalidate(&self, func_id: u64, reason: InvalidationReason) -> Vec<InvalidationEvent> {
        let mut events = Vec::new();
        let mut visited = HashSet::new();
        self.invalidate_recursive(func_id, reason, &mut events, &mut visited);
        events
    }

    fn invalidate_recursive(
        &self,
        func_id: u64,
        reason: InvalidationReason,
        events: &mut Vec<InvalidationEvent>,
        visited: &mut HashSet<u64>,
    ) {
        // Avoid cycles
        if !visited.insert(func_id) {
            return;
        }

        // Increment version
        let (old_version, new_version) = self
            .versions
            .entry(func_id)
            .or_insert_with(VersionInfo::new)
            .increment(reason);

        // Create event
        let timestamp = self.event_counter.fetch_add(1, Ordering::Relaxed);
        let event = InvalidationEvent {
            func_id,
            reason,
            old_version,
            new_version,
            timestamp,
        };

        // Record in history
        if let Ok(mut history) = self.event_history.write() {
            history.push(event.clone());
            if history.len() > self.max_history {
                history.remove(0);
            }
        }

        events.push(event);

        // Cascade to dependents if needed
        if reason.should_cascade() {
            let dependents = self.get_dependents(func_id);
            for dependent in dependents {
                self.invalidate_recursive(
                    dependent,
                    InvalidationReason::CalleeCascade,
                    events,
                    visited,
                );
            }
        }
    }

    /// Invalidate multiple functions at once.
    pub fn invalidate_batch(
        &self,
        func_ids: &[u64],
        reason: InvalidationReason,
    ) -> Vec<InvalidationEvent> {
        let mut all_events = Vec::new();
        let mut visited = HashSet::new();

        for &func_id in func_ids {
            self.invalidate_recursive(func_id, reason, &mut all_events, &mut visited);
        }

        all_events
    }

    // =========================================================================
    // Query
    // =========================================================================

    /// Get the last invalidation reason for a function.
    pub fn get_last_reason(&self, func_id: u64) -> Option<InvalidationReason> {
        self.versions
            .get(&func_id)
            .and_then(|v| v.last_reason.read().ok().and_then(|g| *g))
    }

    /// Get recent invalidation events.
    pub fn get_recent_events(&self, count: usize) -> Vec<InvalidationEvent> {
        if let Ok(history) = self.event_history.read() {
            let start = history.len().saturating_sub(count);
            history[start..].to_vec()
        } else {
            Vec::new()
        }
    }

    /// Get events for a specific function.
    pub fn get_function_events(&self, func_id: u64) -> Vec<InvalidationEvent> {
        if let Ok(history) = self.event_history.read() {
            history
                .iter()
                .filter(|e| e.func_id == func_id)
                .cloned()
                .collect()
        } else {
            Vec::new()
        }
    }

    /// Get the total number of invalidations.
    pub fn total_invalidations(&self) -> u64 {
        self.event_counter.load(Ordering::Relaxed)
    }

    /// Get the number of tracked functions.
    pub fn function_count(&self) -> usize {
        self.versions.len()
    }

    /// Get the number of inlining dependencies.
    pub fn dependency_count(&self) -> usize {
        self.dependencies.iter().map(|d| d.callers.len()).sum()
    }

    // =========================================================================
    // Cleanup
    // =========================================================================

    /// Remove a function from tracking.
    pub fn remove(&self, func_id: u64) {
        // Clear dependencies first
        self.clear_dependencies(func_id);

        // Remove from all callers' callee lists
        if let Some(dep) = self.dependencies.get(&func_id) {
            for caller in dep.callers.iter() {
                if let Some(mut caller_dep) = self.dependencies.get_mut(caller) {
                    caller_dep.callees.remove(&func_id);
                }
            }
        }

        // Remove entirely
        self.versions.remove(&func_id);
        self.dependencies.remove(&func_id);
    }

    /// Clear all data.
    pub fn clear(&self) {
        self.versions.clear();
        self.dependencies.clear();
        if let Ok(mut history) = self.event_history.write() {
            history.clear();
        }
    }

    /// Get a summary of the registry state.
    pub fn summary(&self) -> InvalidationSummary {
        let mut deopt_count = 0;
        let mut tier_upgrade_count = 0;

        if let Ok(history) = self.event_history.read() {
            for event in history.iter() {
                match event.reason {
                    InvalidationReason::Deoptimization => deopt_count += 1,
                    InvalidationReason::TierUpgrade => tier_upgrade_count += 1,
                    _ => {}
                }
            }
        }

        InvalidationSummary {
            function_count: self.function_count(),
            dependency_count: self.dependency_count(),
            total_invalidations: self.total_invalidations(),
            deoptimization_count: deopt_count,
            tier_upgrade_count: tier_upgrade_count,
        }
    }
}

impl Default for InvalidationRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl Clone for InvalidationRegistry {
    fn clone(&self) -> Self {
        let versions = DashMap::new();
        for entry in self.versions.iter() {
            versions.insert(*entry.key(), entry.value().clone());
        }

        let dependencies = DashMap::new();
        for entry in self.dependencies.iter() {
            dependencies.insert(*entry.key(), entry.value().clone());
        }

        let history = self
            .event_history
            .read()
            .map(|h| h.clone())
            .unwrap_or_default();

        Self {
            versions,
            dependencies,
            event_counter: AtomicU64::new(self.event_counter.load(Ordering::Relaxed)),
            event_history: std::sync::RwLock::new(history),
            max_history: self.max_history,
        }
    }
}

// =============================================================================
// Summary
// =============================================================================

/// Summary of invalidation registry state.
#[derive(Debug, Clone, Default)]
pub struct InvalidationSummary {
    /// Number of tracked functions.
    pub function_count: usize,
    /// Number of inlining dependencies.
    pub dependency_count: usize,
    /// Total invalidation events.
    pub total_invalidations: u64,
    /// Number of deoptimization events.
    pub deoptimization_count: usize,
    /// Number of tier upgrade events.
    pub tier_upgrade_count: usize,
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;

    // =========================================================================
    // InvalidationReason Tests
    // =========================================================================

    #[test]
    fn test_reason_should_cascade() {
        assert!(InvalidationReason::TierUpgrade.should_cascade());
        assert!(InvalidationReason::Deoptimization.should_cascade());
        assert!(InvalidationReason::SourceChanged.should_cascade());
        assert!(InvalidationReason::CalleeCascade.should_cascade());
        assert!(InvalidationReason::Manual.should_cascade());
        assert!(!InvalidationReason::MemoryPressure.should_cascade());
    }

    #[test]
    fn test_reason_is_error() {
        assert!(!InvalidationReason::TierUpgrade.is_error());
        assert!(InvalidationReason::Deoptimization.is_error());
        assert!(!InvalidationReason::Manual.is_error());
    }

    // =========================================================================
    // Registry Basic Tests
    // =========================================================================

    #[test]
    fn test_registry_new() {
        let reg = InvalidationRegistry::new();
        assert_eq!(reg.function_count(), 0);
        assert_eq!(reg.dependency_count(), 0);
    }

    #[test]
    fn test_register_function() {
        let reg = InvalidationRegistry::new();
        reg.register(1);
        assert_eq!(reg.function_count(), 1);
        assert_eq!(reg.get_version(1), 0);
    }

    #[test]
    fn test_unregistered_version() {
        let reg = InvalidationRegistry::new();
        assert_eq!(reg.get_version(999), 0);
    }

    // =========================================================================
    // Version Tests
    // =========================================================================

    #[test]
    fn test_invalidation_increments_version() {
        let reg = InvalidationRegistry::new();
        reg.register(1);

        assert_eq!(reg.get_version(1), 0);
        reg.invalidate(1, InvalidationReason::Manual);
        assert_eq!(reg.get_version(1), 1);
    }

    #[test]
    fn test_multiple_invalidations() {
        let reg = InvalidationRegistry::new();
        reg.register(1);

        for i in 1..=5 {
            reg.invalidate(1, InvalidationReason::Manual);
            assert_eq!(reg.get_version(1), i);
        }
    }

    #[test]
    fn test_is_stale() {
        let reg = InvalidationRegistry::new();
        reg.register(1);

        assert!(!reg.is_stale(1, 0));
        reg.invalidate(1, InvalidationReason::Manual);
        assert!(reg.is_stale(1, 0));
        assert!(!reg.is_stale(1, 1));
    }

    // =========================================================================
    // Dependency Tests
    // =========================================================================

    #[test]
    fn test_record_inline() {
        let reg = InvalidationRegistry::new();
        reg.record_inline(1, 2); // 1 inlined 2

        let dependents = reg.get_dependents(2);
        assert_eq!(dependents.len(), 1);
        assert!(dependents.contains(&1));

        let callees = reg.get_callees(1);
        assert_eq!(callees.len(), 1);
        assert!(callees.contains(&2));
    }

    #[test]
    fn test_multiple_dependents() {
        let reg = InvalidationRegistry::new();
        reg.record_inline(1, 10);
        reg.record_inline(2, 10);
        reg.record_inline(3, 10);

        let dependents = reg.get_dependents(10);
        assert_eq!(dependents.len(), 3);
        assert!(dependents.contains(&1));
        assert!(dependents.contains(&2));
        assert!(dependents.contains(&3));
    }

    #[test]
    fn test_multiple_callees() {
        let reg = InvalidationRegistry::new();
        reg.record_inline(1, 10);
        reg.record_inline(1, 11);
        reg.record_inline(1, 12);

        let callees = reg.get_callees(1);
        assert_eq!(callees.len(), 3);
    }

    #[test]
    fn test_remove_inline() {
        let reg = InvalidationRegistry::new();
        reg.record_inline(1, 2);
        reg.remove_inline(1, 2);

        assert!(reg.get_dependents(2).is_empty());
        assert!(reg.get_callees(1).is_empty());
    }

    #[test]
    fn test_clear_dependencies() {
        let reg = InvalidationRegistry::new();
        reg.record_inline(1, 10);
        reg.record_inline(1, 11);
        reg.record_inline(1, 12);

        reg.clear_dependencies(1);

        assert!(reg.get_callees(1).is_empty());
        assert!(reg.get_dependents(10).is_empty());
        assert!(reg.get_dependents(11).is_empty());
        assert!(reg.get_dependents(12).is_empty());
    }

    // =========================================================================
    // Cascade Tests
    // =========================================================================

    #[test]
    fn test_cascade_invalidation() {
        let reg = InvalidationRegistry::new();
        reg.record_inline(1, 10); // 1 inlined 10

        // Invalidate 10, should cascade to 1
        let events = reg.invalidate(10, InvalidationReason::TierUpgrade);

        assert_eq!(events.len(), 2);
        assert!(reg.is_stale(10, 0));
        assert!(reg.is_stale(1, 0));
    }

    #[test]
    fn test_multi_level_cascade() {
        let reg = InvalidationRegistry::new();
        // A inlined B, B inlined C
        reg.record_inline(1, 2);
        reg.record_inline(2, 3);

        // Invalidate C, should cascade to B, then to A
        let events = reg.invalidate(3, InvalidationReason::Deoptimization);

        assert_eq!(events.len(), 3);
        assert!(reg.is_stale(3, 0));
        assert!(reg.is_stale(2, 0));
        assert!(reg.is_stale(1, 0));
    }

    #[test]
    fn test_no_cascade_memory_pressure() {
        let reg = InvalidationRegistry::new();
        reg.record_inline(1, 10);

        let events = reg.invalidate(10, InvalidationReason::MemoryPressure);

        // Only 10 invalidated, not 1
        assert_eq!(events.len(), 1);
        assert!(reg.is_stale(10, 0));
        assert!(!reg.is_stale(1, 0));
    }

    #[test]
    fn test_cascade_diamond_pattern() {
        let reg = InvalidationRegistry::new();
        // Diamond: A and B both inline C
        reg.record_inline(1, 3);
        reg.record_inline(2, 3);

        let events = reg.invalidate(3, InvalidationReason::Manual);

        assert_eq!(events.len(), 3); // 3, 1, 2 (or 3, 2, 1)
    }

    #[test]
    fn test_cascade_cycle() {
        let reg = InvalidationRegistry::new();
        // Cycle: A -> B -> C -> A
        reg.record_inline(1, 2);
        reg.record_inline(2, 3);
        reg.record_inline(3, 1);

        // Should not infinite loop
        let events = reg.invalidate(1, InvalidationReason::Manual);

        assert_eq!(events.len(), 3);
    }

    // =========================================================================
    // Batch Tests
    // =========================================================================

    #[test]
    fn test_invalidate_batch() {
        let reg = InvalidationRegistry::new();
        reg.register(1);
        reg.register(2);
        reg.register(3);

        let events = reg.invalidate_batch(&[1, 2, 3], InvalidationReason::Manual);

        assert_eq!(events.len(), 3);
        assert!(reg.is_stale(1, 0));
        assert!(reg.is_stale(2, 0));
        assert!(reg.is_stale(3, 0));
    }

    #[test]
    fn test_invalidate_batch_dedupes() {
        let reg = InvalidationRegistry::new();
        reg.record_inline(1, 3);
        reg.record_inline(2, 3);

        // Invalidating 3 cascades to 1 and 2
        // Then invalidating 1 and 2 directly shouldn't double-count
        let events = reg.invalidate_batch(&[3, 1, 2], InvalidationReason::Manual);

        // Each function only once
        let func_ids: HashSet<_> = events.iter().map(|e| e.func_id).collect();
        assert_eq!(func_ids.len(), events.len());
    }

    // =========================================================================
    // Event History Tests
    // =========================================================================

    #[test]
    fn test_event_history() {
        let reg = InvalidationRegistry::new();
        reg.register(1);

        reg.invalidate(1, InvalidationReason::TierUpgrade);
        reg.invalidate(1, InvalidationReason::Deoptimization);

        let events = reg.get_recent_events(10);
        assert_eq!(events.len(), 2);
        assert_eq!(events[0].reason, InvalidationReason::TierUpgrade);
        assert_eq!(events[1].reason, InvalidationReason::Deoptimization);
    }

    #[test]
    fn test_function_events() {
        let reg = InvalidationRegistry::new();
        reg.register(1);
        reg.register(2);

        reg.invalidate(1, InvalidationReason::Manual);
        reg.invalidate(2, InvalidationReason::Manual);
        reg.invalidate(1, InvalidationReason::Manual);

        let events = reg.get_function_events(1);
        assert_eq!(events.len(), 2);
    }

    #[test]
    fn test_total_invalidations() {
        let reg = InvalidationRegistry::new();
        reg.register(1);

        assert_eq!(reg.total_invalidations(), 0);
        reg.invalidate(1, InvalidationReason::Manual);
        assert_eq!(reg.total_invalidations(), 1);
    }

    #[test]
    fn test_history_limit() {
        let reg = InvalidationRegistry::with_history_size(5);
        reg.register(1);

        for _ in 0..10 {
            reg.invalidate(1, InvalidationReason::Manual);
        }

        let events = reg.get_recent_events(100);
        assert!(events.len() <= 5);
    }

    // =========================================================================
    // Last Reason Tests
    // =========================================================================

    #[test]
    fn test_get_last_reason() {
        let reg = InvalidationRegistry::new();
        reg.register(1);

        assert!(reg.get_last_reason(1).is_none());

        reg.invalidate(1, InvalidationReason::TierUpgrade);
        assert_eq!(
            reg.get_last_reason(1),
            Some(InvalidationReason::TierUpgrade)
        );

        reg.invalidate(1, InvalidationReason::Deoptimization);
        assert_eq!(
            reg.get_last_reason(1),
            Some(InvalidationReason::Deoptimization)
        );
    }

    // =========================================================================
    // Cleanup Tests
    // =========================================================================

    #[test]
    fn test_remove() {
        let reg = InvalidationRegistry::new();
        reg.record_inline(1, 2);

        reg.remove(2);

        assert!(reg.get_dependents(2).is_empty());
        assert!(reg.get_callees(1).is_empty());
        assert_eq!(reg.function_count(), 1);
    }

    #[test]
    fn test_clear() {
        let reg = InvalidationRegistry::new();
        reg.record_inline(1, 2);
        reg.invalidate(1, InvalidationReason::Manual);

        reg.clear();

        assert_eq!(reg.function_count(), 0);
        assert_eq!(reg.dependency_count(), 0);
        assert!(reg.get_recent_events(100).is_empty());
    }

    // =========================================================================
    // Summary Tests
    // =========================================================================

    #[test]
    fn test_summary() {
        let reg = InvalidationRegistry::new();
        reg.record_inline(1, 10);
        reg.record_inline(2, 10);

        reg.invalidate(10, InvalidationReason::TierUpgrade);
        reg.invalidate(1, InvalidationReason::Deoptimization);

        let summary = reg.summary();
        assert_eq!(summary.function_count, 3);
        assert_eq!(summary.dependency_count, 2);
        assert!(summary.tier_upgrade_count >= 1);
        assert!(summary.deoptimization_count >= 1);
    }

    // =========================================================================
    // Clone Tests
    // =========================================================================

    #[test]
    fn test_clone() {
        let reg = InvalidationRegistry::new();
        reg.record_inline(1, 2);
        reg.invalidate(1, InvalidationReason::Manual);

        let cloned = reg.clone();

        assert_eq!(cloned.get_version(1), 1);
        assert!(!cloned.get_dependents(2).is_empty());
    }

    // =========================================================================
    // Thread Safety Tests
    // =========================================================================

    #[test]
    fn test_concurrent_invalidation() {
        let reg = std::sync::Arc::new(InvalidationRegistry::new());

        for i in 0..10 {
            reg.register(i);
        }

        let mut handles = vec![];

        for i in 0..8 {
            let r = reg.clone();
            handles.push(thread::spawn(move || {
                for j in 0..100 {
                    r.invalidate((j % 10) as u64, InvalidationReason::Manual);
                    if i % 2 == 0 {
                        r.record_inline(i as u64, (i + 1) as u64);
                    }
                }
            }));
        }

        for h in handles {
            h.join().unwrap();
        }

        // Should complete without panicking
        let _ = reg.summary();
    }

    #[test]
    fn test_concurrent_dependency_tracking() {
        let reg = std::sync::Arc::new(InvalidationRegistry::new());
        let mut handles = vec![];

        for i in 0..4 {
            let r = reg.clone();
            handles.push(thread::spawn(move || {
                for j in 0..50 {
                    r.record_inline(i as u64, (i * 100 + j) as u64);
                }
            }));
        }

        for h in handles {
            h.join().unwrap();
        }

        // Each thread added 50 dependencies
        assert!(reg.dependency_count() >= 200);
    }

    // =========================================================================
    // Event Timestamp Tests
    // =========================================================================

    #[test]
    fn test_event_timestamps_monotonic() {
        let reg = InvalidationRegistry::new();
        reg.register(1);
        reg.register(2);

        let events1 = reg.invalidate(1, InvalidationReason::Manual);
        let events2 = reg.invalidate(2, InvalidationReason::Manual);

        assert!(events2[0].timestamp > events1[0].timestamp);
    }

    // =========================================================================
    // Edge Case Tests
    // =========================================================================

    #[test]
    fn test_self_inline() {
        let reg = InvalidationRegistry::new();
        reg.record_inline(1, 1); // Self-reference

        // Should not infinite loop
        let events = reg.invalidate(1, InvalidationReason::Manual);
        assert_eq!(events.len(), 1);
    }

    #[test]
    fn test_invalidate_unregistered() {
        let reg = InvalidationRegistry::new();

        // Should handle gracefully
        let events = reg.invalidate(999, InvalidationReason::Manual);
        assert_eq!(events.len(), 1);
        assert_eq!(reg.get_version(999), 1);
    }

    #[test]
    fn test_dependency_count() {
        let reg = InvalidationRegistry::new();
        reg.record_inline(1, 10);
        reg.record_inline(2, 10);
        reg.record_inline(1, 20);

        // 3 total inlining relationships
        assert_eq!(reg.dependency_count(), 3);
    }
}
