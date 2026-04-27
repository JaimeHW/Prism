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
