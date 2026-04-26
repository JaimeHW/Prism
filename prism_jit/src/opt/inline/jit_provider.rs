//! JIT Callee Provider for Function Inlining
//!
//! This module provides the main callee provider for JIT compilation that
//! integrates with the compilation tiers. Features:
//!
//! - Tier-aware graph management (interpreted, tier-1, tier-2)
//! - Hotness-based inlining priority  
//! - Version-based invalidation with cascade
//! - LRU eviction for memory management
//! - Thread-safe concurrent access
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────┐
//! │ JitCalleeProvider│
//! ├─────────────────┤
//! │ CalleeRegistry   │ ← Graph storage
//! │ HotnessTracker   │ ← Call frequency
//! │ InvalidationReg  │ ← Versioning
//! │ CompilationState │ ← Per-function state
//! │ LRU Cache        │ ← Memory management
//! └─────────────────┘
//! ```
//!
//! # Usage
//!
//! ```ignore
//! let provider = JitCalleeProvider::new();
//!
//! // Register compiled function
//! provider.register_compiled(func_id, graph, CompilationTier::Tier2);
//!
//! // Record calls for hotness
//! provider.record_call(func_id);
//!
//! // Get graph for inlining
//! if let Some(graph) = provider.get_graph(func_id) {
//!     // Use for inlining
//! }
//!
//! // Handle deoptimization
//! provider.handle_deoptimization(func_id, DeoptReason::TypeCheckFailed);
//! ```

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use dashmap::DashMap;

use super::callee::{CalleeGraph, CalleeProvider, CalleeRegistry, InlineHint};
use super::hotness::{HotnessConfig, HotnessLevel, HotnessTracker};
use super::invalidation::{InvalidationReason, InvalidationRegistry, InvalidationSummary};

// =============================================================================
// Compilation Tier
// =============================================================================

/// The compilation tier of a function.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum CompilationTier {
    /// Not yet JIT compiled - interpreter only.
    Interpreted,
    /// Baseline JIT - fast compilation, minimal optimization.
    Tier1,
    /// Optimizing JIT - thorough compilation, full optimization.
    Tier2,
}

impl CompilationTier {
    /// Get the inline hint based on compilation tier.
    pub fn inline_hint(self) -> InlineHint {
        match self {
            CompilationTier::Interpreted => InlineHint::Cold,
            CompilationTier::Tier1 => InlineHint::Default,
            CompilationTier::Tier2 => InlineHint::Hot,
        }
    }

    /// Check if this tier produces optimized code suitable for inlining.
    pub fn is_optimized(self) -> bool {
        self == CompilationTier::Tier2
    }

    /// Get priority multiplier for this tier.
    pub fn priority_multiplier(self) -> f64 {
        match self {
            CompilationTier::Interpreted => 0.0,
            CompilationTier::Tier1 => 0.5,
            CompilationTier::Tier2 => 1.0,
        }
    }
}

impl Default for CompilationTier {
    fn default() -> Self {
        CompilationTier::Interpreted
    }
}

// =============================================================================
// Compilation State
// =============================================================================

/// The current compilation state of a function.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CompilationState {
    /// Function has not been compiled yet.
    NotCompiled,
    /// Function is currently being compiled.
    Compiling { tier: CompilationTier },
    /// Function is compiled and ready.
    Compiled { tier: CompilationTier, version: u32 },
    /// Function was deoptimized.
    Deoptimized { reason: DeoptReason, version: u32 },
}

impl Default for CompilationState {
    fn default() -> Self {
        CompilationState::NotCompiled
    }
}

// =============================================================================
// Deoptimization Reason
// =============================================================================

/// Reason for function deoptimization.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DeoptReason {
    /// Type speculation failed.
    TypeCheckFailed,
    /// Bounds check failed.
    BoundsCheckFailed,
    /// Division by zero.
    DivisionByZero,
    /// Stack overflow.
    StackOverflow,
    /// Null pointer dereference.
    NullPointer,
    /// Invalidation cascade from callee.
    CalleeInvalidated,
    /// OSR entry point mismatch.
    OsrMismatch,
    /// Generic bailout.
    GenericBailout,
    /// Unknown reason.
    Unknown,
}

// =============================================================================
// Configuration
// =============================================================================

/// Configuration for the JIT callee provider.
#[derive(Debug, Clone)]
pub struct JitProviderConfig {
    /// Maximum number of graphs to cache.
    pub max_cached_graphs: usize,
    /// Evict graphs colder than this level.
    pub eviction_threshold: HotnessLevel,
    /// Prefer tier-2 graphs over tier-1 for inlining.
    pub prefer_optimized: bool,
    /// Maximum inlining depth for recursive calls.
    pub max_inline_depth: usize,
    /// Apply decay after this many calls.
    pub decay_interval: u64,
    /// Hotness configuration.
    pub hotness_config: HotnessConfig,
}

impl Default for JitProviderConfig {
    fn default() -> Self {
        Self {
            max_cached_graphs: 10_000,
            eviction_threshold: HotnessLevel::Cold,
            prefer_optimized: true,
            max_inline_depth: 10,
            decay_interval: 10_000,
            hotness_config: HotnessConfig::default(),
        }
    }
}

impl JitProviderConfig {
    /// Create aggressive configuration for maximum inlining.
    pub fn aggressive() -> Self {
        Self {
            max_cached_graphs: 20_000,
            eviction_threshold: HotnessLevel::Cold,
            prefer_optimized: true,
            max_inline_depth: 15,
            decay_interval: 20_000,
            hotness_config: HotnessConfig::aggressive(),
        }
    }

    /// Create conservative configuration for minimal memory.
    pub fn conservative() -> Self {
        Self {
            max_cached_graphs: 2_000,
            eviction_threshold: HotnessLevel::Warm,
            prefer_optimized: true,
            max_inline_depth: 5,
            decay_interval: 5_000,
            hotness_config: HotnessConfig::conservative(),
        }
    }

    /// Create tier-1 configuration (fast but limited).
    pub fn tier1() -> Self {
        Self {
            max_cached_graphs: 5_000,
            eviction_threshold: HotnessLevel::Cold,
            prefer_optimized: false,
            max_inline_depth: 3,
            decay_interval: 5_000,
            hotness_config: HotnessConfig::tier1(),
        }
    }

    /// Create tier-2 configuration (thorough).
    pub fn tier2() -> Self {
        Self::default()
    }
}

// =============================================================================
// Cached Graph Entry
// =============================================================================

/// A cached graph with metadata.
#[derive(Debug)]
struct CachedGraph {
    /// The actual graph.
    graph: Arc<CalleeGraph>,
    /// Compilation tier.
    tier: CompilationTier,
    /// Version when cached.
    version: u32,
    /// Last access time (monotonic counter).
    last_access: AtomicU64,
}

impl CachedGraph {
    fn new(graph: Arc<CalleeGraph>, tier: CompilationTier, version: u32) -> Self {
        Self {
            graph,
            tier,
            version,
            last_access: AtomicU64::new(0),
        }
    }

    fn touch(&self, timestamp: u64) {
        self.last_access.store(timestamp, Ordering::Relaxed);
    }

    fn get_last_access(&self) -> u64 {
        self.last_access.load(Ordering::Relaxed)
    }
}

impl Clone for CachedGraph {
    fn clone(&self) -> Self {
        Self {
            graph: self.graph.clone(),
            tier: self.tier,
            version: self.version,
            last_access: AtomicU64::new(self.last_access.load(Ordering::Relaxed)),
        }
    }
}

// =============================================================================
// JIT Callee Provider
// =============================================================================

/// Thread-safe callee provider with hotness tracking and tier management.
///
/// This is the main integration point between the JIT compiler and the
/// inlining pass. It tracks:
/// - Compiled function graphs at each tier
/// - Call frequency for inlining priority
/// - Version numbers for invalidation
/// - Memory usage for eviction
#[derive(Debug)]
pub struct JitCalleeProvider {
    /// Base registry for graph storage.
    registry: CalleeRegistry,
    /// Cached graphs with tier info.
    cached: DashMap<u64, CachedGraph>,
    /// Compilation state per function.
    states: DashMap<u64, CompilationState>,
    /// Hotness tracking.
    hotness: HotnessTracker,
    /// Invalidation registry.
    invalidation: InvalidationRegistry,
    /// Configuration.
    config: JitProviderConfig,
    /// Access counter for LRU.
    access_counter: AtomicU64,
    /// Total call counter for decay.
    call_counter: AtomicU64,
}

impl JitCalleeProvider {
    /// Create a new JIT callee provider with default configuration.
    pub fn new() -> Self {
        Self::with_config(JitProviderConfig::default())
    }

    /// Create with custom configuration.
    pub fn with_config(config: JitProviderConfig) -> Self {
        Self {
            registry: CalleeRegistry::new(),
            cached: DashMap::new(),
            states: DashMap::new(),
            hotness: HotnessTracker::with_config(config.hotness_config.clone()),
            invalidation: InvalidationRegistry::new(),
            config,
            access_counter: AtomicU64::new(0),
            call_counter: AtomicU64::new(0),
        }
    }

    // =========================================================================
    // Registration
    // =========================================================================

    /// Register a compiled function graph.
    pub fn register_compiled(
        &self,
        func_id: u64,
        graph: CalleeGraph,
        tier: CompilationTier,
    ) -> u32 {
        let version = self.invalidation.get_version(func_id);

        // Register in base registry
        let arc = Arc::new(graph);
        self.registry.register_arc(func_id, arc.clone());

        // Cache with metadata
        self.cached
            .insert(func_id, CachedGraph::new(arc, tier, version));

        // Update state
        self.states
            .insert(func_id, CompilationState::Compiled { tier, version });

        // Register for invalidation tracking
        self.invalidation.register(func_id);

        // Maybe evict cold graphs
        self.maybe_evict();

        version
    }

    /// Mark a function as being compiled.
    pub fn mark_compiling(&self, func_id: u64, tier: CompilationTier) {
        self.states
            .insert(func_id, CompilationState::Compiling { tier });
    }

    /// Upgrade a function to a higher tier.
    pub fn upgrade_tier(&self, func_id: u64, graph: CalleeGraph, new_tier: CompilationTier) -> u32 {
        // Invalidate old version
        self.invalidation
            .invalidate(func_id, InvalidationReason::TierUpgrade);

        // Register new version
        self.register_compiled(func_id, graph, new_tier)
    }

    // =========================================================================
    // Call Recording
    // =========================================================================

    /// Record a call to a function.
    #[inline]
    pub fn record_call(&self, func_id: u64) {
        self.hotness.record_call(func_id);

        // Periodically apply decay
        let count = self.call_counter.fetch_add(1, Ordering::Relaxed);
        if count > 0 && count % self.config.decay_interval == 0 {
            self.hotness.apply_decay();
        }
    }

    /// Record multiple calls at once.
    #[inline]
    pub fn record_calls(&self, func_id: u64, count: u64) {
        self.hotness.record_calls(func_id, count);
    }

    // =========================================================================
    // Graph Retrieval
    // =========================================================================

    /// Get a cached graph if valid.
    pub fn get_cached(&self, func_id: u64) -> Option<Arc<CalleeGraph>> {
        let entry = self.cached.get(&func_id)?;

        // Check staleness
        if self.invalidation.is_stale(func_id, entry.version) {
            return None;
        }

        // Update LRU timestamp
        let timestamp = self.access_counter.fetch_add(1, Ordering::Relaxed);
        entry.touch(timestamp);

        Some(entry.graph.clone())
    }

    /// Get compilation tier for a function.
    pub fn get_tier(&self, func_id: u64) -> Option<CompilationTier> {
        self.cached.get(&func_id).map(|e| e.tier)
    }

    /// Get compilation state for a function.
    pub fn get_state(&self, func_id: u64) -> CompilationState {
        self.states
            .get(&func_id)
            .map(|s| s.clone())
            .unwrap_or(CompilationState::NotCompiled)
    }

    /// Get current version for a function.
    pub fn get_version(&self, func_id: u64) -> u32 {
        self.invalidation.get_version(func_id)
    }

    // =========================================================================
    // Hotness Queries
    // =========================================================================

    /// Get the hotness level for a function.
    pub fn get_hotness(&self, func_id: u64) -> HotnessLevel {
        self.hotness.get_hotness(func_id)
    }

    /// Get inline priority (0.0 to 1.0).
    pub fn get_inline_priority(&self, func_id: u64) -> f64 {
        let hotness_priority = self.hotness.get_inline_priority(func_id);
        let tier_priority = self
            .get_tier(func_id)
            .map(|t| t.priority_multiplier())
            .unwrap_or(0.0);

        // Combine hotness and tier
        hotness_priority * (1.0 + tier_priority) / 2.0
    }

    /// Get all hot functions.
    pub fn get_hot_functions(&self) -> Vec<u64> {
        self.hotness.get_hot_functions()
    }

    // =========================================================================
    // Invalidation
    // =========================================================================

    /// Handle function deoptimization.
    pub fn handle_deoptimization(&self, func_id: u64, reason: DeoptReason) {
        let version = self.invalidation.get_version(func_id);

        // Update state
        self.states
            .insert(func_id, CompilationState::Deoptimized { reason, version });

        // Remove cached graph
        self.cached.remove(&func_id);
        self.registry.unregister(func_id);

        // Invalidate with cascade
        self.invalidation
            .invalidate(func_id, InvalidationReason::Deoptimization);
    }

    /// Record that a caller inlined a callee.
    pub fn record_inline(&self, caller: u64, callee: u64) {
        self.invalidation.record_inline(caller, callee);
    }

    /// Clear inlining dependencies for a function (before recompilation).
    pub fn clear_inline_dependencies(&self, func_id: u64) {
        self.invalidation.clear_dependencies(func_id);
    }

    /// Check if a cached version is stale.
    pub fn is_stale(&self, func_id: u64, cached_version: u32) -> bool {
        self.invalidation.is_stale(func_id, cached_version)
    }

    // =========================================================================
    // Eviction
    // =========================================================================

    fn maybe_evict(&self) {
        if self.cached.len() <= self.config.max_cached_graphs {
            return;
        }

        // Find candidates for eviction
        let mut candidates: Vec<(u64, u64, HotnessLevel)> = self
            .cached
            .iter()
            .map(|e| {
                let id = *e.key();
                let access = e.get_last_access();
                let hotness = self.hotness.get_hotness(id);
                (id, access, hotness)
            })
            .filter(|(_, _, h)| *h <= self.config.eviction_threshold)
            .collect();

        // Sort by last access (oldest first)
        candidates.sort_by_key(|(_, access, _)| *access);

        // Evict oldest entries
        let to_remove = self.cached.len() - self.config.max_cached_graphs + 100;
        for (id, _, _) in candidates.into_iter().take(to_remove) {
            self.evict(id);
        }
    }

    fn evict(&self, func_id: u64) {
        self.cached.remove(&func_id);
        self.registry.unregister(func_id);
        self.invalidation
            .invalidate(func_id, InvalidationReason::MemoryPressure);
    }

    /// Force eviction of cold graphs.
    pub fn evict_cold(&self) {
        let cold_funcs: Vec<u64> = self
            .cached
            .iter()
            .filter(|e| self.hotness.get_hotness(*e.key()) == HotnessLevel::Cold)
            .map(|e| *e.key())
            .collect();

        for id in cold_funcs {
            self.evict(id);
        }
    }

    // =========================================================================
    // Statistics
    // =========================================================================

    /// Get the number of cached graphs.
    pub fn cached_count(&self) -> usize {
        self.cached.len()
    }

    /// Get the number of tracked functions.
    pub fn function_count(&self) -> usize {
        self.states.len()
    }

    /// Get tier-2 function count.
    pub fn tier2_count(&self) -> usize {
        self.cached
            .iter()
            .filter(|e| e.tier == CompilationTier::Tier2)
            .count()
    }

    /// Get comprehensive summary.
    pub fn summary(&self) -> ProviderSummary {
        let invalidation = self.invalidation.summary();

        let mut tier_counts = [0usize; 3];
        for entry in self.cached.iter() {
            match entry.tier {
                CompilationTier::Interpreted => tier_counts[0] += 1,
                CompilationTier::Tier1 => tier_counts[1] += 1,
                CompilationTier::Tier2 => tier_counts[2] += 1,
            }
        }

        ProviderSummary {
            cached_graphs: self.cached.len(),
            tracked_functions: self.states.len(),
            hot_functions: self.hotness.get_hot_functions().len(),
            interpreted_count: tier_counts[0],
            tier1_count: tier_counts[1],
            tier2_count: tier_counts[2],
            total_calls: self.call_counter.load(Ordering::Relaxed),
            invalidation,
        }
    }

    /// Get configuration.
    pub fn config(&self) -> &JitProviderConfig {
        &self.config
    }

    /// Access hotness tracker directly.
    pub fn hotness(&self) -> &HotnessTracker {
        &self.hotness
    }

    /// Access invalidation registry directly.
    pub fn invalidation(&self) -> &InvalidationRegistry {
        &self.invalidation
    }

    // =========================================================================
    // Cleanup
    // =========================================================================

    /// Remove a function entirely.
    pub fn remove(&self, func_id: u64) {
        self.cached.remove(&func_id);
        self.states.remove(&func_id);
        self.registry.unregister(func_id);
        self.hotness.remove(func_id);
        self.invalidation.remove(func_id);
    }

    /// Clear all data.
    pub fn clear(&self) {
        self.cached.clear();
        self.states.clear();
        self.registry.clear();
        self.hotness.reset_all();
        self.invalidation.clear();
    }
}

impl Default for JitCalleeProvider {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// CalleeProvider Implementation
// =============================================================================

impl CalleeProvider for JitCalleeProvider {
    fn get_graph(&self, func_id: u64) -> Option<Arc<CalleeGraph>> {
        self.get_cached(func_id)
    }

    fn has_function(&self, func_id: u64) -> bool {
        self.cached.contains_key(&func_id)
    }

    fn param_count(&self, func_id: u64) -> Option<usize> {
        self.cached.get(&func_id).map(|e| e.graph.param_count)
    }

    fn inline_hint(&self, func_id: u64) -> InlineHint {
        // Combine tier hint with hotness
        let tier_hint = self
            .get_tier(func_id)
            .map(|t| t.inline_hint())
            .unwrap_or(InlineHint::Cold);

        let hotness = self.get_hotness(func_id);

        match (tier_hint, hotness) {
            (InlineHint::Cold, _) => InlineHint::Cold,
            (_, HotnessLevel::VeryHot) => InlineHint::Always,
            (_, HotnessLevel::Hot) => InlineHint::Hot,
            (_, HotnessLevel::Cold) => InlineHint::Cold,
            (hint, _) => hint,
        }
    }

    fn is_intrinsic(&self, func_id: u64) -> bool {
        self.cached
            .get(&func_id)
            .map(|e| e.graph.is_intrinsic)
            .unwrap_or(false)
    }
}

// =============================================================================
// Summary
// =============================================================================

/// Summary of JIT provider state.
#[derive(Debug, Clone, Default)]
pub struct ProviderSummary {
    /// Number of cached graphs.
    pub cached_graphs: usize,
    /// Number of tracked functions.
    pub tracked_functions: usize,
    /// Number of hot functions.
    pub hot_functions: usize,
    /// Functions at interpreted tier.
    pub interpreted_count: usize,
    /// Functions at tier-1.
    pub tier1_count: usize,
    /// Functions at tier-2.
    pub tier2_count: usize,
    /// Total recorded calls.
    pub total_calls: u64,
    /// Invalidation statistics.
    pub invalidation: InvalidationSummary,
}
