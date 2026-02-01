//! Inline Cache Manager with State Machine Transitions.
//!
//! This module implements a production-grade inline cache management system
//! with proper state transitions (Empty → Monomorphic → Polymorphic → Megamorphic),
//! profiler integration, and type feedback collection.
//!
//! # Architecture
//!
//! ```text
//!                    ┌─────────────┐
//!                    │    Empty    │
//!                    └──────┬──────┘
//!                           │ first hit
//!                    ╔══════▼══════╗
//!                    ║ Monomorphic ║
//!                    ╚══════╤══════╝
//!                           │ miss (different type)
//!                    ╔══════▼══════╗
//!                    ║ Polymorphic ║  (up to 4 types)
//!                    ╚══════╤══════╝
//!                           │ 5th type
//!                    ┌──────▼──────┐
//!                    │ Megamorphic │
//!                    └─────────────┘
//! ```
//!
//! # Performance Characteristics
//!
//! - **Monomorphic**: O(1) single type check + direct slot access
//! - **Polymorphic**: O(4) unrolled linear search (cache-friendly)
//! - **Megamorphic**: Full slow-path dispatch (no caching)
//!
//! # Thread Safety
//!
//! The IC Manager is designed for single-threaded use within a VM instance.
//! For multi-threaded scenarios, each thread should have its own IC store.

use super::inline_cache::{ICState, MonoIC, PolyIC, TypeId};
use crate::profiler::{CodeId, Profiler, TypeBitmap};
use rustc_hash::FxHashMap;

// =============================================================================
// IC Site Identifier
// =============================================================================

/// Unique identifier for an inline cache site within a code object.
///
/// Combines code ID (function identity) with bytecode offset to uniquely
/// identify each caching opportunity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ICSiteId {
    /// The code object containing this IC site.
    pub code_id: CodeId,
    /// Bytecode offset of the instruction.
    pub bc_offset: u32,
}

impl ICSiteId {
    /// Create a new IC site identifier.
    #[inline]
    pub const fn new(code_id: CodeId, bc_offset: u32) -> Self {
        Self { code_id, bc_offset }
    }
}

// =============================================================================
// IC Entry (Unified State + Metadata)
// =============================================================================

/// A single inline cache entry with full state and profiling metadata.
///
/// This is the core structure that tracks:
/// - Current IC state (mono/poly/mega)
/// - Access statistics
/// - Type feedback for speculative optimization
#[repr(C)]
#[derive(Debug, Clone)]
pub struct ICEntry {
    /// Current cache state.
    state: ICState,
    /// Total accesses (for hit rate calculation).
    total_accesses: u64,
    /// Cache hits.
    hits: u64,
    /// Last recorded type (for fast monomorphic check).
    last_type: TypeId,
    /// Attribute name index (for attribute ICs).
    name_idx: u32,
    /// Flags: Bit 0 = is_getattr, Bit 1 = is_setattr, Bit 2 = is_call.
    flags: u8,
}

impl ICEntry {
    /// IC is for GetAttr operation.
    const FLAG_GETATTR: u8 = 1 << 0;
    /// IC is for SetAttr operation.
    const FLAG_SETATTR: u8 = 1 << 1;
    /// IC is for Call operation.
    const FLAG_CALL: u8 = 1 << 2;
    /// IC deoptimization has been triggered.
    const FLAG_DEOPT: u8 = 1 << 3;

    /// Create a new empty IC entry for attribute access.
    #[inline]
    pub fn new_attr(name_idx: u32, is_get: bool) -> Self {
        Self {
            state: ICState::Empty,
            total_accesses: 0,
            hits: 0,
            last_type: 0,
            name_idx,
            flags: if is_get {
                Self::FLAG_GETATTR
            } else {
                Self::FLAG_SETATTR
            },
        }
    }

    /// Create a new empty IC entry for calls.
    #[inline]
    pub fn new_call() -> Self {
        Self {
            state: ICState::Empty,
            total_accesses: 0,
            hits: 0,
            last_type: 0,
            name_idx: 0,
            flags: Self::FLAG_CALL,
        }
    }

    /// Check if IC is for GetAttr.
    #[inline]
    pub fn is_getattr(&self) -> bool {
        self.flags & Self::FLAG_GETATTR != 0
    }

    /// Check if IC has been deoptimized.
    #[inline]
    pub fn is_deoptimized(&self) -> bool {
        self.flags & Self::FLAG_DEOPT != 0
    }

    /// Mark IC as deoptimized (will not be used in JIT anymore).
    #[inline]
    pub fn mark_deoptimized(&mut self) {
        self.flags |= Self::FLAG_DEOPT;
    }

    /// Get hit rate as a percentage (0.0-100.0).
    #[inline]
    pub fn hit_rate(&self) -> f32 {
        if self.total_accesses == 0 {
            0.0
        } else {
            (self.hits as f32 / self.total_accesses as f32) * 100.0
        }
    }

    /// Access the IC with a type, returning the cached slot if hit.
    ///
    /// This is the hot path - must be as fast as possible.
    #[inline(always)]
    pub fn access(&mut self, type_id: TypeId) -> ICAccessResult {
        self.total_accesses += 1;

        // Fast path: check last type (monomorphic hot path)
        if self.last_type == type_id && self.last_type != 0 {
            self.hits += 1;
            match &self.state {
                ICState::Monomorphic(mono) => {
                    return ICAccessResult::Hit(mono.cached_slot);
                }
                ICState::Polymorphic(poly) => {
                    // Must search poly entries
                    if let Some(slot) = poly.lookup(type_id) {
                        return ICAccessResult::Hit(slot);
                    }
                }
                _ => {}
            }
        }

        // Slow path: full lookup and potential state transition
        self.access_slow(type_id)
    }

    /// Slow path for IC access when fast path fails.
    #[cold]
    fn access_slow(&mut self, type_id: TypeId) -> ICAccessResult {
        match &mut self.state {
            ICState::Empty => ICAccessResult::Miss,

            ICState::Monomorphic(mono) => {
                if let Some(slot) = mono.check(type_id) {
                    mono.record_hit();
                    self.hits += 1;
                    self.last_type = type_id;
                    ICAccessResult::Hit(slot)
                } else {
                    mono.record_miss();
                    ICAccessResult::Miss
                }
            }

            ICState::Polymorphic(poly) => {
                if let Some(slot) = poly.lookup(type_id) {
                    poly.record(true);
                    self.hits += 1;
                    self.last_type = type_id;
                    ICAccessResult::Hit(slot)
                } else {
                    poly.record(false);
                    ICAccessResult::Miss
                }
            }

            ICState::Megamorphic => ICAccessResult::Megamorphic,
        }
    }

    /// Record a successful lookup result and potentially update state.
    ///
    /// Called after slow-path resolution to cache the result for future lookups.
    pub fn record(&mut self, type_id: TypeId, slot: u32) {
        self.last_type = type_id;

        match &mut self.state {
            ICState::Empty => {
                // Transition to Monomorphic
                let mut mono = MonoIC::empty();
                mono.update(type_id, slot);
                self.state = ICState::Monomorphic(mono);
            }

            ICState::Monomorphic(mono) => {
                if mono.cached_type != type_id {
                    // Transition to Polymorphic
                    let mut poly = PolyIC::empty();
                    poly.add(mono.cached_type, mono.cached_slot);
                    poly.add(type_id, slot);
                    self.state = ICState::Polymorphic(poly);
                }
            }

            ICState::Polymorphic(poly) => {
                if poly.lookup(type_id).is_none() {
                    if !poly.add(type_id, slot) {
                        // Transition to Megamorphic
                        self.state = ICState::Megamorphic;
                    }
                }
            }

            ICState::Megamorphic => {
                // Stay megamorphic - no recovery
            }
        }
    }

    /// Get the current IC state classification.
    #[inline]
    pub fn classification(&self) -> ICClassification {
        match &self.state {
            ICState::Empty => ICClassification::Uninitialized,
            ICState::Monomorphic(_) => ICClassification::Monomorphic,
            ICState::Polymorphic(p) if p.count <= 2 => ICClassification::Bimorphic,
            ICState::Polymorphic(_) => ICClassification::Polymorphic,
            ICState::Megamorphic => ICClassification::Megamorphic,
        }
    }

    /// Get number of types seen by this IC.
    #[inline]
    pub fn type_count(&self) -> u8 {
        match &self.state {
            ICState::Empty => 0,
            ICState::Monomorphic(_) => 1,
            ICState::Polymorphic(p) => p.count,
            ICState::Megamorphic => u8::MAX,
        }
    }
}

impl Default for ICEntry {
    fn default() -> Self {
        Self {
            state: ICState::Empty,
            total_accesses: 0,
            hits: 0,
            last_type: 0,
            name_idx: 0,
            flags: 0,
        }
    }
}

// =============================================================================
// IC Access Result
// =============================================================================

/// Result of an IC access operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ICAccessResult {
    /// Cache hit - slot is valid.
    Hit(u32),
    /// Cache miss - need slow path.
    Miss,
    /// Megamorphic - don't cache.
    Megamorphic,
}

impl ICAccessResult {
    /// Check if this is a hit.
    #[inline]
    pub fn is_hit(&self) -> bool {
        matches!(self, ICAccessResult::Hit(_))
    }

    /// Get the slot if hit.
    #[inline]
    pub fn slot(&self) -> Option<u32> {
        match self {
            ICAccessResult::Hit(slot) => Some(*slot),
            _ => None,
        }
    }
}

// =============================================================================
// IC Classification
// =============================================================================

/// Classification of an IC for optimization decisions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ICClassification {
    /// Never accessed.
    Uninitialized,
    /// Single type - best for optimization.
    Monomorphic,
    /// Two types - still good for optimization.
    Bimorphic,
    /// 3-4 types - marginal optimization value.
    Polymorphic,
    /// Too many types - use generic dispatch.
    Megamorphic,
}

// =============================================================================
// IC Manager
// =============================================================================

/// Central manager for all inline caches in a VM.
///
/// Provides:
/// - O(1) IC lookup by site ID
/// - Centralized statistics collection
/// - Integration with the profiler for type feedback
/// - JIT compilation support (queries for speculative optimization)
#[derive(Debug, Default)]
pub struct ICManager {
    /// All IC entries keyed by site ID.
    entries: FxHashMap<ICSiteId, ICEntry>,
    /// Total IC accesses (global counter).
    total_accesses: u64,
    /// Total IC hits (global counter).
    total_hits: u64,
}

impl ICManager {
    /// Create a new IC manager.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create with pre-allocated capacity.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            entries: FxHashMap::with_capacity_and_hasher(capacity, Default::default()),
            total_accesses: 0,
            total_hits: 0,
        }
    }

    // =========================================================================
    // IC Access (Hot Path)
    // =========================================================================

    /// Access an IC site, returning hit result.
    ///
    /// If IC doesn't exist, returns Miss (will be created on record).
    #[inline(always)]
    pub fn access(&mut self, site_id: ICSiteId, type_id: TypeId) -> ICAccessResult {
        self.total_accesses += 1;

        if let Some(entry) = self.entries.get_mut(&site_id) {
            let result = entry.access(type_id);
            if result.is_hit() {
                self.total_hits += 1;
            }
            result
        } else {
            ICAccessResult::Miss
        }
    }

    /// Record a lookup result at an IC site.
    ///
    /// Creates the IC if it doesn't exist.
    #[inline]
    pub fn record(
        &mut self,
        site_id: ICSiteId,
        type_id: TypeId,
        slot: u32,
        name_idx: u32,
        is_get: bool,
    ) {
        let entry = self
            .entries
            .entry(site_id)
            .or_insert_with(|| ICEntry::new_attr(name_idx, is_get));
        entry.record(type_id, slot);
    }

    /// Record a call site result.
    #[inline]
    pub fn record_call(&mut self, site_id: ICSiteId, func_type: TypeId, dispatched_to: u32) {
        let entry = self
            .entries
            .entry(site_id)
            .or_insert_with(ICEntry::new_call);
        entry.record(func_type, dispatched_to);
    }

    /// Record a binary operation type access (for type feedback).
    ///
    /// This is a simplified recording method for binary operations where
    /// we only care about operand types, not slot offsets.
    #[inline]
    pub fn record_binary_op(&mut self, site_id: ICSiteId, operand_pair: u32) {
        let entry = self.entries.entry(site_id).or_insert_with(|| {
            // Create a binary op IC entry (similar to call but for type tracking)
            ICEntry::new_call() // Reuse call entry type for now
        });
        // Record with slot=0 since we don't need slot for binary ops
        // TypeId is usize, so cast operand_pair
        entry.record(operand_pair as usize, 0);
    }

    // =========================================================================
    // IC Query (For JIT)
    // =========================================================================

    /// Get the classification of an IC site.
    pub fn get_classification(&self, site_id: ICSiteId) -> ICClassification {
        self.entries
            .get(&site_id)
            .map(|e| e.classification())
            .unwrap_or(ICClassification::Uninitialized)
    }

    /// Get the single type for a monomorphic IC (for JIT speculation).
    pub fn get_monomorphic_type(&self, site_id: ICSiteId) -> Option<(TypeId, u32)> {
        match self.entries.get(&site_id)?.state {
            ICState::Monomorphic(ref mono) if mono.cached_type != 0 => {
                Some((mono.cached_type, mono.cached_slot))
            }
            _ => None,
        }
    }

    /// Get all types for a polymorphic IC.
    pub fn get_polymorphic_types(&self, site_id: ICSiteId) -> Option<Vec<(TypeId, u32)>> {
        match self.entries.get(&site_id)?.state {
            ICState::Polymorphic(ref poly) => {
                let mut result = Vec::with_capacity(poly.count as usize);
                for i in 0..poly.count as usize {
                    let (type_id, slot) = poly.entries[i];
                    if type_id != 0 {
                        result.push((type_id, slot));
                    }
                }
                Some(result)
            }
            _ => None,
        }
    }

    /// Check if an IC site is stable enough for JIT specialization.
    ///
    /// Requires sufficient samples and a good hit rate.
    pub fn is_stable_for_jit(&self, site_id: ICSiteId) -> bool {
        if let Some(entry) = self.entries.get(&site_id) {
            // Need at least 100 accesses and >80% hit rate
            entry.total_accesses >= 100 && entry.hit_rate() >= 80.0
        } else {
            false
        }
    }

    /// Mark an IC site as deoptimized (JIT bailout occurred).
    pub fn mark_deoptimized(&mut self, site_id: ICSiteId) {
        if let Some(entry) = self.entries.get_mut(&site_id) {
            entry.mark_deoptimized();
        }
    }

    // =========================================================================
    // Statistics
    // =========================================================================

    /// Get global hit rate.
    pub fn global_hit_rate(&self) -> f32 {
        if self.total_accesses == 0 {
            0.0
        } else {
            (self.total_hits as f32 / self.total_accesses as f32) * 100.0
        }
    }

    /// Get number of IC sites.
    pub fn site_count(&self) -> usize {
        self.entries.len()
    }

    /// Get breakdown of IC classifications.
    pub fn classification_breakdown(&self) -> ICStats {
        let mut stats = ICStats::default();
        for entry in self.entries.values() {
            match entry.classification() {
                ICClassification::Uninitialized => stats.uninitialized += 1,
                ICClassification::Monomorphic => stats.monomorphic += 1,
                ICClassification::Bimorphic => stats.bimorphic += 1,
                ICClassification::Polymorphic => stats.polymorphic += 1,
                ICClassification::Megamorphic => stats.megamorphic += 1,
            }
        }
        stats
    }

    /// Clear all IC data (used for re-profiling).
    pub fn clear(&mut self) {
        self.entries.clear();
        self.total_accesses = 0;
        self.total_hits = 0;
    }
}

// =============================================================================
// IC Statistics
// =============================================================================

/// Statistics about IC classifications.
#[derive(Debug, Clone, Copy, Default)]
pub struct ICStats {
    /// Number of uninitialized ICs.
    pub uninitialized: usize,
    /// Number of monomorphic ICs.
    pub monomorphic: usize,
    /// Number of bimorphic ICs.
    pub bimorphic: usize,
    /// Number of polymorphic ICs.
    pub polymorphic: usize,
    /// Number of megamorphic ICs.
    pub megamorphic: usize,
}

impl ICStats {
    /// Total IC sites.
    pub fn total(&self) -> usize {
        self.uninitialized + self.monomorphic + self.bimorphic + self.polymorphic + self.megamorphic
    }

    /// Percentage that are monomorphic (ideal for JIT).
    pub fn monomorphic_percentage(&self) -> f32 {
        let total = self.total();
        if total == 0 {
            0.0
        } else {
            (self.monomorphic as f32 / total as f32) * 100.0
        }
    }
}

// =============================================================================
// Type Feedback Integration
// =============================================================================

/// Syncs IC data to the profiler's type feedback system.
///
/// This bridges the inline cache observations with the profiler's
/// type feedback that the JIT uses for speculative optimization.
pub fn sync_ic_to_profiler(ic_manager: &ICManager, profiler: &mut Profiler) {
    for (site_id, entry) in &ic_manager.entries {
        // Only sync if we have enough samples
        if entry.total_accesses < 10 {
            continue;
        }

        // Convert IC state to TypeBitmap
        let type_bits = match &entry.state {
            ICState::Empty => 0,
            ICState::Monomorphic(mono) => type_id_to_bitmap(mono.cached_type),
            ICState::Polymorphic(poly) => {
                let mut bits = 0u16;
                for i in 0..poly.count as usize {
                    bits |= type_id_to_bitmap(poly.entries[i].0);
                }
                bits
            }
            ICState::Megamorphic => {
                // Use all bits for megamorphic
                0xFFFF
            }
        };

        // Record in profiler
        if type_bits != 0 {
            profiler.record_type(site_id.code_id, site_id.bc_offset, type_bits);
        }
    }
}

/// Convert TypeId (object type identifier) to TypeBitmap bit.
fn type_id_to_bitmap(type_id: TypeId) -> u16 {
    // Map runtime TypeId to profiler's TypeBitmap
    // This is a simplified mapping - in production, use a lookup table
    match type_id {
        0 => TypeBitmap::NONE,
        1 => TypeBitmap::INT,
        2 => TypeBitmap::FLOAT,
        3 => TypeBitmap::BOOL,
        4 => TypeBitmap::STRING,
        5 => TypeBitmap::LIST,
        6 => TypeBitmap::TUPLE,
        7 => TypeBitmap::DICT,
        8 => TypeBitmap::SET,
        9 => TypeBitmap::FUNCTION,
        _ => TypeBitmap::OBJECT,
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ic_entry_state_transitions() {
        let mut entry = ICEntry::new_attr(0, true);

        // Start empty
        assert_eq!(entry.classification(), ICClassification::Uninitialized);

        // First access: becomes monomorphic
        entry.record(1, 10);
        assert_eq!(entry.classification(), ICClassification::Monomorphic);

        // Same type: still monomorphic
        entry.record(1, 10);
        assert_eq!(entry.classification(), ICClassification::Monomorphic);

        // Different type: becomes bimorphic
        entry.record(2, 20);
        assert_eq!(entry.classification(), ICClassification::Bimorphic);

        // Third type: becomes polymorphic
        entry.record(3, 30);
        assert_eq!(entry.classification(), ICClassification::Polymorphic);

        // Fourth type: still polymorphic
        entry.record(4, 40);
        assert_eq!(entry.classification(), ICClassification::Polymorphic);
        assert_eq!(entry.type_count(), 4);

        // Fifth type: becomes megamorphic
        entry.record(5, 50);
        assert_eq!(entry.classification(), ICClassification::Megamorphic);
    }

    #[test]
    fn test_ic_entry_hit_tracking() {
        let mut entry = ICEntry::new_attr(0, true);

        // Record initial type
        entry.record(42, 100);

        // Access with same type should hit
        for _ in 0..100 {
            let result = entry.access(42);
            assert_eq!(result, ICAccessResult::Hit(100));
        }

        // Hit rate should be 100%
        assert!(entry.hit_rate() > 99.0);
    }

    #[test]
    fn test_ic_manager_basic() {
        let mut manager = ICManager::new();
        let site = ICSiteId::new(CodeId(1), 10);

        // Initial access: miss
        let result = manager.access(site, 1);
        assert_eq!(result, ICAccessResult::Miss);

        // Record the result
        manager.record(site, 1, 100, 0, true);

        // Now access should hit
        let result = manager.access(site, 1);
        assert_eq!(result, ICAccessResult::Hit(100));

        // Different type: miss again
        let result = manager.access(site, 2);
        assert_eq!(result, ICAccessResult::Miss);
    }

    #[test]
    fn test_ic_manager_jit_queries() {
        let mut manager = ICManager::new();
        let site = ICSiteId::new(CodeId(1), 20);

        // Record monomorphic case
        manager.record(site, 5, 50, 0, true);

        // Check classification
        assert_eq!(
            manager.get_classification(site),
            ICClassification::Monomorphic
        );

        // Get monomorphic type for JIT
        let (type_id, slot) = manager.get_monomorphic_type(site).unwrap();
        assert_eq!(type_id, 5);
        assert_eq!(slot, 50);

        // Add more types to make polymorphic
        manager.record(site, 6, 60, 0, true);
        manager.record(site, 7, 70, 0, true);

        let types = manager.get_polymorphic_types(site).unwrap();
        assert_eq!(types.len(), 3);
    }

    #[test]
    fn test_ic_stats() {
        let mut manager = ICManager::new();

        // Create various IC states
        for i in 0..10 {
            let site = ICSiteId::new(CodeId(i), 0);
            manager.record(site, 1, 100, 0, true);
        }

        // Add polymorphic sites
        for i in 10..15 {
            let site = ICSiteId::new(CodeId(i), 0);
            manager.record(site, 1, 100, 0, true);
            manager.record(site, 2, 200, 0, true);
            manager.record(site, 3, 300, 0, true);
        }

        let stats = manager.classification_breakdown();
        assert_eq!(stats.monomorphic, 10);
        assert_eq!(stats.polymorphic, 5);
    }
}
