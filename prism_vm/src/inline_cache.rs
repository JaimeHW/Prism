//! Inline caching for attribute access and method calls.
//!
//! Inline caches store type information at call sites to enable
//! fast-path lookups without full method resolution.

use std::ptr;

/// Type identifier for inline cache checks.
/// Uses raw pointer for fast equality comparison.
pub type TypeId = usize;

/// Monomorphic inline cache for single-type optimization.
///
/// Stores the last-seen type and its cached lookup result.
/// On hit, provides O(1) access; on miss, falls back to slow path.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct MonoIC {
    /// Cached type ID (0 = empty/invalid).
    pub cached_type: TypeId,
    /// Cached slot offset or method pointer.
    pub cached_slot: u32,
    /// Hit counter for profiling.
    pub hits: u32,
    /// Miss counter for upgrade decisions.
    pub misses: u32,
}

impl MonoIC {
    /// Create an empty (invalid) cache.
    #[inline]
    pub const fn empty() -> Self {
        Self {
            cached_type: 0,
            cached_slot: 0,
            hits: 0,
            misses: 0,
        }
    }

    /// Check if cache is valid for the given type.
    #[inline(always)]
    pub fn check(&self, type_id: TypeId) -> Option<u32> {
        if self.cached_type == type_id && self.cached_type != 0 {
            Some(self.cached_slot)
        } else {
            None
        }
    }

    /// Update cache with new type and slot.
    #[inline]
    pub fn update(&mut self, type_id: TypeId, slot: u32) {
        self.cached_type = type_id;
        self.cached_slot = slot;
    }

    /// Record a hit.
    #[inline(always)]
    pub fn record_hit(&mut self) {
        self.hits = self.hits.saturating_add(1);
    }

    /// Record a miss.
    #[inline(always)]
    pub fn record_miss(&mut self) {
        self.misses = self.misses.saturating_add(1);
    }

    /// Check if cache should be upgraded to polymorphic.
    #[inline]
    pub fn should_upgrade(&self) -> bool {
        // Upgrade if we have significant misses relative to hits
        self.misses > 10 && self.misses > self.hits / 4
    }

    /// Get hit rate as a percentage.
    #[inline]
    pub fn hit_rate(&self) -> f32 {
        let total = self.hits + self.misses;
        if total == 0 {
            0.0
        } else {
            (self.hits as f32) / (total as f32) * 100.0
        }
    }
}

impl Default for MonoIC {
    fn default() -> Self {
        Self::empty()
    }
}

/// Maximum entries in a polymorphic inline cache.
pub const POLY_IC_SIZE: usize = 4;

/// Polymorphic inline cache for multiple types.
///
/// Stores up to 4 type→slot mappings for call sites that see
/// multiple types. Beyond 4 types, falls back to megamorphic (no cache).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PolyIC {
    /// Type→slot entries.
    pub entries: [(TypeId, u32); POLY_IC_SIZE],
    /// Number of valid entries.
    pub count: u8,
    /// Total lookups for profiling.
    pub lookups: u32,
    /// Cache hits.
    pub hits: u32,
}

impl PolyIC {
    /// Create an empty polymorphic cache.
    #[inline]
    pub const fn empty() -> Self {
        Self {
            entries: [(0, 0); POLY_IC_SIZE],
            count: 0,
            lookups: 0,
            hits: 0,
        }
    }

    /// Look up a type in the cache.
    #[inline(always)]
    pub fn lookup(&self, type_id: TypeId) -> Option<u32> {
        // Unrolled loop for performance
        let count = self.count as usize;
        if count > 0 && self.entries[0].0 == type_id {
            return Some(self.entries[0].1);
        }
        if count > 1 && self.entries[1].0 == type_id {
            return Some(self.entries[1].1);
        }
        if count > 2 && self.entries[2].0 == type_id {
            return Some(self.entries[2].1);
        }
        if count > 3 && self.entries[3].0 == type_id {
            return Some(self.entries[3].1);
        }
        None
    }

    /// Add a new entry, returning false if cache is full.
    #[inline]
    pub fn add(&mut self, type_id: TypeId, slot: u32) -> bool {
        if self.count as usize >= POLY_IC_SIZE {
            return false;
        }
        self.entries[self.count as usize] = (type_id, slot);
        self.count += 1;
        true
    }

    /// Check if cache is full (megamorphic).
    #[inline]
    pub fn is_full(&self) -> bool {
        self.count as usize >= POLY_IC_SIZE
    }

    /// Record a lookup result.
    #[inline]
    pub fn record(&mut self, hit: bool) {
        self.lookups = self.lookups.saturating_add(1);
        if hit {
            self.hits = self.hits.saturating_add(1);
        }
    }

    /// Get hit rate.
    #[inline]
    pub fn hit_rate(&self) -> f32 {
        if self.lookups == 0 {
            0.0
        } else {
            (self.hits as f32) / (self.lookups as f32) * 100.0
        }
    }
}

impl Default for PolyIC {
    fn default() -> Self {
        Self::empty()
    }
}

/// Inline cache state (transitions mono → poly → mega).
#[derive(Debug, Clone, Copy)]
pub enum ICState {
    /// Uninitialized
    Empty,
    /// Single type cached
    Monomorphic(MonoIC),
    /// Multiple types cached
    Polymorphic(PolyIC),
    /// Too many types, no caching
    Megamorphic,
}

impl Default for ICState {
    fn default() -> Self {
        ICState::Empty
    }
}

/// Call site inline cache for function calls.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct CallIC {
    /// Cached function pointer (null = invalid).
    pub cached_func: *const (),
    /// Expected argument count.
    pub cached_argc: u8,
    /// Hit counter.
    pub hits: u32,
}

impl CallIC {
    /// Create an empty call cache.
    #[inline]
    pub const fn empty() -> Self {
        Self {
            cached_func: ptr::null(),
            cached_argc: 0,
            hits: 0,
        }
    }

    /// Check if cache matches the given function.
    #[inline(always)]
    pub fn check(&self, func_ptr: *const (), argc: u8) -> bool {
        !self.cached_func.is_null() && self.cached_func == func_ptr && self.cached_argc == argc
    }

    /// Update cache with new function.
    #[inline]
    pub fn update(&mut self, func_ptr: *const (), argc: u8) {
        self.cached_func = func_ptr;
        self.cached_argc = argc;
        self.hits = 0;
    }

    /// Record a hit.
    #[inline(always)]
    pub fn record_hit(&mut self) {
        self.hits = self.hits.saturating_add(1);
    }
}

impl Default for CallIC {
    fn default() -> Self {
        Self::empty()
    }
}

/// Inline cache storage for a code object.
///
/// Stores caches indexed by instruction offset.
#[derive(Debug, Default)]
pub struct InlineCacheStore {
    /// Attribute access caches.
    pub attr_caches: Vec<MonoIC>,
    /// Call site caches.
    pub call_caches: Vec<CallIC>,
}

impl InlineCacheStore {
    /// Create storage with capacity for the given code size.
    pub fn new(instruction_count: usize) -> Self {
        // Estimate: ~10% of instructions are cacheable
        let cache_count = instruction_count / 10 + 1;
        Self {
            attr_caches: Vec::with_capacity(cache_count),
            call_caches: Vec::with_capacity(cache_count),
        }
    }

    /// Get or create an attribute cache for the given instruction offset.
    pub fn get_attr_cache(&mut self, offset: usize) -> &mut MonoIC {
        while self.attr_caches.len() <= offset {
            self.attr_caches.push(MonoIC::empty());
        }
        &mut self.attr_caches[offset]
    }

    /// Get or create a call cache for the given instruction offset.
    pub fn get_call_cache(&mut self, offset: usize) -> &mut CallIC {
        while self.call_caches.len() <= offset {
            self.call_caches.push(CallIC::empty());
        }
        &mut self.call_caches[offset]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mono_ic_hit() {
        let mut ic = MonoIC::empty();
        ic.update(42, 10);

        assert_eq!(ic.check(42), Some(10));
        assert_eq!(ic.check(43), None);
    }

    #[test]
    fn test_mono_ic_counters() {
        let mut ic = MonoIC::empty();
        ic.update(1, 0);

        for _ in 0..100 {
            ic.record_hit();
        }
        for _ in 0..10 {
            ic.record_miss();
        }

        assert_eq!(ic.hits, 100);
        assert_eq!(ic.misses, 10);
        assert!(ic.hit_rate() > 90.0);
    }

    #[test]
    fn test_poly_ic() {
        let mut ic = PolyIC::empty();

        assert!(ic.add(1, 10));
        assert!(ic.add(2, 20));
        assert!(ic.add(3, 30));
        assert!(ic.add(4, 40));
        assert!(!ic.add(5, 50)); // Full

        assert_eq!(ic.lookup(1), Some(10));
        assert_eq!(ic.lookup(3), Some(30));
        assert_eq!(ic.lookup(5), None);
        assert!(ic.is_full());
    }

    #[test]
    fn test_call_ic() {
        let mut ic = CallIC::empty();
        let func: fn() = || {};
        let ptr = func as *const ();

        ic.update(ptr, 2);
        assert!(ic.check(ptr, 2));
        assert!(!ic.check(ptr, 3)); // Wrong argc
    }
}
