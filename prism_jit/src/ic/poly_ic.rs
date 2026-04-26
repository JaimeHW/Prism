//! Polymorphic Inline Cache
//!
//! Generic polymorphic IC that can handle up to 4 different type cases.
//! Used when a call site observes multiple but limited type diversity.
//!
//! # Design
//!
//! The poly IC stores entries inline in a fixed-size array (4 entries).
//! This is cache-friendly and avoids allocation on the hot path.
//! Linear scan is used for lookup - optimal for small N.

use super::POLY_IC_ENTRIES;
use prism_runtime::object::shape::ShapeId;

// =============================================================================
// Generic Poly IC Entry
// =============================================================================

/// A single entry in a polymorphic cache.
///
/// Generic over the handler type T (e.g., slot offset, function pointer).
#[derive(Debug, Clone, Copy)]
pub struct PolyIcEntry<T: Copy + Default> {
    /// Shape key for this entry.
    pub shape_id: ShapeId,
    /// Cached handler data.
    pub handler: T,
    /// Access count for LRU-style replacement.
    pub access_count: u32,
}

impl<T: Copy + Default> PolyIcEntry<T> {
    /// Create a new entry.
    #[inline]
    pub const fn new(shape_id: ShapeId, handler: T) -> Self {
        Self {
            shape_id,
            handler,
            access_count: 0,
        }
    }

    /// Create an empty entry.
    #[inline]
    pub fn empty() -> Self {
        Self {
            shape_id: ShapeId(0),
            handler: T::default(),
            access_count: 0,
        }
    }

    /// Check if this entry is empty (shape_id == 0).
    #[inline]
    pub const fn is_empty(&self) -> bool {
        self.shape_id.0 == 0
    }

    /// Check if this entry matches the given shape.
    #[inline]
    pub fn matches(&self, shape_id: ShapeId) -> bool {
        self.shape_id == shape_id
    }

    /// Increment access count.
    #[inline]
    pub fn touch(&mut self) {
        self.access_count = self.access_count.saturating_add(1);
    }
}

impl<T: Copy + Default> Default for PolyIcEntry<T> {
    fn default() -> Self {
        Self::empty()
    }
}

// =============================================================================
// Generic Poly IC
// =============================================================================

/// Generic polymorphic inline cache.
///
/// Stores up to 4 (key, handler) pairs for fast inline lookup.
#[derive(Debug, Clone)]
pub struct PolyIc<T: Copy + Default> {
    /// Fixed-size entry array.
    entries: [PolyIcEntry<T>; POLY_IC_ENTRIES],
    /// Number of valid entries.
    count: u8,
    /// Miss count since last growth.
    miss_count: u8,
}

impl<T: Copy + Default> PolyIc<T> {
    /// Create an empty polymorphic cache.
    #[inline]
    pub fn new() -> Self {
        Self {
            entries: [PolyIcEntry::empty(); POLY_IC_ENTRIES],
            count: 0,
            miss_count: 0,
        }
    }

    /// Lookup an entry by shape ID.
    ///
    /// Returns the handler if found, None otherwise.
    /// O(n) where n ≤ 4.
    #[inline]
    pub fn lookup(&self, shape_id: ShapeId) -> Option<T> {
        for i in 0..(self.count as usize) {
            if self.entries[i].matches(shape_id) {
                return Some(self.entries[i].handler);
            }
        }
        None
    }

    /// Lookup with access count update.
    #[inline]
    pub fn lookup_and_touch(&mut self, shape_id: ShapeId) -> Option<T> {
        for i in 0..(self.count as usize) {
            if self.entries[i].matches(shape_id) {
                self.entries[i].touch();
                return Some(self.entries[i].handler);
            }
        }
        None
    }

    /// Try to insert a new entry.
    ///
    /// Returns true if inserted, false if cache is full.
    pub fn try_insert(&mut self, shape_id: ShapeId, handler: T) -> bool {
        // Check if already present
        for i in 0..(self.count as usize) {
            if self.entries[i].matches(shape_id) {
                self.entries[i].handler = handler;
                return true;
            }
        }

        // Try to add new entry
        if (self.count as usize) < POLY_IC_ENTRIES {
            let idx = self.count as usize;
            self.entries[idx] = PolyIcEntry::new(shape_id, handler);
            self.count += 1;
            self.miss_count = 0;
            return true;
        }

        false
    }

    /// Replace the least recently used entry.
    pub fn replace_lru(&mut self, shape_id: ShapeId, handler: T) {
        let mut min_idx = 0;
        let mut min_count = u32::MAX;

        for i in 0..POLY_IC_ENTRIES {
            if self.entries[i].access_count < min_count {
                min_count = self.entries[i].access_count;
                min_idx = i;
            }
        }

        self.entries[min_idx] = PolyIcEntry::new(shape_id, handler);
    }

    /// Check if the cache is full.
    #[inline]
    pub const fn is_full(&self) -> bool {
        (self.count as usize) >= POLY_IC_ENTRIES
    }

    /// Get number of valid entries.
    #[inline]
    pub const fn len(&self) -> usize {
        self.count as usize
    }

    /// Check if empty.
    #[inline]
    pub const fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// Record a miss.
    #[inline]
    pub fn record_miss(&mut self) {
        self.miss_count = self.miss_count.saturating_add(1);
    }

    /// Get miss count.
    #[inline]
    pub const fn miss_count(&self) -> u8 {
        self.miss_count
    }

    /// Clear all entries.
    pub fn clear(&mut self) {
        for entry in &mut self.entries {
            *entry = PolyIcEntry::empty();
        }
        self.count = 0;
        self.miss_count = 0;
    }

    /// Iterate over valid entries.
    pub fn iter(&self) -> impl Iterator<Item = &PolyIcEntry<T>> {
        self.entries[..self.count as usize].iter()
    }
}

impl<T: Copy + Default> Default for PolyIc<T> {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests;
