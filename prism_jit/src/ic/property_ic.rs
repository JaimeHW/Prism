//! Property Inline Cache
//!
//! Implements caching for GetAttr/SetAttr operations using Shape-based dispatch.
//!
//! # Cache Structure
//!
//! ```text
//! PropertyIc (64 bytes, 1 cache line)
//! ├── MonoData (16 bytes) - Monomorphic cache
//! │   ├── shape_id: u32
//! │   ├── slot_offset: u16
//! │   ├── flags: u16
//! │   └── padding: u64
//! └── PolyData (48 bytes) - Polymorphic cache
//!     └── entries[4]: PolyEntry (12 bytes each)
//! ```
//!
//! # Access Pattern
//!
//! 1. Check if monomorphic: `if obj.shape_id == cached.shape_id`
//! 2. Direct slot access: `obj.slots[cached.slot_offset]`
//! 3. On miss → polymorphic search or cache update

use super::{IcState, POLY_IC_ENTRIES};
use prism_runtime::object::shape::{PropertyFlags, ShapeId};

// =============================================================================
// Slot Info
// =============================================================================

/// Information about a property slot.
///
/// Contains everything needed for O(1) property access.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SlotInfo {
    /// Offset in the object's property storage array.
    pub offset: u16,
    /// Property flags (writable, enumerable, etc.).
    pub flags: PropertyFlags,
    /// Whether this is an inline slot or dictionary entry.
    pub is_inline: bool,
}

impl SlotInfo {
    /// Create slot info for an inline property.
    #[inline]
    pub const fn inline(offset: u16, flags: PropertyFlags) -> Self {
        Self {
            offset,
            flags,
            is_inline: true,
        }
    }

    /// Create slot info for a dictionary property.
    #[inline]
    pub const fn dictionary(offset: u16, flags: PropertyFlags) -> Self {
        Self {
            offset,
            flags,
            is_inline: false,
        }
    }

    /// Check if property is writable.
    #[inline]
    pub const fn is_writable(&self) -> bool {
        self.flags.contains(PropertyFlags::WRITABLE)
    }

    /// Check if property is a data property (not accessor).
    #[inline]
    pub const fn is_data(&self) -> bool {
        self.flags.contains(PropertyFlags::DATA)
    }
}

impl Default for SlotInfo {
    #[inline]
    fn default() -> Self {
        Self {
            offset: 0,
            flags: PropertyFlags::default(),
            is_inline: true,
        }
    }
}

// =============================================================================
// Monomorphic Cache Entry
// =============================================================================

/// Monomorphic property IC data (16 bytes, aligned).
///
/// Used when a single shape is consistently observed at an IC site.
#[repr(C, align(16))]
#[derive(Debug, Clone, Copy)]
pub struct MonoPropertyData {
    /// Expected object shape.
    pub shape_id: ShapeId,
    /// Property slot offset.
    pub slot_offset: u16,
    /// Property flags for quick checks.
    pub flags: u16,
    /// Padding for alignment.
    _pad: [u8; 6],
}

impl MonoPropertyData {
    /// Create a new monomorphic entry.
    #[inline]
    pub const fn new(shape_id: ShapeId, slot_offset: u16, flags: PropertyFlags) -> Self {
        Self {
            shape_id,
            slot_offset,
            flags: flags.bits() as u16,
            _pad: [0; 6],
        }
    }

    /// Create an uninitialized entry.
    #[inline]
    pub const fn uninitialized() -> Self {
        Self {
            shape_id: ShapeId(0),
            slot_offset: 0,
            flags: 0,
            _pad: [0; 6],
        }
    }

    /// Get property flags.
    #[inline]
    pub fn property_flags(&self) -> PropertyFlags {
        PropertyFlags::from_bits_truncate(self.flags as u8)
    }

    /// Check if this entry matches a shape.
    #[inline]
    pub fn matches(&self, shape_id: ShapeId) -> bool {
        self.shape_id == shape_id
    }
}

impl Default for MonoPropertyData {
    #[inline]
    fn default() -> Self {
        Self::uninitialized()
    }
}

// =============================================================================
// Polymorphic Cache Entry
// =============================================================================

/// A single entry in a polymorphic property IC (12 bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PolyPropertyEntry {
    /// Expected object shape.
    pub shape_id: ShapeId,
    /// Property slot offset.
    pub slot_offset: u16,
    /// Property flags.
    pub flags: u16,
    /// Access count for LRU (optional optimization).
    pub access_count: u32,
}

impl PolyPropertyEntry {
    /// Create a new entry.
    #[inline]
    pub const fn new(shape_id: ShapeId, slot_offset: u16, flags: PropertyFlags) -> Self {
        Self {
            shape_id,
            slot_offset,
            flags: flags.bits() as u16,
            access_count: 0,
        }
    }

    /// Create an empty entry.
    #[inline]
    pub const fn empty() -> Self {
        Self {
            shape_id: ShapeId(0),
            slot_offset: 0,
            flags: 0,
            access_count: 0,
        }
    }

    /// Check if this entry is empty.
    #[inline]
    pub const fn is_empty(&self) -> bool {
        self.shape_id.0 == 0
    }

    /// Check if this entry matches a shape.
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

impl Default for PolyPropertyEntry {
    #[inline]
    fn default() -> Self {
        Self::empty()
    }
}

// =============================================================================
// Polymorphic Cache Data
// =============================================================================

/// Polymorphic property IC data (48 bytes).
///
/// Used when 2-4 different shapes are observed at an IC site.
#[repr(C)]
#[derive(Debug, Clone)]
pub struct PolyPropertyData {
    /// Fixed-size array of entries.
    pub entries: [PolyPropertyEntry; POLY_IC_ENTRIES],
    /// Number of valid entries.
    pub count: u8,
    /// Miss count since last entry was added.
    pub miss_since_grow: u8,
    /// Reserved for future use.
    _reserved: [u8; 2],
}

impl PolyPropertyData {
    /// Create a new empty polymorphic cache.
    #[inline]
    pub const fn new() -> Self {
        Self {
            entries: [PolyPropertyEntry::empty(); POLY_IC_ENTRIES],
            count: 0,
            miss_since_grow: 0,
            _reserved: [0; 2],
        }
    }

    /// Search for a shape in the cache.
    ///
    /// Returns the slot offset and flags if found.
    #[inline]
    pub fn lookup(&self, shape_id: ShapeId) -> Option<(u16, PropertyFlags)> {
        // Linear search through entries (cache-friendly for small N)
        for i in 0..(self.count as usize).min(POLY_IC_ENTRIES) {
            if self.entries[i].matches(shape_id) {
                return Some((
                    self.entries[i].slot_offset,
                    PropertyFlags::from_bits_truncate(self.entries[i].flags as u8),
                ));
            }
        }
        None
    }

    /// Search and update access count (for LRU).
    #[inline]
    pub fn lookup_and_touch(&mut self, shape_id: ShapeId) -> Option<(u16, PropertyFlags)> {
        for i in 0..(self.count as usize).min(POLY_IC_ENTRIES) {
            if self.entries[i].matches(shape_id) {
                self.entries[i].touch();
                return Some((
                    self.entries[i].slot_offset,
                    PropertyFlags::from_bits_truncate(self.entries[i].flags as u8),
                ));
            }
        }
        None
    }

    /// Try to add a new entry.
    ///
    /// Returns true if added, false if cache is full.
    pub fn try_add(&mut self, shape_id: ShapeId, slot_offset: u16, flags: PropertyFlags) -> bool {
        if (self.count as usize) >= POLY_IC_ENTRIES {
            return false;
        }

        let idx = self.count as usize;
        self.entries[idx] = PolyPropertyEntry::new(shape_id, slot_offset, flags);
        self.count += 1;
        self.miss_since_grow = 0;
        true
    }

    /// Replace the least recently used entry.
    ///
    /// Used when cache is full and we want to add a new entry.
    pub fn replace_lru(&mut self, shape_id: ShapeId, slot_offset: u16, flags: PropertyFlags) {
        // Find entry with lowest access count
        let mut min_idx = 0;
        let mut min_count = u32::MAX;

        for i in 0..POLY_IC_ENTRIES {
            if self.entries[i].access_count < min_count {
                min_count = self.entries[i].access_count;
                min_idx = i;
            }
        }

        // Replace it
        self.entries[min_idx] = PolyPropertyEntry::new(shape_id, slot_offset, flags);
    }

    /// Check if cache is full.
    #[inline]
    pub const fn is_full(&self) -> bool {
        (self.count as usize) >= POLY_IC_ENTRIES
    }

    /// Get number of entries.
    #[inline]
    pub const fn len(&self) -> usize {
        self.count as usize
    }

    /// Check if empty.
    #[inline]
    pub const fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// Clear all entries.
    #[inline]
    pub fn clear(&mut self) {
        self.entries = [PolyPropertyEntry::empty(); POLY_IC_ENTRIES];
        self.count = 0;
        self.miss_since_grow = 0;
    }

    /// Record a miss.
    #[inline]
    pub fn record_miss(&mut self) {
        self.miss_since_grow = self.miss_since_grow.saturating_add(1);
    }
}

impl Default for PolyPropertyData {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// Property IC Data
// =============================================================================

/// Complete data for a property IC (union of mono and poly).
#[derive(Debug, Clone)]
pub struct PropertyIcData {
    /// Monomorphic cache data.
    pub mono: MonoPropertyData,
    /// Polymorphic cache data.
    pub poly: PolyPropertyData,
}

impl PropertyIcData {
    /// Create new IC data.
    #[inline]
    pub const fn new() -> Self {
        Self {
            mono: MonoPropertyData::uninitialized(),
            poly: PolyPropertyData::new(),
        }
    }

    /// Clear all cached data.
    #[inline]
    pub fn clear(&mut self) {
        self.mono = MonoPropertyData::uninitialized();
        self.poly.clear();
    }
}

impl Default for PropertyIcData {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// Property IC
// =============================================================================

/// Complete property inline cache.
///
/// Manages state transitions and provides lookup interface.
#[derive(Debug)]
pub struct PropertyIc {
    /// Current IC state.
    state: IcState,
    /// Cached data.
    data: PropertyIcData,
    /// Total hits.
    hits: u64,
    /// Total misses.
    misses: u64,
}

impl PropertyIc {
    /// Create a new uninitialized property IC.
    #[inline]
    pub const fn new() -> Self {
        Self {
            state: IcState::Uninitialized,
            data: PropertyIcData::new(),
            hits: 0,
            misses: 0,
        }
    }

    /// Get current state.
    #[inline]
    pub const fn state(&self) -> IcState {
        self.state
    }

    /// Get the cached data.
    #[inline]
    pub const fn data(&self) -> &PropertyIcData {
        &self.data
    }

    /// Get mutable cached data.
    #[inline]
    pub fn data_mut(&mut self) -> &mut PropertyIcData {
        &mut self.data
    }

    /// Lookup a property in the cache.
    ///
    /// Returns slot info if found, None on miss.
    #[inline]
    pub fn lookup(&mut self, shape_id: ShapeId) -> Option<SlotInfo> {
        match self.state {
            IcState::Uninitialized => {
                self.misses += 1;
                None
            }

            IcState::Monomorphic => {
                if self.data.mono.matches(shape_id) {
                    self.hits += 1;
                    Some(SlotInfo {
                        offset: self.data.mono.slot_offset,
                        flags: self.data.mono.property_flags(),
                        is_inline: true,
                    })
                } else {
                    self.misses += 1;
                    None
                }
            }

            IcState::Polymorphic => {
                if let Some((offset, flags)) = self.data.poly.lookup_and_touch(shape_id) {
                    self.hits += 1;
                    Some(SlotInfo {
                        offset,
                        flags,
                        is_inline: true,
                    })
                } else {
                    self.misses += 1;
                    self.data.poly.record_miss();
                    None
                }
            }

            IcState::Megamorphic => {
                self.misses += 1;
                None // Megamorphic uses global cache, not inline
            }
        }
    }

    /// Update the cache after a miss.
    ///
    /// Handles state transitions automatically.
    pub fn update(&mut self, shape_id: ShapeId, slot_offset: u16, flags: PropertyFlags) {
        match self.state {
            IcState::Uninitialized => {
                // Transition to monomorphic
                self.data.mono = MonoPropertyData::new(shape_id, slot_offset, flags);
                self.state = IcState::Monomorphic;
            }

            IcState::Monomorphic => {
                // Different shape - transition to polymorphic
                if self.data.mono.shape_id != shape_id {
                    // Copy mono entry to poly
                    self.data.poly.try_add(
                        self.data.mono.shape_id,
                        self.data.mono.slot_offset,
                        self.data.mono.property_flags(),
                    );
                    // Add new entry
                    self.data.poly.try_add(shape_id, slot_offset, flags);
                    self.state = IcState::Polymorphic;
                }
                // Same shape - just update (shouldn't happen often)
            }

            IcState::Polymorphic => {
                // Try to add new entry
                if !self.data.poly.try_add(shape_id, slot_offset, flags) {
                    // Cache full - transition to megamorphic
                    self.state = IcState::Megamorphic;
                }
            }

            IcState::Megamorphic => {
                // No inline caching - rely on global megamorphic cache
            }
        }
    }

    /// Force transition to a specific state (for testing/debugging).
    pub fn force_state(&mut self, state: IcState) {
        self.state = state;
    }

    /// Reset the IC to uninitialized state.
    pub fn reset(&mut self) {
        self.state = IcState::Uninitialized;
        self.data.clear();
        self.hits = 0;
        self.misses = 0;
    }

    /// Get hit count.
    #[inline]
    pub const fn hits(&self) -> u64 {
        self.hits
    }

    /// Get miss count.
    #[inline]
    pub const fn misses(&self) -> u64 {
        self.misses
    }

    /// Calculate hit rate.
    #[inline]
    pub fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 {
            0.0
        } else {
            self.hits as f64 / total as f64
        }
    }
}

impl Default for PropertyIc {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests;
