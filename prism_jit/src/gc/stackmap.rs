//! Stack map generation and lookup for GC safepoints.
//!
//! Stack maps record which stack slots and registers contain GC-managed
//! pointers at each safepoint. This enables precise garbage collection
//! during JIT execution.
//!
//! # Design
//!
//! Stack maps use compact bitmap encoding for maximum performance:
//! - 64-bit bitmap for stack slots (handles 64 slots per frame)
//! - 16-bit bitmap for registers (all x64 GPRs)
//! - Sorted storage for O(log n) lookup
//!
//! # Example
//!
//! ```ignore
//! use prism_jit::gc::{StackMapBuilder, StackMapRegistry};
//!
//! // During compilation, build stack map
//! let mut builder = StackMapBuilder::new();
//! builder.add_safepoint(0x10, 0b0011, 0b00001111); // offset, regs, stack
//! builder.add_safepoint(0x20, 0b0001, 0b00000011);
//!
//! let stack_map = builder.finish(code_start, code_size);
//!
//! // Register with global registry
//! let mut registry = StackMapRegistry::new();
//! registry.insert(stack_map);
//!
//! // During GC, lookup by return address
//! if let Some(safepoint) = registry.lookup(return_address) {
//!     // Trace live references
//!     for slot in safepoint.live_stack_slots() {
//!         tracer.trace_value(frame[slot]);
//!     }
//! }
//! ```

use std::collections::BTreeMap;
use std::ptr::NonNull;
use std::sync::RwLock;

// =============================================================================
// SafePoint
// =============================================================================

/// A single safepoint where GC can occur during JIT execution.
///
/// Safepoints are placed at:
/// - Function calls (before and after)
/// - Loop back-edges
/// - Allocation sites
/// - Deoptimization points
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(C)]
pub struct SafePoint {
    /// Code offset from function start (4 bytes).
    pub code_offset: u32,
    /// Bitmap of live registers containing pointers (2 bytes).
    /// Bit 0 = RAX, Bit 1 = RCX, ..., Bit 15 = R15
    pub register_bitmap: u16,
    /// Padding for alignment (2 bytes).
    _padding: u16,
    /// Bitmap of live stack slots containing pointers (8 bytes).
    /// Bit 0 = [RBP-8], Bit 1 = [RBP-16], etc.
    pub stack_bitmap: u64,
}

impl SafePoint {
    /// Create a new safepoint.
    #[inline]
    pub const fn new(code_offset: u32, register_bitmap: u16, stack_bitmap: u64) -> Self {
        Self {
            code_offset,
            register_bitmap,
            _padding: 0,
            stack_bitmap,
        }
    }

    /// Check if a register contains a live pointer.
    #[inline]
    pub const fn is_register_live(&self, reg_index: u8) -> bool {
        (self.register_bitmap & (1 << reg_index)) != 0
    }

    /// Check if a stack slot contains a live pointer.
    #[inline]
    pub const fn is_stack_slot_live(&self, slot_index: u8) -> bool {
        (self.stack_bitmap & (1 << slot_index)) != 0
    }

    /// Count live registers.
    #[inline]
    pub const fn live_register_count(&self) -> u32 {
        self.register_bitmap.count_ones()
    }

    /// Count live stack slots.
    #[inline]
    pub const fn live_stack_slot_count(&self) -> u32 {
        self.stack_bitmap.count_ones()
    }

    /// Iterate over live register indices.
    #[inline]
    pub fn live_registers(&self) -> LiveBitmapIter<u16> {
        LiveBitmapIter::new(self.register_bitmap)
    }

    /// Iterate over live stack slot indices.
    #[inline]
    pub fn live_stack_slots(&self) -> LiveBitmapIter<u64> {
        LiveBitmapIter::new(self.stack_bitmap)
    }
}

// =============================================================================
// LiveBitmapIter
// =============================================================================

/// Iterator over set bits in a bitmap.
#[derive(Debug, Clone)]
pub struct LiveBitmapIter<T> {
    bitmap: T,
}

impl<T> LiveBitmapIter<T> {
    #[inline]
    fn new(bitmap: T) -> Self {
        Self { bitmap }
    }
}

impl Iterator for LiveBitmapIter<u16> {
    type Item = u8;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.bitmap == 0 {
            return None;
        }
        let trailing = self.bitmap.trailing_zeros() as u8;
        self.bitmap &= self.bitmap - 1; // Clear lowest set bit
        Some(trailing)
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let count = self.bitmap.count_ones() as usize;
        (count, Some(count))
    }
}

impl ExactSizeIterator for LiveBitmapIter<u16> {}

impl Iterator for LiveBitmapIter<u64> {
    type Item = u8;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.bitmap == 0 {
            return None;
        }
        let trailing = self.bitmap.trailing_zeros() as u8;
        self.bitmap &= self.bitmap - 1; // Clear lowest set bit
        Some(trailing)
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let count = self.bitmap.count_ones() as usize;
        (count, Some(count))
    }
}

impl ExactSizeIterator for LiveBitmapIter<u64> {}

// =============================================================================
// StackMap
// =============================================================================

/// Stack map for an entire compiled function.
///
/// Contains sorted safepoints for O(log n) lookup by code offset.
#[derive(Debug, Clone)]
pub struct StackMap {
    /// Start address of the compiled code.
    pub code_start: usize,
    /// Size of the compiled code in bytes.
    pub code_size: u32,
    /// Frame size in bytes (for stack walking).
    pub frame_size: u32,
    /// Sorted safepoints (by code_offset for binary search).
    safepoints: Box<[SafePoint]>,
}

impl StackMap {
    /// Create a new stack map with the given safepoints.
    ///
    /// # Panics
    ///
    /// Panics if safepoints are not sorted by code_offset.
    pub fn new(
        code_start: usize,
        code_size: u32,
        frame_size: u32,
        mut safepoints: Vec<SafePoint>,
    ) -> Self {
        // Sort by code offset for binary search
        safepoints.sort_by_key(|sp| sp.code_offset);

        // Verify sorted (debug only)
        debug_assert!(
            safepoints
                .windows(2)
                .all(|w| w[0].code_offset <= w[1].code_offset)
        );

        Self {
            code_start,
            code_size,
            frame_size,
            safepoints: safepoints.into_boxed_slice(),
        }
    }

    /// Check if an address falls within this function's code.
    #[inline]
    pub fn contains_address(&self, addr: usize) -> bool {
        addr >= self.code_start && addr < self.code_start + self.code_size as usize
    }

    /// Lookup safepoint by code offset within function.
    ///
    /// Returns the safepoint at or immediately before the given offset.
    #[inline]
    pub fn lookup_offset(&self, offset: u32) -> Option<&SafePoint> {
        if self.safepoints.is_empty() {
            return None;
        }

        // Binary search for exact match or predecessor
        match self
            .safepoints
            .binary_search_by_key(&offset, |sp| sp.code_offset)
        {
            Ok(idx) => Some(&self.safepoints[idx]),
            Err(idx) => {
                // idx is insertion point; predecessor is at idx - 1
                if idx > 0 {
                    Some(&self.safepoints[idx - 1])
                } else {
                    None
                }
            }
        }
    }

    /// Lookup safepoint by absolute address.
    #[inline]
    pub fn lookup_address(&self, addr: usize) -> Option<&SafePoint> {
        if !self.contains_address(addr) {
            return None;
        }
        let offset = (addr - self.code_start) as u32;
        self.lookup_offset(offset)
    }

    /// Get the number of safepoints.
    #[inline]
    pub fn safepoint_count(&self) -> usize {
        self.safepoints.len()
    }

    /// Iterate over all safepoints.
    #[inline]
    pub fn safepoints(&self) -> &[SafePoint] {
        &self.safepoints
    }

    /// Get the code end address.
    #[inline]
    pub fn code_end(&self) -> usize {
        self.code_start + self.code_size as usize
    }
}

// =============================================================================
// StackMapBuilder
// =============================================================================

/// Builder for constructing stack maps during compilation.
#[derive(Debug, Default)]
pub struct StackMapBuilder {
    safepoints: Vec<SafePoint>,
}

impl StackMapBuilder {
    /// Create a new stack map builder.
    #[inline]
    pub fn new() -> Self {
        Self {
            safepoints: Vec::with_capacity(16),
        }
    }

    /// Add a safepoint at the given code offset.
    #[inline]
    pub fn add_safepoint(&mut self, code_offset: u32, register_bitmap: u16, stack_bitmap: u64) {
        self.safepoints
            .push(SafePoint::new(code_offset, register_bitmap, stack_bitmap));
    }

    /// Add a safepoint from a SafePoint struct.
    #[inline]
    pub fn add(&mut self, safepoint: SafePoint) {
        self.safepoints.push(safepoint);
    }

    /// Build the final stack map.
    #[inline]
    pub fn finish(self, code_start: usize, code_size: u32, frame_size: u32) -> StackMap {
        StackMap::new(code_start, code_size, frame_size, self.safepoints)
    }

    /// Get the current number of safepoints.
    #[inline]
    pub fn len(&self) -> usize {
        self.safepoints.len()
    }

    /// Check if empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.safepoints.is_empty()
    }
}

// =============================================================================
// StackMapRegistry
// =============================================================================

/// Global registry of all JIT stack maps.
///
/// Uses a sorted BTreeMap for O(log n) lookup by code address.
/// Thread-safe with RwLock for concurrent access.
#[derive(Debug)]
pub struct StackMapRegistry {
    /// Maps code start address → stack map.
    /// Using BTreeMap for ordered iteration and range queries.
    maps: RwLock<BTreeMap<usize, StackMap>>,
}

impl StackMapRegistry {
    /// Create a new empty registry.
    pub fn new() -> Self {
        Self {
            maps: RwLock::new(BTreeMap::new()),
        }
    }

    /// Insert a stack map into the registry.
    pub fn insert(&self, map: StackMap) {
        let mut maps = self.maps.write().unwrap();
        maps.insert(map.code_start, map);
    }

    /// Remove a stack map from the registry.
    pub fn remove(&self, code_start: usize) -> Option<StackMap> {
        let mut maps = self.maps.write().unwrap();
        maps.remove(&code_start)
    }

    /// Lookup stack map by return address.
    ///
    /// Uses binary search to find the containing code range.
    pub fn lookup(&self, addr: usize) -> Option<StackMapRef> {
        let maps = self.maps.read().unwrap();

        // Find the entry with the largest key <= addr
        if let Some((&code_start, map)) = maps.range(..=addr).next_back() {
            if map.contains_address(addr) {
                // Calculate offset and lookup safepoint
                let offset = (addr - code_start) as u32;
                if let Some(safepoint) = map.lookup_offset(offset) {
                    return Some(StackMapRef {
                        frame_size: map.frame_size,
                        safepoint: *safepoint,
                    });
                }
            }
        }
        None
    }

    /// Get the number of registered stack maps.
    pub fn len(&self) -> usize {
        self.maps.read().unwrap().len()
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.maps.read().unwrap().is_empty()
    }

    /// Clear all stack maps.
    pub fn clear(&self) {
        self.maps.write().unwrap().clear();
    }
}

impl Default for StackMapRegistry {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// StackMapRef
// =============================================================================

/// Reference to a safepoint with frame info (returned from lookup).
#[derive(Debug, Clone, Copy)]
pub struct StackMapRef {
    /// Frame size for stack walking.
    pub frame_size: u32,
    /// The safepoint data.
    pub safepoint: SafePoint,
}

// =============================================================================
// CompactStackMap (for hot path)
// =============================================================================

/// Compact, cache-line aligned stack map array for hot-path lookup.
///
/// This is an alternative to StackMapRegistry optimized for the common case
/// of looking up safepoints during GC. Uses a single sorted array instead
/// of a BTreeMap for better cache locality.
#[derive(Debug)]
pub struct CompactStackMapArray {
    /// Sorted array of (code_start, code_end, safepoints_ptr, safepoint_count, frame_size).
    entries: Vec<CompactEntry>,
}

#[derive(Debug, Clone, Copy)]
#[repr(C)]
struct CompactEntry {
    code_start: usize,
    code_end: usize,
    safepoints: NonNull<SafePoint>,
    safepoint_count: u32,
    frame_size: u32,
}

// Safety: CompactEntry is Send/Sync because it only contains raw pointers
// to data that we own and manage ourselves.
unsafe impl Send for CompactEntry {}
unsafe impl Sync for CompactEntry {}

impl CompactStackMapArray {
    /// Create from a collection of stack maps.
    pub fn from_maps(maps: Vec<StackMap>) -> Self {
        let mut entries: Vec<CompactEntry> = maps
            .into_iter()
            .filter(|m| !m.safepoints.is_empty())
            .map(|m| {
                let safepoints_vec = m.safepoints.into_vec();
                let safepoint_count = safepoints_vec.len() as u32;
                let safepoints = Box::into_raw(safepoints_vec.into_boxed_slice());
                CompactEntry {
                    code_start: m.code_start,
                    code_end: m.code_start + m.code_size as usize,
                    safepoints: NonNull::new(safepoints as *mut SafePoint).unwrap(),
                    safepoint_count,
                    frame_size: m.frame_size,
                }
            })
            .collect();

        entries.sort_by_key(|e| e.code_start);
        Self { entries }
    }

    /// Lookup safepoint by address (hot path).
    #[inline]
    pub fn lookup(&self, addr: usize) -> Option<StackMapRef> {
        // Binary search for the entry containing addr
        let idx = self
            .entries
            .binary_search_by(|e| {
                if addr < e.code_start {
                    std::cmp::Ordering::Greater
                } else if addr >= e.code_end {
                    std::cmp::Ordering::Less
                } else {
                    std::cmp::Ordering::Equal
                }
            })
            .ok()?;

        let entry = &self.entries[idx];
        let offset = (addr - entry.code_start) as u32;

        // Binary search safepoints
        let safepoints = unsafe {
            std::slice::from_raw_parts(entry.safepoints.as_ptr(), entry.safepoint_count as usize)
        };

        let sp_idx = match safepoints.binary_search_by_key(&offset, |sp| sp.code_offset) {
            Ok(i) => i,
            Err(i) if i > 0 => i - 1,
            Err(_) => return None,
        };

        Some(StackMapRef {
            frame_size: entry.frame_size,
            safepoint: safepoints[sp_idx],
        })
    }

    /// Get the number of entries.
    #[inline]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Check if empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

impl Drop for CompactStackMapArray {
    fn drop(&mut self) {
        for entry in &self.entries {
            unsafe {
                let slice = std::slice::from_raw_parts_mut(
                    entry.safepoints.as_ptr(),
                    entry.safepoint_count as usize,
                );
                drop(Box::from_raw(slice as *mut [SafePoint]));
            }
        }
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_safepoint_creation() {
        let sp = SafePoint::new(0x10, 0b0011, 0b00001111);
        assert_eq!(sp.code_offset, 0x10);
        assert_eq!(sp.register_bitmap, 0b0011);
        assert_eq!(sp.stack_bitmap, 0b00001111);
    }

    #[test]
    fn test_safepoint_live_checks() {
        let sp = SafePoint::new(0, 0b0101, 0b10010001);

        // Register checks (0b0101 = bits 0 and 2)
        assert!(sp.is_register_live(0));
        assert!(!sp.is_register_live(1));
        assert!(sp.is_register_live(2));
        assert!(!sp.is_register_live(3));

        // Stack checks (0b10010001 = bits 0, 4, 7)
        assert!(sp.is_stack_slot_live(0));
        assert!(!sp.is_stack_slot_live(1));
        assert!(sp.is_stack_slot_live(4));
        assert!(sp.is_stack_slot_live(7));
    }

    #[test]
    fn test_live_bitmap_iter() {
        let sp = SafePoint::new(0, 0b1010, 0b10100101);

        let regs: Vec<u8> = sp.live_registers().collect();
        assert_eq!(regs, vec![1, 3]);

        let slots: Vec<u8> = sp.live_stack_slots().collect();
        assert_eq!(slots, vec![0, 2, 5, 7]);
    }

    #[test]
    fn test_stackmap_lookup() {
        let map = StackMap::new(
            0x1000,
            0x100,
            64,
            vec![
                SafePoint::new(0x10, 0b0001, 0b0001),
                SafePoint::new(0x30, 0b0010, 0b0010),
                SafePoint::new(0x50, 0b0100, 0b0100),
            ],
        );

        // Exact match
        let sp = map.lookup_offset(0x30).unwrap();
        assert_eq!(sp.register_bitmap, 0b0010);

        // Before first safepoint
        assert!(map.lookup_offset(0x05).is_none());

        // Between safepoints (returns predecessor)
        let sp = map.lookup_offset(0x40).unwrap();
        assert_eq!(sp.code_offset, 0x30);

        // Address lookup
        let sp = map.lookup_address(0x1050).unwrap();
        assert_eq!(sp.code_offset, 0x50);

        // Out of range
        assert!(map.lookup_address(0x2000).is_none());
    }

    #[test]
    fn test_stackmap_builder() {
        let mut builder = StackMapBuilder::new();
        builder.add_safepoint(0x20, 0b0001, 0b0001);
        builder.add_safepoint(0x10, 0b0010, 0b0010); // Unsorted input
        builder.add_safepoint(0x30, 0b0100, 0b0100);

        let map = builder.finish(0x1000, 0x100, 48);

        // Should be sorted
        assert_eq!(map.safepoints()[0].code_offset, 0x10);
        assert_eq!(map.safepoints()[1].code_offset, 0x20);
        assert_eq!(map.safepoints()[2].code_offset, 0x30);
    }

    #[test]
    fn test_registry_lookup() {
        let registry = StackMapRegistry::new();

        // Insert multiple maps
        registry.insert(StackMap::new(
            0x1000,
            0x100,
            64,
            vec![SafePoint::new(0x10, 0b0001, 0b0001)],
        ));
        registry.insert(StackMap::new(
            0x2000,
            0x200,
            128,
            vec![SafePoint::new(0x20, 0b0010, 0b0010)],
        ));

        // Lookup in first map
        let result = registry.lookup(0x1010).unwrap();
        assert_eq!(result.safepoint.register_bitmap, 0b0001);
        assert_eq!(result.frame_size, 64);

        // Lookup in second map
        let result = registry.lookup(0x2020).unwrap();
        assert_eq!(result.safepoint.register_bitmap, 0b0010);
        assert_eq!(result.frame_size, 128);

        // Not found
        assert!(registry.lookup(0x3000).is_none());
    }

    #[test]
    fn test_compact_array_lookup() {
        let maps = vec![
            StackMap::new(
                0x1000,
                0x100,
                64,
                vec![SafePoint::new(0x10, 0b0001, 0b0001)],
            ),
            StackMap::new(
                0x2000,
                0x200,
                128,
                vec![
                    SafePoint::new(0x10, 0b0010, 0b0010),
                    SafePoint::new(0x30, 0b0100, 0b0100),
                ],
            ),
        ];

        let compact = CompactStackMapArray::from_maps(maps);
        assert_eq!(compact.len(), 2);

        // Lookup
        let result = compact.lookup(0x1010).unwrap();
        assert_eq!(result.safepoint.register_bitmap, 0b0001);

        let result = compact.lookup(0x2025).unwrap();
        assert_eq!(result.safepoint.code_offset, 0x10); // Predecessor

        let result = compact.lookup(0x2030).unwrap();
        assert_eq!(result.safepoint.code_offset, 0x30);
    }

    #[test]
    fn test_safepoint_counts() {
        let sp = SafePoint::new(0, 0b10101010, 0xFF00FF00FF00FF00);
        assert_eq!(sp.live_register_count(), 4);
        assert_eq!(sp.live_stack_slot_count(), 32);
    }
}
