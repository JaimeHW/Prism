//! Core object header and reference types.
//!
//! All Python objects share a common header layout for uniform GC handling
//! and fast type checking.

pub mod registry;
pub mod shape;
pub mod shaped_object;
pub mod type_obj;

use crate::object::type_obj::TypeId;
use std::sync::atomic::{AtomicU32, Ordering};

// =============================================================================
// GC Flags
// =============================================================================

/// GC mark colors for tri-color marking.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GcColor {
    /// Not yet visited.
    White = 0,
    /// In processing queue.
    Gray = 1,
    /// Fully scanned.
    Black = 2,
}

/// GC flag bits packed into u32.
#[derive(Debug, Clone, Copy)]
pub struct GcFlags(u32);

impl GcFlags {
    /// Create new flags (white, not pinned).
    #[inline]
    pub const fn new() -> Self {
        Self(0)
    }

    /// Get GC color.
    #[inline]
    pub fn color(self) -> GcColor {
        match self.0 & 0x3 {
            0 => GcColor::White,
            1 => GcColor::Gray,
            2 => GcColor::Black,
            _ => GcColor::White, // Should not happen
        }
    }

    /// Set GC color.
    #[inline]
    pub fn set_color(self, color: GcColor) -> Self {
        Self((self.0 & !0x3) | (color as u32))
    }

    /// Check if object is pinned (cannot be moved).
    #[inline]
    pub fn is_pinned(self) -> bool {
        (self.0 & 0x4) != 0
    }

    /// Set pinned flag.
    #[inline]
    pub fn set_pinned(self, pinned: bool) -> Self {
        if pinned {
            Self(self.0 | 0x4)
        } else {
            Self(self.0 & !0x4)
        }
    }

    /// Check if object is finalized.
    #[inline]
    pub fn is_finalized(self) -> bool {
        (self.0 & 0x8) != 0
    }

    /// Raw value.
    #[inline]
    pub fn raw(self) -> u32 {
        self.0
    }
}

impl Default for GcFlags {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// Object Header
// =============================================================================

/// Hash value constant for "not yet computed".
pub const HASH_NOT_COMPUTED: u64 = u64::MAX;

/// Object header - 16 bytes, cache-line aligned fields.
///
/// All heap-allocated Python objects begin with this header.
/// The layout is designed for:
/// - Fast type checking via TypeId (4 bytes, no pointer chase)
/// - GC metadata in gc_flags (4 bytes)
/// - Cached hash value (8 bytes, avoids recomputation)
#[repr(C, align(8))]
pub struct ObjectHeader {
    /// Type identifier for fast dispatch.
    pub type_id: TypeId,
    /// GC flags (color, pinned, finalized).
    pub gc_flags: AtomicU32,
    /// Cached hash value (HASH_NOT_COMPUTED = not yet computed).
    pub hash: u64,
}

impl ObjectHeader {
    /// Create a new object header.
    #[inline]
    pub fn new(type_id: TypeId) -> Self {
        Self {
            type_id,
            gc_flags: AtomicU32::new(0),
            hash: HASH_NOT_COMPUTED,
        }
    }

    /// Get GC flags.
    #[inline]
    pub fn gc_flags(&self) -> GcFlags {
        GcFlags(self.gc_flags.load(Ordering::Relaxed))
    }

    /// Set GC flags atomically.
    #[inline]
    pub fn set_gc_flags(&self, flags: GcFlags) {
        self.gc_flags.store(flags.raw(), Ordering::Relaxed);
    }

    /// Get GC color.
    #[inline]
    pub fn gc_color(&self) -> GcColor {
        self.gc_flags().color()
    }

    /// Set GC color atomically.
    #[inline]
    pub fn set_gc_color(&self, color: GcColor) {
        let old = self.gc_flags.load(Ordering::Relaxed);
        let new = (old & !0x3) | (color as u32);
        self.gc_flags.store(new, Ordering::Relaxed);
    }

    /// Check if hash is computed.
    #[inline]
    pub fn has_hash(&self) -> bool {
        self.hash != HASH_NOT_COMPUTED
    }

    /// Get cached hash (or None if not computed).
    #[inline]
    pub fn cached_hash(&self) -> Option<u64> {
        if self.has_hash() {
            Some(self.hash)
        } else {
            None
        }
    }
}

impl std::fmt::Debug for ObjectHeader {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ObjectHeader")
            .field("type_id", &self.type_id)
            .field("gc_flags", &self.gc_flags())
            .field("hash", &self.hash)
            .finish()
    }
}

// =============================================================================
// Object Trait
// =============================================================================

/// Trait for all Python objects.
///
/// Provides access to the object header and basic operations.
pub trait PyObject: Send + Sync {
    /// Get the object header.
    fn header(&self) -> &ObjectHeader;

    /// Get mutable header (for GC).
    fn header_mut(&mut self) -> &mut ObjectHeader;

    /// Get the type ID.
    #[inline]
    fn type_id(&self) -> TypeId {
        self.header().type_id
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_header_size() {
        // Header should be 16 bytes
        assert_eq!(std::mem::size_of::<ObjectHeader>(), 16);
    }

    #[test]
    fn test_gc_flags() {
        let flags = GcFlags::new();
        assert_eq!(flags.color(), GcColor::White);
        assert!(!flags.is_pinned());

        let flags = flags.set_color(GcColor::Gray);
        assert_eq!(flags.color(), GcColor::Gray);

        let flags = flags.set_pinned(true);
        assert!(flags.is_pinned());
        assert_eq!(flags.color(), GcColor::Gray);
    }

    #[test]
    fn test_object_header() {
        let header = ObjectHeader::new(TypeId::LIST);
        assert_eq!(header.type_id, TypeId::LIST);
        assert_eq!(header.gc_color(), GcColor::White);
        assert!(!header.has_hash());
    }
}
