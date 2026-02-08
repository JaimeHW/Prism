//! Heap memory management.
//!
//! The heap is divided into three spaces:
//! - Nursery: Young generation with bump-pointer allocation
//! - Old Space: Tenured generation with block-based allocation
//! - Large Object Space: Large objects that bypass copying

mod block;
mod large_object_space;
mod nursery;
mod old_space;

pub use block::{Block, BlockHeader, BlockState};
pub use large_object_space::LargeObjectSpace;
pub use nursery::Nursery;
pub use old_space::OldSpace;

use crate::barrier::RememberedSet;
use crate::config::GcConfig;
use crate::stats::GcStats;
use crate::Generation;

use std::ptr::NonNull;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Main heap structure managing all memory spaces.
pub struct GcHeap {
    /// Configuration parameters.
    config: GcConfig,

    /// Young generation (bump-pointer allocation).
    nursery: Nursery,

    /// Old generation (block-based allocation).
    old_space: OldSpace,

    /// Large object space (direct allocation).
    large_objects: LargeObjectSpace,

    /// GC statistics.
    stats: GcStats,

    /// Total bytes allocated since last GC.
    bytes_since_gc: AtomicUsize,

    /// Remembered set for tracking old→young references.
    /// Used by the write barrier and drained during minor GC.
    remembered_set: RememberedSet,
}

impl GcHeap {
    /// Create a new heap with the given configuration.
    pub fn new(config: GcConfig) -> Self {
        config.validate().expect("Invalid GC configuration");

        let nursery = Nursery::new(config.nursery_size);
        let old_space = OldSpace::new(config.initial_old_size, config.block_size);
        let large_objects = LargeObjectSpace::new();

        Self {
            config,
            nursery,
            old_space,
            large_objects,
            stats: GcStats::new(),
            bytes_since_gc: AtomicUsize::new(0),
            remembered_set: RememberedSet::new(),
        }
    }

    /// Create a heap with default configuration.
    pub fn with_defaults() -> Self {
        Self::new(GcConfig::default())
    }

    // =========================================================================
    // Allocation
    // =========================================================================

    /// Allocate memory for an object of the given size.
    ///
    /// Returns a pointer to uninitialized memory that can hold `size` bytes.
    /// The caller must initialize the object before the next GC.
    ///
    /// # Safety
    ///
    /// - The returned memory is uninitialized
    /// - Must be initialized before GC can trace it
    /// - Size must include the object header
    #[inline]
    pub fn alloc(&self, size: usize) -> Option<NonNull<u8>> {
        let aligned_size = align_up(size, 8);

        // Large object allocation
        if aligned_size >= self.config.large_object_threshold {
            return self.alloc_large(aligned_size);
        }

        // Try nursery first (fast path)
        if let Some(ptr) = self.nursery.alloc(aligned_size) {
            self.stats.record_allocation(aligned_size);
            self.bytes_since_gc
                .fetch_add(aligned_size, Ordering::Relaxed);
            return Some(ptr);
        }

        // Nursery full - caller should trigger minor GC
        None
    }

    /// Allocate in the large object space.
    fn alloc_large(&self, size: usize) -> Option<NonNull<u8>> {
        let ptr = self.large_objects.alloc(size)?;
        self.stats.record_allocation(size);
        self.stats
            .large_object_usage
            .fetch_add(size as u64, Ordering::Relaxed);
        Some(ptr)
    }

    /// Allocate directly in the old generation.
    ///
    /// Used during promotion from nursery.
    pub fn alloc_tenured(&self, size: usize) -> Option<NonNull<u8>> {
        let aligned_size = align_up(size, 8);
        let ptr = self.old_space.alloc(aligned_size)?;
        self.stats.record_promotion(aligned_size);
        Some(ptr)
    }

    // =========================================================================
    // Space Queries
    // =========================================================================

    /// Check if a pointer is in the nursery.
    #[inline]
    pub fn is_young(&self, ptr: *const ()) -> bool {
        self.nursery.contains(ptr)
    }

    /// Check if a pointer is in the old generation.
    #[inline]
    pub fn is_old(&self, ptr: *const ()) -> bool {
        self.old_space.contains(ptr) || self.large_objects.contains(ptr)
    }

    /// Check if a pointer is managed by this heap.
    #[inline]
    pub fn contains(&self, ptr: *const ()) -> bool {
        self.nursery.contains(ptr)
            || self.old_space.contains(ptr)
            || self.large_objects.contains(ptr)
    }

    /// Get the generation of an object.
    pub fn generation_of(&self, ptr: *const ()) -> Option<Generation> {
        if self.nursery.contains(ptr) {
            Some(Generation::Nursery)
        } else if self.old_space.contains(ptr) {
            Some(Generation::Tenured)
        } else if self.large_objects.contains(ptr) {
            Some(Generation::LargeObject)
        } else {
            None
        }
    }

    // =========================================================================
    // Collection Triggers
    // =========================================================================

    /// Check if a minor GC should be triggered.
    #[inline]
    pub fn should_minor_collect(&self) -> bool {
        self.bytes_since_gc.load(Ordering::Relaxed) >= self.config.minor_gc_trigger
    }

    /// Check if a major GC should be triggered.
    pub fn should_major_collect(&self) -> bool {
        let usage = self.old_space.usage() as f64;
        let capacity = self.old_space.capacity() as f64;
        if capacity > 0.0 {
            (usage / capacity) >= self.config.major_gc_threshold
        } else {
            false
        }
    }

    /// Reset bytes-since-gc counter after collection.
    pub fn reset_gc_counter(&self) {
        self.bytes_since_gc.store(0, Ordering::Relaxed);
    }

    // =========================================================================
    // Accessors
    // =========================================================================

    /// Get the configuration.
    pub fn config(&self) -> &GcConfig {
        &self.config
    }

    /// Get GC statistics.
    pub fn stats(&self) -> &GcStats {
        &self.stats
    }

    /// Get the nursery.
    pub fn nursery(&self) -> &Nursery {
        &self.nursery
    }

    /// Get mutable nursery access.
    pub fn nursery_mut(&mut self) -> &mut Nursery {
        &mut self.nursery
    }

    /// Get the old space.
    pub fn old_space(&self) -> &OldSpace {
        &self.old_space
    }

    /// Get mutable old space access.
    pub fn old_space_mut(&mut self) -> &mut OldSpace {
        &mut self.old_space
    }

    /// Get the large object space.
    pub fn large_objects(&self) -> &LargeObjectSpace {
        &self.large_objects
    }

    /// Get mutable large object space access.
    pub fn large_objects_mut(&mut self) -> &mut LargeObjectSpace {
        &mut self.large_objects
    }

    // =========================================================================
    // Remembered Set (Write Barrier Integration)
    // =========================================================================

    /// Get the remembered set for write barrier marking.
    ///
    /// The write barrier calls this to record old→young references.
    #[inline]
    pub fn remembered_set(&self) -> &RememberedSet {
        &self.remembered_set
    }

    /// Drain the remembered set for GC root scanning.
    ///
    /// Called during minor collection to get all old→young references
    /// that need to be treated as roots.
    pub fn drain_remembered_set(&self) -> Vec<crate::barrier::RememberedEntry> {
        self.remembered_set.drain()
    }

    /// Clear the remembered set.
    pub fn clear_remembered_set(&self) {
        self.remembered_set.clear();
    }
}

/// Align a size up to the given alignment.
#[inline]
pub const fn align_up(size: usize, align: usize) -> usize {
    (size + align - 1) & !(align - 1)
}

/// Align a pointer up to the given alignment.
#[inline]
pub fn align_ptr_up(ptr: *mut u8, align: usize) -> *mut u8 {
    let addr = ptr as usize;
    let aligned = align_up(addr, align);
    aligned as *mut u8
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_align_up() {
        assert_eq!(align_up(0, 8), 0);
        assert_eq!(align_up(1, 8), 8);
        assert_eq!(align_up(7, 8), 8);
        assert_eq!(align_up(8, 8), 8);
        assert_eq!(align_up(9, 8), 16);
    }

    #[test]
    fn test_heap_creation() {
        let heap = GcHeap::with_defaults();
        assert!(!heap.should_minor_collect());
    }
}
