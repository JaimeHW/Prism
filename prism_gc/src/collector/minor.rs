//! Minor (nursery) garbage collection.
//!
//! Minor GC uses a copying collection algorithm:
//! 1. Scan roots to find live objects in nursery
//! 2. Copy live objects to to-space (or promote to tenured)
//! 3. Update all references to point to new locations
//! 4. Swap from-space and to-space
//!
//! # Performance Characteristics
//!
//! - **Time**: O(live data), not O(heap size)
//! - **Latency**: Typically < 1ms for small nurseries
//! - **Throughput**: High due to bump-pointer allocation
//!
//! # Algorithm: Cheney's Copying Collection
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────────┐
//! │  FROM-SPACE (Nursery)          │  TO-SPACE                              │
//! │  ┌─────┬─────┬─────┬───────┐   │  ┌─────┬─────┬───────────────────────┐ │
//! │  │  A  │  B  │  C  │ free  │   │  │  A' │  C' │      free             │ │
//! │  │alive│dead │alive│       │──▶│  │copy │copy │                       │ │
//! │  └─────┴─────┴─────┴───────┘   │  └─────┴─────┴───────────────────────┘ │
//! │                                │                                        │
//! │  B is unreachable, not copied  │  Only live objects are copied          │
//! └─────────────────────────────────────────────────────────────────────────┘
//! ```

use crate::heap::GcHeap;
use crate::roots::RootSet;
use crate::trace::{ObjectTracer, Tracer};
use prism_core::Value;
use rustc_hash::FxHashMap;
use std::collections::VecDeque;

/// Result of a minor collection.
#[derive(Debug, Default)]
pub struct MinorResult {
    /// Bytes freed (objects that died in nursery).
    pub bytes_freed: usize,
    /// Objects freed.
    pub objects_freed: usize,
    /// Bytes promoted to old generation.
    pub bytes_promoted: usize,
    /// Objects promoted.
    pub objects_promoted: usize,
    /// Live bytes in nursery after collection.
    pub live_bytes: usize,
}

/// Minor (scavenge) collector for the nursery.
///
/// This collector implements Cheney's semi-space copying algorithm:
/// - Objects are allocated in from-space using bump allocation
/// - During collection, live objects are copied to to-space
/// - Objects surviving multiple collections are promoted to tenured space
pub struct MinorCollector {
    /// Work queue of objects to process (Cheney's scan pointer).
    worklist: VecDeque<*const ()>,
    /// Forwarding pointers: old address → new address.
    /// Using FxHashMap for O(1) lookup with minimal overhead.
    forwarding: FxHashMap<usize, usize>,
    /// Promotion age threshold (objects surviving this many GCs are promoted).
    promotion_age: u8,
}

impl MinorCollector {
    /// Create a new minor collector.
    #[inline]
    pub fn new() -> Self {
        Self::with_promotion_age(2)
    }

    /// Create with custom promotion age.
    ///
    /// Objects surviving `promotion_age` minor collections are promoted
    /// to the tenured generation.
    #[inline]
    pub fn with_promotion_age(promotion_age: u8) -> Self {
        Self {
            worklist: VecDeque::with_capacity(1024),
            forwarding: FxHashMap::default(),
            promotion_age,
        }
    }

    /// Perform minor collection.
    ///
    /// # Arguments
    /// - `heap`: The GC heap
    /// - `roots`: Root set containing stack roots, globals, etc.
    /// - `tracer`: Object tracer for type-aware tracing
    ///
    /// # Algorithm
    /// 1. Scan roots and copy live objects to to-space
    /// 2. Process worklist: trace each copied object's children
    /// 3. Swap from-space and to-space
    /// 4. Calculate statistics
    pub fn collect<T: ObjectTracer>(
        &mut self,
        heap: &mut GcHeap,
        roots: &RootSet,
        object_tracer: &T,
    ) -> MinorResult {
        let mut result = MinorResult::default();

        // Clear state from previous collection
        self.worklist.clear();
        self.forwarding.clear();

        let allocated_before = heap.nursery().allocated();

        // Phase 1: Scan roots and copy live objects to to-space
        {
            let mut tracer = ScavengeTracer {
                heap,
                collector: self,
                result: &mut result,
            };
            roots.trace(&mut tracer);
        }

        // Phase 2: Process worklist (Cheney's algorithm)
        // Each object in the worklist has been copied to to-space.
        // We need to trace its children and copy them too.
        while let Some(obj_ptr) = self.worklist.pop_front() {
            // Create a tracer for this object's children
            let mut tracer = ScavengeTracer {
                heap,
                collector: self,
                result: &mut result,
            };

            // Trace the object's children using the runtime's dispatch
            // SAFETY: obj_ptr points to a valid object in to-space
            unsafe {
                object_tracer.trace_object(obj_ptr, &mut tracer);
            }
        }

        // Phase 3: Swap spaces
        heap.nursery_mut().swap_spaces();

        // Calculate bytes freed
        result.bytes_freed =
            allocated_before.saturating_sub(result.live_bytes + result.bytes_promoted);

        result
    }

    /// Collect without object tracing (for tests or when runtime is unavailable).
    ///
    /// This version only processes roots but doesn't trace object children.
    pub fn collect_roots_only(&mut self, heap: &mut GcHeap, roots: &RootSet) -> MinorResult {
        let mut result = MinorResult::default();

        self.worklist.clear();
        self.forwarding.clear();

        let allocated_before = heap.nursery().allocated();

        {
            let mut tracer = ScavengeTracer {
                heap,
                collector: self,
                result: &mut result,
            };
            roots.trace(&mut tracer);
        }

        // Process worklist without tracing (objects are already copied,
        // but their children won't be traced)
        while let Some(_obj_ptr) = self.worklist.pop_front() {
            // No tracing without object_tracer
        }

        heap.nursery_mut().swap_spaces();
        result.bytes_freed =
            allocated_before.saturating_sub(result.live_bytes + result.bytes_promoted);

        result
    }

    /// Copy an object from from-space to to-space or tenured space.
    ///
    /// Returns the new location of the object, or None if allocation failed.
    ///
    /// # Arguments
    /// - `heap`: The GC heap
    /// - `ptr`: Pointer to the object in from-space
    /// - `size`: Size of the object in bytes
    /// - `age`: Current age of the object (number of survived GCs)
    /// - `result`: Statistics accumulator
    #[inline]
    fn copy_object(
        &mut self,
        heap: &mut GcHeap,
        ptr: *const (),
        size: usize,
        age: u8,
        result: &mut MinorResult,
    ) -> Option<*const ()> {
        // Fast path: check if already forwarded
        if let Some(&new_addr) = self.forwarding.get(&(ptr as usize)) {
            return Some(new_addr as *const ());
        }

        // Decide: promote to tenured or copy to survivor
        let new_ptr = if age >= self.promotion_age {
            // Promote to tenured space
            let new_ptr = heap.alloc_tenured(size)?;
            result.bytes_promoted += size;
            result.objects_promoted += 1;
            new_ptr
        } else {
            // Copy to to-space (survivor)
            let new_ptr = heap.nursery().alloc_to_space(size)?;
            result.live_bytes += size;
            new_ptr
        };

        // Copy object data
        // SAFETY: Both pointers are valid, non-overlapping, and properly sized
        unsafe {
            std::ptr::copy_nonoverlapping(ptr as *const u8, new_ptr.as_ptr(), size);
        }

        // Record forwarding pointer for future lookups
        let new_addr = new_ptr.as_ptr() as usize;
        self.forwarding.insert(ptr as usize, new_addr);

        // Add to worklist for tracing children
        self.worklist.push_back(new_addr as *const ());

        Some(new_addr as *const ())
    }

    /// Check if an object has been forwarded.
    #[inline]
    fn get_forwarding(&self, ptr: *const ()) -> Option<*const ()> {
        self.forwarding
            .get(&(ptr as usize))
            .map(|&addr| addr as *const ())
    }

    /// Get the promotion age threshold.
    #[inline]
    pub fn promotion_age(&self) -> u8 {
        self.promotion_age
    }
}

impl Default for MinorCollector {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// Scavenge Tracer
// =============================================================================

/// Tracer for scavenge (copying) collection.
///
/// This tracer is called during root scanning and when tracing copied objects.
/// It identifies nursery objects and triggers copying to to-space.
struct ScavengeTracer<'a> {
    heap: &'a mut GcHeap,
    collector: &'a mut MinorCollector,
    result: &'a mut MinorResult,
}

impl<'a> Tracer for ScavengeTracer<'a> {
    #[inline]
    fn trace_value(&mut self, value: Value) {
        if let Some(ptr) = value.as_object_ptr() {
            self.trace_ptr(ptr);
        }
    }

    #[inline]
    fn trace_ptr(&mut self, ptr: *const ()) {
        if ptr.is_null() {
            return;
        }

        // Only process objects in the nursery's from-space
        if !self.heap.nursery().in_from_space(ptr) {
            return;
        }

        // Fast path: check if already forwarded
        if let Some(_new_ptr) = self.collector.get_forwarding(ptr) {
            // TODO: Update the reference to point to new location
            // This requires mutable access to the source slot
            return;
        }

        // Object is in nursery and not yet copied
        // For now, add to worklist for processing
        // In a complete implementation, we would:
        // 1. Read the object header to get size and age
        // 2. Call copy_object to move it
        // 3. Update the reference to point to new location
        self.collector.worklist.push_back(ptr);
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::GcConfig;
    use crate::trace::NoopObjectTracer;

    #[test]
    fn test_minor_collector_creation() {
        let collector = MinorCollector::new();
        assert!(collector.worklist.is_empty());
        assert!(collector.forwarding.is_empty());
        assert_eq!(collector.promotion_age, 2);
    }

    #[test]
    fn test_minor_collector_custom_promotion_age() {
        let collector = MinorCollector::with_promotion_age(5);
        assert_eq!(collector.promotion_age(), 5);
    }

    #[test]
    fn test_minor_collection_empty() {
        let mut collector = MinorCollector::new();
        let mut heap = GcHeap::new(GcConfig::default());
        let roots = RootSet::new();

        let result = collector.collect(&mut heap, &roots, &NoopObjectTracer);

        assert_eq!(result.bytes_freed, 0);
        assert_eq!(result.objects_promoted, 0);
    }

    #[test]
    fn test_minor_collection_roots_only() {
        let mut collector = MinorCollector::new();
        let mut heap = GcHeap::new(GcConfig::default());
        let roots = RootSet::new();

        let result = collector.collect_roots_only(&mut heap, &roots);

        assert_eq!(result.bytes_freed, 0);
        assert_eq!(result.objects_promoted, 0);
    }

    #[test]
    fn test_forwarding_lookup() {
        let mut collector = MinorCollector::new();

        // Initially empty
        assert!(collector.get_forwarding(0x1000 as *const ()).is_none());

        // Add forwarding entry
        collector.forwarding.insert(0x1000, 0x2000);

        // Should find it
        let forwarded = collector.get_forwarding(0x1000 as *const ());
        assert_eq!(forwarded, Some(0x2000 as *const ()));
    }
}
