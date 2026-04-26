//! Major (full) garbage collection.
//!
//! Major GC uses a tri-color mark-sweep algorithm:
//! 1. Mark phase: Trace from roots, mark all reachable objects
//! 2. Sweep phase: Free all unmarked (white) objects
//!
//! # Tri-Color Invariant
//!
//! The tri-color abstraction maintains:
//! - **White**: Not yet visited (potentially unreachable)
//! - **Gray**: Reachable, but children not yet scanned
//! - **Black**: Reachable and all children have been scanned
//!
//! The invariant: No black object points directly to a white object.
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────────┐
//! │  MARK PHASE                                                              │
//! │                                                                         │
//! │  ┌─────┐     ┌─────┐     ┌─────┐                                        │
//! │  │ Root│────▶│  A  │────▶│  B  │                                        │
//! │  └─────┘     │gray │     │white│                                        │
//! │              └─────┘     └─────┘                                        │
//! │                 │           ▲                                           │
//! │                 │ trace()   │ discovered                                │
//! │                 ▼           │                                           │
//! │              ┌─────┐     ┌─────┐                                        │
//! │              │  A  │     │  B  │                                        │
//! │              │black│     │gray │                                        │
//! │              └─────┘     └─────┘                                        │
//! └─────────────────────────────────────────────────────────────────────────┘
//! ```

use crate::heap::GcHeap;
use crate::roots::RootSet;
use crate::trace::{ObjectTracer, Tracer};
use prism_core::Value;
use rustc_hash::FxHashSet;
use std::collections::VecDeque;

/// Result of a major collection.
#[derive(Debug, Default)]
pub struct MajorResult {
    /// Bytes freed.
    pub bytes_freed: usize,
    /// Objects freed.
    pub objects_freed: usize,
    /// Live bytes after collection.
    pub live_bytes: usize,
    /// Objects marked as live.
    pub objects_marked: usize,
}

/// Major (mark-sweep) collector for the full heap.
///
/// This collector implements a tri-color mark-sweep algorithm:
/// - Uses a gray worklist for incremental-safe marking
/// - Supports concurrent marking (future enhancement)
/// - Integrates with write barriers for generational correctness
pub struct MajorCollector {
    /// Gray worklist for incremental marking.
    /// Objects in this queue have been discovered but not yet traced.
    worklist: VecDeque<*const ()>,
    /// Set of marked object addresses (white → gray/black).
    /// Using FxHashSet for O(1) lookup with minimal overhead.
    marked: FxHashSet<usize>,
}

impl MajorCollector {
    /// Create a new major collector.
    #[inline]
    pub fn new() -> Self {
        Self {
            worklist: VecDeque::with_capacity(4096),
            marked: FxHashSet::default(),
        }
    }

    /// Perform a full mark-sweep collection.
    ///
    /// # Arguments
    /// - `heap`: The GC heap
    /// - `roots`: Root set containing stack roots, globals, etc.
    /// - `object_tracer`: Object tracer for type-aware tracing
    ///
    /// # Algorithm
    /// 1. Clear all marks (all objects white)
    /// 2. Mark from roots (gray → black)
    /// 3. Process worklist until empty
    /// 4. Sweep unmarked objects
    pub fn collect<T: ObjectTracer>(
        &mut self,
        heap: &mut GcHeap,
        roots: &RootSet,
        object_tracer: &T,
    ) -> MajorResult {
        let mut result = MajorResult::default();

        // Clear state from previous collection
        self.worklist.clear();
        self.marked.clear();

        // Phase 1: Clear marks (all objects white)
        heap.clear_major_marks();

        // Phase 2: Mark from roots
        {
            let mut tracer = MarkingTracer {
                collector: self,
                heap,
            };
            roots.trace(&mut tracer);
        }

        // Phase 3: Process worklist (gray → black)
        while let Some(obj_ptr) = self.worklist.pop_front() {
            // Object is now being processed (going from gray to black)
            result.objects_marked += 1;

            // Create a tracer for this object's children
            let mut tracer = MarkingTracer {
                collector: self,
                heap,
            };

            // Trace the object's children using the runtime's dispatch
            // SAFETY: obj_ptr points to a valid object
            unsafe {
                object_tracer.trace_object(obj_ptr, &mut tracer);
            }
        }

        // Phase 4: Sweep major-collected spaces.
        //
        // Old-space reclamation is currently block-granular. A live object keeps
        // its whole block, while blocks with no marked roots are reset for reuse.
        let (old_freed, old_objects) = heap.old_space_mut().sweep();
        result.bytes_freed += old_freed;
        result.objects_freed += old_objects;

        // Sweep large object space
        let (los_freed, los_objects) = heap.large_objects().sweep();
        result.bytes_freed += los_freed;
        result.objects_freed += los_objects;

        // Calculate live bytes
        result.live_bytes = heap.old_space().usage() + heap.large_objects().usage();

        result
    }

    /// Collect without object tracing (for tests or when runtime is unavailable).
    pub fn collect_roots_only(&mut self, heap: &mut GcHeap, roots: &RootSet) -> MajorResult {
        let mut result = MajorResult::default();

        self.worklist.clear();
        self.marked.clear();

        heap.clear_major_marks();

        {
            let mut tracer = MarkingTracer {
                collector: self,
                heap,
            };
            roots.trace(&mut tracer);
        }

        // Process worklist without tracing children
        while let Some(_obj_ptr) = self.worklist.pop_front() {
            result.objects_marked += 1;
        }

        let (old_freed, old_objects) = heap.old_space_mut().sweep();
        result.bytes_freed += old_freed;
        result.objects_freed += old_objects;

        let (los_freed, los_objects) = heap.large_objects().sweep();
        result.bytes_freed += los_freed;
        result.objects_freed += los_objects;

        result.live_bytes = heap.old_space().usage() + heap.large_objects().usage();

        result
    }

    /// Mark an object gray (add to worklist).
    ///
    /// Returns true if the object was newly marked.
    #[inline]
    fn mark_gray(&mut self, ptr: *const ()) -> bool {
        if ptr.is_null() {
            return false;
        }

        let addr = ptr as usize;

        // Fast path: already marked
        if self.marked.contains(&addr) {
            return false;
        }

        // Mark as seen and add to worklist
        self.marked.insert(addr);
        self.worklist.push_back(ptr);
        true
    }

    /// Check if an object has been marked.
    #[inline]
    pub fn is_marked(&self, ptr: *const ()) -> bool {
        self.marked.contains(&(ptr as usize))
    }

    /// Get number of marked objects (for debugging/stats).
    #[inline]
    pub fn marked_count(&self) -> usize {
        self.marked.len()
    }
}

impl Default for MajorCollector {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// Marking Tracer
// =============================================================================

/// Tracer for the mark phase.
///
/// This tracer is called during root scanning and when tracing marked objects.
/// It identifies old-generation objects and marks them gray.
struct MarkingTracer<'a> {
    collector: &'a mut MajorCollector,
    heap: &'a GcHeap,
}

impl<'a> Tracer for MarkingTracer<'a> {
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

        // Only mark objects in old generation or LOS. Young generation objects
        // are handled by minor GC.
        if !self.heap.mark_major_live(ptr) {
            return;
        }

        // Mark gray and add to worklist
        self.collector.mark_gray(ptr);
    }
}
