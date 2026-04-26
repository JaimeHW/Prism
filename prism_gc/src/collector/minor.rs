//! Minor nursery collection safepoint.
//!
//! Prism keeps nursery allocation on a fast bump-pointer path, but nursery
//! collection is intentionally non-moving until the runtime exposes exact,
//! rewriteable roots and object reference slots. Resetting from-space without
//! rewriting every live `Value` is unsound, so this collector preserves all
//! nursery objects and reports live-byte accounting. Runtime allocation falls
//! back to tenured space when the nursery is full, keeping VM allocations inside
//! one managed heap without corrupting references.

use crate::heap::GcHeap;
use crate::roots::RootSet;
use crate::trace::ObjectTracer;

/// Result of a minor collection.
#[derive(Debug, Default)]
pub struct MinorResult {
    /// Bytes freed from the nursery.
    pub bytes_freed: usize,
    /// Objects freed from the nursery.
    pub objects_freed: usize,
    /// Bytes promoted to old generation.
    pub bytes_promoted: usize,
    /// Objects promoted.
    pub objects_promoted: usize,
    /// Live bytes in nursery after collection.
    pub live_bytes: usize,
}

/// Minor collector for nursery safepoints.
///
/// The promotion-age field is retained as part of the public collector tuning
/// contract; it becomes active again when the exact moving collector is enabled.
pub struct MinorCollector {
    /// Promotion age threshold for the future moving collector.
    promotion_age: u8,
}

impl MinorCollector {
    /// Create a new minor collector.
    #[inline]
    pub fn new() -> Self {
        Self::with_promotion_age(2)
    }

    /// Create with custom promotion age.
    #[inline]
    pub const fn with_promotion_age(promotion_age: u8) -> Self {
        Self { promotion_age }
    }

    /// Perform a nursery safepoint with object tracing available.
    ///
    /// The tracer and roots are accepted to keep the collector API stable, but
    /// the nursery is not moved or reset until all references can be updated.
    pub fn collect<T: ObjectTracer>(
        &mut self,
        heap: &mut GcHeap,
        roots: &RootSet,
        object_tracer: &T,
    ) -> MinorResult {
        let _ = (roots, object_tracer);
        self.preserve_nursery(heap)
    }

    /// Perform a nursery safepoint without object tracing.
    pub fn collect_roots_only(&mut self, heap: &mut GcHeap, roots: &RootSet) -> MinorResult {
        let _ = roots;
        self.preserve_nursery(heap)
    }

    #[inline]
    fn preserve_nursery(&self, heap: &GcHeap) -> MinorResult {
        MinorResult {
            live_bytes: heap.nursery().allocated(),
            ..MinorResult::default()
        }
    }

    /// Get promotion age threshold.
    #[inline]
    pub const fn promotion_age(&self) -> u8 {
        self.promotion_age
    }
}

impl Default for MinorCollector {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests;
