//! Tracer interface for object graph traversal.

use prism_core::Value;

/// Tracer interface for visiting object references during GC.
///
/// The GC calls `trace()` on each live object, which then uses
/// this tracer to report its references.
///
/// # Implementation Notes
///
/// Tracers may:
/// - Add objects to a mark worklist
/// - Copy objects to a new location (for copying collection)
/// - Update statistics
///
/// # Example
///
/// ```ignore
/// struct MarkingTracer<'a> {
///     worklist: &'a mut Vec<*const ()>,
///     heap: &'a Heap,
/// }
///
/// impl Tracer for MarkingTracer<'_> {
///     fn trace_value(&mut self, value: Value) {
///         if let Some(ptr) = value.as_object_ptr() {
///             self.trace_ptr(ptr);
///         }
///     }
///
///     fn trace_ptr(&mut self, ptr: *const ()) {
///         if self.heap.mark_gray(ptr) {
///             self.worklist.push(ptr);
///         }
///     }
/// }
/// ```
pub trait Tracer {
    /// Trace a Value that may contain an object reference.
    ///
    /// The implementation should extract any object pointer from
    /// the value and mark it as reachable.
    fn trace_value(&mut self, value: Value);

    /// Trace a raw object pointer.
    ///
    /// The implementation should mark this object as reachable.
    /// It's safe to call this with null pointers (they're ignored).
    fn trace_ptr(&mut self, ptr: *const ());
}

/// A null tracer that does nothing (for testing).
pub struct NullTracer;

impl Tracer for NullTracer {
    #[inline]
    fn trace_value(&mut self, _value: Value) {}

    #[inline]
    fn trace_ptr(&mut self, _ptr: *const ()) {}
}

/// A counting tracer for debugging and statistics.
pub struct CountingTracer {
    /// Number of values traced.
    pub value_count: usize,
    /// Number of pointers traced.
    pub ptr_count: usize,
}

impl CountingTracer {
    /// Create a new counting tracer.
    pub fn new() -> Self {
        Self {
            value_count: 0,
            ptr_count: 0,
        }
    }

    /// Get total number of references traced.
    pub fn total(&self) -> usize {
        self.value_count + self.ptr_count
    }
}

impl Default for CountingTracer {
    fn default() -> Self {
        Self::new()
    }
}

impl Tracer for CountingTracer {
    fn trace_value(&mut self, _value: Value) {
        self.value_count += 1;
    }

    fn trace_ptr(&mut self, _ptr: *const ()) {
        self.ptr_count += 1;
    }
}
