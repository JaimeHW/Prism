//! Object tracing for garbage collection.
//!
//! The `Trace` trait is the core interface between user objects and the GC.
//! Every GC-managed type must implement `Trace` to enable the collector
//! to discover reachable objects.

pub mod object_tracer;
pub mod tracer;

pub use object_tracer::{NoopObjectTracer, ObjectTracer};
pub use tracer::Tracer;

use prism_core::Value;

/// Trait for types that can be traced by the garbage collector.
///
/// # Safety
///
/// This trait is unsafe because incorrect implementations can cause:
/// - Memory leaks (failing to trace reachable objects)
/// - Use-after-free (tracing dangling pointers)
/// - Undefined behavior during collection
///
/// Implementations must:
/// 1. Trace ALL object references that this object holds
/// 2. Never trace the same reference twice in a single call
/// 3. Never access freed memory during tracing
///
/// # Example
///
/// ```ignore
/// use prism_gc::{Trace, Tracer};
///
/// struct MyList {
///     items: Vec<Value>,
///     parent: Option<*const MyList>,
/// }
///
/// unsafe impl Trace for MyList {
///     fn trace(&self, tracer: &mut dyn Tracer) {
///         // Trace all contained values
///         for item in &self.items {
///             tracer.trace_value(*item);
///         }
///         // Trace optional reference
///         if let Some(parent) = self.parent {
///             tracer.trace_ptr(parent as *const ());
///         }
///     }
/// }
/// ```
pub unsafe trait Trace {
    /// Visit all object references held by this object.
    ///
    /// Called during the marking phase of garbage collection.
    /// The implementation must call `tracer.trace_value()` or
    /// `tracer.trace_ptr()` for each reference this object holds.
    fn trace(&self, tracer: &mut dyn Tracer);

    /// Called when the object is about to be collected.
    ///
    /// Use this for running finalizers (Python `__del__`),
    /// releasing external resources, or cleanup.
    ///
    /// # Safety
    ///
    /// During finalization:
    /// - Other objects may already be finalized or freed
    /// - Do not access other GC-managed objects
    /// - Keep finalization as simple as possible
    ///
    /// Default implementation does nothing.
    fn finalize(&mut self) {}

    /// Returns true if this object needs finalization.
    ///
    /// Objects without finalizers can skip the finalization queue,
    /// improving collection performance.
    ///
    /// Default: false
    #[inline]
    fn needs_finalization(&self) -> bool {
        false
    }

    /// Get the size of this object in bytes (for statistics).
    ///
    /// Should return the total memory footprint including any
    /// heap-allocated data owned by this object.
    ///
    /// Default: size_of::<Self>()
    #[inline]
    fn size_of(&self) -> usize {
        std::mem::size_of_val(self)
    }
}

// =============================================================================
// Trace implementations for primitives
// =============================================================================

/// Safety: Primitives hold no references.
unsafe impl Trace for () {
    #[inline]
    fn trace(&self, _tracer: &mut dyn Tracer) {}
}

unsafe impl Trace for bool {
    #[inline]
    fn trace(&self, _tracer: &mut dyn Tracer) {}
}

unsafe impl Trace for i64 {
    #[inline]
    fn trace(&self, _tracer: &mut dyn Tracer) {}
}

unsafe impl Trace for f64 {
    #[inline]
    fn trace(&self, _tracer: &mut dyn Tracer) {}
}

unsafe impl Trace for String {
    #[inline]
    fn trace(&self, _tracer: &mut dyn Tracer) {}

    #[inline]
    fn size_of(&self) -> usize {
        std::mem::size_of::<String>() + self.capacity()
    }
}

// =============================================================================
// Trace implementations for containers
// =============================================================================

unsafe impl<T: Trace> Trace for Vec<T> {
    fn trace(&self, tracer: &mut dyn Tracer) {
        for item in self {
            item.trace(tracer);
        }
    }

    fn size_of(&self) -> usize {
        std::mem::size_of::<Vec<T>>() + self.capacity() * std::mem::size_of::<T>()
    }
}

unsafe impl<T: Trace> Trace for Option<T> {
    fn trace(&self, tracer: &mut dyn Tracer) {
        if let Some(inner) = self {
            inner.trace(tracer);
        }
    }
}

unsafe impl<T: Trace> Trace for Box<T> {
    fn trace(&self, tracer: &mut dyn Tracer) {
        (**self).trace(tracer);
    }

    fn size_of(&self) -> usize {
        std::mem::size_of::<Box<T>>() + (**self).size_of()
    }
}

unsafe impl<T: Trace> Trace for std::sync::Arc<T> {
    fn trace(&self, tracer: &mut dyn Tracer) {
        (**self).trace(tracer);
    }
}

unsafe impl<T: Trace> Trace for [T] {
    fn trace(&self, tracer: &mut dyn Tracer) {
        for item in self {
            item.trace(tracer);
        }
    }
}

// =============================================================================
// Trace for Value
// =============================================================================

/// Safety: Traces the object pointer if Value contains one.
unsafe impl Trace for Value {
    fn trace(&self, tracer: &mut dyn Tracer) {
        tracer.trace_value(*self);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_primitive_trace() {
        struct NullTracer;
        impl Tracer for NullTracer {
            fn trace_value(&mut self, _value: Value) {}
            fn trace_ptr(&mut self, _ptr: *const ()) {}
        }

        let mut tracer = NullTracer;

        // Primitives should trace without panicking
        true.trace(&mut tracer);
        42i64.trace(&mut tracer);
        3.14f64.trace(&mut tracer);
        "hello".to_string().trace(&mut tracer);
    }

    #[test]
    fn test_container_trace() {
        struct CountingTracer {
            count: usize,
        }
        impl Tracer for CountingTracer {
            fn trace_value(&mut self, _value: Value) {
                self.count += 1;
            }
            fn trace_ptr(&mut self, _ptr: *const ()) {
                self.count += 1;
            }
        }

        let mut tracer = CountingTracer { count: 0 };

        let values: Vec<Value> = vec![Value::none(), Value::bool(true), Value::int(42).unwrap()];
        values.trace(&mut tracer);

        assert_eq!(tracer.count, 3);
    }
}
