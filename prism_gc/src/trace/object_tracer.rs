//! Object tracing callback interface.
//!
//! This module defines the callback interface that allows the GC collector
//! to trace objects without depending on the runtime's concrete types.
//!
//! # Architecture
//!
//! The `ObjectTracer` trait provides an abstraction between the GC and
//! the runtime's type system:
//!
//! ```text
//! ┌─────────────────┐         ┌─────────────────────────┐
//! │    prism_gc     │         │     prism_runtime       │
//! │                 │         │                         │
//! │   Collector     │◀────────│   RuntimeObjectTracer   │
//! │                 │         │   (impl ObjectTracer)   │
//! │   uses:         │         │                         │
//! │   ObjectTracer  │         │   dispatches to:        │
//! │   trait         │         │   gc_dispatch table     │
//! └─────────────────┘         └─────────────────────────┘
//! ```
//!
//! The runtime provides an implementation of `ObjectTracer` that uses
//! the dispatch table to call the correct trace function for each type.

use crate::trace::Tracer;

/// Callback interface for object tracing.
///
/// The GC collector uses this trait to trace objects during collection
/// without needing to know their concrete types.
///
/// # Safety
///
/// Implementations must ensure that:
/// - `trace_object` correctly identifies and traces all object references
/// - `finalize_object` is called at most once per object
/// - Object pointers passed to these methods are valid and properly aligned
pub trait ObjectTracer: Send + Sync {
    /// Trace all references in an object.
    ///
    /// # Arguments
    /// - `ptr`: Raw pointer to the object header
    /// - `tracer`: The tracer to call for each child reference
    ///
    /// # Safety
    /// - `ptr` must point to a valid, live GC object
    /// - The object must remain valid for the duration of the call
    unsafe fn trace_object(&self, ptr: *const (), tracer: &mut dyn Tracer);

    /// Get the size of an object.
    ///
    /// # Arguments
    /// - `ptr`: Raw pointer to the object header
    ///
    /// # Returns
    /// The total allocation size of the object in bytes.
    ///
    /// # Safety
    /// - `ptr` must point to a valid, live GC object
    unsafe fn size_of_object(&self, ptr: *const ()) -> usize;

    /// Finalize (drop) an object before reclaiming its memory.
    ///
    /// # Arguments
    /// - `ptr`: Raw pointer to the object header
    ///
    /// # Safety
    /// - `ptr` must point to a valid, live GC object
    /// - Must be called exactly once per object before memory is reclaimed
    unsafe fn finalize_object(&self, ptr: *mut ());
}

/// No-op object tracer for testing.
#[derive(Default)]
pub struct NoopObjectTracer;

impl ObjectTracer for NoopObjectTracer {
    unsafe fn trace_object(&self, _ptr: *const (), _tracer: &mut dyn Tracer) {
        // No-op
    }

    unsafe fn size_of_object(&self, _ptr: *const ()) -> usize {
        0
    }

    unsafe fn finalize_object(&self, _ptr: *mut ()) {
        // No-op
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_noop_tracer() {
        let tracer = NoopObjectTracer;
        unsafe {
            tracer.trace_object(std::ptr::null(), &mut NoopTracerImpl);
            assert_eq!(tracer.size_of_object(std::ptr::null()), 0);
        }
    }

    struct NoopTracerImpl;
    impl Tracer for NoopTracerImpl {
        fn trace_value(&mut self, _value: prism_core::Value) {}
        fn trace_ptr(&mut self, _ptr: *const ()) {}
    }
}
