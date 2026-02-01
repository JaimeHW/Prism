//! Runtime object tracer implementation.
//!
//! This module provides the `RuntimeObjectTracer` which implements
//! the `ObjectTracer` trait from `prism_gc`, enabling the GC to
//! trace Prism runtime objects without knowing their concrete types.
//!
//! # Architecture
//!
//! The tracer works by:
//! 1. Reading the `ObjectHeader` from the raw pointer
//! 2. Extracting the `TypeId` from the header
//! 3. Using the dispatch table to call the correct trace function
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────┐
//! │                     OBJECT MEMORY LAYOUT                             │
//! │  ┌────────────────────────────────────────────────────────────────┐ │
//! │  │  ObjectHeader (16 bytes)           │  Object Data              │ │
//! │  │  ┌──────────────────────────────┐  │                           │ │
//! │  │  │ type_id (4)  │ gc_flags (4)  │  │                           │ │
//! │  │  │ hash (8)                     │  │                           │ │
//! │  │  └──────────────────────────────┘  │                           │ │
//! │  └────────────────────────────────────┴───────────────────────────┘ │
//! │       ↑                                                             │
//! │       │                                                             │
//! │       ptr                                                           │
//! └─────────────────────────────────────────────────────────────────────┘
//! ```

use crate::gc_dispatch::{finalize_object, size_of_object, trace_object};
use crate::object::ObjectHeader;
use prism_gc::trace::{ObjectTracer, Tracer};

/// Runtime object tracer that uses the dispatch table for type-based tracing.
///
/// This is a zero-sized type that provides the implementation of `ObjectTracer`
/// for the Prism runtime. It is thread-safe and can be used during concurrent
/// garbage collection.
///
/// # Example
///
/// ```ignore
/// use prism_runtime::RuntimeObjectTracer;
/// use prism_gc::ObjectTracer;
///
/// let tracer = RuntimeObjectTracer::new();
///
/// // During GC, the collector can use this to trace objects
/// unsafe {
///     tracer.trace_object(object_ptr, &mut gc_tracer);
/// }
/// ```
#[derive(Debug, Clone, Copy, Default)]
pub struct RuntimeObjectTracer;

impl RuntimeObjectTracer {
    /// Create a new runtime object tracer.
    #[inline]
    pub const fn new() -> Self {
        Self
    }
}

/// Implementation of `ObjectTracer` for the Prism runtime.
///
/// # Safety
///
/// This implementation assumes that:
/// - All GC-managed objects start with an `ObjectHeader`
/// - The `type_id` in the header correctly identifies the object type
/// - The dispatch table has been initialized before any tracing occurs
impl ObjectTracer for RuntimeObjectTracer {
    /// Trace all references in an object.
    ///
    /// Reads the object header to determine the type, then dispatches
    /// to the appropriate trace function.
    ///
    /// # Safety
    /// - `ptr` must point to a valid, live GC object with an `ObjectHeader`
    unsafe fn trace_object(&self, ptr: *const (), tracer: &mut dyn Tracer) {
        if ptr.is_null() {
            return;
        }

        // SAFETY: Caller guarantees ptr points to valid object with header
        let header = unsafe { &*(ptr as *const ObjectHeader) };
        let type_id = header.type_id;

        // Calculate pointer to object data (after header)
        // For now, we pass the header pointer and let the dispatch handle offset
        // Most of our objects include the header as the first field
        // SAFETY: Caller guarantees valid object
        unsafe {
            trace_object(ptr, type_id, tracer);
        }
    }

    /// Get the size of an object.
    ///
    /// Reads the object header to determine the type, then returns
    /// the size of that type.
    ///
    /// # Safety
    /// - `ptr` must point to a valid, live GC object with an `ObjectHeader`
    unsafe fn size_of_object(&self, ptr: *const ()) -> usize {
        if ptr.is_null() {
            return 0;
        }

        // SAFETY: Caller guarantees ptr points to valid object with header
        let header = unsafe { &*(ptr as *const ObjectHeader) };
        let type_id = header.type_id;

        // SAFETY: Caller guarantees valid object
        unsafe { size_of_object(ptr, type_id) }
    }

    /// Finalize (drop) an object.
    ///
    /// Reads the object header to determine the type, then calls
    /// the appropriate finalize function.
    ///
    /// # Safety
    /// - `ptr` must point to a valid, live GC object with an `ObjectHeader`
    /// - Must be called exactly once before memory is reclaimed
    unsafe fn finalize_object(&self, ptr: *mut ()) {
        if ptr.is_null() {
            return;
        }

        // SAFETY: Caller guarantees ptr points to valid object with header
        let header = unsafe { &*(ptr as *const ObjectHeader) };
        let type_id = header.type_id;

        // SAFETY: Caller guarantees valid object
        unsafe {
            finalize_object(ptr, type_id);
        }
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gc_dispatch::init_gc_dispatch;
    use crate::types::list::ListObject;
    use prism_core::Value;
    use std::mem;

    #[test]
    fn test_runtime_object_tracer_creation() {
        let tracer = RuntimeObjectTracer::new();
        // Should be zero-sized
        assert_eq!(mem::size_of_val(&tracer), 0);
    }

    #[test]
    fn test_trace_list_through_tracer() {
        init_gc_dispatch();

        let tracer = RuntimeObjectTracer::new();
        let mut list = ListObject::new();
        list.push(Value::int(42).unwrap());

        struct CountingTracer {
            value_count: usize,
        }
        impl Tracer for CountingTracer {
            fn trace_value(&mut self, _value: Value) {
                self.value_count += 1;
            }
            fn trace_ptr(&mut self, _ptr: *const ()) {}
        }

        let mut counting = CountingTracer { value_count: 0 };

        // Get pointer to list (the header is embedded in the ListObject)
        let ptr = &list as *const ListObject as *const ();

        unsafe {
            tracer.trace_object(ptr, &mut counting);
        }

        // Should trace 1 value
        assert_eq!(counting.value_count, 1);
    }

    #[test]
    fn test_size_through_tracer() {
        init_gc_dispatch();

        let tracer = RuntimeObjectTracer::new();
        let list = ListObject::new();

        let ptr = &list as *const ListObject as *const ();

        let size = unsafe { tracer.size_of_object(ptr) };
        assert_eq!(size, mem::size_of::<ListObject>());
    }

    #[test]
    fn test_null_pointer_handling() {
        let tracer = RuntimeObjectTracer::new();

        struct PanicTracer;
        impl Tracer for PanicTracer {
            fn trace_value(&mut self, _value: Value) {
                panic!("Should not be called");
            }
            fn trace_ptr(&mut self, _ptr: *const ()) {
                panic!("Should not be called");
            }
        }

        let mut panic_tracer = PanicTracer;

        // Should handle null pointer gracefully
        unsafe {
            tracer.trace_object(std::ptr::null(), &mut panic_tracer);
            assert_eq!(tracer.size_of_object(std::ptr::null()), 0);
        }
    }

    #[test]
    fn test_tracer_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<RuntimeObjectTracer>();
    }
}
