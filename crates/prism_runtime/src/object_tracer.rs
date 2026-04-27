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
