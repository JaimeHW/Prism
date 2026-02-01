//! High-performance Python runtime system for Prism.
//!
//! This crate provides:
//! - Python object model (ObjectHeader, TypeObject, slots)
//! - Core types (function, list, dict, tuple, etc.)
//! - Type registry for fast dispatch
//! - Garbage collection integration (Trace trait implementations)
//! - GC dispatch system for type-based tracing
//! - Runtime object tracer for GC integration
//! - Write barrier utilities for container mutations

#![deny(unsafe_op_in_unsafe_fn)]

pub mod gc;
pub mod gc_barrier;
pub mod gc_dispatch;
pub mod gc_trace;
pub mod object;
pub mod object_tracer;
pub mod types;

// Re-export commonly used items
pub use object::registry::{TypeRegistry, global_registry, init_builtin_types};
pub use object::type_obj::{TypeFlags, TypeId, TypeObject, TypeSlots};
pub use object::{GcColor, GcFlags, HASH_NOT_COMPUTED, ObjectHeader, PyObject};

// Re-export Trace trait from prism_gc for convenience
pub use prism_gc::trace::Trace;

// Re-export GC dispatch functions
pub use gc_dispatch::{finalize_object, init_gc_dispatch, size_of_object, trace_object};

// Re-export RuntimeObjectTracer
pub use object_tracer::RuntimeObjectTracer;

// Re-export barrier utilities
pub use gc_barrier::{
    BarrierMut, container_write_barrier, container_write_barrier_ptr, container_write_barrier_slice,
};
