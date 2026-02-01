//! Write barrier integration for the Prism runtime.
//!
//! This module provides write barrier utilities for container types
//! to maintain the generational GC invariant:
//!
//! > No old-generation object may point to a young-generation object
//! > without that reference being tracked.
//!
//! # Architecture
//!
//! Write barriers are placed after every pointer store operation that
//! could create an old→young reference:
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────┐
//! │                     WRITE BARRIER FLOW                               │
//! │                                                                     │
//! │   ┌─────────────────────────────────────────────────────────────┐   │
//! │   │ 1. Store value to field                                      │   │
//! │   │    obj.items[i] = new_value;                                │   │
//! │   └─────────────────────────────────────────────────────────────┘   │
//! │                          │                                         │
//! │                          ▼                                         │
//! │   ┌─────────────────────────────────────────────────────────────┐   │
//! │   │ 2. Write barrier (fast path check)                          │   │
//! │   │    if holder.is_old() && new_value.is_young() {             │   │
//! │   │        mark_card_dirty(holder);                             │   │
//! │   │    }                                                        │   │
//! │   └─────────────────────────────────────────────────────────────┘   │
//! └─────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Usage
//!
//! Container types should call `write_barrier` after storing object references:
//!
//! ```ignore
//! use prism_runtime::gc_barrier::write_barrier;
//!
//! fn push(&mut self, heap: &GcHeap, value: Value) {
//!     self.items.push(value);
//!     write_barrier(heap, self.as_ptr(), value);
//! }
//! ```
//!
//! # Performance
//!
//! The fast path (young holder or non-object value) is a single branch.
//! Barriers are inline and optimized for minimal overhead on the hot path.

use prism_core::Value;
use prism_gc::barrier::{write_barrier, write_barrier_ptr};
use prism_gc::heap::GcHeap;

// Re-export barrier functions from prism_gc for convenience
pub use prism_gc::barrier::write_barrier as gc_write_barrier;
pub use prism_gc::barrier::write_barrier_ptr as gc_write_barrier_ptr;

/// Write barrier for storing a value into a container.
///
/// Call this after storing a value that may contain an object reference.
///
/// # Arguments
/// - `heap`: The GC heap
/// - `holder`: Pointer to the object containing the field
/// - `value`: The value that was stored
///
/// # Example
/// ```ignore
/// list.items.push(value);
/// container_write_barrier(heap, list as *const _, value);
/// ```
#[inline(always)]
pub fn container_write_barrier(heap: &GcHeap, holder: *const (), value: Value) {
    write_barrier(heap, holder, value);
}

/// Write barrier for storing multiple values into a container.
///
/// Call this after batch operations like extend, slice assignment, etc.
///
/// # Arguments
/// - `heap`: The GC heap
/// - `holder`: Pointer to the object containing the fields
/// - `values`: Slice of values that were stored
#[inline]
pub fn container_write_barrier_slice(heap: &GcHeap, holder: *const (), values: &[Value]) {
    for &value in values {
        write_barrier(heap, holder, value);
    }
}

/// Write barrier for storing a raw pointer into a container.
///
/// Use this when storing object pointers directly (not as Value).
///
/// # Arguments
/// - `heap`: The GC heap  
/// - `holder`: Pointer to the object containing the field
/// - `new_ptr`: The object pointer that was stored
#[inline(always)]
pub fn container_write_barrier_ptr(heap: &GcHeap, holder: *const (), new_ptr: *const ()) {
    write_barrier_ptr(heap, holder, new_ptr);
}

/// Trait for types that need write barriers on mutation.
///
/// Container types can implement this trait to provide a standard
/// interface for barrier-aware mutations.
pub trait BarrierMut {
    /// Get a pointer to this object for barrier purposes.
    fn barrier_self_ptr(&self) -> *const ();

    /// Called after storing a value reference.
    ///
    /// Default implementation delegates to `container_write_barrier`.
    #[inline]
    fn after_store(&self, heap: &GcHeap, value: Value) {
        container_write_barrier(heap, self.barrier_self_ptr(), value);
    }

    /// Called after storing multiple value references.
    #[inline]
    fn after_store_slice(&self, heap: &GcHeap, values: &[Value]) {
        container_write_barrier_slice(heap, self.barrier_self_ptr(), values);
    }
}

/// Macro to emit a write barrier after a container mutation.
///
/// # Example
/// ```ignore
/// barrier_after_store!(heap, list_obj, new_value);
/// ```
#[macro_export]
macro_rules! barrier_after_store {
    ($heap:expr, $holder:expr, $value:expr) => {
        $crate::gc_barrier::container_write_barrier($heap, $holder as *const _ as *const (), $value)
    };
}

/// Macro to emit write barriers for a slice of values.
#[macro_export]
macro_rules! barrier_after_store_slice {
    ($heap:expr, $holder:expr, $values:expr) => {
        $crate::gc_barrier::container_write_barrier_slice(
            $heap,
            $holder as *const _ as *const (),
            $values,
        )
    };
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use prism_gc::config::GcConfig;

    #[test]
    fn test_container_write_barrier_no_panic() {
        let heap = GcHeap::new(GcConfig::default());

        // Should not panic with null pointers or non-object values
        container_write_barrier(&heap, std::ptr::null(), Value::none());
        container_write_barrier(&heap, std::ptr::null(), Value::int(42).unwrap());
        container_write_barrier(&heap, std::ptr::null(), Value::bool(true));
    }

    #[test]
    fn test_container_write_barrier_slice() {
        let heap = GcHeap::new(GcConfig::default());

        let values = [
            Value::int(1).unwrap(),
            Value::int(2).unwrap(),
            Value::none(),
        ];

        // Should not panic
        container_write_barrier_slice(&heap, std::ptr::null(), &values);
    }

    #[test]
    fn test_container_write_barrier_ptr() {
        let heap = GcHeap::new(GcConfig::default());

        // Should not panic
        container_write_barrier_ptr(&heap, std::ptr::null(), std::ptr::null());
    }

    #[test]
    fn test_barrier_trait() {
        struct TestContainer {
            #[allow(dead_code)]
            value: Value,
        }

        impl BarrierMut for TestContainer {
            fn barrier_self_ptr(&self) -> *const () {
                self as *const _ as *const ()
            }
        }

        let heap = GcHeap::new(GcConfig::default());
        let container = TestContainer {
            value: Value::none(),
        };

        // Should not panic
        container.after_store(&heap, Value::int(42).unwrap());
    }

    #[test]
    fn test_barrier_macros() {
        let heap = GcHeap::new(GcConfig::default());
        let dummy = 42u64; // Just need a holder address

        barrier_after_store!(&heap, &dummy, Value::int(1).unwrap());
        barrier_after_store_slice!(&heap, &dummy, &[Value::int(2).unwrap()]);
    }
}
