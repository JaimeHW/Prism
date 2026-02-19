//! GC object dispatch system.
//!
//! This module provides high-performance type dispatch for garbage collection,
//! enabling the collector to trace objects without knowing their concrete types.
//!
//! # Architecture
//!
//! The dispatch system uses a static function pointer table indexed by `TypeId`
//! for O(1) dispatch without virtual call overhead:
//!
//! ```text
//! ┌──────────────────────────────────────────────────────────────────────────┐
//! │                     GC DISPATCH TABLE                                     │
//! │  ┌────────────────────────────────────────────────────────────────────┐  │
//! │  │ TypeId │ trace_fn          │ size_fn           │ finalize_fn       │  │
//! │  ├────────┼───────────────────┼───────────────────┼───────────────────┤  │
//! │  │   4    │ trace_string      │ size_string       │ finalize_string   │  │
//! │  │   6    │ trace_list        │ size_list         │ finalize_list     │  │
//! │  │   7    │ trace_tuple       │ size_tuple        │ finalize_tuple    │  │
//! │  │   8    │ trace_dict        │ size_dict         │ finalize_dict     │  │
//! │  │  ...   │ ...               │ ...               │ ...               │  │
//! │  └────────┴───────────────────┴───────────────────┴───────────────────┘  │
//! └──────────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Thread Safety
//!
//! The dispatch table is initialized once at startup (via `init_gc_dispatch`)
//! and read-only thereafter, making it safe for concurrent access during GC.

use crate::object::type_obj::TypeId;
use crate::types::bytes::BytesObject;
use crate::types::dict::DictObject;
use crate::types::function::FunctionObject;
use crate::types::iter::IteratorObject;
use crate::types::list::ListObject;
use crate::types::range::RangeObject;
use crate::types::set::SetObject;
use crate::types::string::StringObject;
use crate::types::tuple::TupleObject;
use prism_gc::Trace;
use prism_gc::trace::Tracer;
use std::mem;
use std::sync::OnceLock;

// =============================================================================
// Type Dispatch Entry
// =============================================================================

/// Function pointer type for tracing an object.
///
/// # Safety
/// - `ptr` must point to a valid object of the type this function handles
/// - The object must remain valid for the duration of the trace
pub type TraceFn = unsafe fn(ptr: *const (), tracer: &mut dyn Tracer);

/// Function pointer type for getting object size.
///
/// # Safety
/// - `ptr` must point to a valid object of the type this function handles
pub type SizeFn = unsafe fn(ptr: *const ()) -> usize;

/// Function pointer type for finalizing (dropping) an object.
///
/// # Safety
/// - `ptr` must point to a valid object of the type this function handles
/// - Must be called exactly once before memory is reclaimed
pub type FinalizeFn = unsafe fn(ptr: *mut ());

/// Dispatch entry for a single type.
///
/// Each entry provides the function pointers needed for GC operations.
#[derive(Clone, Copy)]
pub struct DispatchEntry {
    /// Trace all child references.
    pub trace: TraceFn,
    /// Get the object's allocation size.
    pub size: SizeFn,
    /// Finalize (drop) the object.
    pub finalize: FinalizeFn,
}

// =============================================================================
// Dispatch Table
// =============================================================================

/// Maximum number of built-in types (matches TypeId::FIRST_USER_TYPE).
const MAX_BUILTIN_TYPES: usize = 256;

/// Static dispatch table for built-in types.
///
/// Indexed by `TypeId.raw()` for O(1) lookup.
/// Entries for unused type IDs are set to no-op functions.
struct DispatchTable {
    entries: [DispatchEntry; MAX_BUILTIN_TYPES],
}

impl DispatchTable {
    /// Create a new dispatch table with all entries initialized to no-ops.
    const fn new() -> Self {
        Self {
            entries: [DispatchEntry {
                trace: trace_noop,
                size: size_noop,
                finalize: finalize_noop,
            }; MAX_BUILTIN_TYPES],
        }
    }

    /// Register an entry for a type.
    fn register(&mut self, type_id: TypeId, entry: DispatchEntry) {
        let idx = type_id.raw() as usize;
        if idx < MAX_BUILTIN_TYPES {
            self.entries[idx] = entry;
        }
    }

    /// Look up an entry by type ID.
    #[inline(always)]
    fn get(&self, type_id: TypeId) -> &DispatchEntry {
        let idx = type_id.raw() as usize;
        if idx < MAX_BUILTIN_TYPES {
            &self.entries[idx]
        } else {
            // For user types, fall back to no-op
            // A real implementation would use a secondary hashmap
            &self.entries[0]
        }
    }
}

// =============================================================================
// Global Dispatch Table
// =============================================================================

/// Global dispatch table singleton.
static DISPATCH_TABLE: OnceLock<DispatchTable> = OnceLock::new();

/// Get the global dispatch table.
#[inline(always)]
fn dispatch_table() -> &'static DispatchTable {
    DISPATCH_TABLE.get_or_init(init_dispatch_table)
}

/// Initialize the dispatch table with all built-in types.
fn init_dispatch_table() -> DispatchTable {
    let mut table = DispatchTable::new();

    // Register all built-in types
    table.register(
        TypeId::STR,
        DispatchEntry {
            trace: trace_string,
            size: size_string,
            finalize: finalize_string,
        },
    );

    table.register(
        TypeId::BYTES,
        DispatchEntry {
            trace: trace_bytes,
            size: size_bytes,
            finalize: finalize_bytes,
        },
    );

    table.register(
        TypeId::BYTEARRAY,
        DispatchEntry {
            trace: trace_bytes,
            size: size_bytes,
            finalize: finalize_bytes,
        },
    );

    table.register(
        TypeId::LIST,
        DispatchEntry {
            trace: trace_list,
            size: size_list,
            finalize: finalize_list,
        },
    );

    table.register(
        TypeId::TUPLE,
        DispatchEntry {
            trace: trace_tuple,
            size: size_tuple,
            finalize: finalize_tuple,
        },
    );

    table.register(
        TypeId::DICT,
        DispatchEntry {
            trace: trace_dict,
            size: size_dict,
            finalize: finalize_dict,
        },
    );

    table.register(
        TypeId::SET,
        DispatchEntry {
            trace: trace_set,
            size: size_set,
            finalize: finalize_set,
        },
    );

    table.register(
        TypeId::FUNCTION,
        DispatchEntry {
            trace: trace_function,
            size: size_function,
            finalize: finalize_function,
        },
    );

    table.register(
        TypeId::RANGE,
        DispatchEntry {
            trace: trace_range,
            size: size_range,
            finalize: finalize_range,
        },
    );

    table.register(
        TypeId::ITERATOR,
        DispatchEntry {
            trace: trace_iterator,
            size: size_iterator,
            finalize: finalize_iterator,
        },
    );

    table
}

/// Initialize GC dispatch.
///
/// Call this once at runtime startup to ensure the dispatch table is ready.
pub fn init_gc_dispatch() {
    let _ = dispatch_table();
}

// =============================================================================
// Public API
// =============================================================================

/// Trace an object given its pointer and type ID.
///
/// # Safety
/// - `ptr` must point to a valid, live object
/// - The `type_id` must match the actual type of the object
#[inline]
pub unsafe fn trace_object(ptr: *const (), type_id: TypeId, tracer: &mut dyn Tracer) {
    let entry = dispatch_table().get(type_id);
    // SAFETY: Caller guarantees ptr points to valid object of correct type
    unsafe { (entry.trace)(ptr, tracer) };
}

/// Get the size of an object given its pointer and type ID.
///
/// # Safety
/// - `ptr` must point to a valid, live object
/// - The `type_id` must match the actual type of the object
#[inline]
pub unsafe fn size_of_object(ptr: *const (), type_id: TypeId) -> usize {
    let entry = dispatch_table().get(type_id);
    // SAFETY: Caller guarantees ptr points to valid object of correct type
    unsafe { (entry.size)(ptr) }
}

/// Finalize (drop) an object given its pointer and type ID.
///
/// # Safety
/// - `ptr` must point to a valid, live object
/// - The `type_id` must match the actual type of the object
/// - Must be called exactly once before memory is reclaimed
#[inline]
pub unsafe fn finalize_object(ptr: *mut (), type_id: TypeId) {
    let entry = dispatch_table().get(type_id);
    // SAFETY: Caller guarantees ptr points to valid object of correct type
    unsafe { (entry.finalize)(ptr) };
}

// =============================================================================
// No-op Functions (for primitive/untraced types)
// =============================================================================

/// No-op trace function for leaf types.
unsafe fn trace_noop(_ptr: *const (), _tracer: &mut dyn Tracer) {
    // Nothing to trace
}

/// No-op size function (returns 0).
unsafe fn size_noop(_ptr: *const ()) -> usize {
    0
}

/// No-op finalize function.
unsafe fn finalize_noop(_ptr: *mut ()) {
    // Nothing to finalize
}

// =============================================================================
// StringObject Dispatch
// =============================================================================

unsafe fn trace_string(ptr: *const (), tracer: &mut dyn Tracer) {
    // SAFETY: Caller guarantees ptr points to valid StringObject
    let obj = unsafe { &*(ptr as *const StringObject) };
    obj.trace(tracer);
}

unsafe fn size_string(_ptr: *const ()) -> usize {
    mem::size_of::<StringObject>()
}

unsafe fn finalize_string(ptr: *mut ()) {
    // SAFETY: Caller guarantees ptr points to valid StringObject
    unsafe { std::ptr::drop_in_place(ptr as *mut StringObject) };
}

// =============================================================================
// BytesObject Dispatch
// =============================================================================

unsafe fn trace_bytes(ptr: *const (), tracer: &mut dyn Tracer) {
    // SAFETY: Caller guarantees ptr points to valid BytesObject
    let obj = unsafe { &*(ptr as *const BytesObject) };
    obj.trace(tracer);
}

unsafe fn size_bytes(ptr: *const ()) -> usize {
    // SAFETY: Caller guarantees ptr points to valid BytesObject
    let obj = unsafe { &*(ptr as *const BytesObject) };
    std::mem::size_of::<BytesObject>() + obj.len()
}

unsafe fn finalize_bytes(ptr: *mut ()) {
    // SAFETY: Caller guarantees ptr points to valid BytesObject
    unsafe { std::ptr::drop_in_place(ptr as *mut BytesObject) };
}

// =============================================================================
// ListObject Dispatch
// =============================================================================

unsafe fn trace_list(ptr: *const (), tracer: &mut dyn Tracer) {
    // SAFETY: Caller guarantees ptr points to valid ListObject
    let obj = unsafe { &*(ptr as *const ListObject) };
    obj.trace(tracer);
}

unsafe fn size_list(_ptr: *const ()) -> usize {
    mem::size_of::<ListObject>()
}

unsafe fn finalize_list(ptr: *mut ()) {
    // SAFETY: Caller guarantees ptr points to valid ListObject
    unsafe { std::ptr::drop_in_place(ptr as *mut ListObject) };
}

// =============================================================================
// TupleObject Dispatch
// =============================================================================

unsafe fn trace_tuple(ptr: *const (), tracer: &mut dyn Tracer) {
    // SAFETY: Caller guarantees ptr points to valid TupleObject
    let obj = unsafe { &*(ptr as *const TupleObject) };
    obj.trace(tracer);
}

unsafe fn size_tuple(_ptr: *const ()) -> usize {
    mem::size_of::<TupleObject>()
}

unsafe fn finalize_tuple(ptr: *mut ()) {
    // SAFETY: Caller guarantees ptr points to valid TupleObject
    unsafe { std::ptr::drop_in_place(ptr as *mut TupleObject) };
}

// =============================================================================
// DictObject Dispatch
// =============================================================================

unsafe fn trace_dict(ptr: *const (), tracer: &mut dyn Tracer) {
    // SAFETY: Caller guarantees ptr points to valid DictObject
    let obj = unsafe { &*(ptr as *const DictObject) };
    obj.trace(tracer);
}

unsafe fn size_dict(_ptr: *const ()) -> usize {
    mem::size_of::<DictObject>()
}

unsafe fn finalize_dict(ptr: *mut ()) {
    // SAFETY: Caller guarantees ptr points to valid DictObject
    unsafe { std::ptr::drop_in_place(ptr as *mut DictObject) };
}

// =============================================================================
// SetObject Dispatch
// =============================================================================

unsafe fn trace_set(ptr: *const (), tracer: &mut dyn Tracer) {
    // SAFETY: Caller guarantees ptr points to valid SetObject
    let obj = unsafe { &*(ptr as *const SetObject) };
    obj.trace(tracer);
}

unsafe fn size_set(_ptr: *const ()) -> usize {
    mem::size_of::<SetObject>()
}

unsafe fn finalize_set(ptr: *mut ()) {
    // SAFETY: Caller guarantees ptr points to valid SetObject
    unsafe { std::ptr::drop_in_place(ptr as *mut SetObject) };
}

// =============================================================================
// FunctionObject Dispatch
// =============================================================================

unsafe fn trace_function(ptr: *const (), tracer: &mut dyn Tracer) {
    // SAFETY: Caller guarantees ptr points to valid FunctionObject
    let obj = unsafe { &*(ptr as *const FunctionObject) };
    obj.trace(tracer);
}

unsafe fn size_function(_ptr: *const ()) -> usize {
    mem::size_of::<FunctionObject>()
}

unsafe fn finalize_function(ptr: *mut ()) {
    // SAFETY: Caller guarantees ptr points to valid FunctionObject
    unsafe { std::ptr::drop_in_place(ptr as *mut FunctionObject) };
}

// =============================================================================
// RangeObject Dispatch
// =============================================================================

unsafe fn trace_range(ptr: *const (), tracer: &mut dyn Tracer) {
    // SAFETY: Caller guarantees ptr points to valid RangeObject
    let obj = unsafe { &*(ptr as *const RangeObject) };
    obj.trace(tracer);
}

unsafe fn size_range(_ptr: *const ()) -> usize {
    mem::size_of::<RangeObject>()
}

unsafe fn finalize_range(ptr: *mut ()) {
    // SAFETY: Caller guarantees ptr points to valid RangeObject
    unsafe { std::ptr::drop_in_place(ptr as *mut RangeObject) };
}

// =============================================================================
// IteratorObject Dispatch
// =============================================================================

unsafe fn trace_iterator(ptr: *const (), tracer: &mut dyn Tracer) {
    // SAFETY: Caller guarantees ptr points to valid IteratorObject
    let obj = unsafe { &*(ptr as *const IteratorObject) };
    obj.trace(tracer);
}

unsafe fn size_iterator(_ptr: *const ()) -> usize {
    mem::size_of::<IteratorObject>()
}

unsafe fn finalize_iterator(ptr: *mut ()) {
    // SAFETY: Caller guarantees ptr points to valid IteratorObject
    unsafe { std::ptr::drop_in_place(ptr as *mut IteratorObject) };
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::bytes::BytesObject;
    use crate::types::list::ListObject;
    use prism_core::Value;

    #[test]
    fn test_dispatch_table_initialization() {
        init_gc_dispatch();

        // Verify all built-in types are registered
        let table = dispatch_table();

        // ListObject should have a valid entry
        let list_entry = table.get(TypeId::LIST);
        assert!(list_entry.size as usize != size_noop as usize);
    }

    #[test]
    fn test_trace_list_object() {
        init_gc_dispatch();

        let list = ListObject::new();
        let ptr = &list as *const ListObject as *const ();

        // Create a counting tracer
        struct CountingTracer {
            count: usize,
        }
        impl Tracer for CountingTracer {
            fn trace_value(&mut self, value: Value) {
                if value.as_object_ptr().is_some() {
                    self.count += 1;
                }
            }
            fn trace_ptr(&mut self, ptr: *const ()) {
                if !ptr.is_null() {
                    self.count += 1;
                }
            }
        }

        let mut tracer = CountingTracer { count: 0 };

        unsafe {
            trace_object(ptr, TypeId::LIST, &mut tracer);
        }

        // Empty list traces no children
        assert_eq!(tracer.count, 0);
    }

    #[test]
    fn test_size_of_list() {
        init_gc_dispatch();

        let list = ListObject::new();
        let ptr = &list as *const ListObject as *const ();

        let size = unsafe { size_of_object(ptr, TypeId::LIST) };
        assert_eq!(size, mem::size_of::<ListObject>());
    }

    #[test]
    fn test_size_of_dict() {
        init_gc_dispatch();

        let dict = DictObject::new();
        let ptr = &dict as *const DictObject as *const ();

        let size = unsafe { size_of_object(ptr, TypeId::DICT) };
        assert_eq!(size, mem::size_of::<DictObject>());
    }

    #[test]
    fn test_size_of_bytes() {
        init_gc_dispatch();

        let bytes = BytesObject::from_slice(b"hello");
        let ptr = &bytes as *const BytesObject as *const ();

        let size = unsafe { size_of_object(ptr, TypeId::BYTES) };
        assert_eq!(size, mem::size_of::<BytesObject>() + 5);
    }

    #[test]
    fn test_noop_for_unknown_type() {
        init_gc_dispatch();

        // Type ID 100 is not registered
        let entry = dispatch_table().get(TypeId(100));

        // Should return noop functions
        unsafe {
            let size = (entry.size)(std::ptr::null());
            assert_eq!(size, 0);
        }
    }

    #[test]
    fn test_dispatch_table_all_types() {
        init_gc_dispatch();

        let table = dispatch_table();

        // Verify known types have non-noop entries
        let type_ids = [
            TypeId::STR,
            TypeId::BYTES,
            TypeId::BYTEARRAY,
            TypeId::LIST,
            TypeId::TUPLE,
            TypeId::DICT,
            TypeId::SET,
            TypeId::FUNCTION,
            TypeId::RANGE,
            TypeId::ITERATOR,
        ];

        for type_id in type_ids {
            let entry = table.get(type_id);
            // Size function should not be noop
            assert!(
                entry.size as usize != size_noop as usize,
                "Type {:?} should have non-noop size function",
                type_id
            );
        }
    }

    #[test]
    fn test_trace_list_with_elements() {
        init_gc_dispatch();

        let mut list = ListObject::new();
        list.push(Value::int(42).unwrap());
        list.push(Value::bool(true));
        list.push(Value::none());

        let ptr = &list as *const ListObject as *const ();

        struct CountingTracer {
            value_count: usize,
        }
        impl Tracer for CountingTracer {
            fn trace_value(&mut self, _value: Value) {
                self.value_count += 1;
            }
            fn trace_ptr(&mut self, _ptr: *const ()) {}
        }

        let mut tracer = CountingTracer { value_count: 0 };

        unsafe {
            trace_object(ptr, TypeId::LIST, &mut tracer);
        }

        // Should trace 3 values
        assert_eq!(tracer.value_count, 3);
    }
}
