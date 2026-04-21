//! Thread-local access to the active VM-managed heap.
//!
//! Runtime helpers in `prism_runtime` frequently need to materialize heap
//! objects without having a direct `&VirtualMachine` parameter available.
//! During real VM execution we bind the current GC heap into thread-local
//! state so these helpers can still allocate into the managed heap instead of
//! falling back to process-lifetime storage.

use prism_core::Value;
use prism_gc::heap::GcHeap;
use prism_gc::trace::Trace;
use std::alloc::Layout;
use std::cell::RefCell;
use std::sync::atomic::{AtomicUsize, Ordering};

#[derive(Clone, Copy)]
struct HeapBindingEntry {
    id: usize,
    heap: *const GcHeap,
}

thread_local! {
    static CURRENT_HEAP_BINDINGS: RefCell<Vec<HeapBindingEntry>> = const {
        RefCell::new(Vec::new())
    };
}

static NEXT_BINDING_ID: AtomicUsize = AtomicUsize::new(1);

/// Registration handle for the active runtime heap binding.
///
/// While this handle is alive, allocations requested through
/// [`alloc_value_in_current_heap`] will target the bound heap on the current
/// thread.
pub struct RuntimeHeapBinding {
    id: usize,
}

impl RuntimeHeapBinding {
    /// Register a GC heap as the current runtime allocation target.
    pub fn register(heap: &GcHeap) -> Self {
        let id = NEXT_BINDING_ID.fetch_add(1, Ordering::Relaxed);
        CURRENT_HEAP_BINDINGS.with(|bindings| {
            bindings.borrow_mut().push(HeapBindingEntry {
                id,
                heap: heap as *const GcHeap,
            });
        });
        Self { id }
    }
}

impl Drop for RuntimeHeapBinding {
    fn drop(&mut self) {
        CURRENT_HEAP_BINDINGS.with(|bindings| {
            let mut bindings = bindings.borrow_mut();
            if let Some(index) = bindings.iter().rposition(|entry| entry.id == self.id) {
                bindings.remove(index);
            }
        });
    }
}

/// Allocate a traceable object in the currently bound VM heap, if one exists.
#[inline]
pub fn alloc_value_in_current_heap<T: Trace>(value: T) -> Option<Value> {
    CURRENT_HEAP_BINDINGS.with(|bindings| {
        let heap = bindings.borrow().last().copied()?;
        let heap = unsafe { &*heap.heap };
        alloc_value_in_heap(heap, value)
    })
}

#[inline]
fn alloc_value_in_heap<T: Trace>(heap: &GcHeap, value: T) -> Option<Value> {
    let layout = Layout::new::<T>();
    let size = layout.size().max(8);
    let ptr = heap.alloc(size)?;
    let typed_ptr = ptr.as_ptr() as *mut T;

    unsafe {
        std::ptr::write(typed_ptr, value);
    }

    Some(Value::object_ptr(typed_ptr as *const ()))
}

/// Returns true when the current thread has a managed runtime heap bound.
#[inline]
pub fn has_current_heap_binding() -> bool {
    CURRENT_HEAP_BINDINGS.with(|bindings| !bindings.borrow().is_empty())
}

#[cfg(test)]
pub fn current_heap_binding_depth() -> usize {
    CURRENT_HEAP_BINDINGS.with(|bindings| bindings.borrow().len())
}
