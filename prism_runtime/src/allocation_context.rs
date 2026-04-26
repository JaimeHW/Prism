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
    try_alloc_value_in_current_heap(value).ok()
}

/// Allocate a traceable object in the current VM heap without consuming the
/// object when no bound heap has capacity.
#[inline]
pub fn try_alloc_value_in_current_heap<T: Trace>(value: T) -> Result<Value, T> {
    CURRENT_HEAP_BINDINGS.with(|bindings| {
        let Some(heap) = bindings.borrow().last().copied() else {
            return Err(value);
        };
        let heap = unsafe { &*heap.heap };
        alloc_value_in_heap_or_value(heap, value)
    })
}

/// Allocate a traceable object in the bound VM heap when possible, otherwise
/// fall back to a stable boxed pointer for standalone runtime helpers.
#[inline]
pub fn alloc_value_in_current_heap_or_box<T: Trace>(value: T) -> Value {
    let value = match try_alloc_value_in_current_heap(value) {
        Ok(value) => return value,
        Err(value) => value,
    };
    let ptr = Box::into_raw(Box::new(value)) as *const ();
    Value::object_ptr(ptr)
}

#[inline]
fn alloc_value_in_heap_or_value<T: Trace>(heap: &GcHeap, value: T) -> Result<Value, T> {
    let layout = Layout::new::<T>();
    let size = layout.size().max(8);
    let Some(ptr) = heap.alloc(size) else {
        return Err(value);
    };
    let typed_ptr = ptr.as_ptr() as *mut T;

    unsafe {
        std::ptr::write(typed_ptr, value);
    }

    Ok(Value::object_ptr(typed_ptr as *const ()))
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::dict::DictObject;

    #[test]
    fn alloc_value_in_current_heap_or_box_uses_bound_heap() {
        let heap = GcHeap::with_defaults();
        let _binding = RuntimeHeapBinding::register(&heap);

        let value = alloc_value_in_current_heap_or_box(DictObject::new());
        let ptr = value
            .as_object_ptr()
            .expect("allocated value should be an object pointer");

        assert!(heap.contains(ptr));
    }

    #[test]
    fn try_alloc_value_in_current_heap_preserves_value_on_exhaustion() {
        let config = prism_gc::GcConfig {
            nursery_size: 64 * 1024,
            minor_gc_trigger: 64 * 1024,
            ..prism_gc::GcConfig::default()
        };
        let heap = GcHeap::new(config);
        let _binding = RuntimeHeapBinding::register(&heap);

        let mut exhausted = false;
        for _ in 0..10_000 {
            if alloc_value_in_current_heap(DictObject::new()).is_none() {
                exhausted = true;
                break;
            }
        }
        assert!(exhausted, "tiny nursery should exhaust during the test");

        let value = DictObject::new();
        let recovered = try_alloc_value_in_current_heap(value)
            .expect_err("failed allocation should return the original object");
        assert_eq!(recovered.len(), 0);
    }
}
