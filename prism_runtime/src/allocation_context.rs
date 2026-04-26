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

trait StandaloneObject: Trace {}

impl<T: Trace + 'static> StandaloneObject for T {}

#[derive(Clone, Copy)]
struct HeapBindingEntry {
    id: usize,
    heap: *const GcHeap,
}

thread_local! {
    static CURRENT_HEAP_BINDINGS: RefCell<Vec<HeapBindingEntry>> = const {
        RefCell::new(Vec::new())
    };
    static STANDALONE_ALLOCATIONS: RefCell<Vec<Box<dyn StandaloneObject>>> = const {
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

/// Allocate a traceable object in the bound VM heap when possible.
///
/// Runtime helpers outside a VM keep standalone allocations in a thread-local
/// owner so their pointers remain stable without leaking. When a VM heap is
/// bound, this function never falls back outside the managed heap: if nursery
/// allocation fails it uses the tenured space, preserving a single ownership
/// domain for VM-created objects.
#[inline]
pub fn alloc_value_in_current_heap_or_box<T: Trace + 'static>(value: T) -> Value {
    if !has_current_heap_binding() {
        return alloc_standalone_value(value);
    }

    let _value = match try_alloc_value_in_current_heap(value) {
        Ok(value) => return value,
        Err(value) => value,
    };

    panic!("managed heap allocation failed after nursery and tenured allocation attempts")
}

#[inline]
fn alloc_value_in_heap_or_value<T: Trace>(heap: &GcHeap, value: T) -> Result<Value, T> {
    let layout = Layout::new::<T>();
    let Some(ptr) = heap
        .alloc_layout(layout)
        .or_else(|| heap.alloc_tenured_layout(layout))
    else {
        return Err(value);
    };
    let typed_ptr = ptr.as_ptr() as *mut T;

    unsafe {
        std::ptr::write(typed_ptr, value);
    }

    Ok(Value::object_ptr(typed_ptr as *const ()))
}

#[inline]
fn alloc_standalone_value<T: Trace + 'static>(value: T) -> Value {
    let mut boxed = Box::new(value);
    let ptr = (&mut *boxed) as *mut T as *const ();
    STANDALONE_ALLOCATIONS.with(|allocations| {
        allocations.borrow_mut().push(boxed);
    });
    Value::object_ptr(ptr)
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
pub fn standalone_allocation_count() -> usize {
    STANDALONE_ALLOCATIONS.with(|allocations| allocations.borrow().len())
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
    fn alloc_value_in_current_heap_uses_tenured_after_nursery_exhaustion() {
        let config = prism_gc::GcConfig {
            nursery_size: 64 * 1024,
            minor_gc_trigger: 64 * 1024,
            ..prism_gc::GcConfig::default()
        };
        let heap = GcHeap::new(config);
        let _binding = RuntimeHeapBinding::register(&heap);

        while heap.alloc(1024).is_some() {}
        assert!(
            heap.should_minor_collect(),
            "test setup should exhaust the nursery"
        );

        let value = alloc_value_in_current_heap(DictObject::new())
            .expect("managed allocation should fall back to tenured space");
        let ptr = value
            .as_object_ptr()
            .expect("allocated value should be an object pointer");
        assert!(heap.contains(ptr));
        assert!(heap.is_old(ptr));
    }

    #[test]
    fn alloc_value_without_bound_heap_is_thread_local_owned() {
        let before = standalone_allocation_count();

        let value = alloc_value_in_current_heap_or_box(DictObject::new());
        let ptr = value
            .as_object_ptr()
            .expect("standalone allocation should return an object pointer");

        assert!(!ptr.is_null());
        assert_eq!(standalone_allocation_count(), before + 1);
    }

    #[test]
    fn try_alloc_value_in_current_heap_preserves_value_without_binding() {
        let value = DictObject::new();
        let recovered = try_alloc_value_in_current_heap(value)
            .expect_err("unbound allocation should return the original object");
        assert_eq!(recovered.len(), 0);
    }

    #[repr(align(32))]
    struct AlignedObject {
        value: u64,
    }

    unsafe impl Trace for AlignedObject {
        fn trace(&self, _tracer: &mut dyn prism_gc::Tracer) {}
    }

    #[test]
    fn typed_heap_allocation_preserves_rust_alignment() {
        let heap = GcHeap::with_defaults();
        let _binding = RuntimeHeapBinding::register(&heap);

        let value = alloc_value_in_current_heap_or_box(AlignedObject { value: 7 });
        let ptr = value
            .as_object_ptr()
            .expect("allocated value should be an object pointer");

        assert_eq!((ptr as usize) % std::mem::align_of::<AlignedObject>(), 0);
        unsafe {
            assert_eq!((*(ptr as *const AlignedObject)).value, 7);
        }
    }
}
