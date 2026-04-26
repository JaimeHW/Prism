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
fn alloc_static_value_ignores_bound_heap() {
    let heap = GcHeap::with_defaults();
    let _binding = RuntimeHeapBinding::register(&heap);

    let value = alloc_static_value(DictObject::new());
    let ptr = value
        .as_object_ptr()
        .expect("static allocation should return an object pointer");

    assert!(!heap.contains(ptr));
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
