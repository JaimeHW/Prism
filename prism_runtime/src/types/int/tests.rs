use super::*;
use crate::allocation_context::{
    RuntimeHeapBinding, current_heap_binding_depth, standalone_allocation_count,
};
use prism_gc::config::GcConfig;
use prism_gc::heap::GcHeap;
use std::alloc::Layout;

#[test]
fn test_bigint_to_value_keeps_small_ints_inline() {
    let value = bigint_to_value(BigInt::from(42_i64));
    assert_eq!(value.as_int(), Some(42));
    assert!(value_as_heap_int(value).is_none());
}

#[test]
fn test_bigint_to_value_promotes_large_values() {
    let big = BigInt::from(1_u8) << 100_u32;
    let value = bigint_to_value(big.clone());

    let obj = value_as_heap_int(value).expect("large integer should allocate");
    assert_eq!(obj.value(), &big);
    assert_eq!(value_to_bigint(value), Some(big));
}

#[test]
fn test_int_value_to_string_formats_heap_backed_values() {
    let big = BigInt::from(1_u8) << 90_u32;
    let value = bigint_to_value(big.clone());
    assert_eq!(int_value_to_string(value), Some(big.to_string()));
}

#[test]
fn test_bigint_to_value_uses_bound_vm_heap_when_available() {
    assert_eq!(current_heap_binding_depth(), 0);
    let heap = GcHeap::new(GcConfig::default());
    let _binding = RuntimeHeapBinding::register(&heap);

    let baseline = standalone_allocation_count();
    let big = BigInt::from(1_u8) << 100_u32;
    let value = bigint_to_value(big.clone());

    assert_eq!(standalone_allocation_count(), baseline);
    assert!(heap.contains(value.as_object_ptr().expect("bigint should allocate")));
    let obj = value_as_heap_int(value).expect("bound heap should allocate managed int");
    assert_eq!(obj.value(), &big);
}

#[test]
fn test_bigint_to_value_uses_tenured_after_nursery_exhaustion() {
    let heap = GcHeap::new(GcConfig {
        nursery_size: 64 * 1024,
        minor_gc_trigger: 64 * 1024,
        large_object_threshold: 128 * 1024,
        ..GcConfig::default()
    });
    heap.alloc_layout(Layout::from_size_align(64 * 1024 - 8, 8).unwrap())
        .expect("nursery filler allocation should succeed");

    let _binding = RuntimeHeapBinding::register(&heap);
    let baseline = standalone_allocation_count();

    let big = BigInt::from(1_u8) << 100_u32;
    let value = bigint_to_value(big.clone());
    assert_eq!(standalone_allocation_count(), baseline);
    assert!(heap.contains(value.as_object_ptr().expect("bigint should allocate")));
    let obj = value_as_heap_int(value).expect("large integer should survive exhaustion");
    assert_eq!(obj.value(), &big);
}

#[test]
fn test_value_to_bigint_reads_int_subclass_native_storage() {
    let big = BigInt::from(1_u8) << 88_u32;
    let object = ShapedObject::new_int_backed(
        TypeId::from_raw(512),
        crate::object::shape::Shape::empty(),
        big.clone(),
    );
    let ptr = Box::into_raw(Box::new(object));
    let value = Value::object_ptr(ptr as *const ());

    assert!(is_int_value(value));
    assert_eq!(value_to_bigint(value), Some(big.clone()));
    assert_eq!(int_value_to_string(value), Some(big.to_string()));

    unsafe { drop(Box::from_raw(ptr)) };
}
