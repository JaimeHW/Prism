use super::*;
use crate::allocation_context::RuntimeHeapBinding;
use prism_core::Value;
use prism_gc::heap::GcHeap;

#[test]
fn test_unsigned_byte_view_indexes_and_slices() {
    let view = MemoryViewObject::from_bytes(Value::none(), b"abcdef", true);

    assert_eq!(view.len(), 6);
    assert_eq!(view.ndim(), 1);
    assert_eq!(view.shape(), &[6]);
    assert_eq!(view.strides(), vec![1]);
    assert_eq!(view.nbytes(), 6);
    assert_eq!(view.get(1).and_then(|value| value.as_int()), Some(98));

    let sliced = view.slice(&SliceObject::new(Some(1), Some(6), Some(2)));
    assert_eq!(sliced.as_bytes(), b"bdf");
}

#[test]
fn test_cast_native_unsigned_int_decodes_elements() {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&7_u32.to_ne_bytes());
    bytes.extend_from_slice(&11_u32.to_ne_bytes());
    let view = MemoryViewObject::from_bytes(Value::none(), &bytes, true)
        .cast(MemoryViewFormat::UnsignedInt)
        .expect("byte length is divisible by u32 itemsize");

    assert_eq!(view.format_str(), "I");
    assert_eq!(view.item_size(), 4);
    assert_eq!(view.len(), 2);
    assert_eq!(view.get(0).and_then(|value| value.as_int()), Some(7));
    assert_eq!(view.get(-1).and_then(|value| value.as_int()), Some(11));
}

#[test]
fn test_char_view_decodes_bytes_into_bound_heap() {
    let heap = GcHeap::with_defaults();
    let _binding = RuntimeHeapBinding::register(&heap);
    let view = MemoryViewObject::from_bytes(Value::none(), b"abc", true)
        .cast(MemoryViewFormat::Char)
        .expect("byte length is divisible by char itemsize");

    let value = view.get(0).expect("char view should decode one-byte bytes");
    let ptr = value
        .as_object_ptr()
        .expect("char view should decode to a bytes object");
    let bytes = unsafe { &*(ptr as *const BytesObject) };

    assert!(heap.contains(ptr));
    assert_eq!(bytes.as_bytes(), b"a");
}

#[test]
fn test_cast_with_shape_records_multidimensional_metadata() {
    let view = MemoryViewObject::from_bytes(Value::none(), b"abcd", true)
        .cast_with_shape(MemoryViewFormat::UnsignedByte, Some(vec![2, 2]))
        .expect("shape product matches byte length");

    assert_eq!(view.ndim(), 2);
    assert_eq!(view.shape(), &[2, 2]);
    assert_eq!(view.strides(), vec![2, 1]);
    assert_eq!(view.as_bytes(), b"abcd");
}

#[test]
fn test_released_flag_is_mutable() {
    let mut view = MemoryViewObject::from_bytes(Value::none(), b"x", true);
    assert!(!view.released());
    view.release();
    assert!(view.released());
}
