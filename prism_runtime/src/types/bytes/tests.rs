use super::*;
use crate::object::shape::Shape;

#[test]
fn test_new_bytes_empty() {
    let bytes = BytesObject::new();
    assert_eq!(bytes.header.type_id, TypeId::BYTES);
    assert!(bytes.is_empty());
    assert!(!bytes.is_bytearray());
}

#[test]
fn test_new_bytearray_empty() {
    let bytearray = BytesObject::new_bytearray();
    assert_eq!(bytearray.header.type_id, TypeId::BYTEARRAY);
    assert!(bytearray.is_empty());
    assert!(bytearray.is_bytearray());
}

#[test]
fn test_from_slice_and_len() {
    let bytes = BytesObject::from_slice(&[65, 66, 67]);
    assert_eq!(bytes.len(), 3);
    assert_eq!(bytes.as_bytes(), b"ABC");
}

#[test]
fn test_bytearray_from_slice_type() {
    let bytearray = BytesObject::bytearray_from_slice(&[1, 2, 3]);
    assert_eq!(bytearray.header.type_id, TypeId::BYTEARRAY);
    assert_eq!(bytearray.as_bytes(), &[1, 2, 3]);
}

#[test]
fn test_repeat_and_repeat_with_type() {
    let bytes = BytesObject::repeat(0, 5);
    assert_eq!(bytes.header.type_id, TypeId::BYTES);
    assert_eq!(bytes.as_bytes(), &[0, 0, 0, 0, 0]);

    let bytearray = BytesObject::repeat_with_type(255, 3, TypeId::BYTEARRAY);
    assert_eq!(bytearray.header.type_id, TypeId::BYTEARRAY);
    assert_eq!(bytearray.as_bytes(), &[255, 255, 255]);
}

#[test]
fn test_repeat_sequence_preserves_type_and_contents() {
    let bytes = BytesObject::from_slice(b"ab")
        .repeat_sequence(3)
        .expect("bytes repeat should not overflow");
    assert_eq!(bytes.header.type_id, TypeId::BYTES);
    assert_eq!(bytes.as_bytes(), b"ababab");

    let bytearray = BytesObject::bytearray_from_slice(b"xy")
        .repeat_sequence(2)
        .expect("bytearray repeat should not overflow");
    assert_eq!(bytearray.header.type_id, TypeId::BYTEARRAY);
    assert_eq!(bytearray.as_bytes(), b"xyxy");
}

#[test]
fn test_repeat_sequence_zero_count_returns_empty_same_type() {
    let bytes = BytesObject::from_slice(b"ab")
        .repeat_sequence(0)
        .expect("zero repeat should be representable");
    assert_eq!(bytes.header.type_id, TypeId::BYTES);
    assert!(bytes.is_empty());

    let bytearray = BytesObject::bytearray_from_slice(b"xy")
        .repeat_sequence(0)
        .expect("zero repeat should be representable");
    assert_eq!(bytearray.header.type_id, TypeId::BYTEARRAY);
    assert!(bytearray.is_empty());
}

#[test]
fn test_get_positive_and_negative_index() {
    let bytes = BytesObject::from_slice(&[10, 20, 30]);
    assert_eq!(bytes.get(0), Some(10));
    assert_eq!(bytes.get(2), Some(30));
    assert_eq!(bytes.get(-1), Some(30));
    assert_eq!(bytes.get(-3), Some(10));
}

#[test]
fn test_get_out_of_bounds() {
    let bytes = BytesObject::from_slice(&[10, 20, 30]);
    assert_eq!(bytes.get(3), None);
    assert_eq!(bytes.get(-4), None);
}

#[test]
fn test_iter() {
    let bytes = BytesObject::from_slice(&[1, 2, 3, 4]);
    let collected: Vec<u8> = bytes.iter().collect();
    assert_eq!(collected, vec![1, 2, 3, 4]);
}

#[test]
fn test_slice_preserves_bytes_type_and_supports_steps() {
    let bytes = BytesObject::from_slice(b"abcdef");
    let slice = SliceObject::full(0, 6, 2);
    let sliced = bytes.slice(&slice);

    assert_eq!(sliced.header.type_id, TypeId::BYTES);
    assert_eq!(sliced.as_bytes(), b"ace");
}

#[test]
fn test_slice_preserves_bytearray_type_and_supports_reverse() {
    let bytes = BytesObject::bytearray_from_slice(b"abcd");
    let slice = SliceObject::new(None, None, Some(-1));
    let sliced = bytes.slice(&slice);

    assert_eq!(sliced.header.type_id, TypeId::BYTEARRAY);
    assert_eq!(sliced.as_bytes(), b"dcba");
}

#[test]
fn test_concat_preserves_bytes_type() {
    let bytes = BytesObject::from_slice(b"abc");
    let result = bytes.concat(b"def");
    assert_eq!(result.header.type_id, TypeId::BYTES);
    assert_eq!(result.as_bytes(), b"abcdef");
}

#[test]
fn test_concat_preserves_bytearray_type() {
    let bytearray = BytesObject::bytearray_from_slice(b"abc");
    let result = bytearray.concat(b"def");
    assert_eq!(result.header.type_id, TypeId::BYTEARRAY);
    assert_eq!(result.as_bytes(), b"abcdef");
}

#[test]
fn test_push_immutable_bytes_rejected() {
    let mut bytes = BytesObject::from_slice(&[1, 2, 3]);
    assert!(!bytes.push(4));
    assert_eq!(bytes.as_bytes(), &[1, 2, 3]);
}

#[test]
fn test_push_mutable_bytearray() {
    let mut bytearray = BytesObject::new_bytearray();
    assert!(bytearray.push(1));
    assert!(bytearray.push(2));
    assert_eq!(bytearray.as_bytes(), &[1, 2]);
}

#[test]
fn test_extend_immutable_bytes_rejected() {
    let mut bytes = BytesObject::from_slice(&[1, 2, 3]);
    assert!(!bytes.extend_from_slice(&[4, 5]));
    assert_eq!(bytes.as_bytes(), &[1, 2, 3]);
}

#[test]
fn test_extend_mutable_bytearray() {
    let mut bytearray = BytesObject::bytearray_from_slice(&[1, 2]);
    assert!(bytearray.extend_from_slice(&[3, 4]));
    assert_eq!(bytearray.as_bytes(), &[1, 2, 3, 4]);
}

#[test]
fn test_set_immutable_bytes_rejected() {
    let mut bytes = BytesObject::from_slice(&[1, 2, 3]);
    assert!(!bytes.set(0, 9));
    assert_eq!(bytes.as_bytes(), &[1, 2, 3]);
}

#[test]
fn test_set_mutable_bytearray_positive_and_negative() {
    let mut bytearray = BytesObject::bytearray_from_slice(&[1, 2, 3]);
    assert!(bytearray.set(0, 9));
    assert!(bytearray.set(-1, 7));
    assert_eq!(bytearray.as_bytes(), &[9, 2, 7]);
}

#[test]
fn test_set_mutable_out_of_bounds() {
    let mut bytearray = BytesObject::bytearray_from_slice(&[1, 2, 3]);
    assert!(!bytearray.set(3, 9));
    assert!(!bytearray.set(-4, 9));
    assert_eq!(bytearray.as_bytes(), &[1, 2, 3]);
}

#[test]
fn test_assign_bytearray_contiguous_slice_can_resize() {
    let mut bytearray = BytesObject::bytearray_from_slice(b"abcdef");
    bytearray
        .assign_slice(&SliceObject::start_stop(2, 4), b"XYZ")
        .expect("contiguous bytearray slice assignment should resize");
    assert_eq!(bytearray.as_bytes(), b"abXYZef");

    bytearray
        .assign_slice(&SliceObject::new(Some(-5), None, None), b"12345")
        .expect("negative slice bounds should be normalized");
    assert_eq!(bytearray.as_bytes(), b"ab12345");
}

#[test]
fn test_assign_bytearray_extended_slice_requires_matching_length() {
    let mut bytearray = BytesObject::bytearray_from_slice(b"abcdef");
    bytearray
        .assign_slice(&SliceObject::full(0, 6, 2), b"XYZ")
        .expect("matching extended slice assignment should succeed");
    assert_eq!(bytearray.as_bytes(), b"XbYdZf");

    let err = bytearray
        .assign_slice(&SliceObject::full(0, 6, 2), b"12")
        .expect_err("extended bytearray slice size mismatch should fail");
    assert_eq!(
        err,
        ByteSliceAssignError::ExtendedSliceSizeMismatch {
            expected: 3,
            actual: 2
        }
    );
}

#[test]
fn test_clone_preserves_type_and_contents() {
    let original = BytesObject::bytearray_from_slice(&[1, 2, 3]);
    let cloned = original.clone();
    assert_eq!(cloned.header.type_id, TypeId::BYTEARRAY);
    assert_eq!(cloned.as_bytes(), &[1, 2, 3]);
}

#[test]
fn test_debug_includes_kind() {
    let bytes = BytesObject::from_slice(&[1, 2, 3]);
    let dbg = format!("{:?}", bytes);
    assert!(dbg.contains("bytes"));
}

#[test]
fn test_value_as_bytes_ref_reads_heap_subclass_native_storage() {
    let object = ShapedObject::new_bytes_backed(
        TypeId::from_raw(TypeId::FIRST_USER_TYPE + 50),
        Shape::empty(),
        BytesObject::from_slice(b"payload"),
    );
    let value = Value::object_ptr(Box::into_raw(Box::new(object)) as *const ());

    assert_eq!(
        value_as_bytes_ref(value)
            .expect("bytes backing should be visible")
            .as_bytes(),
        b"payload"
    );

    unsafe {
        drop(Box::from_raw(
            value.as_object_ptr().unwrap() as *mut ShapedObject
        ));
    }
}
