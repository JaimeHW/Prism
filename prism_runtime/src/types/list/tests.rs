use super::*;
use crate::object::shape::Shape;

#[test]
fn test_list_basic() {
    let mut list = ListObject::new();
    assert!(list.is_empty());

    list.push(Value::int(1).unwrap());
    list.push(Value::int(2).unwrap());
    list.push(Value::int(3).unwrap());

    assert_eq!(list.len(), 3);
    assert_eq!(list.get(0).unwrap().as_int(), Some(1));
    assert_eq!(list.get(1).unwrap().as_int(), Some(2));
    assert_eq!(list.get(2).unwrap().as_int(), Some(3));
}

#[test]
fn test_list_negative_index() {
    let list = ListObject::from_slice(&[
        Value::int(1).unwrap(),
        Value::int(2).unwrap(),
        Value::int(3).unwrap(),
    ]);

    assert_eq!(list.get(-1).unwrap().as_int(), Some(3));
    assert_eq!(list.get(-2).unwrap().as_int(), Some(2));
    assert_eq!(list.get(-3).unwrap().as_int(), Some(1));
    assert!(list.get(-4).is_none());
}

#[test]
fn test_list_pop() {
    let mut list = ListObject::from_slice(&[Value::int(1).unwrap(), Value::int(2).unwrap()]);

    assert_eq!(list.pop().unwrap().as_int(), Some(2));
    assert_eq!(list.pop().unwrap().as_int(), Some(1));
    assert!(list.pop().is_none());
}

#[test]
fn test_list_insert() {
    let mut list = ListObject::from_slice(&[Value::int(1).unwrap(), Value::int(3).unwrap()]);

    list.insert(1, Value::int(2).unwrap());
    assert_eq!(list.get(0).unwrap().as_int(), Some(1));
    assert_eq!(list.get(1).unwrap().as_int(), Some(2));
    assert_eq!(list.get(2).unwrap().as_int(), Some(3));
}

#[test]
fn test_list_repeat() {
    let list = ListObject::from_slice(&[Value::int(1).unwrap(), Value::int(2).unwrap()]);
    let repeated = list.repeat(3);

    assert_eq!(repeated.len(), 6);
    assert_eq!(
        repeated.as_slice(),
        &[
            Value::int_unchecked(1),
            Value::int_unchecked(2),
            Value::int_unchecked(1),
            Value::int_unchecked(2),
            Value::int_unchecked(1),
            Value::int_unchecked(2),
        ]
    );
}

#[test]
fn test_list_repeat_zero_returns_empty() {
    let list = ListObject::from_slice(&[Value::int(1).unwrap()]);
    let repeated = list.repeat(0);

    assert!(repeated.is_empty());
}

#[test]
fn test_assign_slice_replaces_contiguous_region() {
    let mut list = ListObject::from_slice(&[
        Value::int(0).unwrap(),
        Value::int(1).unwrap(),
        Value::int(2).unwrap(),
        Value::int(3).unwrap(),
    ]);

    list.assign_slice(
        &SliceObject::start_stop(1, 3),
        [Value::int(10).unwrap(), Value::int(11).unwrap()],
    )
    .expect("contiguous slice assignment should succeed");

    assert_eq!(
        list.as_slice(),
        &[
            Value::int_unchecked(0),
            Value::int_unchecked(10),
            Value::int_unchecked(11),
            Value::int_unchecked(3)
        ]
    );
}

#[test]
fn test_assign_slice_can_grow_list() {
    let mut list = ListObject::from_slice(&[
        Value::int(0).unwrap(),
        Value::int(1).unwrap(),
        Value::int(2).unwrap(),
    ]);

    list.assign_slice(
        &SliceObject::start_stop(1, 2),
        [
            Value::int(10).unwrap(),
            Value::int(11).unwrap(),
            Value::int(12).unwrap(),
        ],
    )
    .expect("slice assignment should allow growth");

    assert_eq!(
        list.as_slice(),
        &[
            Value::int_unchecked(0),
            Value::int_unchecked(10),
            Value::int_unchecked(11),
            Value::int_unchecked(12),
            Value::int_unchecked(2),
        ]
    );
}

#[test]
fn test_assign_slice_can_insert_into_empty_region() {
    let mut list = ListObject::from_slice(&[
        Value::int(0).unwrap(),
        Value::int(1).unwrap(),
        Value::int(2).unwrap(),
    ]);

    list.assign_slice(&SliceObject::start_stop(2, 1), [Value::int(99).unwrap()])
        .expect("empty-slice assignment should behave as insertion");

    assert_eq!(
        list.as_slice(),
        &[
            Value::int_unchecked(0),
            Value::int_unchecked(1),
            Value::int_unchecked(99),
            Value::int_unchecked(2),
        ]
    );
}

#[test]
fn test_assign_slice_replaces_extended_slice_in_place() {
    let mut list = ListObject::from_slice(&[
        Value::int(0).unwrap(),
        Value::int(1).unwrap(),
        Value::int(2).unwrap(),
        Value::int(3).unwrap(),
        Value::int(4).unwrap(),
        Value::int(5).unwrap(),
    ]);

    list.assign_slice(
        &SliceObject::full(0, 6, 2),
        [
            Value::int(10).unwrap(),
            Value::int(11).unwrap(),
            Value::int(12).unwrap(),
        ],
    )
    .expect("extended slice assignment should succeed when lengths match");

    assert_eq!(
        list.as_slice(),
        &[
            Value::int_unchecked(10),
            Value::int_unchecked(1),
            Value::int_unchecked(11),
            Value::int_unchecked(3),
            Value::int_unchecked(12),
            Value::int_unchecked(5),
        ]
    );
}

#[test]
fn test_assign_slice_rejects_mismatched_extended_slice_lengths() {
    let mut list = ListObject::from_slice(&[
        Value::int(0).unwrap(),
        Value::int(1).unwrap(),
        Value::int(2).unwrap(),
        Value::int(3).unwrap(),
    ]);

    let error = list
        .assign_slice(&SliceObject::full(0, 4, 2), [Value::int(10).unwrap()])
        .expect_err("extended slice assignment should validate replacement length");

    assert_eq!(
        error,
        ListSliceAssignError::ExtendedSliceSizeMismatch {
            expected: 2,
            actual: 1,
        }
    );
}

#[test]
fn test_delete_slice_removes_contiguous_region() {
    let mut list = ListObject::from_slice(&[
        Value::int(0).unwrap(),
        Value::int(1).unwrap(),
        Value::int(2).unwrap(),
        Value::int(3).unwrap(),
    ]);

    list.delete_slice(&SliceObject::start_stop(1, 3));

    assert_eq!(
        list.as_slice(),
        &[Value::int_unchecked(0), Value::int_unchecked(3)]
    );
}

#[test]
fn test_delete_slice_removes_extended_indices() {
    let mut list = ListObject::from_slice(&[
        Value::int(0).unwrap(),
        Value::int(1).unwrap(),
        Value::int(2).unwrap(),
        Value::int(3).unwrap(),
        Value::int(4).unwrap(),
        Value::int(5).unwrap(),
    ]);

    list.delete_slice(&SliceObject::full(0, 6, 2));

    assert_eq!(
        list.as_slice(),
        &[
            Value::int_unchecked(1),
            Value::int_unchecked(3),
            Value::int_unchecked(5),
        ]
    );
}

#[test]
fn test_object_ptr_as_list_ref_supports_heap_list_subclasses() {
    let object = Box::into_raw(Box::new(ShapedObject::new_list_backed(
        TypeId::from_raw(512),
        Shape::empty(),
    )));
    unsafe { &mut *object }
        .list_backing_mut()
        .expect("list backing should exist")
        .push(Value::int(5).unwrap());

    let list = object_ptr_as_list_ref(object as *const ())
        .expect("heap list subclass should expose native list storage");
    assert_eq!(list.as_slice(), &[Value::int(5).unwrap()]);

    unsafe {
        drop(Box::from_raw(object));
    }
}
