use super::*;
use crate::object::shape::Shape;
use crate::object::shaped_object::ShapedObject;

#[test]
fn test_tuple_basic() {
    let tuple = TupleObject::from_slice(&[
        Value::int(1).unwrap(),
        Value::int(2).unwrap(),
        Value::int(3).unwrap(),
    ]);

    assert_eq!(tuple.len(), 3);
    assert_eq!(tuple.get(0).unwrap().as_int(), Some(1));
    assert_eq!(tuple.get(1).unwrap().as_int(), Some(2));
    assert_eq!(tuple.get(2).unwrap().as_int(), Some(3));
}

#[test]
fn test_object_ptr_as_tuple_ref_supports_tuple_backed_objects() {
    let object = Box::into_raw(Box::new(ShapedObject::new_tuple_backed(
        TypeId::OBJECT,
        Shape::empty(),
        TupleObject::from_slice(&[Value::int(5).unwrap(), Value::int(8).unwrap()]),
    )));

    let tuple = object_ptr_as_tuple_ref(object as *const ())
        .expect("tuple-backed object should expose native tuple storage");
    assert_eq!(tuple.len(), 2);
    assert_eq!(tuple.get(0).and_then(|value| value.as_int()), Some(5));
    assert_eq!(tuple.get(1).and_then(|value| value.as_int()), Some(8));

    unsafe {
        drop(Box::from_raw(object));
    }
}

#[test]
fn test_tuple_negative_index() {
    let tuple = TupleObject::from_slice(&[Value::int(1).unwrap(), Value::int(2).unwrap()]);

    assert_eq!(tuple.get(-1).unwrap().as_int(), Some(2));
    assert_eq!(tuple.get(-2).unwrap().as_int(), Some(1));
    assert!(tuple.get(-3).is_none());
}

#[test]
fn test_tuple_concat() {
    let t1 = TupleObject::from_slice(&[Value::int(1).unwrap()]);
    let t2 = TupleObject::from_slice(&[Value::int(2).unwrap()]);
    let t3 = t1.concat(&t2);

    assert_eq!(t3.len(), 2);
    assert_eq!(t3.get(0).unwrap().as_int(), Some(1));
    assert_eq!(t3.get(1).unwrap().as_int(), Some(2));
}

#[test]
fn test_tuple_repeat() {
    let t = TupleObject::from_slice(&[Value::int(1).unwrap()]);
    let t3 = t.repeat(3);

    assert_eq!(t3.len(), 3);
    assert_eq!(t3.get(0).unwrap().as_int(), Some(1));
    assert_eq!(t3.get(1).unwrap().as_int(), Some(1));
    assert_eq!(t3.get(2).unwrap().as_int(), Some(1));
}

#[test]
fn test_empty_tuple() {
    let t = TupleObject::empty();
    assert!(t.is_empty());
    assert_eq!(t.len(), 0);
}
