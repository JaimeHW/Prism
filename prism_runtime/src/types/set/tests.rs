use super::*;
use crate::types::string::StringObject;
use crate::types::tuple::TupleObject;
use prism_core::intern::intern;

#[test]
fn test_set_basic() {
    let mut set = SetObject::new();
    assert!(set.is_empty());

    set.add(Value::int(1).unwrap());
    set.add(Value::int(2).unwrap());
    set.add(Value::int(3).unwrap());

    assert_eq!(set.len(), 3);
    assert!(set.contains(Value::int(1).unwrap()));
    assert!(set.contains(Value::int(2).unwrap()));
    assert!(set.contains(Value::int(3).unwrap()));
    assert!(!set.contains(Value::int(4).unwrap()));
}

#[test]
fn test_set_duplicates() {
    let mut set = SetObject::new();
    assert!(set.add(Value::int(1).unwrap())); // First insert
    assert!(!set.add(Value::int(1).unwrap())); // Duplicate
    assert_eq!(set.len(), 1);
}

#[test]
fn test_set_remove() {
    let mut set = SetObject::from_slice(&[
        Value::int(1).unwrap(),
        Value::int(2).unwrap(),
        Value::int(3).unwrap(),
    ]);

    assert!(set.remove(Value::int(2).unwrap()));
    assert!(!set.contains(Value::int(2).unwrap()));
    assert_eq!(set.len(), 2);

    assert!(!set.remove(Value::int(2).unwrap())); // Already removed
}

#[test]
fn test_set_pop() {
    let mut set = SetObject::from_slice(&[Value::int(42).unwrap()]);
    let popped = set.pop();
    assert!(popped.is_some());
    assert_eq!(popped.unwrap().as_int(), Some(42));
    assert!(set.is_empty());
    assert!(set.pop().is_none());
}

#[test]
fn test_set_none_element() {
    let mut set = SetObject::new();
    set.add(Value::none());
    assert!(set.contains(Value::none()));
    assert_eq!(set.len(), 1);
}

#[test]
fn test_set_union() {
    let set1 = SetObject::from_slice(&[Value::int(1).unwrap(), Value::int(2).unwrap()]);
    let set2 = SetObject::from_slice(&[Value::int(2).unwrap(), Value::int(3).unwrap()]);

    let union = set1.union(&set2);
    assert_eq!(union.len(), 3);
    assert!(union.contains(Value::int(1).unwrap()));
    assert!(union.contains(Value::int(2).unwrap()));
    assert!(union.contains(Value::int(3).unwrap()));
}

#[test]
fn test_set_intersection() {
    let set1 = SetObject::from_slice(&[
        Value::int(1).unwrap(),
        Value::int(2).unwrap(),
        Value::int(3).unwrap(),
    ]);
    let set2 = SetObject::from_slice(&[
        Value::int(2).unwrap(),
        Value::int(3).unwrap(),
        Value::int(4).unwrap(),
    ]);

    let inter = set1.intersection(&set2);
    assert_eq!(inter.len(), 2);
    assert!(!inter.contains(Value::int(1).unwrap()));
    assert!(inter.contains(Value::int(2).unwrap()));
    assert!(inter.contains(Value::int(3).unwrap()));
    assert!(!inter.contains(Value::int(4).unwrap()));
}

#[test]
fn test_set_difference() {
    let set1 = SetObject::from_slice(&[
        Value::int(1).unwrap(),
        Value::int(2).unwrap(),
        Value::int(3).unwrap(),
    ]);
    let set2 = SetObject::from_slice(&[Value::int(2).unwrap(), Value::int(4).unwrap()]);

    let diff = set1.difference(&set2);
    assert_eq!(diff.len(), 2);
    assert!(diff.contains(Value::int(1).unwrap()));
    assert!(!diff.contains(Value::int(2).unwrap()));
    assert!(diff.contains(Value::int(3).unwrap()));
}

#[test]
fn test_set_symmetric_difference() {
    let set1 = SetObject::from_slice(&[Value::int(1).unwrap(), Value::int(2).unwrap()]);
    let set2 = SetObject::from_slice(&[Value::int(2).unwrap(), Value::int(3).unwrap()]);

    let sym_diff = set1.symmetric_difference(&set2);
    assert_eq!(sym_diff.len(), 2);
    assert!(sym_diff.contains(Value::int(1).unwrap()));
    assert!(!sym_diff.contains(Value::int(2).unwrap()));
    assert!(sym_diff.contains(Value::int(3).unwrap()));
}

#[test]
fn test_set_subset_superset() {
    let set1 = SetObject::from_slice(&[Value::int(1).unwrap(), Value::int(2).unwrap()]);
    let set2 = SetObject::from_slice(&[
        Value::int(1).unwrap(),
        Value::int(2).unwrap(),
        Value::int(3).unwrap(),
    ]);

    assert!(set1.is_subset(&set2));
    assert!(!set2.is_subset(&set1));
    assert!(set2.is_superset(&set1));
    assert!(!set1.is_superset(&set2));
}

#[test]
fn test_set_disjoint() {
    let set1 = SetObject::from_slice(&[Value::int(1).unwrap(), Value::int(2).unwrap()]);
    let set2 = SetObject::from_slice(&[Value::int(3).unwrap(), Value::int(4).unwrap()]);
    let set3 = SetObject::from_slice(&[Value::int(2).unwrap(), Value::int(3).unwrap()]);

    assert!(set1.is_disjoint(&set2));
    assert!(!set1.is_disjoint(&set3));
}

#[test]
fn test_set_update() {
    let mut set1 = SetObject::from_slice(&[Value::int(1).unwrap()]);
    let set2 = SetObject::from_slice(&[Value::int(2).unwrap(), Value::int(3).unwrap()]);

    set1.update(&set2);
    assert_eq!(set1.len(), 3);
    assert!(set1.contains(Value::int(1).unwrap()));
    assert!(set1.contains(Value::int(2).unwrap()));
    assert!(set1.contains(Value::int(3).unwrap()));
}

#[test]
fn test_set_clone() {
    let set1 = SetObject::from_slice(&[Value::int(1).unwrap(), Value::int(2).unwrap()]);
    let set2 = set1.clone();

    assert_eq!(set1.len(), set2.len());
    assert!(set2.contains(Value::int(1).unwrap()));
    assert!(set2.contains(Value::int(2).unwrap()));
}

#[test]
fn test_set_with_floats() {
    let mut set = SetObject::new();
    set.add(Value::float(1.5));
    set.add(Value::float(2.5));
    set.add(Value::float(1.5)); // Duplicate

    assert_eq!(set.len(), 2);
    assert!(set.contains(Value::float(1.5)));
    assert!(set.contains(Value::float(2.5)));
}

#[test]
fn test_set_matches_heap_and_interned_strings_by_content() {
    let mut set = SetObject::new();
    let heap_ptr = Box::into_raw(Box::new(StringObject::new("while")));
    set.add(Value::object_ptr(heap_ptr as *const ()));

    assert!(set.contains(Value::string(intern("while"))));
}

#[test]
fn test_set_matches_tuple_members_structurally() {
    let mut set = SetObject::new();
    let left_ptr = Box::into_raw(Box::new(StringObject::new("while")));
    let right_ptr = Box::into_raw(Box::new(StringObject::new("while")));
    let tuple_a = Box::into_raw(Box::new(TupleObject::from_slice(&[
        Value::object_ptr(left_ptr as *const ()),
        Value::int_unchecked(1),
    ])));
    let tuple_b = Box::into_raw(Box::new(TupleObject::from_slice(&[
        Value::object_ptr(right_ptr as *const ()),
        Value::int_unchecked(1),
    ])));

    set.add(Value::object_ptr(tuple_a as *const ()));
    assert!(set.contains(Value::object_ptr(tuple_b as *const ())));
}

#[test]
fn test_set_iter() {
    let set = SetObject::from_slice(&[
        Value::int(1).unwrap(),
        Value::int(2).unwrap(),
        Value::int(3).unwrap(),
    ]);

    let collected: Vec<_> = set.iter().collect();
    assert_eq!(collected.len(), 3);
}
