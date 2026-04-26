use super::*;
use crate::types::string::StringObject;
use crate::types::tuple::TupleObject;
use prism_core::intern::intern;

#[test]
fn test_dict_basic() {
    let mut dict = DictObject::new();
    assert!(dict.is_empty());

    let key1 = Value::int(1).unwrap();
    let key2 = Value::int(2).unwrap();
    let val1 = Value::int(100).unwrap();
    let val2 = Value::int(200).unwrap();

    dict.set(key1, val1);
    dict.set(key2, val2);

    assert_eq!(dict.len(), 2);
    assert_eq!(dict.get(key1).unwrap().as_int(), Some(100));
    assert_eq!(dict.get(key2).unwrap().as_int(), Some(200));
}

#[test]
fn test_dict_overwrite() {
    let mut dict = DictObject::new();
    let key = Value::int(1).unwrap();

    dict.set(key, Value::int(100).unwrap());
    dict.set(key, Value::int(200).unwrap());

    assert_eq!(dict.len(), 1);
    assert_eq!(dict.get(key).unwrap().as_int(), Some(200));
}

#[test]
fn test_dict_iter_preserves_insertion_order() {
    let mut dict = DictObject::new();
    let alpha = Value::string(intern("alpha"));
    let beta = Value::string(intern("beta"));
    let gamma = Value::string(intern("gamma"));
    dict.set(alpha, Value::int(1).unwrap());
    dict.set(beta, Value::int(2).unwrap());
    dict.set(gamma, Value::int(3).unwrap());

    let items = dict
        .iter()
        .map(|(key, value)| (key, value.as_int()))
        .collect::<Vec<_>>();
    assert_eq!(
        items,
        vec![(alpha, Some(1)), (beta, Some(2)), (gamma, Some(3)),]
    );
}

#[test]
fn test_dict_delete_and_reinsert_moves_key_to_end() {
    let mut dict = DictObject::new();
    let alpha = Value::string(intern("alpha"));
    let beta = Value::string(intern("beta"));
    let gamma = Value::string(intern("gamma"));

    dict.set(alpha, Value::int(1).unwrap());
    dict.set(beta, Value::int(2).unwrap());
    dict.set(gamma, Value::int(3).unwrap());
    assert_eq!(dict.remove(beta).and_then(|value| value.as_int()), Some(2));
    dict.set(beta, Value::int(4).unwrap());

    let keys = dict.keys().collect::<Vec<_>>();
    assert_eq!(keys, vec![alpha, gamma, beta]);
}

#[test]
fn test_dict_popitem_uses_lifo_order() {
    let mut dict = DictObject::new();
    dict.set(Value::string(intern("alpha")), Value::int(1).unwrap());
    dict.set(Value::string(intern("beta")), Value::int(2).unwrap());

    let (key, value) = dict.popitem().expect("popitem should return newest item");
    assert_eq!(key, Value::string(intern("beta")));
    assert_eq!(value.as_int(), Some(2));
}

#[test]
fn test_dict_remove() {
    let mut dict = DictObject::new();
    let key = Value::int(1).unwrap();

    dict.set(key, Value::int(100).unwrap());
    assert!(dict.contains_key(key));

    let removed = dict.remove(key);
    assert_eq!(removed.unwrap().as_int(), Some(100));
    assert!(!dict.contains_key(key));
}

#[test]
fn test_dict_none_key() {
    let mut dict = DictObject::new();
    let key = Value::none();

    dict.set(key, Value::int(42).unwrap());
    assert_eq!(dict.get(key).unwrap().as_int(), Some(42));
}

#[test]
fn test_dict_interned_string_key_roundtrip() {
    let mut dict = DictObject::new();
    let key = Value::string(intern("key"));

    dict.set(key, Value::int(123).unwrap());
    assert_eq!(
        dict.get(Value::string(intern("key"))).unwrap().as_int(),
        Some(123)
    );
}

#[test]
fn test_dict_int_float_key_alias() {
    let mut dict = DictObject::new();
    dict.set(Value::int_unchecked(1), Value::int_unchecked(99));

    assert_eq!(dict.get(Value::float(1.0)).unwrap().as_int(), Some(99));
}

#[test]
fn test_dict_matches_heap_and_interned_string_keys_by_content() {
    let mut dict = DictObject::new();
    let heap_ptr = Box::into_raw(Box::new(StringObject::new("while")));
    let heap_key = Value::object_ptr(heap_ptr as *const ());

    dict.set(heap_key, Value::int_unchecked(7));
    assert_eq!(
        dict.get(Value::string(intern("while"))).unwrap().as_int(),
        Some(7)
    );
}

#[test]
fn test_dict_matches_tuple_keys_structurally() {
    let mut dict = DictObject::new();
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

    dict.set(
        Value::object_ptr(tuple_a as *const ()),
        Value::int_unchecked(9),
    );
    assert_eq!(
        dict.get(Value::object_ptr(tuple_b as *const ()))
            .unwrap()
            .as_int(),
        Some(9)
    );
}

#[test]
fn test_dict_setdefault_inserts_default_once() {
    let mut dict = DictObject::new();
    let key = Value::string(intern("token"));

    let inserted = dict.setdefault(key, Value::int_unchecked(7));
    let existing = dict.setdefault(key, Value::int_unchecked(99));

    assert_eq!(inserted.as_int(), Some(7));
    assert_eq!(existing.as_int(), Some(7));
    assert_eq!(dict.get(key).unwrap().as_int(), Some(7));
}
