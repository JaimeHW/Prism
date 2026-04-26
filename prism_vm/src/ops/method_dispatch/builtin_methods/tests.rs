use super::*;
use crate::builtins::iterator_to_value;
use crate::error::RuntimeErrorKind;
use crate::stdlib::exceptions::ExceptionTypeId;
use prism_code::CodeObject;
use prism_core::intern::{intern, interned_by_ptr};
use prism_runtime::object::class::PyClassObject;
use prism_runtime::object::descriptor::BoundMethod;
use prism_runtime::object::shape::Shape;
use prism_runtime::object::shaped_object::ShapedObject;
use prism_runtime::object::type_obj::TypeId;
use prism_runtime::object::views::MappingProxyObject;
use prism_runtime::types::function::FunctionObject;
use prism_runtime::types::iter::IteratorObject;
use std::sync::Arc;

mod binary;

fn boxed_list_value(list: ListObject) -> (Value, *mut ListObject) {
    let ptr = Box::into_raw(Box::new(list));
    (Value::object_ptr(ptr as *const ()), ptr)
}

fn boxed_tuple_value(tuple: TupleObject) -> (Value, *mut TupleObject) {
    let ptr = Box::into_raw(Box::new(tuple));
    (Value::object_ptr(ptr as *const ()), ptr)
}

fn list_values(ptr: *const ListObject) -> Vec<i64> {
    let list = unsafe { &*ptr };
    list.as_slice()
        .iter()
        .map(|value| value.as_int().expect("expected tagged int"))
        .collect()
}

fn string_value(value: Value) -> String {
    if value.is_string() {
        let ptr = value
            .as_string_object_ptr()
            .expect("tagged string should have a pointer");
        return interned_by_ptr(ptr as *const u8)
            .expect("tagged string pointer should resolve")
            .as_str()
            .to_string();
    }

    let ptr = value
        .as_object_ptr()
        .expect("string result should be an object");
    let string = unsafe { &*(ptr as *const StringObject) };
    string.as_str().to_string()
}

fn builtin_from_value(value: Value) -> &'static BuiltinFunctionObject {
    unsafe {
        &*(value
            .as_object_ptr()
            .expect("builtin method should be materialized")
            as *const BuiltinFunctionObject)
    }
}

fn property_echo_getter(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() != 1 {
        return Err(BuiltinError::TypeError(format!(
            "getter expected 1 argument, got {}",
            args.len()
        )));
    }
    Ok(args[0])
}

fn property_accepting_setter(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() != 2 {
        return Err(BuiltinError::TypeError(format!(
            "setter expected 2 arguments, got {}",
            args.len()
        )));
    }
    Ok(Value::string(intern("setter return is ignored")))
}

fn property_accepting_deleter(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() != 1 {
        return Err(BuiltinError::TypeError(format!(
            "deleter expected 1 argument, got {}",
            args.len()
        )));
    }
    Ok(Value::string(intern("deleter return is ignored")))
}

fn byte_values(value: Value) -> Vec<u8> {
    let ptr = value
        .as_object_ptr()
        .expect("byte result should be an object");
    let bytes = unsafe { &*(ptr as *const BytesObject) };
    bytes.as_bytes().to_vec()
}

fn assert_unicode_encode_error(err: BuiltinError, expected_message: &str) {
    match err {
        BuiltinError::Raised(runtime_err) => match runtime_err.kind() {
            RuntimeErrorKind::Exception { type_id, message } => {
                assert_eq!(*type_id, ExceptionTypeId::UnicodeEncodeError.as_u8() as u16);
                assert_eq!(message.as_ref(), expected_message);
            }
            kind => panic!("expected UnicodeEncodeError, got {kind:?}"),
        },
        other => panic!("expected UnicodeEncodeError, got {other:?}"),
    }
}

#[test]
fn test_resolve_list_method_returns_builtin_for_sequence_protocol_and_mutators() {
    let iter = resolve_list_method("__iter__").expect("__iter__ should resolve");
    let len = resolve_list_method("__len__").expect("__len__ should resolve");
    let getitem = resolve_list_method("__getitem__").expect("__getitem__ should resolve");
    let append = resolve_list_method("append").expect("append should resolve");
    let extend = resolve_list_method("extend").expect("extend should resolve");
    let insert = resolve_list_method("insert").expect("insert should resolve");
    let remove = resolve_list_method("remove").expect("remove should resolve");
    let pop = resolve_list_method("pop").expect("pop should resolve");
    let copy = resolve_list_method("copy").expect("copy should resolve");
    let clear = resolve_list_method("clear").expect("clear should resolve");
    let reverse = resolve_list_method("reverse").expect("reverse should resolve");
    assert!(iter.method.as_object_ptr().is_some());
    assert!(len.method.as_object_ptr().is_some());
    assert!(getitem.method.as_object_ptr().is_some());
    assert!(append.method.as_object_ptr().is_some());
    assert!(extend.method.as_object_ptr().is_some());
    assert!(insert.method.as_object_ptr().is_some());
    assert!(remove.method.as_object_ptr().is_some());
    assert!(pop.method.as_object_ptr().is_some());
    assert!(copy.method.as_object_ptr().is_some());
    assert!(clear.method.as_object_ptr().is_some());
    assert!(reverse.method.as_object_ptr().is_some());
    assert!(!append.is_descriptor);
    assert!(!extend.is_descriptor);
    assert!(!insert.is_descriptor);
    assert!(!remove.is_descriptor);
    assert!(!pop.is_descriptor);
    assert!(!copy.is_descriptor);
    assert!(!clear.is_descriptor);
    assert!(!reverse.is_descriptor);
}

#[test]
fn test_resolve_tuple_method_returns_builtin_for_sequence_protocol() {
    for name in ["__iter__", "__len__", "__getitem__", "count", "index"] {
        let method = resolve_tuple_method(name).unwrap_or_else(|| panic!("{name} should resolve"));
        assert!(method.method.as_object_ptr().is_some());
        assert!(!method.is_descriptor);
    }
}

#[test]
fn test_tuple_methods_use_native_storage() {
    let (tuple_value, tuple_ptr) = boxed_tuple_value(TupleObject::from_slice(&[
        Value::int(1).unwrap(),
        Value::int(2).unwrap(),
        Value::int(1).unwrap(),
    ]));

    assert_eq!(
        tuple_len(&[tuple_value])
            .expect("tuple.__len__ should succeed")
            .as_int(),
        Some(3)
    );
    assert_eq!(
        tuple_getitem(&[tuple_value, Value::int(-1).unwrap()])
            .expect("tuple.__getitem__ should accept negative indices")
            .as_int(),
        Some(1)
    );
    assert_eq!(
        tuple_count(&[tuple_value, Value::int(1).unwrap()])
            .expect("tuple.count should succeed")
            .as_int(),
        Some(2)
    );
    assert_eq!(
        tuple_index(&[tuple_value, Value::int(1).unwrap(), Value::int(1).unwrap(),])
            .expect("tuple.index should honor the start bound")
            .as_int(),
        Some(2)
    );

    let slice_ptr = Box::into_raw(Box::new(SliceObject::new(Some(0), Some(3), Some(2))));
    let sliced = tuple_getitem(&[tuple_value, Value::object_ptr(slice_ptr as *const ())])
        .expect("tuple.__getitem__ should accept slice objects");
    let sliced_ptr = sliced
        .as_object_ptr()
        .expect("tuple slice should return a tuple object")
        as *mut TupleObject;
    let sliced_tuple = unsafe { &*sliced_ptr };
    assert_eq!(sliced_tuple.len(), 2);
    assert_eq!(sliced_tuple.as_slice()[0].as_int(), Some(1));
    assert_eq!(sliced_tuple.as_slice()[1].as_int(), Some(1));

    let iter_value = tuple_iter(&[tuple_value]).expect("tuple.__iter__ should succeed");
    let iter = get_iterator_mut(&iter_value).expect("tuple.__iter__ should return iterator");
    assert_eq!(iter.next().and_then(|value| value.as_int()), Some(1));
    assert_eq!(iter.next().and_then(|value| value.as_int()), Some(2));
    assert_eq!(iter.next().and_then(|value| value.as_int()), Some(1));
    assert!(iter.next().is_none());

    unsafe {
        drop(Box::from_raw(
            iter_value.as_object_ptr().unwrap() as *mut IteratorObject
        ));
        drop(Box::from_raw(sliced_ptr));
        drop(Box::from_raw(slice_ptr));
        drop(Box::from_raw(tuple_ptr));
    }
}

#[test]
fn test_list_append_mutates_list_and_returns_none() {
    let (list_value, list_ptr) = boxed_list_value(ListObject::new());
    let result = list_append(&[list_value, Value::int(7).unwrap()]).expect("append should work");
    assert!(result.is_none());
    assert_eq!(list_values(list_ptr), vec![7]);

    unsafe {
        drop(Box::from_raw(list_ptr));
    }
}

#[test]
fn test_list_clear_removes_all_items_and_returns_none() {
    let (list_value, list_ptr) = boxed_list_value(ListObject::from_slice(&[
        Value::int(1).unwrap(),
        Value::int(2).unwrap(),
        Value::int(3).unwrap(),
    ]));

    let result = list_clear(&[list_value]).expect("clear should work");

    assert!(result.is_none());
    assert_eq!(list_values(list_ptr), Vec::<i64>::new());

    unsafe {
        drop(Box::from_raw(list_ptr));
    }
}

#[test]
fn test_list_methods_accept_heap_list_subclasses_with_native_backing() {
    let object = Box::into_raw(Box::new(ShapedObject::new_list_backed(
        TypeId::from_raw(512),
        Shape::empty(),
    )));
    let value = Value::object_ptr(object as *const ());

    list_append(&[value, Value::int(7).unwrap()]).expect("append should work on subclasses");
    list_append(&[value, Value::int(11).unwrap()]).expect("append should work on subclasses");
    let len = list_len(&[value]).expect("__len__ should work on subclasses");
    let first = list_getitem(&[value, Value::int(0).unwrap()])
        .expect("__getitem__ should work on subclasses");
    let iter = list_iter(&[value]).expect("__iter__ should work on subclasses");
    let copied = list_copy(&[value]).expect("copy should work on subclasses");
    list_clear(&[value]).expect("clear should work on subclasses");
    let copied_ptr = copied
        .as_object_ptr()
        .expect("list.copy should still return a concrete list")
        as *mut ListObject;
    let iter_ptr =
        iter.as_object_ptr()
            .expect("list.__iter__ should return an iterator") as *mut IteratorObject;
    let iter_ref = unsafe { &mut *iter_ptr };

    let backing = unsafe { &*object }
        .list_backing()
        .expect("list backing should exist");
    assert!(backing.as_slice().is_empty());
    assert_eq!(len.as_int(), Some(2));
    assert_eq!(first.as_int(), Some(7));
    assert_eq!(iter_ref.next().and_then(|value| value.as_int()), Some(7));
    assert_eq!(iter_ref.next().and_then(|value| value.as_int()), Some(11));
    assert!(iter_ref.next().is_none());
    assert_eq!(list_values(copied_ptr), vec![7, 11]);

    unsafe {
        drop(Box::from_raw(iter_ptr));
        drop(Box::from_raw(copied_ptr));
        drop(Box::from_raw(object));
    }
}

#[test]
fn test_list_copy_returns_distinct_shallow_copy() {
    let (list_value, list_ptr) = boxed_list_value(ListObject::from_slice(&[
        Value::int(1).unwrap(),
        Value::int(2).unwrap(),
    ]));

    let copied = list_copy(&[list_value]).expect("copy should work");
    let copied_ptr = copied
        .as_object_ptr()
        .expect("list.copy should return a list object") as *mut ListObject;

    assert_eq!(list_values(list_ptr), vec![1, 2]);
    assert_eq!(list_values(copied_ptr), vec![1, 2]);
    assert_ne!(list_ptr, copied_ptr);

    unsafe {
        drop(Box::from_raw(copied_ptr));
        drop(Box::from_raw(list_ptr));
    }
}

#[test]
fn test_list_count_counts_matching_values() {
    let (list_value, list_ptr) = boxed_list_value(ListObject::from_slice(&[
        Value::int(1).unwrap(),
        Value::int(2).unwrap(),
        Value::int(1).unwrap(),
        Value::int(3).unwrap(),
    ]));

    assert_eq!(
        list_count(&[list_value, Value::int(1).unwrap()])
            .expect("list.count should succeed")
            .as_int(),
        Some(2)
    );

    unsafe {
        drop(Box::from_raw(list_ptr));
    }
}

#[test]
fn test_list_index_honors_optional_bounds() {
    let (list_value, list_ptr) = boxed_list_value(ListObject::from_slice(&[
        Value::string(intern("alpha")),
        Value::string(intern("beta")),
        Value::string(intern("alpha")),
        Value::string(intern("gamma")),
    ]));

    assert_eq!(
        list_index(&[list_value, Value::string(intern("alpha"))])
            .expect("list.index should find the first match")
            .as_int(),
        Some(0)
    );
    assert_eq!(
        list_index(&[
            list_value,
            Value::string(intern("alpha")),
            Value::int(1).unwrap(),
        ])
        .expect("list.index should honor the start bound")
        .as_int(),
        Some(2)
    );
    let err = list_index(&[
        list_value,
        Value::string(intern("alpha")),
        Value::int(1).unwrap(),
        Value::int(2).unwrap(),
    ])
    .expect_err("list.index should honor the stop bound");
    assert_eq!(err.to_string(), "ValueError: list.index(x): x not in list");

    unsafe {
        drop(Box::from_raw(list_ptr));
    }
}

#[test]
fn test_list_remove_deletes_first_matching_value_and_returns_none() {
    let (list_value, list_ptr) = boxed_list_value(ListObject::from_slice(&[
        Value::int(1).unwrap(),
        Value::int(2).unwrap(),
        Value::int(2).unwrap(),
        Value::int(3).unwrap(),
    ]));

    let result = list_remove(&[list_value, Value::int(2).unwrap()])
        .expect("remove should delete the first matching value");
    assert!(result.is_none());
    assert_eq!(list_values(list_ptr), vec![1, 2, 3]);

    unsafe {
        drop(Box::from_raw(list_ptr));
    }
}

#[test]
fn test_list_pop_without_index_returns_last_item() {
    let (list_value, list_ptr) = boxed_list_value(ListObject::from_slice(&[
        Value::int(1).unwrap(),
        Value::int(2).unwrap(),
        Value::int(3).unwrap(),
    ]));

    let popped = list_pop(&[list_value]).expect("pop should remove the tail element");
    assert_eq!(popped.as_int(), Some(3));
    assert_eq!(list_values(list_ptr), vec![1, 2]);

    unsafe {
        drop(Box::from_raw(list_ptr));
    }
}

#[test]
fn test_list_pop_with_index_supports_negative_offsets() {
    let (list_value, list_ptr) = boxed_list_value(ListObject::from_slice(&[
        Value::int(10).unwrap(),
        Value::int(20).unwrap(),
        Value::int(30).unwrap(),
    ]));

    let popped =
        list_pop(&[list_value, Value::int(-2).unwrap()]).expect("pop should honor negatives");
    assert_eq!(popped.as_int(), Some(20));
    assert_eq!(list_values(list_ptr), vec![10, 30]);

    unsafe {
        drop(Box::from_raw(list_ptr));
    }
}

#[test]
fn test_list_pop_empty_and_out_of_range_match_cpython_messages() {
    let (empty_value, empty_ptr) = boxed_list_value(ListObject::new());
    let empty_err = list_pop(&[empty_value]).expect_err("empty pop should fail");
    match empty_err {
        BuiltinError::IndexError(message) => assert_eq!(message, "pop from empty list"),
        other => panic!("expected IndexError, got {other:?}"),
    }

    let indexed_empty_err = list_pop(&[empty_value, Value::int(0).unwrap()])
        .expect_err("indexed empty pop should fail");
    match indexed_empty_err {
        BuiltinError::IndexError(message) => assert_eq!(message, "pop from empty list"),
        other => panic!("expected IndexError, got {other:?}"),
    }

    let (list_value, list_ptr) =
        boxed_list_value(ListObject::from_slice(&[Value::int(1).unwrap()]));
    let range_err =
        list_pop(&[list_value, Value::int(4).unwrap()]).expect_err("out-of-range pop should fail");
    match range_err {
        BuiltinError::IndexError(message) => assert_eq!(message, "pop index out of range"),
        other => panic!("expected IndexError, got {other:?}"),
    }
    assert_eq!(list_values(list_ptr), vec![1]);

    unsafe {
        drop(Box::from_raw(list_ptr));
        drop(Box::from_raw(empty_ptr));
    }
}

#[test]
fn test_list_insert_inserts_at_requested_position_and_returns_none() {
    let (list_value, list_ptr) = boxed_list_value(ListObject::from_slice(&[
        Value::int(1).unwrap(),
        Value::int(3).unwrap(),
    ]));

    let result = list_insert(&[list_value, Value::int(1).unwrap(), Value::int(2).unwrap()])
        .expect("insert should place the value before the requested index");
    assert!(result.is_none());
    assert_eq!(list_values(list_ptr), vec![1, 2, 3]);

    unsafe {
        drop(Box::from_raw(list_ptr));
    }
}

#[test]
fn test_list_insert_clamps_indices_to_list_bounds() {
    let (list_value, list_ptr) = boxed_list_value(ListObject::from_slice(&[
        Value::int(2).unwrap(),
        Value::int(3).unwrap(),
    ]));

    list_insert(&[list_value, Value::int(-10).unwrap(), Value::int(1).unwrap()])
        .expect("insert should clamp large negative indices to the start");
    list_insert(&[list_value, Value::int(99).unwrap(), Value::int(4).unwrap()])
        .expect("insert should clamp large positive indices to the end");
    assert_eq!(list_values(list_ptr), vec![1, 2, 3, 4]);

    unsafe {
        drop(Box::from_raw(list_ptr));
    }
}

#[test]
fn test_list_insert_rejects_non_integer_indices() {
    let (list_value, list_ptr) =
        boxed_list_value(ListObject::from_slice(&[Value::int(1).unwrap()]));

    let err = list_insert(&[
        list_value,
        Value::string(intern("bad")),
        Value::int(2).unwrap(),
    ])
    .expect_err("insert should reject non-integer indices");
    match err {
        BuiltinError::TypeError(message) => {
            assert_eq!(message, "'str' object cannot be interpreted as an integer");
        }
        other => panic!("expected TypeError, got {other:?}"),
    }
    assert_eq!(list_values(list_ptr), vec![1]);

    unsafe {
        drop(Box::from_raw(list_ptr));
    }
}

#[test]
fn test_list_remove_uses_runtime_string_equality() {
    let (list_value, list_ptr) = boxed_list_value(ListObject::from_slice(&[
        Value::string(intern("needle")),
        Value::string(intern("other")),
    ]));
    let needle_ptr = Box::into_raw(Box::new(StringObject::new("needle")));
    let needle = Value::object_ptr(needle_ptr as *const ());

    let result = list_remove(&[list_value, needle])
        .expect("remove should match strings across runtime representations");
    assert!(result.is_none());

    let remaining = unsafe { &*list_ptr }
        .as_slice()
        .iter()
        .copied()
        .map(string_value)
        .collect::<Vec<_>>();
    assert_eq!(remaining, vec!["other".to_string()]);

    unsafe {
        drop(Box::from_raw(needle_ptr));
        drop(Box::from_raw(list_ptr));
    }
}

#[test]
fn test_list_remove_raises_value_error_when_item_is_missing() {
    let (list_value, list_ptr) = boxed_list_value(ListObject::from_slice(&[
        Value::int(1).unwrap(),
        Value::int(2).unwrap(),
    ]));

    let err = list_remove(&[list_value, Value::int(9).unwrap()])
        .expect_err("remove should fail when the value is absent");
    match err {
        BuiltinError::ValueError(message) => {
            assert_eq!(message, "list.remove(x): x not in list");
        }
        other => panic!("expected ValueError, got {other:?}"),
    }
    assert_eq!(list_values(list_ptr), vec![1, 2]);

    unsafe {
        drop(Box::from_raw(list_ptr));
    }
}

#[test]
fn test_list_reverse_mutates_in_place_and_returns_none() {
    let (list_value, list_ptr) = boxed_list_value(ListObject::from_slice(&[
        Value::int(1).unwrap(),
        Value::int(2).unwrap(),
        Value::int(3).unwrap(),
    ]));

    let result = list_reverse(&[list_value]).expect("reverse should work");
    assert!(result.is_none());
    assert_eq!(list_values(list_ptr), vec![3, 2, 1]);

    unsafe {
        drop(Box::from_raw(list_ptr));
    }
}

#[test]
fn test_list_sort_orders_numeric_values_in_place() {
    let (list_value, list_ptr) = boxed_list_value(ListObject::from_slice(&[
        Value::int(3).unwrap(),
        Value::int(1).unwrap(),
        Value::int(2).unwrap(),
    ]));
    let mut vm = VirtualMachine::default();

    let result = list_sort_with_vm(&mut vm, &[list_value], &[]).expect("sort should succeed");
    assert!(result.is_none());
    assert_eq!(list_values(list_ptr), vec![1, 2, 3]);

    unsafe {
        drop(Box::from_raw(list_ptr));
    }
}

#[test]
fn test_list_sort_honors_reverse_keyword() {
    let (list_value, list_ptr) = boxed_list_value(ListObject::from_slice(&[
        Value::int(3).unwrap(),
        Value::int(1).unwrap(),
        Value::int(2).unwrap(),
    ]));
    let mut vm = VirtualMachine::default();

    let result = list_sort_with_vm(&mut vm, &[list_value], &[("reverse", Value::bool(true))])
        .expect("sort(reverse=True) should succeed");
    assert!(result.is_none());
    assert_eq!(list_values(list_ptr), vec![3, 2, 1]);

    unsafe {
        drop(Box::from_raw(list_ptr));
    }
}

#[test]
fn test_list_sort_honors_key_keyword_with_builtin_callable() {
    let (list_value, list_ptr) = boxed_list_value(ListObject::from_slice(&[
        Value::int(2).unwrap(),
        Value::int(0).unwrap(),
        Value::int(3).unwrap(),
        Value::int(1).unwrap(),
    ]));
    let mut vm = VirtualMachine::default();

    let result = list_sort_with_vm(
        &mut vm,
        &[list_value],
        &[(
            "key",
            crate::builtins::builtin_type_object_for_type_id(TypeId::BOOL),
        )],
    )
    .expect("sort(key=bool) should succeed");
    assert!(result.is_none());
    assert_eq!(list_values(list_ptr), vec![0, 2, 3, 1]);

    unsafe {
        drop(Box::from_raw(list_ptr));
    }
}

#[test]
fn test_resolve_deque_method_returns_builtin_methods() {
    let append = resolve_deque_method("append").expect("deque.append should resolve");
    let pop = resolve_deque_method("pop").expect("deque.pop should resolve");
    assert!(append.method.as_object_ptr().is_some());
    assert!(pop.method.as_object_ptr().is_some());
    assert!(!append.is_descriptor);
    assert!(!pop.is_descriptor);
}

#[test]
fn test_deque_append_and_pop_round_trip() {
    let deque_ptr = Box::into_raw(Box::new(DequeObject::new()));
    let deque_value = Value::object_ptr(deque_ptr as *const ());

    let appended =
        deque_append(&[deque_value, Value::int(11).unwrap()]).expect("deque.append should succeed");
    assert!(appended.is_none());
    assert_eq!(unsafe { &*deque_ptr }.len(), 1);

    let popped = deque_pop(&[deque_value]).expect("deque.pop should succeed");
    assert_eq!(popped.as_int(), Some(11));
    assert!(unsafe { &*deque_ptr }.is_empty());

    unsafe {
        drop(Box::from_raw(deque_ptr));
    }
}

#[test]
fn test_resolve_dict_method_returns_builtin_for_views() {
    let keys = resolve_dict_method("keys").expect("keys should resolve");
    let get = resolve_dict_method("get").expect("get should resolve");
    let len = resolve_dict_method("__len__").expect("__len__ should resolve");
    let contains = resolve_dict_method("__contains__").expect("__contains__ should resolve");
    let values = resolve_dict_method("values").expect("values should resolve");
    let items = resolve_dict_method("items").expect("items should resolve");
    let getitem = resolve_dict_method("__getitem__").expect("__getitem__ should resolve");
    let setitem = resolve_dict_method("__setitem__").expect("__setitem__ should resolve");
    let delitem = resolve_dict_method("__delitem__").expect("__delitem__ should resolve");
    let pop = resolve_dict_method("pop").expect("pop should resolve");
    let popitem = resolve_dict_method("popitem").expect("popitem should resolve");
    let setdefault = resolve_dict_method("setdefault").expect("setdefault should resolve");
    let clear = resolve_dict_method("clear").expect("clear should resolve");
    let update = resolve_dict_method("update").expect("update should resolve");
    let copy = resolve_dict_method("copy").expect("copy should resolve");
    assert!(keys.method.as_object_ptr().is_some());
    assert!(get.method.as_object_ptr().is_some());
    assert!(len.method.as_object_ptr().is_some());
    assert!(contains.method.as_object_ptr().is_some());
    assert!(values.method.as_object_ptr().is_some());
    assert!(items.method.as_object_ptr().is_some());
    assert!(getitem.method.as_object_ptr().is_some());
    assert!(setitem.method.as_object_ptr().is_some());
    assert!(delitem.method.as_object_ptr().is_some());
    assert!(pop.method.as_object_ptr().is_some());
    assert!(popitem.method.as_object_ptr().is_some());
    assert!(setdefault.method.as_object_ptr().is_some());
    assert!(clear.method.as_object_ptr().is_some());
    assert!(update.method.as_object_ptr().is_some());
    assert!(copy.method.as_object_ptr().is_some());
    assert!(!keys.is_descriptor);
    assert!(!get.is_descriptor);
    assert!(!len.is_descriptor);
    assert!(!contains.is_descriptor);
    assert!(!values.is_descriptor);
    assert!(!items.is_descriptor);
    assert!(!getitem.is_descriptor);
    assert!(!setitem.is_descriptor);
    assert!(!delitem.is_descriptor);
    assert!(!pop.is_descriptor);
    assert!(!popitem.is_descriptor);
    assert!(!setdefault.is_descriptor);
    assert!(!clear.is_descriptor);
    assert!(!update.is_descriptor);
    assert!(!copy.is_descriptor);
}

#[test]
fn test_dict_popitem_returns_latest_entry_and_rejects_empty_dict() {
    let mut dict = DictObject::new();
    dict.set(Value::string(intern("alpha")), Value::int(1).unwrap());
    dict.set(Value::string(intern("beta")), Value::int(2).unwrap());
    let ptr = Box::into_raw(Box::new(dict));
    let value = Value::object_ptr(ptr as *const ());

    let popped = dict_popitem(&[value]).expect("popitem should succeed");
    let popped_ptr = popped
        .as_object_ptr()
        .expect("popitem should return a tuple") as *mut TupleObject;
    let tuple = unsafe { &*popped_ptr };
    assert_eq!(tuple.as_slice()[0], Value::string(intern("beta")));
    assert_eq!(tuple.as_slice()[1].as_int(), Some(2));
    assert_eq!(unsafe { &*ptr }.len(), 1);

    let second = dict_popitem(&[value]).expect("second popitem should also succeed");
    let second_ptr = second
        .as_object_ptr()
        .expect("second popitem should return a tuple") as *mut TupleObject;
    let err = dict_popitem(&[value]).expect_err("empty popitem should fail");
    assert_eq!(err.to_string(), "KeyError: popitem(): dictionary is empty");

    unsafe {
        drop(Box::from_raw(second_ptr));
        drop(Box::from_raw(popped_ptr));
        drop(Box::from_raw(ptr));
    }
}

#[test]
fn test_dict_methods_reject_unhashable_keys() {
    let dict = DictObject::new();
    let dict_ptr = Box::into_raw(Box::new(dict));
    let dict_value = Value::object_ptr(dict_ptr as *const ());
    let key = to_object_value(ListObject::from_slice(&[Value::int(1).unwrap()]));

    for err in [
        dict_contains(&[dict_value, key]).unwrap_err(),
        dict_get(&[dict_value, key]).unwrap_err(),
        dict_setitem(&[dict_value, key, Value::int(1).unwrap()]).unwrap_err(),
    ] {
        assert!(err.to_string().contains("unhashable type: 'list'"));
    }

    unsafe {
        drop(Box::from_raw(
            key.as_object_ptr().unwrap() as *mut ListObject
        ));
        drop(Box::from_raw(dict_ptr));
    }
}

#[test]
fn test_dict_methods_accept_heap_dict_subclasses_with_native_backing() {
    let mut instance = ShapedObject::new_dict_backed(TypeId::from_raw(700), Shape::empty());
    instance
        .dict_backing_mut()
        .expect("dict backing should exist")
        .set(Value::string(intern("existing")), Value::int(1).unwrap());
    let ptr = Box::into_raw(Box::new(instance));
    let value = Value::object_ptr(ptr as *const ());

    dict_setitem(&[
        value,
        Value::string(intern("added")),
        Value::int(9).unwrap(),
    ])
    .expect("dict.__setitem__ should accept dict subclasses");
    assert_eq!(
        dict_contains(&[value, Value::string(intern("existing"))]).unwrap(),
        Value::bool(true)
    );
    assert_eq!(
        dict_getitem(&[value, Value::string(intern("added"))]).unwrap(),
        Value::int(9).unwrap()
    );

    unsafe {
        drop(Box::from_raw(ptr));
    }
}

#[test]
fn test_resolve_mapping_proxy_method_returns_builtin_for_mapping_surface() {
    let keys = resolve_mapping_proxy_method("keys").expect("keys should resolve");
    let values = resolve_mapping_proxy_method("values").expect("values should resolve");
    let items = resolve_mapping_proxy_method("items").expect("items should resolve");
    let get = resolve_mapping_proxy_method("get").expect("get should resolve");
    let len = resolve_mapping_proxy_method("__len__").expect("__len__ should resolve");
    let contains =
        resolve_mapping_proxy_method("__contains__").expect("__contains__ should resolve");
    let copy = resolve_mapping_proxy_method("copy").expect("copy should resolve");

    for method in [keys, values, items, get, len, contains, copy] {
        assert!(method.method.as_object_ptr().is_some());
        assert!(!method.is_descriptor);
    }
}

#[test]
fn test_mappingproxy_methods_cover_heap_class_proxies() {
    let class = Arc::new(PyClassObject::new_simple(intern("ProxyMapping")));
    class.set_attr(intern("token"), Value::int(7).unwrap());
    class.set_attr(intern("label"), Value::string(intern("ready")));

    let proxy_ptr = Box::into_raw(Box::new(MappingProxyObject::for_user_class(Arc::as_ptr(
        &class,
    ))));
    let proxy_value = Value::object_ptr(proxy_ptr as *const ());

    let keys = mappingproxy_keys(&[proxy_value]).expect("keys should succeed");
    let values = mappingproxy_values(&[proxy_value]).expect("values should succeed");
    let items = mappingproxy_items(&[proxy_value]).expect("items should succeed");
    for (value, expected_type) in [
        (keys, TypeId::DICT_KEYS),
        (values, TypeId::DICT_VALUES),
        (items, TypeId::DICT_ITEMS),
    ] {
        let ptr = value.as_object_ptr().expect("view should be object");
        let header = unsafe { &*(ptr as *const ObjectHeader) };
        assert_eq!(header.type_id, expected_type);
        unsafe {
            drop(Box::from_raw(ptr as *mut DictViewObject));
        }
    }

    assert_eq!(
        mappingproxy_get(&[proxy_value, Value::string(intern("token"))])
            .expect("get should succeed")
            .as_int(),
        Some(7)
    );
    assert_eq!(
        mappingproxy_get(&[
            proxy_value,
            Value::string(intern("missing")),
            Value::int(99).unwrap(),
        ])
        .expect("get should return provided default")
        .as_int(),
        Some(99)
    );
    assert_eq!(
        mappingproxy_len(&[proxy_value])
            .expect("__len__ should succeed")
            .as_int(),
        Some(2)
    );
    assert_eq!(
        mappingproxy_contains(&[proxy_value, Value::string(intern("label"))])
            .expect("__contains__ should succeed"),
        Value::bool(true)
    );

    let copy = mappingproxy_copy(&[proxy_value]).expect("copy should succeed");
    let copy_ptr = copy.as_object_ptr().expect("copy should be a dict");
    let copy_dict = unsafe { &*(copy_ptr as *const DictObject) };
    assert_eq!(
        copy_dict
            .get(Value::string(intern("token")))
            .expect("copied token should exist")
            .as_int(),
        Some(7)
    );

    unsafe {
        drop(Box::from_raw(copy_ptr as *mut DictObject));
        drop(Box::from_raw(proxy_ptr));
    }
}

#[test]
fn test_resolve_object_method_returns_builtin_for_rich_comparisons() {
    let eq = resolve_object_method("__eq__").expect("__eq__ should resolve");
    let ne = resolve_object_method("__ne__").expect("__ne__ should resolve");
    assert!(eq.method.as_object_ptr().is_some());
    assert!(ne.method.as_object_ptr().is_some());
    assert!(!eq.is_descriptor);
    assert!(!ne.is_descriptor);
    assert!(resolve_object_method("__lt__").is_none());
}

#[test]
fn test_resolve_object_method_returns_builtin_for_representation_dunders() {
    let repr_method = resolve_object_method("__repr__").expect("__repr__ should resolve");
    let str_method = resolve_object_method("__str__").expect("__str__ should resolve");
    let format_method = resolve_object_method("__format__").expect("__format__ should resolve");

    let repr_builtin = unsafe {
        &*(repr_method
            .method
            .as_object_ptr()
            .expect("__repr__ should be allocated") as *const BuiltinFunctionObject)
    };
    let str_builtin = unsafe {
        &*(str_method
            .method
            .as_object_ptr()
            .expect("__str__ should be allocated") as *const BuiltinFunctionObject)
    };
    let format_builtin = unsafe {
        &*(format_method
            .method
            .as_object_ptr()
            .expect("__format__ should be allocated") as *const BuiltinFunctionObject)
    };

    assert_eq!(repr_builtin.name(), "object.__repr__");
    assert_eq!(str_builtin.name(), "object.__str__");
    assert_eq!(format_builtin.name(), "object.__format__");
}

#[test]
fn test_object_attribute_mutator_wrappers_use_default_attribute_storage() {
    let mut vm = VirtualMachine::new();
    let object = crate::builtins::builtin_object(&[]).expect("object() should succeed");
    let object_ptr = object
        .as_object_ptr()
        .expect("object() should allocate a shaped object");

    object_setattr(
        &mut vm,
        &[
            object,
            Value::string(intern("token")),
            Value::int(42).expect("token should fit"),
        ],
    )
    .expect("object.__setattr__ should set default attributes");

    let shaped = unsafe { &*(object_ptr as *const ShapedObject) };
    assert_eq!(
        shaped
            .get_property("token")
            .expect("token should be stored")
            .as_int(),
        Some(42)
    );

    object_delattr(&mut vm, &[object, Value::string(intern("token"))])
        .expect("object.__delattr__ should delete default attributes");
    let shaped = unsafe { &*(object_ptr as *const ShapedObject) };
    assert!(shaped.get_property("token").is_none());
}

#[test]
fn test_resolve_type_method_renders_builtin_type_repr() {
    let method = resolve_type_method("__repr__").expect("type.__repr__ should resolve");
    let builtin = unsafe {
        &*(method
            .method
            .as_object_ptr()
            .expect("type.__repr__ should be allocated") as *const BuiltinFunctionObject)
    };
    let rendered = builtin
        .call(&[crate::builtins::builtin_type_object_for_type_id(
            TypeId::INT,
        )])
        .expect("type.__repr__(int) should succeed");

    assert_eq!(string_value(rendered), "<class 'int'>");
}

#[test]
fn test_numeric_and_string_resolvers_expose_representation_dunders() {
    for (resolver, owner) in [
        (
            resolve_int_method as fn(&str) -> Option<CachedMethod>,
            "int",
        ),
        (
            resolve_bool_method as fn(&str) -> Option<CachedMethod>,
            "bool",
        ),
        (
            resolve_float_method as fn(&str) -> Option<CachedMethod>,
            "float",
        ),
        (
            resolve_str_method as fn(&str) -> Option<CachedMethod>,
            "str",
        ),
    ] {
        let repr_builtin = unsafe {
            &*(resolver("__repr__")
                .expect("__repr__ should resolve")
                .method
                .as_object_ptr()
                .expect("__repr__ should be allocated")
                as *const BuiltinFunctionObject)
        };
        let str_builtin = unsafe {
            &*(resolver("__str__")
                .expect("__str__ should resolve")
                .method
                .as_object_ptr()
                .expect("__str__ should be allocated")
                as *const BuiltinFunctionObject)
        };
        let format_builtin = unsafe {
            &*(resolver("__format__")
                .expect("__format__ should resolve")
                .method
                .as_object_ptr()
                .expect("__format__ should be allocated")
                as *const BuiltinFunctionObject)
        };

        assert_eq!(repr_builtin.name(), format!("{owner}.__repr__"));
        assert_eq!(str_builtin.name(), format!("{owner}.__str__"));
        assert_eq!(format_builtin.name(), format!("{owner}.__format__"));
    }
}

#[test]
fn test_resolve_int_method_exposes_bit_operations() {
    let bit_length = unsafe {
        &*(resolve_int_method("bit_length")
            .expect("bit_length should resolve")
            .method
            .as_object_ptr()
            .expect("bit_length should be allocated") as *const BuiltinFunctionObject)
    };
    let bit_count = unsafe {
        &*(resolve_int_method("bit_count")
            .expect("bit_count should resolve")
            .method
            .as_object_ptr()
            .expect("bit_count should be allocated") as *const BuiltinFunctionObject)
    };

    assert_eq!(bit_length.name(), "int.bit_length");
    assert_eq!(bit_count.name(), "int.bit_count");
}

#[test]
fn test_resolve_int_method_exposes_add_wrapper() {
    let add = unsafe {
        &*(resolve_int_method("__add__")
            .expect("__add__ should resolve")
            .method
            .as_object_ptr()
            .expect("__add__ should be allocated") as *const BuiltinFunctionObject)
    };

    assert_eq!(add.name(), "int.__add__");
}

#[test]
fn test_int_index_matches_python_descriptor_contract() {
    let int_result = int_index(&[Value::int(42).expect("value should fit")])
        .expect("int.__index__ should accept exact ints");
    assert_eq!(int_result.as_int(), Some(42));

    let bool_result = int_index(&[Value::bool(true)]).expect("bool should inherit int.__index__");
    assert_eq!(bool_result.as_int(), Some(1));

    let error = int_index(&[Value::none()])
        .expect_err("non-int receiver should fail descriptor validation");
    assert!(matches!(error, BuiltinError::TypeError(_)));
}

#[test]
fn test_int_add_matches_python_descriptor_contract() {
    let result = int_add(&[
        bigint_to_value(num_bigint::BigInt::from(i64::MAX)),
        Value::int(1).expect("one should fit"),
    ])
    .expect("int.__add__ should accept integer operands");
    assert_eq!(
        prism_runtime::types::int::value_to_bigint(result),
        Some(num_bigint::BigInt::from(i64::MAX) + num_bigint::BigInt::from(1_i64))
    );

    let bool_result = int_add(&[Value::bool(true), Value::int(2).expect("two should fit")])
        .expect("bool should be accepted as an int receiver");
    assert_eq!(bool_result.as_int(), Some(3));

    let unsupported = int_add(&[Value::int(1).expect("one should fit"), Value::none()])
        .expect("unsupported rhs should return NotImplemented");
    assert_eq!(unsupported, builtin_not_implemented_value());

    let error = int_add(&[Value::none(), Value::int(1).expect("one should fit")])
        .expect_err("non-int receiver should fail descriptor validation");
    assert!(matches!(error, BuiltinError::TypeError(_)));
}

#[test]
fn test_int_bit_operations_match_python_magnitude_rules() {
    let bit_length =
        int_bit_length(&[Value::int(-37).unwrap()]).expect("bit_length should accept ints");
    assert_eq!(bit_length.as_int(), Some(6));

    let bit_count =
        int_bit_count(&[Value::int(-37).unwrap()]).expect("bit_count should accept ints");
    assert_eq!(bit_count.as_int(), Some(3));

    let boolean = int_bit_length(&[Value::bool(true)]).expect("bool should inherit int APIs");
    assert_eq!(boolean.as_int(), Some(1));
}

#[test]
fn test_resolve_exception_method_returns_builtin_for_with_traceback() {
    let with_traceback =
        resolve_exception_method("with_traceback").expect("with_traceback should resolve");
    let ptr = with_traceback
        .method
        .as_object_ptr()
        .expect("with_traceback should be allocated");
    let builtin = unsafe { &*(ptr as *const BuiltinFunctionObject) };
    assert_eq!(builtin.name(), "BaseException.with_traceback");
    assert!(!with_traceback.is_descriptor);
}

#[test]
fn test_object_rich_comparisons_follow_identity_default() {
    let mut vm = VirtualMachine::new();
    let same = Value::int(7).unwrap();
    assert_eq!(
        object_eq(&[same, same]).expect("__eq__ should accept two operands"),
        Value::bool(true)
    );
    assert_eq!(
        object_ne(&mut vm, &[same, same]).expect("__ne__ should accept two operands"),
        Value::bool(false)
    );

    let lhs = Value::int(7).unwrap();
    let rhs = Value::int(8).unwrap();
    assert_eq!(
        object_eq(&[lhs, rhs]).expect("__eq__ should accept mismatched operands"),
        builtin_not_implemented_value()
    );
    assert_eq!(
        object_ne(&mut vm, &[lhs, rhs]).expect("__ne__ should accept mismatched operands"),
        Value::bool(true)
    );
}

#[test]
fn test_resolve_str_method_returns_builtin_for_upper() {
    let upper = resolve_str_method("upper").expect("upper should resolve");
    assert!(upper.method.as_object_ptr().is_some());
    assert!(!upper.is_descriptor);
}

#[test]
fn test_resolve_str_method_returns_builtin_for_lower() {
    let lower = resolve_str_method("lower").expect("lower should resolve");
    assert!(lower.method.as_object_ptr().is_some());
    assert!(!lower.is_descriptor);
}

#[test]
fn test_resolve_str_method_returns_builtin_for_capitalize() {
    let capitalize = resolve_str_method("capitalize").expect("capitalize should resolve");
    assert!(capitalize.method.as_object_ptr().is_some());
    assert!(!capitalize.is_descriptor);
}

#[test]
fn test_resolve_str_method_returns_builtin_for_replace() {
    let replace = resolve_str_method("replace").expect("replace should resolve");
    assert!(replace.method.as_object_ptr().is_some());
    assert!(!replace.is_descriptor);
}

#[test]
fn test_resolve_str_method_returns_builtin_for_remove_affix_methods() {
    for name in ["removeprefix", "removesuffix"] {
        let method = resolve_str_method(name).unwrap_or_else(|| panic!("{name} should resolve"));
        assert!(method.method.as_object_ptr().is_some());
        assert!(!method.is_descriptor);
    }
}

#[test]
fn test_resolve_str_method_returns_builtin_for_split() {
    let split = resolve_str_method("split").expect("split should resolve");
    assert!(split.method.as_object_ptr().is_some());
    assert!(!split.is_descriptor);
    let rsplit = resolve_str_method("rsplit").expect("rsplit should resolve");
    assert!(rsplit.method.as_object_ptr().is_some());
    assert!(!rsplit.is_descriptor);
}

#[test]
fn test_resolve_str_method_returns_builtin_for_splitlines() {
    let splitlines = resolve_str_method("splitlines").expect("splitlines should resolve");
    assert!(splitlines.method.as_object_ptr().is_some());
    assert!(!splitlines.is_descriptor);
}

#[test]
fn test_resolve_str_method_returns_builtin_for_expandtabs() {
    let expandtabs = resolve_str_method("expandtabs").expect("expandtabs should resolve");
    assert!(expandtabs.method.as_object_ptr().is_some());
    assert!(!expandtabs.is_descriptor);
}

#[test]
fn test_resolve_str_method_returns_builtin_for_strip_family() {
    for name in ["strip", "lstrip", "rstrip"] {
        let method = resolve_str_method(name).expect("strip-family method should resolve");
        assert!(
            method.method.as_object_ptr().is_some(),
            "{name} should resolve"
        );
        assert!(!method.is_descriptor, "{name} should not be a descriptor");
    }
}

#[test]
fn test_resolve_str_method_returns_builtin_for_join() {
    let join = resolve_str_method("join").expect("join should resolve");
    assert!(join.method.as_object_ptr().is_some());
    assert!(!join.is_descriptor);
}

#[test]
fn test_resolve_str_method_returns_builtin_for_format() {
    let format = resolve_str_method("format").expect("format should resolve");
    assert!(format.method.as_object_ptr().is_some());
    assert!(!format.is_descriptor);
}

#[test]
fn test_resolve_str_method_returns_builtin_for_isidentifier() {
    let method = resolve_str_method("isidentifier").expect("isidentifier should resolve");
    assert!(method.method.as_object_ptr().is_some());
    assert!(!method.is_descriptor);
}

#[test]
fn test_resolve_str_method_returns_builtin_for_startswith_and_endswith() {
    let startswith = resolve_str_method("startswith").expect("startswith should resolve");
    let endswith = resolve_str_method("endswith").expect("endswith should resolve");
    let partition = resolve_str_method("partition").expect("partition should resolve");
    let rpartition = resolve_str_method("rpartition").expect("rpartition should resolve");
    assert!(startswith.method.as_object_ptr().is_some());
    assert!(endswith.method.as_object_ptr().is_some());
    assert!(partition.method.as_object_ptr().is_some());
    assert!(rpartition.method.as_object_ptr().is_some());
    assert!(!startswith.is_descriptor);
    assert!(!endswith.is_descriptor);
    assert!(!partition.is_descriptor);
    assert!(!rpartition.is_descriptor);
}

#[test]
fn test_resolve_str_method_returns_builtin_for_find_family_and_predicates() {
    for name in [
        "find",
        "rfind",
        "index",
        "rindex",
        "count",
        "translate",
        "isascii",
        "isalpha",
        "isdigit",
        "isalnum",
        "isspace",
        "isupper",
        "islower",
        "isdecimal",
        "isnumeric",
        "istitle",
    ] {
        let method = resolve_str_method(name).unwrap_or_else(|| panic!("{name} should resolve"));
        assert!(method.method.as_object_ptr().is_some());
        assert!(!method.is_descriptor);
    }
}

#[test]
fn test_str_upper_returns_uppercased_string() {
    let result =
        str_upper(&[Value::string(prism_core::intern::intern("Path"))]).expect("upper should work");
    assert_eq!(string_value(result), "PATH");
}

#[test]
fn test_str_lower_returns_lowercased_string_and_reuses_unchanged_receiver() {
    let mixed = Value::string(prism_core::intern::intern("Path"));
    let lowered = str_lower(&[mixed]).expect("lower should work");
    assert_eq!(string_value(lowered), "path");

    let already_lower = Value::string(prism_core::intern::intern("path"));
    let unchanged = str_lower(&[already_lower]).expect("lower should preserve lowercase");
    assert_eq!(unchanged, already_lower);
}

#[test]
fn test_str_translate_applies_mapping_values_and_preserves_missing_chars() {
    let mut vm = VirtualMachine::new();
    let mut table = DictObject::new();
    table.set(
        Value::int('a' as i64).unwrap(),
        Value::int('A' as i64).unwrap(),
    );
    table.set(
        Value::int('.' as i64).unwrap(),
        Value::string(intern("\\.")),
    );
    table.set(Value::int('x' as i64).unwrap(), Value::none());
    let table_ptr = Box::into_raw(Box::new(table));
    let table_value = Value::object_ptr(table_ptr as *const ());

    let translated = str_translate(&mut vm, &[Value::string(intern("a.x!")), table_value])
        .expect("translate should apply integer, string, and deletion mappings");
    assert_eq!(string_value(translated), "A\\.!");

    unsafe {
        drop(Box::from_raw(table_ptr));
    }
}

#[test]
fn test_str_translate_reuses_receiver_and_validates_mapping_results() {
    let mut vm = VirtualMachine::new();
    let empty_table_ptr = Box::into_raw(Box::new(DictObject::new()));
    let empty_table_value = Value::object_ptr(empty_table_ptr as *const ());
    let source = Value::string(intern("plain"));

    let unchanged = str_translate(&mut vm, &[source, empty_table_value])
        .expect("missing translations should leave characters unchanged");
    assert_eq!(unchanged, source);

    let mut invalid_table = DictObject::new();
    let invalid_replacement = to_object_value(ListObject::new());
    invalid_table.set(Value::int('p' as i64).unwrap(), invalid_replacement);
    let invalid_table_ptr = Box::into_raw(Box::new(invalid_table));
    let invalid_table_value = Value::object_ptr(invalid_table_ptr as *const ());
    let err = str_translate(&mut vm, &[source, invalid_table_value])
        .expect_err("unsupported mapping result should fail");
    assert_eq!(
        err.to_string(),
        "TypeError: character mapping must return integer, None or str"
    );

    unsafe {
        drop(Box::from_raw(
            invalid_replacement.as_object_ptr().unwrap() as *mut ListObject
        ));
        drop(Box::from_raw(
            invalid_table_value.as_object_ptr().unwrap() as *mut DictObject
        ));
        drop(Box::from_raw(
            empty_table_value.as_object_ptr().unwrap() as *mut DictObject
        ));
    }
}

#[test]
fn test_str_capitalize_uppercases_first_character_and_lowercases_rest() {
    let result = str_capitalize(&[Value::string(intern("hELLO"))]).expect("capitalize should work");
    assert_eq!(string_value(result), "Hello");

    let unchanged = str_capitalize(&[Value::string(intern("Hello"))])
        .expect("capitalize should preserve canonical form");
    assert_eq!(unchanged, Value::string(intern("Hello")));
}

#[test]
fn test_str_replace_replaces_occurrences_and_honors_count() {
    let replaced = str_replace(&[
        Value::string(intern("banana")),
        Value::string(intern("na")),
        Value::string(intern("NA")),
    ])
    .expect("replace should work");
    assert_eq!(string_value(replaced), "baNANA");

    let counted = str_replace(&[
        Value::string(intern("banana")),
        Value::string(intern("na")),
        Value::string(intern("NA")),
        Value::int(1).unwrap(),
    ])
    .expect("counted replace should work");
    assert_eq!(string_value(counted), "baNAna");
}

#[test]
fn test_str_replace_accepts_cpython_keyword_arguments() {
    let replaced = str_replace_kw(
        &[Value::string(intern("banana"))],
        &[
            ("old", Value::string(intern("na"))),
            ("new", Value::string(intern("NA"))),
            ("count", Value::int(1).unwrap()),
        ],
    )
    .expect("keyword replace should work");
    assert_eq!(string_value(replaced), "baNAna");

    let mixed = str_replace_kw(
        &[Value::string(intern("banana")), Value::string(intern("na"))],
        &[("new", Value::string(intern("NA")))],
    )
    .expect("mixed positional and keyword replace should work");
    assert_eq!(string_value(mixed), "baNANA");

    let duplicate = str_replace_kw(
        &[Value::string(intern("banana")), Value::string(intern("na"))],
        &[("old", Value::string(intern("a")))],
    )
    .expect_err("duplicate old argument should fail");
    assert!(duplicate.to_string().contains("multiple values"));
}

#[test]
fn test_str_replace_supports_empty_pattern_and_bool_count() {
    let all = str_replace(&[
        Value::string(intern("abc")),
        Value::string(intern("")),
        Value::string(intern("-")),
    ])
    .expect("replace with empty pattern should work");
    assert_eq!(string_value(all), "-a-b-c-");

    let limited = str_replace(&[
        Value::string(intern("abc")),
        Value::string(intern("")),
        Value::string(intern("-")),
        Value::int(3).unwrap(),
    ])
    .expect("counted empty-pattern replace should work");
    assert_eq!(string_value(limited), "-a-b-c");

    let bool_count = str_replace(&[
        Value::string(intern("aaaa")),
        Value::string(intern("a")),
        Value::string(intern("b")),
        Value::bool(true),
    ])
    .expect("bool count should be treated as an int");
    assert_eq!(string_value(bool_count), "baaa");
}

#[test]
fn test_str_replace_reuses_receiver_when_nothing_changes() {
    let receiver = Value::string(intern("banana"));

    let unchanged = str_replace(&[
        receiver,
        Value::string(intern("x")),
        Value::string(intern("y")),
    ])
    .expect("replace should succeed when there is nothing to replace");
    assert_eq!(unchanged, receiver);

    let zero_count = str_replace(&[
        receiver,
        Value::string(intern("a")),
        Value::string(intern("b")),
        Value::bool(false),
    ])
    .expect("replace with count=False should succeed");
    assert_eq!(zero_count, receiver);
}

#[test]
fn test_str_replace_rejects_non_string_operands_and_non_int_count() {
    let non_string_old = str_replace(&[
        Value::string(intern("banana")),
        Value::int(1).unwrap(),
        Value::string(intern("b")),
    ])
    .expect_err("old pattern must be a string");
    assert!(
        non_string_old
            .to_string()
            .contains("str.replace() argument 1 must be str")
    );

    let non_string_new = str_replace(&[
        Value::string(intern("banana")),
        Value::string(intern("a")),
        Value::int(1).unwrap(),
    ])
    .expect_err("replacement must be a string");
    assert!(
        non_string_new
            .to_string()
            .contains("str.replace() argument 2 must be str")
    );

    let non_int_count = str_replace(&[
        Value::string(intern("banana")),
        Value::string(intern("a")),
        Value::string(intern("b")),
        Value::float(1.5),
    ])
    .expect_err("count must be integer-like");
    assert!(
        non_int_count
            .to_string()
            .contains("str.replace() argument 3 must be int")
    );
}

#[test]
fn test_str_remove_affix_methods_match_python_contract() {
    let without_prefix =
        str_removeprefix(&[Value::string(intern("spam")), Value::string(intern("sp"))])
            .expect("removeprefix should remove matching prefix");
    assert_eq!(string_value(without_prefix), "am");

    let without_suffix =
        str_removesuffix(&[Value::string(intern("spam")), Value::string(intern("am"))])
            .expect("removesuffix should remove matching suffix");
    assert_eq!(string_value(without_suffix), "sp");

    let full_prefix = str_removeprefix(&[
        Value::string(intern("abcde")),
        Value::string(intern("abcde")),
    ])
    .expect("removeprefix should handle full-string matches");
    assert_eq!(string_value(full_prefix), "");

    let full_suffix = str_removesuffix(&[
        Value::string(intern("abcde")),
        Value::string(intern("abcde")),
    ])
    .expect("removesuffix should handle full-string matches");
    assert_eq!(string_value(full_suffix), "");
}

#[test]
fn test_str_remove_affix_reuses_receiver_for_noop_cases() {
    let receiver = Value::string(intern("spam"));

    let missing_prefix = str_removeprefix(&[receiver, Value::string(intern("python"))])
        .expect("missing prefix should be a no-op");
    assert_eq!(missing_prefix, receiver);

    let empty_prefix = str_removeprefix(&[receiver, Value::string(intern(""))])
        .expect("empty prefix should be a no-op");
    assert_eq!(empty_prefix, receiver);

    let missing_suffix = str_removesuffix(&[receiver, Value::string(intern("python"))])
        .expect("missing suffix should be a no-op");
    assert_eq!(missing_suffix, receiver);

    let empty_suffix = str_removesuffix(&[receiver, Value::string(intern(""))])
        .expect("empty suffix should be a no-op");
    assert_eq!(empty_suffix, receiver);
}

#[test]
fn test_str_remove_affix_rejects_invalid_arguments() {
    let missing =
        str_removeprefix(&[Value::string(intern("hello"))]).expect_err("prefix is required");
    assert!(
        missing
            .to_string()
            .contains("str.removeprefix() takes exactly 1 argument")
    );

    let non_string = str_removesuffix(&[Value::string(intern("hello")), Value::int(42).unwrap()])
        .expect_err("suffix must be a string");
    assert!(
        non_string
            .to_string()
            .contains("str.removesuffix() argument 1 must be str")
    );

    let tuple_affix = to_object_value(TupleObject::from_slice(&[
        Value::string(intern("he")),
        Value::string(intern("l")),
    ]));
    let tuple_err = str_removeprefix(&[Value::string(intern("hello")), tuple_affix])
        .expect_err("tuple prefixes are not accepted");
    assert!(
        tuple_err
            .to_string()
            .contains("str.removeprefix() argument 1 must be str")
    );
}

#[test]
fn test_str_split_supports_explicit_separator_and_maxsplit() {
    let result = str_split(&[
        Value::string(intern("a::b::c")),
        Value::string(intern("::")),
        Value::int(1).unwrap(),
    ])
    .expect("split with separator should work");

    let result_ptr = result.as_object_ptr().expect("split should return a list");
    let list = unsafe { &*(result_ptr as *const ListObject) };
    let values: Vec<String> = list.as_slice().iter().copied().map(string_value).collect();
    assert_eq!(values, vec!["a".to_string(), "b::c".to_string()]);
}

#[test]
fn test_str_split_accepts_sep_and_maxsplit_keywords() {
    let result = str_split_kw(
        &[Value::string(intern("3.12.0"))],
        &[
            ("sep", Value::string(intern("."))),
            ("maxsplit", Value::int(1).unwrap()),
        ],
    )
    .expect("split keywords should work");

    let result_ptr = result.as_object_ptr().expect("split should return a list");
    let list = unsafe { &*(result_ptr as *const ListObject) };
    let values: Vec<String> = list.as_slice().iter().copied().map(string_value).collect();
    assert_eq!(values, vec!["3".to_string(), "12.0".to_string()]);
}

#[test]
fn test_str_split_supports_whitespace_splitting_and_none_separator() {
    let whitespace = str_split(&[
        Value::string(intern("  alpha   beta gamma  ")),
        Value::none(),
        Value::int(1).unwrap(),
    ])
    .expect("split with implicit whitespace should work");
    let whitespace_ptr = whitespace
        .as_object_ptr()
        .expect("whitespace split should return a list");
    let whitespace_list = unsafe { &*(whitespace_ptr as *const ListObject) };
    let values: Vec<String> = whitespace_list
        .as_slice()
        .iter()
        .copied()
        .map(string_value)
        .collect();
    assert_eq!(
        values,
        vec!["alpha".to_string(), "beta gamma  ".to_string()]
    );

    let zero = str_split(&[
        Value::string(intern("   keep  spacing")),
        Value::none(),
        Value::int(0).unwrap(),
    ])
    .expect("split with maxsplit=0 should work");
    let zero_ptr = zero
        .as_object_ptr()
        .expect("zero split should return a list");
    let zero_list = unsafe { &*(zero_ptr as *const ListObject) };
    let zero_values: Vec<String> = zero_list
        .as_slice()
        .iter()
        .copied()
        .map(string_value)
        .collect();
    assert_eq!(zero_values, vec!["keep  spacing".to_string()]);
}

#[test]
fn test_str_split_rejects_empty_separator() {
    let err = str_split(&[Value::string(intern("abc")), Value::string(intern(""))])
        .expect_err("empty separator should fail");
    assert!(err.to_string().contains("empty separator"));
}

#[test]
fn test_str_rsplit_supports_explicit_separator_and_maxsplit() {
    let result = str_rsplit(&[
        Value::string(intern("os.confstr")),
        Value::string(intern(".")),
        Value::int(1).unwrap(),
    ])
    .expect("rsplit with separator should work");

    let result_ptr = result.as_object_ptr().expect("rsplit should return a list");
    let list = unsafe { &*(result_ptr as *const ListObject) };
    let values: Vec<String> = list.as_slice().iter().copied().map(string_value).collect();
    assert_eq!(values, vec!["os".to_string(), "confstr".to_string()]);
}

#[test]
fn test_str_rsplit_accepts_keywords_and_rejects_duplicates() {
    let result = str_rsplit_kw(
        &[Value::string(intern("a.b.c"))],
        &[
            ("sep", Value::string(intern("."))),
            ("maxsplit", Value::int(1).unwrap()),
        ],
    )
    .expect("rsplit keywords should work");

    let result_ptr = result.as_object_ptr().expect("rsplit should return a list");
    let list = unsafe { &*(result_ptr as *const ListObject) };
    let values: Vec<String> = list.as_slice().iter().copied().map(string_value).collect();
    assert_eq!(values, vec!["a.b".to_string(), "c".to_string()]);

    let duplicate = str_split_kw(
        &[Value::string(intern("a.b")), Value::string(intern("."))],
        &[("sep", Value::string(intern(",")))],
    )
    .expect_err("duplicate sep must be rejected");
    assert!(
        duplicate
            .to_string()
            .contains("multiple values for argument 'sep'")
    );
}

#[test]
fn test_str_rsplit_preserves_left_whitespace_remainder() {
    let result = str_rsplit(&[
        Value::string(intern("  alpha   beta gamma  ")),
        Value::none(),
        Value::int(1).unwrap(),
    ])
    .expect("rsplit with implicit whitespace should work");

    let result_ptr = result.as_object_ptr().expect("rsplit should return a list");
    let list = unsafe { &*(result_ptr as *const ListObject) };
    let values: Vec<String> = list.as_slice().iter().copied().map(string_value).collect();
    assert_eq!(
        values,
        vec!["alpha   beta".to_string(), "gamma".to_string()]
    );
}

#[test]
fn test_str_splitlines_handles_crlf_and_optional_keepends() {
    let without_keepends = str_splitlines(&[Value::string(intern("alpha\r\nbeta\n"))])
        .expect("splitlines should work");
    let without_ptr = without_keepends
        .as_object_ptr()
        .expect("splitlines should return a list");
    let without_list = unsafe { &*(without_ptr as *const ListObject) };
    let without_values: Vec<String> = without_list
        .iter()
        .map(|value| {
            string_object_from_value(*value)
                .unwrap()
                .as_str()
                .to_string()
        })
        .collect();
    assert_eq!(
        without_values,
        vec!["alpha".to_string(), "beta".to_string()]
    );

    let with_keepends =
        str_splitlines(&[Value::string(intern("alpha\r\nbeta\n")), Value::bool(true)])
            .expect("splitlines keepends should work");
    let with_ptr = with_keepends
        .as_object_ptr()
        .expect("splitlines should return a list");
    let with_list = unsafe { &*(with_ptr as *const ListObject) };
    let with_values: Vec<String> = with_list
        .iter()
        .map(|value| {
            string_object_from_value(*value)
                .unwrap()
                .as_str()
                .to_string()
        })
        .collect();
    assert_eq!(
        with_values,
        vec!["alpha\r\n".to_string(), "beta\n".to_string()]
    );
}

#[test]
fn test_str_splitlines_returns_empty_list_for_empty_string() {
    let result = str_splitlines(&[Value::string(intern(""))]).expect("splitlines should work");
    let ptr = result
        .as_object_ptr()
        .expect("splitlines should return a list");
    let list = unsafe { &*(ptr as *const ListObject) };
    assert!(list.is_empty());
}

#[test]
fn test_str_splitlines_accepts_keepends_keyword_argument() {
    let splitlines = builtin_from_value(
        resolve_str_method("splitlines")
            .expect("splitlines should resolve")
            .method,
    );
    let result = splitlines
        .call_with_keywords(
            &[Value::string(intern("a\r\nb"))],
            &[("keepends", Value::bool(true))],
        )
        .expect("splitlines keyword call should succeed");
    let list = unsafe {
        &*(result
            .as_object_ptr()
            .expect("splitlines should return a list") as *const ListObject)
    };
    let values: Vec<String> = list
        .iter()
        .map(|value| {
            string_object_from_value(*value)
                .unwrap()
                .as_str()
                .to_string()
        })
        .collect();
    assert_eq!(values, vec!["a\r\n".to_string(), "b".to_string()]);
}

#[test]
fn test_str_expandtabs_expands_tabs_and_resets_columns_after_newlines() {
    let result = str_expandtabs(&[Value::string(intern("01\t0123\na\r\tb"))])
        .expect("expandtabs should work");
    assert_eq!(string_value(result), "01      0123\na\r        b");
}

#[test]
fn test_str_expandtabs_supports_negative_and_boolean_tab_sizes() {
    let collapsed = str_expandtabs(&[Value::string(intern("a\tb")), Value::int(-1).unwrap()])
        .expect("negative tabsize should collapse tabs");
    assert_eq!(string_value(collapsed), "ab");

    let boolean = str_expandtabs(&[Value::string(intern("a\tb")), Value::bool(true)])
        .expect("bool tabsize should be treated as an integer");
    assert_eq!(string_value(boolean), "a b");
}

#[test]
fn test_str_expandtabs_accepts_tabsize_keyword_argument() {
    let expandtabs = builtin_from_value(
        resolve_str_method("expandtabs")
            .expect("expandtabs should resolve")
            .method,
    );
    let result = expandtabs
        .call_with_keywords(
            &[Value::string(intern("a\tb"))],
            &[("tabsize", Value::int(4).unwrap())],
        )
        .expect("expandtabs keyword call should succeed");
    assert_eq!(string_value(result), "a   b");
}

#[test]
fn test_str_expandtabs_reuses_receiver_when_no_tabs_are_present() {
    let receiver = Value::string(intern("stable"));
    let result = str_expandtabs(&[receiver]).expect("expandtabs should accept default tabsize");
    assert_eq!(result, receiver);
}

#[test]
fn test_str_partition_matches_python_contract() {
    let value = str_partition(&[
        Value::string(intern("alpha.beta.gamma")),
        Value::string(intern(".")),
    ])
    .expect("partition should succeed");
    let tuple_ptr = value
        .as_object_ptr()
        .expect("partition should return a tuple") as *mut TupleObject;
    let tuple = unsafe { &*tuple_ptr };
    assert_eq!(string_value(tuple.as_slice()[0]), "alpha");
    assert_eq!(string_value(tuple.as_slice()[1]), ".");
    assert_eq!(string_value(tuple.as_slice()[2]), "beta.gamma");

    let missing = str_partition(&[Value::string(intern("alpha")), Value::string(intern("."))])
        .expect("partition should handle missing separator");
    let missing_ptr = missing
        .as_object_ptr()
        .expect("missing partition should return a tuple")
        as *mut TupleObject;
    let missing_tuple = unsafe { &*missing_ptr };
    assert_eq!(string_value(missing_tuple.as_slice()[0]), "alpha");
    assert_eq!(string_value(missing_tuple.as_slice()[1]), "");
    assert_eq!(string_value(missing_tuple.as_slice()[2]), "");

    let empty_separator =
        str_partition(&[Value::string(intern("alpha")), Value::string(intern(""))])
            .expect_err("partition should reject an empty separator");
    assert!(matches!(empty_separator, BuiltinError::ValueError(_)));

    unsafe {
        drop(Box::from_raw(tuple_ptr));
        drop(Box::from_raw(missing_ptr));
    }
}

#[test]
fn test_str_rpartition_matches_python_contract() {
    let value = str_rpartition(&[
        Value::string(intern("alpha.beta.gamma")),
        Value::string(intern(".")),
    ])
    .expect("rpartition should succeed");
    let tuple_ptr = value
        .as_object_ptr()
        .expect("rpartition should return a tuple") as *mut TupleObject;
    let tuple = unsafe { &*tuple_ptr };
    assert_eq!(string_value(tuple.as_slice()[0]), "alpha.beta");
    assert_eq!(string_value(tuple.as_slice()[1]), ".");
    assert_eq!(string_value(tuple.as_slice()[2]), "gamma");

    let missing = str_rpartition(&[Value::string(intern("alpha")), Value::string(intern("."))])
        .expect("rpartition should handle missing separator");
    let missing_ptr = missing
        .as_object_ptr()
        .expect("missing rpartition should return a tuple")
        as *mut TupleObject;
    let missing_tuple = unsafe { &*missing_ptr };
    assert_eq!(string_value(missing_tuple.as_slice()[0]), "");
    assert_eq!(string_value(missing_tuple.as_slice()[1]), "");
    assert_eq!(string_value(missing_tuple.as_slice()[2]), "alpha");

    unsafe {
        drop(Box::from_raw(tuple_ptr));
        drop(Box::from_raw(missing_ptr));
    }
}

#[test]
fn test_str_strip_family_supports_whitespace_none_and_explicit_char_sets() {
    let stripped = str_strip(&[Value::string(intern("  alpha  "))])
        .expect("strip should trim surrounding whitespace");
    assert_eq!(string_value(stripped), "alpha");

    let none_chars = str_lstrip(&[Value::string(intern("\n\t alpha")), Value::none()])
        .expect("lstrip should accept None as the default whitespace matcher");
    assert_eq!(string_value(none_chars), "alpha");

    let explicit = str_rstrip(&[
        Value::string(intern("path\\\\//")),
        Value::string(intern("/\\")),
    ])
    .expect("rstrip should support explicit trim character sets");
    assert_eq!(string_value(explicit), "path");
}

#[test]
fn test_str_strip_family_reuses_receiver_when_nothing_changes_or_chars_are_empty() {
    let receiver = Value::string(intern("stable"));

    let unchanged = str_strip(&[receiver]).expect("strip should succeed");
    assert_eq!(unchanged, receiver);

    let empty_chars = str_rstrip(&[receiver, Value::string(intern(""))])
        .expect("rstrip with empty char set should be a no-op");
    assert_eq!(empty_chars, receiver);
}

#[test]
fn test_str_strip_family_rejects_non_string_char_sets() {
    let err = str_strip(&[Value::string(intern("value")), Value::int(1).unwrap()])
        .expect_err("strip chars must be strings or None");
    assert!(
        err.to_string()
            .contains("str.strip() argument 1 must be str")
    );
}

#[test]
fn test_str_encode_resolves_and_supports_default_and_explicit_codecs() {
    assert!(resolve_str_method("encode").is_some());

    let default_encoded =
        str_encode(&[Value::string(intern("h\u{00e9}"))]).expect("encode should default to utf-8");
    assert_eq!(byte_values(default_encoded), "h\u{00e9}".as_bytes());

    let latin1_encoded = str_encode(&[
        Value::string(intern("\u{00e9}")),
        Value::string(intern("latin-1")),
    ])
    .expect("encode should support latin-1");
    assert_eq!(byte_values(latin1_encoded), vec![0xe9]);

    let ignored = str_encode(&[
        Value::string(intern("A\u{00e9}")),
        Value::string(intern("ascii")),
        Value::string(intern("ignore")),
    ])
    .expect("encode should honor ignore errors");
    assert_eq!(byte_values(ignored), b"A");
}

#[test]
fn test_str_encode_raises_unicode_encode_error_for_strict_failures() {
    let err = str_encode(&[
        Value::string(intern("A\u{00e9}")),
        Value::string(intern("ascii")),
    ])
    .expect_err("strict ascii encoding should fail");

    assert_unicode_encode_error(
        err,
        "'ascii' codec can't encode character '\\xe9' in position 1: ordinal not in range(128)",
    );
}

#[test]
fn test_str_encode_rejects_too_many_arguments() {
    let err = str_encode(&[
        Value::string(intern("abc")),
        Value::string(intern("utf-8")),
        Value::string(intern("strict")),
        Value::string(intern("extra")),
    ])
    .expect_err("encode should reject too many arguments");

    assert_eq!(
        err.to_string(),
        "TypeError: encode() takes at most 2 arguments (3 given)"
    );
}

#[test]
fn test_str_join_concatenates_iterable_of_strings() {
    let parts = prism_runtime::types::tuple::TupleObject::from_slice(&[
        Value::string(intern("hits")),
        Value::string(intern("misses")),
        Value::string(intern("maxsize")),
    ]);
    let parts_ptr = Box::into_raw(Box::new(parts));
    let result = str_join(&[
        Value::string(intern(", ")),
        Value::object_ptr(parts_ptr as *const ()),
    ])
    .expect("join should work");
    assert_eq!(string_value(result), "hits, misses, maxsize");

    unsafe {
        drop(Box::from_raw(parts_ptr));
    }
}

#[test]
fn test_str_join_rejects_non_string_items() {
    let parts = prism_runtime::types::tuple::TupleObject::from_slice(&[
        Value::string(intern("hits")),
        Value::int(1).unwrap(),
    ]);
    let parts_ptr = Box::into_raw(Box::new(parts));
    let err = str_join(&[
        Value::string(intern(", ")),
        Value::object_ptr(parts_ptr as *const ()),
    ])
    .expect_err("join should reject non-strings");
    assert!(err.to_string().contains("sequence item 1"));

    unsafe {
        drop(Box::from_raw(parts_ptr));
    }
}

#[test]
fn test_str_isidentifier_accepts_ascii_and_unicode_identifiers() {
    assert_eq!(
        str_isidentifier(&[Value::string(intern("cache_info"))]).unwrap(),
        Value::bool(true)
    );
    assert_eq!(
        str_isidentifier(&[Value::string(intern("_lru_cache_wrapper"))]).unwrap(),
        Value::bool(true)
    );
    assert_eq!(
        str_isidentifier(&[Value::string(intern("cafe"))]).unwrap(),
        Value::bool(true)
    );

    let unicode =
        Value::object_ptr(Box::into_raw(Box::new(StringObject::new("πcache"))) as *const ());
    let unicode_result = str_isidentifier(&[unicode]).unwrap();
    assert_eq!(unicode_result, Value::bool(true));
    unsafe {
        drop(Box::from_raw(
            unicode.as_object_ptr().unwrap() as *mut StringObject
        ));
    }
}

#[test]
fn test_str_isidentifier_rejects_invalid_identifier_forms() {
    for candidate in ["", "2cache", "cache-info", "cache info"] {
        assert_eq!(
            str_isidentifier(&[Value::string(intern(candidate))]).unwrap(),
            Value::bool(false),
            "{candidate:?} should not be a valid identifier"
        );
    }
}

#[test]
fn test_str_startswith_supports_single_prefix_tuple_prefixes_and_bounds() {
    assert_eq!(
        str_startswith(&[
            Value::string(intern("CacheInfo")),
            Value::string(intern("Cache")),
        ])
        .unwrap(),
        Value::bool(true)
    );

    let prefixes = prism_runtime::types::tuple::TupleObject::from_slice(&[
        Value::string(intern("miss")),
        Value::string(intern("Cache")),
    ]);
    let prefixes_ptr = Box::into_raw(Box::new(prefixes));
    let tuple_result = str_startswith(&[
        Value::string(intern("CacheInfo")),
        Value::object_ptr(prefixes_ptr as *const ()),
    ])
    .unwrap();
    assert_eq!(tuple_result, Value::bool(true));

    let bounded = str_startswith(&[
        Value::string(intern("xxCacheInfo")),
        Value::string(intern("Cache")),
        Value::int(2).unwrap(),
        Value::int(7).unwrap(),
    ])
    .unwrap();
    assert_eq!(bounded, Value::bool(true));

    unsafe {
        drop(Box::from_raw(prefixes_ptr));
    }
}

#[test]
fn test_str_endswith_supports_suffixes_and_negative_bounds() {
    assert_eq!(
        str_endswith(&[
            Value::string(intern("functools.py")),
            Value::string(intern(".py")),
        ])
        .unwrap(),
        Value::bool(true)
    );

    let bounded = str_endswith(&[
        Value::string(intern("prefix_suffix_tail")),
        Value::string(intern("suffix")),
        Value::int(7).unwrap(),
        Value::int(-5).unwrap(),
    ])
    .unwrap();
    assert_eq!(bounded, Value::bool(true));
}

#[test]
fn test_str_find_family_supports_bounds_and_missing_substrings() {
    assert_eq!(
        str_find(&[
            Value::string(intern("prefix_suffix_suffix")),
            Value::string(intern("suffix")),
        ])
        .unwrap(),
        Value::int(7).unwrap()
    );
    assert_eq!(
        str_rfind(&[
            Value::string(intern("prefix_suffix_suffix")),
            Value::string(intern("suffix")),
        ])
        .unwrap(),
        Value::int(14).unwrap()
    );
    assert_eq!(
        str_find(&[
            Value::string(intern("prefix_suffix_suffix")),
            Value::string(intern("suffix")),
            Value::int(8).unwrap(),
            Value::int(20).unwrap(),
        ])
        .unwrap(),
        Value::int(14).unwrap()
    );
    assert_eq!(
        str_find(&[
            Value::string(intern("prefix_suffix")),
            Value::string(intern("missing")),
        ])
        .unwrap(),
        Value::int(-1).unwrap()
    );
    assert_eq!(
        str_count(&[Value::string(intern("aaaa")), Value::string(intern("aa")),]).unwrap(),
        Value::int(2).unwrap()
    );
    assert_eq!(
        str_count(&[Value::string(intern("abc")), Value::string(intern("")),]).unwrap(),
        Value::int(4).unwrap()
    );

    let index_err = str_index(&[
        Value::string(intern("prefix_suffix")),
        Value::string(intern("missing")),
    ])
    .expect_err("index should raise when substring is absent");
    assert!(index_err.to_string().contains("substring not found"));

    let rindex_err = str_rindex(&[
        Value::string(intern("prefix_suffix")),
        Value::string(intern("missing")),
    ])
    .expect_err("rindex should raise when substring is absent");
    assert!(rindex_err.to_string().contains("substring not found"));
}

#[test]
fn test_str_character_predicates_match_python_truthiness_rules() {
    assert_eq!(
        str_isalpha(&[Value::string(intern("Prism"))]).unwrap(),
        Value::bool(true)
    );
    assert_eq!(
        str_isdigit(&[Value::string(intern("0123"))]).unwrap(),
        Value::bool(true)
    );
    assert_eq!(
        str_isdecimal(&[Value::string(intern("0123"))]).unwrap(),
        Value::bool(true)
    );
    assert_eq!(
        str_isnumeric(&[Value::string(intern("0123"))]).unwrap(),
        Value::bool(true)
    );
    assert_eq!(
        str_isalnum(&[Value::string(intern("Prism3"))]).unwrap(),
        Value::bool(true)
    );
    assert_eq!(
        str_isspace(&[Value::string(intern("\u{3000}"))]).unwrap(),
        Value::bool(true)
    );
    assert_eq!(
        str_isupper(&[Value::string(intern("PRISM 3"))]).unwrap(),
        Value::bool(true)
    );
    assert_eq!(
        str_islower(&[Value::string(intern("prism 3"))]).unwrap(),
        Value::bool(true)
    );
    assert_eq!(
        str_istitle(&[Value::string(intern("Prism Runtime"))]).unwrap(),
        Value::bool(true)
    );
    assert_eq!(
        str_istitle(&[Value::string(intern("prism Runtime"))]).unwrap(),
        Value::bool(false)
    );
    assert_eq!(
        str_isalpha(&[Value::string(intern(""))]).unwrap(),
        Value::bool(false)
    );
    assert_eq!(
        str_isascii(&[Value::string(intern(""))]).unwrap(),
        Value::bool(true)
    );
    assert_eq!(
        str_isascii(&[Value::string(intern("Prism123"))]).unwrap(),
        Value::bool(true)
    );
    let unicode =
        Value::object_ptr(Box::into_raw(Box::new(StringObject::new("πcache"))) as *const ());
    assert_eq!(str_isascii(&[unicode]).unwrap(), Value::bool(false));
    unsafe {
        drop(Box::from_raw(
            unicode.as_object_ptr().unwrap() as *mut StringObject
        ));
    }
}

#[test]
fn test_str_startswith_rejects_non_string_affix_members() {
    let prefixes = prism_runtime::types::tuple::TupleObject::from_slice(&[
        Value::string(intern("Miss")),
        Value::int(1).unwrap(),
    ]);
    let prefixes_ptr = Box::into_raw(Box::new(prefixes));
    let err = str_startswith(&[
        Value::string(intern("CacheInfo")),
        Value::object_ptr(prefixes_ptr as *const ()),
    ])
    .expect_err("tuple prefixes should reject non-string members");
    assert!(
        err.to_string()
            .contains("first arg must be str or a tuple of str")
    );

    unsafe {
        drop(Box::from_raw(prefixes_ptr));
    }
}

#[test]
fn test_dict_view_methods_return_specialized_view_objects() {
    let dict = Box::new(prism_runtime::types::dict::DictObject::new());
    let dict_ptr = Box::into_raw(dict);
    let dict_value = Value::object_ptr(dict_ptr as *const ());

    let keys = dict_keys(&[dict_value]).expect("keys should work");
    let values = dict_values(&[dict_value]).expect("values should work");
    let items = dict_items(&[dict_value]).expect("items should work");

    for (value, expected_type) in [
        (keys, TypeId::DICT_KEYS),
        (values, TypeId::DICT_VALUES),
        (items, TypeId::DICT_ITEMS),
    ] {
        let result_ptr = value
            .as_object_ptr()
            .expect("dict view should return an object");
        let header = unsafe { &*(result_ptr as *const ObjectHeader) };
        assert_eq!(header.type_id, expected_type);
        unsafe {
            drop(Box::from_raw(result_ptr as *mut DictViewObject));
        }
    }

    unsafe {
        drop(Box::from_raw(dict_ptr));
    }
}

#[test]
fn test_dict_item_methods_mutate_and_read_entries() {
    let mut dict = DictObject::new();
    let dict_ptr = Box::into_raw(Box::new(dict));
    let dict_value = Value::object_ptr(dict_ptr as *const ());
    let key = Value::string(intern("token"));
    let value = Value::int(7).unwrap();

    dict_setitem(&[dict_value, key, value]).expect("__setitem__ should work");
    assert_eq!(
        dict_getitem(&[dict_value, key])
            .expect("__getitem__ should work")
            .as_int(),
        Some(7)
    );
    assert_eq!(
        dict_pop(&[dict_value, key])
            .expect("pop should work")
            .as_int(),
        Some(7)
    );

    unsafe {
        drop(Box::from_raw(dict_ptr));
    }
}

#[test]
fn test_dict_copy_returns_distinct_mapping_with_same_entries() {
    let mut dict = DictObject::new();
    dict.set(Value::string(intern("a")), Value::int(1).unwrap());
    dict.set(Value::string(intern("b")), Value::int(2).unwrap());
    let dict_ptr = Box::into_raw(Box::new(dict));
    let dict_value = Value::object_ptr(dict_ptr as *const ());

    let copied = dict_copy(&[dict_value]).expect("dict.copy should work");
    let copied_ptr = copied
        .as_object_ptr()
        .expect("dict.copy should return a dict object") as *mut DictObject;

    let original = unsafe { &*dict_ptr };
    let duplicate = unsafe { &*copied_ptr };
    assert_eq!(original.len(), duplicate.len());
    assert_eq!(
        duplicate.get(Value::string(intern("a"))).unwrap().as_int(),
        Some(1)
    );
    assert_eq!(
        duplicate.get(Value::string(intern("b"))).unwrap().as_int(),
        Some(2)
    );
    assert_ne!(dict_ptr, copied_ptr);

    unsafe {
        drop(Box::from_raw(copied_ptr));
        drop(Box::from_raw(dict_ptr));
    }
}

#[test]
fn test_dict_get_returns_existing_and_default_values() {
    let mut dict = DictObject::new();
    dict.set(Value::string(intern("token")), Value::int(7).unwrap());
    let dict_ptr = Box::into_raw(Box::new(dict));
    let dict_value = Value::object_ptr(dict_ptr as *const ());

    assert_eq!(
        dict_get(&[dict_value, Value::string(intern("token"))])
            .expect("get should return existing value")
            .as_int(),
        Some(7)
    );
    assert_eq!(
        dict_get(&[
            dict_value,
            Value::string(intern("missing")),
            Value::int(42).unwrap(),
        ])
        .expect("get should return default")
        .as_int(),
        Some(42)
    );
    assert!(
        dict_get(&[dict_value, Value::string(intern("missing"))])
            .expect("get without default should return None")
            .is_none()
    );

    unsafe {
        drop(Box::from_raw(dict_ptr));
    }
}

#[test]
fn test_dict_len_returns_current_size() {
    let mut dict = DictObject::new();
    dict.set(Value::string(intern("a")), Value::int(1).unwrap());
    dict.set(Value::string(intern("b")), Value::int(2).unwrap());
    let dict_ptr = Box::into_raw(Box::new(dict));
    let dict_value = Value::object_ptr(dict_ptr as *const ());

    assert_eq!(
        dict_len(&[dict_value])
            .expect("__len__ should succeed")
            .as_int(),
        Some(2)
    );

    unsafe {
        drop(Box::from_raw(dict_ptr));
    }
}

#[test]
fn test_dict_delitem_and_clear_update_storage() {
    let mut dict = DictObject::new();
    dict.set(Value::string(intern("a")), Value::int(1).unwrap());
    dict.set(Value::string(intern("b")), Value::int(2).unwrap());
    let dict_ptr = Box::into_raw(Box::new(dict));
    let dict_value = Value::object_ptr(dict_ptr as *const ());

    dict_delitem(&[dict_value, Value::string(intern("a"))]).expect("__delitem__ should work");
    let dict_ref = unsafe { &*(dict_ptr as *const DictObject) };
    assert!(!dict_ref.contains_key(Value::string(intern("a"))));
    assert!(dict_ref.contains_key(Value::string(intern("b"))));

    dict_clear(&[dict_value]).expect("clear should work");
    let dict_ref = unsafe { &*(dict_ptr as *const DictObject) };
    assert!(dict_ref.is_empty());

    unsafe {
        drop(Box::from_raw(dict_ptr));
    }
}

#[test]
fn test_dict_setdefault_inserts_once_and_preserves_existing_value() {
    let mut dict = DictObject::new();
    let existing_key = Value::string(intern("existing"));
    dict.set(existing_key, Value::int_unchecked(1));
    let dict_ptr = Box::into_raw(Box::new(dict));
    let dict_value = Value::object_ptr(dict_ptr as *const ());

    let inserted = dict_setdefault(&[dict_value, Value::string(intern("new"))])
        .expect("setdefault should insert a None default");
    let existing = dict_setdefault(&[dict_value, existing_key, Value::int_unchecked(99)])
        .expect("setdefault should preserve the existing value");

    let dict_ref = unsafe { &*dict_ptr };
    assert!(inserted.is_none());
    assert!(
        dict_ref
            .get(Value::string(intern("new")))
            .unwrap()
            .is_none()
    );
    assert_eq!(existing.as_int(), Some(1));
    assert_eq!(dict_ref.get(existing_key).unwrap().as_int(), Some(1));

    unsafe {
        drop(Box::from_raw(dict_ptr));
    }
}

#[test]
fn test_dict_update_merges_other_dict_entries() {
    let mut target = DictObject::new();
    target.set(Value::string(intern("a")), Value::int(1).unwrap());
    let target_ptr = Box::into_raw(Box::new(target));
    let target_value = Value::object_ptr(target_ptr as *const ());

    let mut source = DictObject::new();
    source.set(Value::string(intern("b")), Value::int(2).unwrap());
    source.set(Value::string(intern("a")), Value::int(3).unwrap());
    let source_ptr = Box::into_raw(Box::new(source));
    let source_value = Value::object_ptr(source_ptr as *const ());

    let mut vm = VirtualMachine::new();
    dict_update_with_vm(&mut vm, &[target_value, source_value]).expect("update should work");

    let target_ref = unsafe { &*target_ptr };
    assert_eq!(
        target_ref.get(Value::string(intern("a"))).unwrap().as_int(),
        Some(3)
    );
    assert_eq!(
        target_ref.get(Value::string(intern("b"))).unwrap().as_int(),
        Some(2)
    );

    unsafe {
        drop(Box::from_raw(source_ptr));
        drop(Box::from_raw(target_ptr));
    }
}

#[test]
fn test_dict_update_accepts_iterable_of_pairs() {
    let target_ptr = Box::into_raw(Box::new(DictObject::new()));
    let target_value = Value::object_ptr(target_ptr as *const ());

    let pair_a = Box::into_raw(Box::new(TupleObject::from_slice(&[
        Value::string(intern("left")),
        Value::int(10).unwrap(),
    ])));
    let pair_b = Box::into_raw(Box::new(TupleObject::from_slice(&[
        Value::string(intern("right")),
        Value::int(20).unwrap(),
    ])));
    let items_ptr = Box::into_raw(Box::new(ListObject::from_slice(&[
        Value::object_ptr(pair_a as *const ()),
        Value::object_ptr(pair_b as *const ()),
    ])));
    let items_value = Value::object_ptr(items_ptr as *const ());

    let mut vm = VirtualMachine::new();
    dict_update_with_vm(&mut vm, &[target_value, items_value])
        .expect("iterable update should work");

    let target_ref = unsafe { &*target_ptr };
    assert_eq!(
        target_ref
            .get(Value::string(intern("left")))
            .unwrap()
            .as_int(),
        Some(10)
    );
    assert_eq!(
        target_ref
            .get(Value::string(intern("right")))
            .unwrap()
            .as_int(),
        Some(20)
    );

    unsafe {
        drop(Box::from_raw(items_ptr));
        drop(Box::from_raw(pair_b));
        drop(Box::from_raw(pair_a));
        drop(Box::from_raw(target_ptr));
    }
}

#[test]
fn test_resolve_set_method_returns_builtin_for_mutation_and_membership_surface() {
    for name in [
        "add",
        "remove",
        "discard",
        "pop",
        "clear",
        "update",
        "difference_update",
        "intersection_update",
        "symmetric_difference_update",
        "copy",
        "union",
        "intersection",
        "difference",
        "symmetric_difference",
        "isdisjoint",
        "issubset",
        "issuperset",
        "__contains__",
    ] {
        let method = resolve_set_method(TypeId::SET, name)
            .unwrap_or_else(|| panic!("set.{name} should resolve"));
        assert!(
            method.method.as_object_ptr().is_some(),
            "set.{name} should be heap backed"
        );
        assert!(
            !method.is_descriptor,
            "set.{name} should not be a descriptor"
        );
    }

    for name in [
        "union",
        "intersection",
        "difference",
        "symmetric_difference",
        "isdisjoint",
        "issubset",
        "issuperset",
        "copy",
        "__contains__",
    ] {
        let method = resolve_set_method(TypeId::FROZENSET, name)
            .unwrap_or_else(|| panic!("frozenset.{name} should resolve"));
        assert!(method.method.as_object_ptr().is_some());
        assert!(!method.is_descriptor);
    }
}

#[test]
fn test_set_add_mutates_receiver_and_returns_none() {
    let set = SetObject::new();
    let ptr = Box::into_raw(Box::new(set));
    let value = Value::object_ptr(ptr as *const ());

    let result = set_add(&[value, Value::int(11).unwrap()]).expect("add should work");
    assert!(result.is_none());
    assert!(unsafe { &*ptr }.contains(Value::int(11).unwrap()));

    unsafe {
        drop(Box::from_raw(ptr));
    }
}

#[test]
fn test_set_remove_discard_and_clear_follow_python_mutation_rules() {
    let set = SetObject::from_slice(&[Value::int(1).unwrap(), Value::int(2).unwrap()]);
    let ptr = Box::into_raw(Box::new(set));
    let value = Value::object_ptr(ptr as *const ());

    assert!(
        set_remove(&[value, Value::int(1).unwrap()])
            .expect("remove should succeed")
            .is_none()
    );
    assert!(!unsafe { &*ptr }.contains(Value::int(1).unwrap()));

    assert!(
        set_discard(&[value, Value::int(9).unwrap()])
            .expect("discard should ignore missing values")
            .is_none()
    );

    let error = set_remove(&[value, Value::int(9).unwrap()]).expect_err("remove should fail");
    assert!(matches!(error, BuiltinError::KeyError(_)));

    assert!(set_clear(&[value]).expect("clear should succeed").is_none());
    assert!(unsafe { &*ptr }.is_empty());

    unsafe {
        drop(Box::from_raw(ptr));
    }
}

#[test]
fn test_set_pop_removes_and_returns_single_member() {
    let set = SetObject::from_slice(&[Value::int(11).unwrap()]);
    let ptr = Box::into_raw(Box::new(set));
    let value = Value::object_ptr(ptr as *const ());

    let popped = set_pop(&[value]).expect("set.pop should work");
    assert_eq!(popped.as_int(), Some(11));
    assert!(unsafe { &*ptr }.is_empty());

    unsafe {
        drop(Box::from_raw(ptr));
    }
}

#[test]
fn test_set_pop_errors_on_empty_set() {
    let set = SetObject::new();
    let ptr = Box::into_raw(Box::new(set));
    let value = Value::object_ptr(ptr as *const ());

    let error = set_pop(&[value]).expect_err("empty set.pop should fail");
    assert!(matches!(error, BuiltinError::KeyError(_)));
    assert_eq!(error.to_string(), "KeyError: pop from an empty set");

    unsafe {
        drop(Box::from_raw(ptr));
    }
}

#[test]
fn test_set_update_family_consumes_iterables_with_vm_support() {
    let set = SetObject::from_slice(&[Value::int(1).unwrap(), Value::int(2).unwrap()]);
    let ptr = Box::into_raw(Box::new(set));
    let value = Value::object_ptr(ptr as *const ());
    let mut vm = VirtualMachine::new();

    let update_items = to_object_value(ListObject::from_slice(&[
        Value::int(2).unwrap(),
        Value::int(3).unwrap(),
    ]));
    set_update_with_vm(&mut vm, &[value, update_items]).expect("update should work");
    assert!(unsafe { &*ptr }.contains(Value::int(3).unwrap()));

    let difference_items = to_object_value(TupleObject::from_slice(&[
        Value::int(2).unwrap(),
        Value::int(7).unwrap(),
    ]));
    set_difference_update_with_vm(&mut vm, &[value, difference_items])
        .expect("difference_update should work");
    assert!(!unsafe { &*ptr }.contains(Value::int(2).unwrap()));

    let intersection_items = to_object_value(ListObject::from_slice(&[
        Value::int(1).unwrap(),
        Value::int(9).unwrap(),
    ]));
    set_intersection_update_with_vm(&mut vm, &[value, intersection_items])
        .expect("intersection_update should work");
    assert!(unsafe { &*ptr }.contains(Value::int(1).unwrap()));
    assert!(!unsafe { &*ptr }.contains(Value::int(3).unwrap()));

    let symmetric_items = to_object_value(ListObject::from_slice(&[
        Value::int(1).unwrap(),
        Value::int(5).unwrap(),
    ]));
    set_symmetric_difference_update_with_vm(&mut vm, &[value, symmetric_items])
        .expect("symmetric_difference_update should work");
    assert!(!unsafe { &*ptr }.contains(Value::int(1).unwrap()));
    assert!(unsafe { &*ptr }.contains(Value::int(5).unwrap()));

    unsafe {
        drop(Box::from_raw(ptr));
    }
}

#[test]
fn test_set_functional_methods_accept_iterables_and_preserve_receiver_type() {
    let set = SetObject::from_slice(&[Value::int(1).unwrap(), Value::int(2).unwrap()]);
    let set_ptr = Box::into_raw(Box::new(set));
    let set_value = Value::object_ptr(set_ptr as *const ());
    let mut vm = VirtualMachine::new();

    let union_items = to_object_value(ListObject::from_slice(&[
        Value::int(2).unwrap(),
        Value::int(3).unwrap(),
    ]));
    let union =
        set_union_with_vm(&mut vm, &[set_value, union_items]).expect("set.union should work");
    let union_ptr = union
        .as_object_ptr()
        .expect("set.union should return a set") as *mut SetObject;
    assert_eq!(unsafe { &*set_ptr }.len(), 2);
    assert!(unsafe { &*union_ptr }.contains(Value::int(3).unwrap()));

    let difference_items = to_object_value(TupleObject::from_slice(&[Value::int(2).unwrap()]));
    let difference = set_difference_with_vm(&mut vm, &[set_value, difference_items])
        .expect("set.difference should work");
    let difference_ptr = difference
        .as_object_ptr()
        .expect("set.difference should return a set") as *mut SetObject;
    assert!(unsafe { &*difference_ptr }.contains(Value::int(1).unwrap()));
    assert!(!unsafe { &*difference_ptr }.contains(Value::int(2).unwrap()));

    let subset_items = to_object_value(ListObject::from_slice(&[
        Value::int(1).unwrap(),
        Value::int(2).unwrap(),
        Value::int(9).unwrap(),
    ]));
    assert_eq!(
        set_issubset_with_vm(&mut vm, &[set_value, subset_items]).unwrap(),
        Value::bool(true)
    );

    unsafe {
        drop(Box::from_raw(union_ptr));
        drop(Box::from_raw(difference_ptr));
        drop(Box::from_raw(set_ptr));
    }

    let mut frozen = SetObject::from_slice(&[Value::int(4).unwrap(), Value::int(5).unwrap()]);
    frozen.header.type_id = TypeId::FROZENSET;
    let frozen_ptr = Box::into_raw(Box::new(frozen));
    let frozen_value = Value::object_ptr(frozen_ptr as *const ());
    let frozen_union_items = to_object_value(ListObject::from_slice(&[Value::int(6).unwrap()]));
    let frozen_union = frozenset_union_with_vm(&mut vm, &[frozen_value, frozen_union_items])
        .expect("frozenset.union should work");
    let frozen_union_ptr = frozen_union
        .as_object_ptr()
        .expect("frozenset.union should return a frozenset");
    assert_eq!(
        unsafe { &*(frozen_union_ptr as *const ObjectHeader) }.type_id,
        TypeId::FROZENSET
    );
    assert!(unsafe { &*(frozen_union_ptr as *const SetObject) }.contains(Value::int(6).unwrap()));

    unsafe {
        drop(Box::from_raw(frozen_union_ptr as *mut SetObject));
        drop(Box::from_raw(frozen_ptr));
    }
}

#[test]
fn test_set_membership_and_mutation_methods_reject_unhashable_values() {
    let set = SetObject::new();
    let set_ptr = Box::into_raw(Box::new(set));
    let set_value = Value::object_ptr(set_ptr as *const ());
    let key = to_object_value(ListObject::from_slice(&[Value::int(1).unwrap()]));

    for err in [
        set_add(&[set_value, key]).unwrap_err(),
        set_contains(&[set_value, key]).unwrap_err(),
        set_discard(&[set_value, key]).unwrap_err(),
    ] {
        assert!(err.to_string().contains("unhashable type: 'list'"));
    }

    unsafe {
        drop(Box::from_raw(
            key.as_object_ptr().unwrap() as *mut ListObject
        ));
        drop(Box::from_raw(set_ptr));
    }
}

#[test]
fn test_set_copy_returns_distinct_set() {
    let set = SetObject::from_slice(&[Value::int(1).unwrap(), Value::int(2).unwrap()]);
    let ptr = Box::into_raw(Box::new(set));
    let value = Value::object_ptr(ptr as *const ());

    let copied = set_copy(&[value]).expect("set.copy should work");
    let copied_ptr = copied
        .as_object_ptr()
        .expect("set.copy should return an object") as *mut SetObject;

    assert!(unsafe { &*copied_ptr }.contains(Value::int(1).unwrap()));
    assert!(unsafe { &*copied_ptr }.contains(Value::int(2).unwrap()));
    assert_ne!(ptr, copied_ptr);

    unsafe {
        drop(Box::from_raw(copied_ptr));
        drop(Box::from_raw(ptr));
    }
}

#[test]
fn test_frozenset_contains_reports_membership() {
    let mut frozenset = SetObject::from_slice(&[Value::int(1).unwrap(), Value::int(2).unwrap()]);
    frozenset.header.type_id = TypeId::FROZENSET;
    let ptr = Box::into_raw(Box::new(frozenset));
    let value = Value::object_ptr(ptr as *const ());

    let present =
        frozenset_contains(&[value, Value::int(2).unwrap()]).expect("contains should work");
    let missing =
        frozenset_contains(&[value, Value::int(9).unwrap()]).expect("contains should work");
    assert_eq!(present.as_bool(), Some(true));
    assert_eq!(missing.as_bool(), Some(false));

    unsafe {
        drop(Box::from_raw(ptr));
    }
}

#[test]
fn test_frozenset_copy_returns_same_object() {
    let mut frozenset = SetObject::from_slice(&[Value::int(1).unwrap(), Value::int(2).unwrap()]);
    frozenset.header.type_id = TypeId::FROZENSET;
    let ptr = Box::into_raw(Box::new(frozenset));
    let value = Value::object_ptr(ptr as *const ());

    let copied = frozenset_copy(&[value]).expect("frozenset.copy should work");
    assert_eq!(copied, value);

    unsafe {
        drop(Box::from_raw(ptr));
    }
}
