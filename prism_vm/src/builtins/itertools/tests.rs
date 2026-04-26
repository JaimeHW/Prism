use super::*;
use crate::builtins::iter_dispatch;
use prism_compiler::Compiler;
use prism_parser::parse;
use prism_runtime::types::int::{bigint_to_value, value_to_i64};
use prism_runtime::types::list::ListObject;
use prism_runtime::types::tuple::TupleObject;
use std::sync::Arc;

fn execute(source: &str) -> Result<Value, String> {
    let module = parse(source).map_err(|err| format!("parse error: {err:?}"))?;
    let code = Compiler::compile_module(&module, "<itertools-test>")
        .map_err(|err| format!("compile error: {err:?}"))?;
    crate::run(Arc::new(code)).map_err(|err| format!("runtime error: {err:?}"))
}

#[test]
fn test_range_validation() {
    // Zero step should error
    let result = builtin_range(&[
        Value::int(0).unwrap(),
        Value::int(10).unwrap(),
        Value::int(0).unwrap(),
    ]);
    assert!(result.is_err());
}

#[test]
fn test_range_type_error() {
    // Non-integer should error
    let result = builtin_range(&[Value::float(3.14)]);
    assert!(result.is_err());
}

#[test]
fn test_range_accepts_heap_backed_bigint_stop() {
    let big_stop = bigint_to_value(num_bigint::BigInt::from(1_u8) << 1000_u32);
    let range = builtin_range(&[big_stop]).expect("range should accept bigint stop");
    let iter = builtin_iter(&[range]).expect("iter(range(bigint)) should succeed");

    let mut iter =
        iter_dispatch::get_iterator_mut(&iter).expect("iter() should return an iterator object");
    assert_eq!(value_to_i64(iter.next().unwrap()), Some(0));
    assert_eq!(value_to_i64(iter.next().unwrap()), Some(1));
    assert_eq!(value_to_i64(iter.next().unwrap()), Some(2));
}

// -------------------------------------------------------------------------
// all() tests
// -------------------------------------------------------------------------

#[test]
fn test_all_empty_list() {
    // all([]) = True (vacuous truth)
    let list = ListObject::new();
    let ptr = Box::leak(Box::new(list)) as *mut ListObject as *const ();
    let value = Value::object_ptr(ptr);

    let result = builtin_all(&[value]).unwrap();
    assert!(result.as_bool().unwrap());
}

#[test]
fn test_all_all_truthy() {
    // all([1, 2, 3]) = True
    let list = ListObject::from_slice(&[
        Value::int(1).unwrap(),
        Value::int(2).unwrap(),
        Value::int(3).unwrap(),
    ]);
    let ptr = Box::leak(Box::new(list)) as *mut ListObject as *const ();
    let value = Value::object_ptr(ptr);

    let result = builtin_all(&[value]).unwrap();
    assert!(result.as_bool().unwrap());
}

#[test]
fn test_all_one_falsy() {
    // all([1, 0, 3]) = False
    let list = ListObject::from_slice(&[
        Value::int(1).unwrap(),
        Value::int(0).unwrap(),
        Value::int(3).unwrap(),
    ]);
    let ptr = Box::leak(Box::new(list)) as *mut ListObject as *const ();
    let value = Value::object_ptr(ptr);

    let result = builtin_all(&[value]).unwrap();
    assert!(!result.as_bool().unwrap());
}

#[test]
fn test_all_first_falsy() {
    // all([0, 1, 2]) = False (early exit)
    let list = ListObject::from_slice(&[
        Value::int(0).unwrap(),
        Value::int(1).unwrap(),
        Value::int(2).unwrap(),
    ]);
    let ptr = Box::leak(Box::new(list)) as *mut ListObject as *const ();
    let value = Value::object_ptr(ptr);

    let result = builtin_all(&[value]).unwrap();
    assert!(!result.as_bool().unwrap());
}

#[test]
fn test_filter_accepts_iterator_input() {
    let list = ListObject::from_slice(&[
        Value::int(0).unwrap(),
        Value::int(1).unwrap(),
        Value::int(2).unwrap(),
    ]);
    let ptr = Box::leak(Box::new(list)) as *mut ListObject as *const ();
    let iterator = builtin_iter(&[Value::object_ptr(ptr)]).expect("iter(list) should succeed");
    let filter =
        builtin_filter(&[Value::none(), iterator]).expect("filter should accept iterator inputs");
    let mut filter =
        iter_dispatch::get_iterator_mut(&filter).expect("filter should return iterator");

    assert_eq!(filter.next().unwrap().as_int(), Some(1));
    assert_eq!(filter.next().unwrap().as_int(), Some(2));
    assert!(filter.next().is_none());
}

#[test]
fn test_enumerate_accepts_iterator_input() {
    let list = ListObject::from_slice(&[Value::int(7).unwrap(), Value::int(8).unwrap()]);
    let ptr = Box::leak(Box::new(list)) as *mut ListObject as *const ();
    let iterator = builtin_iter(&[Value::object_ptr(ptr)]).expect("iter(list) should succeed");
    let enumerate =
        builtin_enumerate(&[iterator]).expect("enumerate should accept iterator inputs");
    let mut enumerate =
        iter_dispatch::get_iterator_mut(&enumerate).expect("enumerate should return iterator");

    let first = enumerate.next().unwrap();
    let first_ptr = first.as_object_ptr().expect("enumerate should yield tuple");
    let first = unsafe { &*(first_ptr as *const TupleObject) };
    assert_eq!(first.get(0).unwrap().as_int(), Some(0));
    assert_eq!(first.get(1).unwrap().as_int(), Some(7));

    let second = enumerate.next().unwrap();
    let second_ptr = second
        .as_object_ptr()
        .expect("enumerate should yield tuple");
    let second = unsafe { &*(second_ptr as *const TupleObject) };
    assert_eq!(second.get(0).unwrap().as_int(), Some(1));
    assert_eq!(second.get(1).unwrap().as_int(), Some(8));
    assert!(enumerate.next().is_none());
}

#[test]
fn test_all_range() {
    // all(range(1, 5)) = True (all positive)
    let range = builtin_range(&[Value::int(1).unwrap(), Value::int(5).unwrap()]).unwrap();
    let result = builtin_all(&[range]).unwrap();
    assert!(result.as_bool().unwrap());
}

#[test]
fn test_all_range_with_zero() {
    // all(range(0, 5)) = False (0 is falsy)
    let range = builtin_range(&[Value::int(0).unwrap(), Value::int(5).unwrap()]).unwrap();
    let result = builtin_all(&[range]).unwrap();
    assert!(!result.as_bool().unwrap());
}

#[test]
fn test_all_wrong_arg_count() {
    let result = builtin_all(&[]);
    assert!(result.is_err());

    let list = ListObject::new();
    let ptr = Box::leak(Box::new(list)) as *mut ListObject as *const ();
    let value = Value::object_ptr(ptr);
    let result = builtin_all(&[value, value]);
    assert!(result.is_err());
}

// -------------------------------------------------------------------------
// any() tests
// -------------------------------------------------------------------------

#[test]
fn test_any_empty_list() {
    // any([]) = False
    let list = ListObject::new();
    let ptr = Box::leak(Box::new(list)) as *mut ListObject as *const ();
    let value = Value::object_ptr(ptr);

    let result = builtin_any(&[value]).unwrap();
    assert!(!result.as_bool().unwrap());
}

#[test]
fn test_any_all_falsy() {
    // any([0, None, False]) = False
    let list = ListObject::from_slice(&[Value::int(0).unwrap(), Value::none(), Value::bool(false)]);
    let ptr = Box::leak(Box::new(list)) as *mut ListObject as *const ();
    let value = Value::object_ptr(ptr);

    let result = builtin_any(&[value]).unwrap();
    assert!(!result.as_bool().unwrap());
}

#[test]
fn test_any_one_truthy() {
    // any([0, 0, 1]) = True
    let list = ListObject::from_slice(&[
        Value::int(0).unwrap(),
        Value::int(0).unwrap(),
        Value::int(1).unwrap(),
    ]);
    let ptr = Box::leak(Box::new(list)) as *mut ListObject as *const ();
    let value = Value::object_ptr(ptr);

    let result = builtin_any(&[value]).unwrap();
    assert!(result.as_bool().unwrap());
}

#[test]
fn test_any_first_truthy() {
    // any([1, 0, 0]) = True (early exit)
    let list = ListObject::from_slice(&[
        Value::int(1).unwrap(),
        Value::int(0).unwrap(),
        Value::int(0).unwrap(),
    ]);
    let ptr = Box::leak(Box::new(list)) as *mut ListObject as *const ();
    let value = Value::object_ptr(ptr);

    let result = builtin_any(&[value]).unwrap();
    assert!(result.as_bool().unwrap());
}

#[test]
fn test_any_range() {
    // any(range(0, 5)) = True (1,2,3,4 are truthy)
    let range = builtin_range(&[Value::int(0).unwrap(), Value::int(5).unwrap()]).unwrap();
    let result = builtin_any(&[range]).unwrap();
    assert!(result.as_bool().unwrap());
}

#[test]
fn test_any_range_empty() {
    // any(range(0, 0)) = False (empty range)
    let range = builtin_range(&[Value::int(0).unwrap(), Value::int(0).unwrap()]).unwrap();
    let result = builtin_any(&[range]).unwrap();
    assert!(!result.as_bool().unwrap());
}

#[test]
fn test_any_wrong_arg_count() {
    let result = builtin_any(&[]);
    assert!(result.is_err());
}

// -------------------------------------------------------------------------
// enumerate() tests
// -------------------------------------------------------------------------

#[test]
fn test_enumerate_basic() {
    // enumerate(['a', 'b']) should produce tuples
    let list = ListObject::from_slice(&[Value::int(10).unwrap(), Value::int(20).unwrap()]);
    let ptr = Box::leak(Box::new(list)) as *mut ListObject as *const ();
    let value = Value::object_ptr(ptr);

    let result = builtin_enumerate(&[value]);
    assert!(result.is_ok());
    // Result is an enumerate iterator
}

#[test]
fn test_enumerate_with_start() {
    let list = ListObject::from_slice(&[Value::int(1).unwrap()]);
    let ptr = Box::leak(Box::new(list)) as *mut ListObject as *const ();
    let value = Value::object_ptr(ptr);

    let result = builtin_enumerate(&[value, Value::int(5).unwrap()]);
    assert!(result.is_ok());
}

#[test]
fn test_enumerate_wrong_arg_count() {
    let result = builtin_enumerate(&[]);
    assert!(result.is_err());

    let list = ListObject::new();
    let ptr = Box::leak(Box::new(list)) as *mut ListObject as *const ();
    let value = Value::object_ptr(ptr);

    let result = builtin_enumerate(&[value, Value::int(0).unwrap(), Value::int(1).unwrap()]);
    assert!(result.is_err());
}

#[test]
fn test_enumerate_invalid_start() {
    let list = ListObject::new();
    let ptr = Box::leak(Box::new(list)) as *mut ListObject as *const ();
    let value = Value::object_ptr(ptr);

    // Non-integer start should error
    let result = builtin_enumerate(&[value, Value::float(1.5)]);
    assert!(result.is_err());
}

#[test]
fn test_enumerate_vm_supports_user_defined_iterables() {
    let result = execute(
        r#"
class Iterable:
    def __iter__(self):
        return iter((10, 20))

it = enumerate(Iterable(), 4)
assert next(it) == (4, 10)
assert next(it) == (5, 20)
try:
    next(it)
except StopIteration:
    pass
else:
    raise AssertionError("enumerate should stop")
"#,
    );

    assert!(
        result.is_ok(),
        "enumerate should honor __iter__ protocol: {result:?}"
    );
}

#[test]
fn test_enumerate_vm_supports_generator_inputs() {
    let result = execute(
        r#"
def numbers():
    yield 1
    yield 2

it = enumerate(numbers(), 9)
assert next(it) == (9, 1)
assert next(it) == (10, 2)
try:
    next(it)
except StopIteration:
    pass
else:
    raise AssertionError("generator-backed enumerate should stop")
"#,
    );

    assert!(
        result.is_ok(),
        "enumerate should preserve lazy generator iteration: {result:?}"
    );
}

// -------------------------------------------------------------------------
// zip() tests
// -------------------------------------------------------------------------

#[test]
fn test_zip_basic() {
    let list1 = ListObject::from_slice(&[Value::int(1).unwrap(), Value::int(2).unwrap()]);
    let ptr1 = Box::leak(Box::new(list1)) as *mut ListObject as *const ();
    let val1 = Value::object_ptr(ptr1);

    let list2 = ListObject::from_slice(&[Value::int(10).unwrap(), Value::int(20).unwrap()]);
    let ptr2 = Box::leak(Box::new(list2)) as *mut ListObject as *const ();
    let val2 = Value::object_ptr(ptr2);

    let result = builtin_zip(&[val1, val2]);
    assert!(result.is_ok());
}

#[test]
fn test_zip_empty() {
    // zip() with no arguments should return empty iterator
    let result = builtin_zip(&[]);
    assert!(result.is_ok());
}

#[test]
fn test_zip_single() {
    let list = ListObject::from_slice(&[Value::int(1).unwrap()]);
    let ptr = Box::leak(Box::new(list)) as *mut ListObject as *const ();
    let value = Value::object_ptr(ptr);

    let result = builtin_zip(&[value]);
    assert!(result.is_ok());
}

// -------------------------------------------------------------------------
// filter() tests
// -------------------------------------------------------------------------

#[test]
fn test_filter_identity() {
    // filter(None, iterable) filters falsy values
    let list = ListObject::from_slice(&[
        Value::int(1).unwrap(),
        Value::int(0).unwrap(),
        Value::int(2).unwrap(),
    ]);
    let ptr = Box::leak(Box::new(list)) as *mut ListObject as *const ();
    let value = Value::object_ptr(ptr);

    let result = builtin_filter(&[Value::none(), value]);
    assert!(result.is_ok());
}

#[test]
fn test_filter_wrong_arg_count() {
    let result = builtin_filter(&[]);
    assert!(result.is_err());

    let result = builtin_filter(&[Value::none()]);
    assert!(result.is_err());
}

// -------------------------------------------------------------------------
// reversed() tests
// -------------------------------------------------------------------------

#[test]
fn test_reversed_basic() {
    let list = ListObject::from_slice(&[
        Value::int(1).unwrap(),
        Value::int(2).unwrap(),
        Value::int(3).unwrap(),
    ]);
    let ptr = Box::leak(Box::new(list)) as *mut ListObject as *const ();
    let value = Value::object_ptr(ptr);

    let result = builtin_reversed(&[value]);
    assert!(result.is_ok());
}

#[test]
fn test_reversed_wrong_arg_count() {
    let result = builtin_reversed(&[]);
    assert!(result.is_err());

    let list = ListObject::new();
    let ptr = Box::leak(Box::new(list)) as *mut ListObject as *const ();
    let value = Value::object_ptr(ptr);

    let result = builtin_reversed(&[value, value]);
    assert!(result.is_err());
}

// -------------------------------------------------------------------------
// map() tests
// -------------------------------------------------------------------------

#[test]
fn test_map_basic() {
    let list = ListObject::from_slice(&[Value::int(1).unwrap()]);
    let ptr = Box::leak(Box::new(list)) as *mut ListObject as *const ();
    let value = Value::object_ptr(ptr);

    // func is a placeholder
    let result = builtin_map(&[Value::none(), value]);
    assert!(result.is_ok());
}

#[test]
fn test_iter_on_iterator_returns_same_object() {
    let list = ListObject::from_slice(&[Value::int(1).unwrap(), Value::int(2).unwrap()]);
    let list_ptr = Box::leak(Box::new(list)) as *mut ListObject as *const ();
    let list_value = Value::object_ptr(list_ptr);

    let iter1 = builtin_iter(&[list_value]).unwrap();
    let iter2 = builtin_iter(&[iter1]).unwrap();
    assert_eq!(iter1, iter2);
}

#[test]
fn test_iter_on_iterator_preserves_state() {
    let list = ListObject::from_slice(&[Value::int(10).unwrap(), Value::int(20).unwrap()]);
    let list_ptr = Box::leak(Box::new(list)) as *mut ListObject as *const ();
    let list_value = Value::object_ptr(list_ptr);

    let iter = builtin_iter(&[list_value]).unwrap();
    let first = builtin_next(&[iter]).unwrap();
    assert_eq!(first.as_int(), Some(10));

    let iter_again = builtin_iter(&[iter]).unwrap();
    assert_eq!(iter_again, iter);

    let second = builtin_next(&[iter_again]).unwrap();
    assert_eq!(second.as_int(), Some(20));
}

#[test]
fn test_next_exhausted_raises_stop_iteration() {
    let list = ListObject::from_slice(&[Value::int(10).unwrap()]);
    let list_ptr = Box::leak(Box::new(list)) as *mut ListObject as *const ();
    let list_value = Value::object_ptr(list_ptr);

    let iter = builtin_iter(&[list_value]).unwrap();
    assert_eq!(builtin_next(&[iter]).unwrap().as_int(), Some(10));

    let err = builtin_next(&[iter]).unwrap_err();
    assert!(matches!(err, BuiltinError::StopIteration));
}

#[test]
fn test_next_exhausted_returns_default() {
    let list = ListObject::from_slice(&[Value::int(10).unwrap()]);
    let list_ptr = Box::leak(Box::new(list)) as *mut ListObject as *const ();
    let list_value = Value::object_ptr(list_ptr);

    let iter = builtin_iter(&[list_value]).unwrap();
    assert_eq!(builtin_next(&[iter]).unwrap().as_int(), Some(10));

    let default = Value::int(99).unwrap();
    assert_eq!(builtin_next(&[iter, default]).unwrap().as_int(), Some(99));
}

#[test]
fn test_map_wrong_arg_count() {
    let result = builtin_map(&[]);
    assert!(result.is_err());

    let result = builtin_map(&[Value::none()]);
    assert!(result.is_err());
}

// -------------------------------------------------------------------------
// sorted() tests
// -------------------------------------------------------------------------

#[test]
fn test_sorted_integers_ascending() {
    // sorted([3, 1, 2]) should return [1, 2, 3]
    let list = ListObject::from_slice(&[
        Value::int(3).unwrap(),
        Value::int(1).unwrap(),
        Value::int(2).unwrap(),
    ]);
    let ptr = Box::leak(Box::new(list)) as *mut ListObject as *const ();
    let value = Value::object_ptr(ptr);

    let result = builtin_sorted(&[value]).unwrap();

    // Result should be a list containing [1, 2, 3]
    let result_ptr = result.as_object_ptr().unwrap();
    let sorted_list = unsafe { &*(result_ptr as *const ListObject) };
    assert_eq!(sorted_list.len(), 3);
    assert_eq!(sorted_list.get(0).unwrap().as_int().unwrap(), 1);
    assert_eq!(sorted_list.get(1).unwrap().as_int().unwrap(), 2);
    assert_eq!(sorted_list.get(2).unwrap().as_int().unwrap(), 3);
}

#[test]
fn test_sorted_integers_descending() {
    // sorted([1, 2, 3], reverse=True) should return [3, 2, 1]
    let list = ListObject::from_slice(&[
        Value::int(1).unwrap(),
        Value::int(2).unwrap(),
        Value::int(3).unwrap(),
    ]);
    let ptr = Box::leak(Box::new(list)) as *mut ListObject as *const ();
    let value = Value::object_ptr(ptr);

    let result = builtin_sorted(&[value, Value::none(), Value::bool(true)]).unwrap();

    let result_ptr = result.as_object_ptr().unwrap();
    let sorted_list = unsafe { &*(result_ptr as *const ListObject) };
    assert_eq!(sorted_list.len(), 3);
    assert_eq!(sorted_list.get(0).unwrap().as_int().unwrap(), 3);
    assert_eq!(sorted_list.get(1).unwrap().as_int().unwrap(), 2);
    assert_eq!(sorted_list.get(2).unwrap().as_int().unwrap(), 1);
}

#[test]
fn test_sorted_range() {
    // sorted(range(5, 0, -1)) should return [1, 2, 3, 4, 5]
    let range = builtin_range(&[
        Value::int(5).unwrap(),
        Value::int(0).unwrap(),
        Value::int(-1).unwrap(),
    ])
    .unwrap();

    let result = builtin_sorted(&[range]).unwrap();

    let result_ptr = result.as_object_ptr().unwrap();
    let sorted_list = unsafe { &*(result_ptr as *const ListObject) };
    assert_eq!(sorted_list.len(), 5);
    assert_eq!(sorted_list.get(0).unwrap().as_int().unwrap(), 1);
    assert_eq!(sorted_list.get(4).unwrap().as_int().unwrap(), 5);
}

#[test]
fn test_sorted_empty() {
    // sorted([]) should return []
    let list = ListObject::new();
    let ptr = Box::leak(Box::new(list)) as *mut ListObject as *const ();
    let value = Value::object_ptr(ptr);

    let result = builtin_sorted(&[value]).unwrap();

    let result_ptr = result.as_object_ptr().unwrap();
    let sorted_list = unsafe { &*(result_ptr as *const ListObject) };
    assert_eq!(sorted_list.len(), 0);
}

#[test]
fn test_sorted_single_element() {
    let list = ListObject::from_slice(&[Value::int(42).unwrap()]);
    let ptr = Box::leak(Box::new(list)) as *mut ListObject as *const ();
    let value = Value::object_ptr(ptr);

    let result = builtin_sorted(&[value]).unwrap();

    let result_ptr = result.as_object_ptr().unwrap();
    let sorted_list = unsafe { &*(result_ptr as *const ListObject) };
    assert_eq!(sorted_list.len(), 1);
    assert_eq!(sorted_list.get(0).unwrap().as_int().unwrap(), 42);
}

#[test]
fn test_sorted_already_sorted() {
    let list = ListObject::from_slice(&[
        Value::int(1).unwrap(),
        Value::int(2).unwrap(),
        Value::int(3).unwrap(),
    ]);
    let ptr = Box::leak(Box::new(list)) as *mut ListObject as *const ();
    let value = Value::object_ptr(ptr);

    let result = builtin_sorted(&[value]).unwrap();

    let result_ptr = result.as_object_ptr().unwrap();
    let sorted_list = unsafe { &*(result_ptr as *const ListObject) };
    assert_eq!(sorted_list.get(0).unwrap().as_int().unwrap(), 1);
    assert_eq!(sorted_list.get(1).unwrap().as_int().unwrap(), 2);
    assert_eq!(sorted_list.get(2).unwrap().as_int().unwrap(), 3);
}

#[test]
fn test_sorted_wrong_arg_count() {
    let result = builtin_sorted(&[]);
    assert!(result.is_err());

    let list = ListObject::new();
    let ptr = Box::leak(Box::new(list)) as *mut ListObject as *const ();
    let value = Value::object_ptr(ptr);

    // Too many arguments
    let result = builtin_sorted(&[
        value,
        Value::none(),
        Value::bool(false),
        Value::int(1).unwrap(),
    ]);
    assert!(result.is_err());
}

#[test]
fn test_sorted_key_not_implemented() {
    // Key function requires VM integration
    let list = ListObject::from_slice(&[Value::int(1).unwrap()]);
    let ptr = Box::leak(Box::new(list)) as *mut ListObject as *const ();
    let value = Value::object_ptr(ptr);

    // Pass a non-None key function
    let result = builtin_sorted(&[value, Value::int(42).unwrap()]);
    assert!(result.is_err()); // Should be NotImplemented
}

// -------------------------------------------------------------------------
// compare_values() tests
// -------------------------------------------------------------------------

#[test]
fn test_compare_values_integers() {
    use std::cmp::Ordering;

    let a = Value::int(1).unwrap();
    let b = Value::int(2).unwrap();
    let c = Value::int(1).unwrap();

    assert_eq!(compare_values(&a, &b), Ordering::Less);
    assert_eq!(compare_values(&b, &a), Ordering::Greater);
    assert_eq!(compare_values(&a, &c), Ordering::Equal);
}

#[test]
fn test_compare_values_floats() {
    use std::cmp::Ordering;

    let a = Value::float(1.5);
    let b = Value::float(2.5);
    let c = Value::float(1.5);

    assert_eq!(compare_values(&a, &b), Ordering::Less);
    assert_eq!(compare_values(&b, &a), Ordering::Greater);
    assert_eq!(compare_values(&a, &c), Ordering::Equal);
}

#[test]
fn test_compare_values_int_float() {
    use std::cmp::Ordering;

    let a = Value::int(2).unwrap();
    let b = Value::float(2.5);

    assert_eq!(compare_values(&a, &b), Ordering::Less);
    assert_eq!(compare_values(&b, &a), Ordering::Greater);
}

#[test]
fn test_compare_values_none() {
    use std::cmp::Ordering;

    let none = Value::none();
    let one = Value::int(1).unwrap();

    // None is smallest
    assert_eq!(compare_values(&none, &one), Ordering::Less);
    assert_eq!(compare_values(&one, &none), Ordering::Greater);
    assert_eq!(compare_values(&none, &none), Ordering::Equal);
}

#[test]
fn test_compare_values_booleans() {
    use std::cmp::Ordering;

    let f = Value::bool(false);
    let t = Value::bool(true);

    assert_eq!(compare_values(&f, &t), Ordering::Less);
    assert_eq!(compare_values(&t, &f), Ordering::Greater);
    assert_eq!(compare_values(&t, &t), Ordering::Equal);
}
