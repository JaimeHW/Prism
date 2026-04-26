
use super::*;
use crate::VirtualMachine;
use crate::builtins::get_iterator_mut;
use crate::ops::iteration::{IterStep, next_step};
use prism_core::intern::intern;

#[test]
fn test_module_name() {
    let m = ItertoolsModule::new();
    assert_eq!(m.name(), "itertools");
}

#[test]
fn test_module_dir() {
    let m = ItertoolsModule::new();
    let attrs = m.dir();
    assert!(attrs.iter().any(|a| a.as_ref() == "count"));
    assert!(attrs.iter().any(|a| a.as_ref() == "chain"));
    assert!(attrs.iter().any(|a| a.as_ref() == "product"));
    assert!(attrs.iter().any(|a| a.as_ref() == "groupby"));
    assert!(attrs.iter().any(|a| a.as_ref() == "pairwise"));
    assert!(attrs.iter().any(|a| a.as_ref() == "batched"));
}

#[test]
fn test_module_exports_bootstrap_callables() {
    let m = ItertoolsModule::new();

    for name in ["chain", "repeat", "starmap", "islice"] {
        assert!(
            m.get_attr(name)
                .expect("callable should exist")
                .as_object_ptr()
                .is_some(),
            "{name} should be exposed as a callable object"
        );
    }
}

#[test]
fn test_module_unknown_attr() {
    let m = ItertoolsModule::new();
    assert!(m.get_attr("nonexistent").is_err());
}

#[test]
fn test_builtin_chain_lazily_yields_each_iterable_in_order() {
    let left = TupleObject::from_slice(&[Value::int_unchecked(1), Value::int_unchecked(2)]);
    let right = TupleObject::from_slice(&[Value::int_unchecked(3)]);
    let left_ptr = Box::leak(Box::new(left)) as *mut TupleObject as *const ();
    let right_ptr = Box::leak(Box::new(right)) as *mut TupleObject as *const ();
    let mut vm = VirtualMachine::new();

    let iter_value = builtin_chain(
        &mut vm,
        &[Value::object_ptr(left_ptr), Value::object_ptr(right_ptr)],
    )
    .expect("chain() should accept iterable arguments");

    let mut values = Vec::new();
    loop {
        match next_step(&mut vm, iter_value).expect("chain iterator should advance") {
            IterStep::Yielded(value) => {
                values.push(value.as_int().expect("chain should yield ints"))
            }
            IterStep::Exhausted => break,
        }
    }

    assert_eq!(values, vec![1, 2, 3]);
}

#[test]
fn test_builtin_chain_without_args_returns_empty_iterator() {
    let mut vm = VirtualMachine::new();
    let iter_value = builtin_chain(&mut vm, &[]).expect("chain() should allow no args");
    let iter = get_iterator_mut(&iter_value).expect("chain() should return an iterator");

    assert!(iter.next().is_none());
    assert!(iter.is_exhausted());
}

#[test]
fn test_builtin_repeat_yields_bounded_repeated_values_lazily() {
    let iter_value = builtin_repeat(&[Value::string(intern("tick")), Value::int_unchecked(3)])
        .expect("repeat() should accept object and count");
    let iter = get_iterator_mut(&iter_value).expect("repeat() should return an iterator");

    assert_eq!(iter.size_hint(), Some(3));
    for expected_remaining in [2, 1, 0] {
        let value = iter.next().expect("repeat() should yield requested values");
        assert_eq!(value, Value::string(intern("tick")));
        assert_eq!(iter.size_hint(), Some(expected_remaining));
    }
    assert!(iter.next().is_none());
}

#[test]
fn test_builtin_repeat_negative_count_is_empty() {
    let iter_value = builtin_repeat(&[Value::int_unchecked(7), Value::int_unchecked(-5)])
        .expect("repeat() should clamp negative counts to zero");
    let iter = get_iterator_mut(&iter_value).expect("repeat() should return an iterator");

    assert_eq!(iter.size_hint(), Some(0));
    assert!(iter.next().is_none());
}

#[test]
fn test_builtin_repeat_without_count_is_unbounded() {
    let iter_value =
        builtin_repeat(&[Value::int_unchecked(4)]).expect("repeat() should allow no count");
    let iter = get_iterator_mut(&iter_value).expect("repeat() should return an iterator");

    assert_eq!(iter.size_hint(), None);
    assert_eq!(iter.next().unwrap().as_int(), Some(4));
    assert_eq!(iter.next().unwrap().as_int(), Some(4));
    assert!(!iter.is_exhausted());
}

#[test]
fn test_builtin_product_returns_cartesian_tuples() {
    let left = TupleObject::from_slice(&[Value::int_unchecked(1), Value::int_unchecked(2)]);
    let right = TupleObject::from_slice(&[Value::string(intern("a")), Value::string(intern("b"))]);
    let left_ptr = Box::leak(Box::new(left)) as *mut TupleObject as *const ();
    let right_ptr = Box::leak(Box::new(right)) as *mut TupleObject as *const ();

    let iter_value = builtin_product(&[Value::object_ptr(left_ptr), Value::object_ptr(right_ptr)])
        .expect("product() should succeed");
    let iter = get_iterator_mut(&iter_value).expect("product() should return an iterator");

    let first = iter.next().expect("expected first tuple");
    let second = iter.next().expect("expected second tuple");
    let first_ptr = first
        .as_object_ptr()
        .expect("product entries should be tuples");
    let second_ptr = second
        .as_object_ptr()
        .expect("product entries should be tuples");
    let first_tuple = unsafe { &*(first_ptr as *const TupleObject) };
    let second_tuple = unsafe { &*(second_ptr as *const TupleObject) };

    assert_eq!(first_tuple.get(0).unwrap().as_int(), Some(1));
    assert!(first_tuple.get(1).unwrap().is_string());
    assert_eq!(second_tuple.get(0).unwrap().as_int(), Some(1));
    assert!(second_tuple.get(1).unwrap().is_string());
}

#[test]
fn test_builtin_product_with_no_args_yields_single_empty_tuple() {
    let iter_value = builtin_product(&[]).expect("product() should succeed");
    let iter = get_iterator_mut(&iter_value).expect("product() should return an iterator");
    let only = iter.next().expect("empty product should yield one tuple");
    let tuple_ptr = only
        .as_object_ptr()
        .expect("product entry should be a tuple");
    let tuple = unsafe { &*(tuple_ptr as *const TupleObject) };
    assert_eq!(tuple.len(), 0);
    assert!(iter.next().is_none());
}

#[test]
fn test_builtin_islice_slices_count_iterators_lazily() {
    let count = builtin_count(&[]).expect("count() should succeed");
    let iter_value = builtin_islice(&[
        count,
        Value::int_unchecked(3),
        Value::int_unchecked(8),
        Value::int_unchecked(2),
    ])
    .expect("islice() should succeed");
    let iter = get_iterator_mut(&iter_value).expect("islice() should return an iterator");

    let mut values = Vec::new();
    while let Some(value) = iter.next() {
        values.push(value.as_int().expect("islice(count()) should yield ints"));
    }

    assert_eq!(values, vec![3, 5, 7]);
}

#[test]
fn test_builtin_islice_supports_none_stop() {
    let tuple = TupleObject::from_slice(&[
        Value::int_unchecked(10),
        Value::int_unchecked(11),
        Value::int_unchecked(12),
        Value::int_unchecked(13),
    ]);
    let tuple_ptr = Box::leak(Box::new(tuple)) as *mut TupleObject as *const ();

    let iter_value = builtin_islice(&[
        Value::object_ptr(tuple_ptr),
        Value::int_unchecked(2),
        Value::none(),
    ])
    .expect("islice() should succeed");
    let iter = get_iterator_mut(&iter_value).expect("islice() should return an iterator");

    assert_eq!(iter.next().unwrap().as_int(), Some(12));
    assert_eq!(iter.next().unwrap().as_int(), Some(13));
    assert!(iter.next().is_none());
}

#[test]
fn test_builtin_permutations_matches_cpython_order() {
    let iter_value = builtin_permutations(&[Value::string(intern("abc"))])
        .expect("permutations() should succeed");
    let iter = get_iterator_mut(&iter_value).expect("permutations() should return an iterator");

    let mut tuples = Vec::new();
    while let Some(value) = iter.next() {
        let tuple_ptr = value
            .as_object_ptr()
            .expect("permutations entries should be tuples");
        let tuple = unsafe { &*(tuple_ptr as *const TupleObject) };
        tuples.push((
            tuple.get(0).unwrap(),
            tuple.get(1).unwrap(),
            tuple.get(2).unwrap(),
        ));
    }

    assert_eq!(tuples.len(), 6);
    assert_eq!(tuples[0].0, Value::string(intern("a")));
    assert_eq!(tuples[0].1, Value::string(intern("b")));
    assert_eq!(tuples[0].2, Value::string(intern("c")));
    assert_eq!(tuples[1].0, Value::string(intern("a")));
    assert_eq!(tuples[1].1, Value::string(intern("c")));
    assert_eq!(tuples[1].2, Value::string(intern("b")));
    assert_eq!(tuples[5].0, Value::string(intern("c")));
    assert_eq!(tuples[5].1, Value::string(intern("b")));
    assert_eq!(tuples[5].2, Value::string(intern("a")));
}

#[test]
fn test_builtin_permutations_rejects_negative_r() {
    let error = builtin_permutations(&[Value::string(intern("ab")), Value::int_unchecked(-1)])
        .expect_err("negative r should fail");
    assert!(matches!(error, BuiltinError::ValueError(_)));
}
