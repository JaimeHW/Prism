
use super::*;
use prism_gc::trace::Tracer;
use prism_runtime::types::list::ListObject;

// =========================================================================
// Construction Tests
// =========================================================================

#[test]
fn test_new_creates_empty_deque() {
    let d = Deque::new();
    assert!(d.is_empty());
    assert_eq!(d.len(), 0);
}

#[test]
fn test_with_capacity_respects_minimum() {
    let d = Deque::with_capacity(4);
    assert!(d.capacity() >= MIN_CAPACITY);
}

#[test]
fn test_with_capacity_rounds_to_power_of_two() {
    let d = Deque::with_capacity(17);
    assert_eq!(d.capacity(), 32);
}

#[test]
fn test_deque_object_trace_visits_live_elements() {
    struct CountingTracer {
        values: usize,
    }

    impl Tracer for CountingTracer {
        fn trace_value(&mut self, _value: Value) {
            self.values += 1;
        }

        fn trace_ptr(&mut self, _ptr: *const ()) {}
    }

    let object = DequeObject::from_deque(Deque::from_iter([
        Value::int_unchecked(1),
        Value::int_unchecked(2),
        Value::int_unchecked(3),
    ]));
    let mut tracer = CountingTracer { values: 0 };

    object.trace(&mut tracer);

    assert_eq!(tracer.values, 3);
}

#[test]
fn test_with_maxlen() {
    let d = Deque::with_maxlen(5);
    assert_eq!(d.maxlen(), Some(5));
}

#[test]
fn test_from_iter_creates_deque() {
    let values = vec![
        Value::int_unchecked(1),
        Value::int_unchecked(2),
        Value::int_unchecked(3),
    ];
    let d = Deque::from_iter(values);
    assert_eq!(d.len(), 3);
    assert_eq!(d.get(0).and_then(|v| v.as_int()), Some(1));
    assert_eq!(d.get(2).and_then(|v| v.as_int()), Some(3));
}

#[test]
fn test_builtin_deque_constructs_empty_native_object() {
    let value = builtin_deque(&[]).expect("deque() should succeed");
    let deque = value_as_deque(&value).expect("deque() should return native deque");
    assert!(deque.is_empty());
    assert_eq!(deque.len(), 0);
}

#[test]
fn test_builtin_deque_consumes_iterable_and_maxlen() {
    let iterable = Value::object_ptr(Box::into_raw(Box::new(ListObject::from_slice(&[
        Value::int_unchecked(1),
        Value::int_unchecked(2),
        Value::int_unchecked(3),
    ]))) as *const ());

    let value = builtin_deque(&[iterable, Value::int_unchecked(2)])
        .expect("deque(iterable, maxlen) should succeed");
    let deque = value_as_deque(&value).expect("constructor should return deque");
    assert_eq!(deque.len(), 2);
    assert_eq!(
        deque.deque().front().and_then(|value| value.as_int()),
        Some(2)
    );
    assert_eq!(
        deque.deque().back().and_then(|value| value.as_int()),
        Some(3)
    );
}

#[test]
fn test_builtin_deque_kw_accepts_maxlen_keyword() {
    let value = builtin_deque_kw(&[], &[("maxlen", Value::int_unchecked(4))])
        .expect("deque(maxlen=...) should succeed");
    let deque = value_as_deque(&value).expect("keyword constructor should return deque");
    assert_eq!(deque.deque().maxlen(), Some(4));
}

// =========================================================================
// Append/Pop Tests
// =========================================================================

#[test]
fn test_append_increases_length() {
    let mut d = Deque::new();
    d.append(Value::int_unchecked(1));
    assert_eq!(d.len(), 1);
    d.append(Value::int_unchecked(2));
    assert_eq!(d.len(), 2);
}

#[test]
fn test_appendleft_increases_length() {
    let mut d = Deque::new();
    d.appendleft(Value::int_unchecked(1));
    assert_eq!(d.len(), 1);
    d.appendleft(Value::int_unchecked(2));
    assert_eq!(d.len(), 2);
}

#[test]
fn test_append_and_pop_order() {
    let mut d = Deque::new();
    d.append(Value::int_unchecked(1));
    d.append(Value::int_unchecked(2));
    d.append(Value::int_unchecked(3));

    assert_eq!(d.pop().and_then(|v| v.as_int()), Some(3));
    assert_eq!(d.pop().and_then(|v| v.as_int()), Some(2));
    assert_eq!(d.pop().and_then(|v| v.as_int()), Some(1));
    assert_eq!(d.pop(), None);
}

#[test]
fn test_appendleft_and_popleft_order() {
    let mut d = Deque::new();
    d.appendleft(Value::int_unchecked(1));
    d.appendleft(Value::int_unchecked(2));
    d.appendleft(Value::int_unchecked(3));

    assert_eq!(d.popleft().and_then(|v| v.as_int()), Some(3));
    assert_eq!(d.popleft().and_then(|v| v.as_int()), Some(2));
    assert_eq!(d.popleft().and_then(|v| v.as_int()), Some(1));
    assert_eq!(d.popleft(), None);
}

#[test]
fn test_mixed_operations() {
    let mut d = Deque::new();
    d.append(Value::int_unchecked(1));
    d.appendleft(Value::int_unchecked(0));
    d.append(Value::int_unchecked(2));

    // [0, 1, 2]
    assert_eq!(d.popleft().and_then(|v| v.as_int()), Some(0));
    assert_eq!(d.pop().and_then(|v| v.as_int()), Some(2));
    assert_eq!(d.popleft().and_then(|v| v.as_int()), Some(1));
}

#[test]
fn test_pop_empty_returns_none() {
    let mut d = Deque::new();
    assert_eq!(d.pop(), None);
    assert_eq!(d.popleft(), None);
}

// =========================================================================
// Maxlen Tests
// =========================================================================

#[test]
fn test_maxlen_drops_from_left_on_append() {
    let mut d = Deque::with_maxlen(3);
    d.append(Value::int_unchecked(1));
    d.append(Value::int_unchecked(2));
    d.append(Value::int_unchecked(3));
    assert_eq!(d.len(), 3);

    d.append(Value::int_unchecked(4));
    assert_eq!(d.len(), 3);
    assert_eq!(d.get(0).and_then(|v| v.as_int()), Some(2));
    assert_eq!(d.get(2).and_then(|v| v.as_int()), Some(4));
}

#[test]
fn test_maxlen_drops_from_right_on_appendleft() {
    let mut d = Deque::with_maxlen(3);
    d.appendleft(Value::int_unchecked(1));
    d.appendleft(Value::int_unchecked(2));
    d.appendleft(Value::int_unchecked(3));
    assert_eq!(d.len(), 3);

    d.appendleft(Value::int_unchecked(4));
    assert_eq!(d.len(), 3);
    assert_eq!(d.get(0).and_then(|v| v.as_int()), Some(4));
    assert_eq!(d.get(2).and_then(|v| v.as_int()), Some(2));
}

#[test]
fn test_maxlen_zero_ignores_appends() {
    let mut d = Deque::with_maxlen(0);
    d.append(Value::int_unchecked(1));
    d.appendleft(Value::int_unchecked(2));
    assert!(d.is_empty());
}

// =========================================================================
// Index Access Tests
// =========================================================================

#[test]
fn test_positive_index() {
    let mut d = Deque::new();
    d.append(Value::int_unchecked(10));
    d.append(Value::int_unchecked(20));
    d.append(Value::int_unchecked(30));

    assert_eq!(d.get(0).and_then(|v| v.as_int()), Some(10));
    assert_eq!(d.get(1).and_then(|v| v.as_int()), Some(20));
    assert_eq!(d.get(2).and_then(|v| v.as_int()), Some(30));
}

#[test]
fn test_negative_index() {
    let mut d = Deque::new();
    d.append(Value::int_unchecked(10));
    d.append(Value::int_unchecked(20));
    d.append(Value::int_unchecked(30));

    assert_eq!(d.get(-1).and_then(|v| v.as_int()), Some(30));
    assert_eq!(d.get(-2).and_then(|v| v.as_int()), Some(20));
    assert_eq!(d.get(-3).and_then(|v| v.as_int()), Some(10));
}

#[test]
fn test_out_of_bounds_index() {
    let mut d = Deque::new();
    d.append(Value::int_unchecked(1));

    assert_eq!(d.get(5), None);
    assert_eq!(d.get(-5), None);
}

#[test]
fn test_front_and_back() {
    let mut d = Deque::new();
    assert_eq!(d.front(), None);
    assert_eq!(d.back(), None);

    d.append(Value::int_unchecked(1));
    d.append(Value::int_unchecked(2));

    assert_eq!(d.front().and_then(|v| v.as_int()), Some(1));
    assert_eq!(d.back().and_then(|v| v.as_int()), Some(2));
}

// =========================================================================
// Rotation Tests
// =========================================================================

#[test]
fn test_rotate_right() {
    let mut d = Deque::new();
    for i in 0..5 {
        d.append(Value::int_unchecked(i));
    }
    // [0, 1, 2, 3, 4]
    d.rotate(2);
    // [3, 4, 0, 1, 2]

    assert_eq!(d.get(0).and_then(|v| v.as_int()), Some(3));
    assert_eq!(d.get(1).and_then(|v| v.as_int()), Some(4));
    assert_eq!(d.get(2).and_then(|v| v.as_int()), Some(0));
}

#[test]
fn test_rotate_left() {
    let mut d = Deque::new();
    for i in 0..5 {
        d.append(Value::int_unchecked(i));
    }
    // [0, 1, 2, 3, 4]
    d.rotate(-2);
    // [2, 3, 4, 0, 1]

    assert_eq!(d.get(0).and_then(|v| v.as_int()), Some(2));
    assert_eq!(d.get(4).and_then(|v| v.as_int()), Some(1));
}

#[test]
fn test_rotate_empty() {
    let mut d = Deque::new();
    d.rotate(5);
    assert!(d.is_empty());
}

#[test]
fn test_rotate_single_element() {
    let mut d = Deque::new();
    d.append(Value::int_unchecked(1));
    d.rotate(100);
    assert_eq!(d.get(0).and_then(|v| v.as_int()), Some(1));
}

// =========================================================================
// Reverse Tests
// =========================================================================

#[test]
fn test_reverse() {
    let mut d = Deque::new();
    for i in 0..5 {
        d.append(Value::int_unchecked(i));
    }
    d.reverse();

    assert_eq!(d.get(0).and_then(|v| v.as_int()), Some(4));
    assert_eq!(d.get(4).and_then(|v| v.as_int()), Some(0));
}

#[test]
fn test_reverse_empty() {
    let mut d = Deque::new();
    d.reverse();
    assert!(d.is_empty());
}

#[test]
fn test_reverse_single() {
    let mut d = Deque::new();
    d.append(Value::int_unchecked(1));
    d.reverse();
    assert_eq!(d.get(0).and_then(|v| v.as_int()), Some(1));
}

// =========================================================================
// Clear Tests
// =========================================================================

#[test]
fn test_clear() {
    let mut d = Deque::new();
    for i in 0..10 {
        d.append(Value::int_unchecked(i));
    }
    d.clear();

    assert!(d.is_empty());
    assert_eq!(d.len(), 0);
}

// =========================================================================
// Remove/Insert Tests
// =========================================================================

#[test]
fn test_remove_existing() {
    let mut d = Deque::new();
    d.append(Value::int_unchecked(1));
    d.append(Value::int_unchecked(2));
    d.append(Value::int_unchecked(3));

    assert!(d.remove(&Value::int_unchecked(2)));
    assert_eq!(d.len(), 2);
    assert_eq!(d.get(0).and_then(|v| v.as_int()), Some(1));
    assert_eq!(d.get(1).and_then(|v| v.as_int()), Some(3));
}

#[test]
fn test_remove_not_found() {
    let mut d = Deque::new();
    d.append(Value::int_unchecked(1));

    assert!(!d.remove(&Value::int_unchecked(99)));
    assert_eq!(d.len(), 1);
}

#[test]
fn test_insert_beginning() {
    let mut d = Deque::new();
    d.append(Value::int_unchecked(2));
    d.append(Value::int_unchecked(3));

    d.insert(0, Value::int_unchecked(1));

    assert_eq!(d.len(), 3);
    assert_eq!(d.get(0).and_then(|v| v.as_int()), Some(1));
}

#[test]
fn test_insert_middle() {
    let mut d = Deque::new();
    d.append(Value::int_unchecked(1));
    d.append(Value::int_unchecked(3));

    d.insert(1, Value::int_unchecked(2));

    assert_eq!(d.len(), 3);
    assert_eq!(d.get(1).and_then(|v| v.as_int()), Some(2));
}

#[test]
fn test_insert_end() {
    let mut d = Deque::new();
    d.append(Value::int_unchecked(1));
    d.append(Value::int_unchecked(2));

    d.insert(2, Value::int_unchecked(3));

    assert_eq!(d.len(), 3);
    assert_eq!(d.get(2).and_then(|v| v.as_int()), Some(3));
}

// =========================================================================
// Count/Index Tests
// =========================================================================

#[test]
fn test_count() {
    let mut d = Deque::new();
    d.append(Value::int_unchecked(1));
    d.append(Value::int_unchecked(2));
    d.append(Value::int_unchecked(1));
    d.append(Value::int_unchecked(1));

    assert_eq!(d.count(&Value::int_unchecked(1)), 3);
    assert_eq!(d.count(&Value::int_unchecked(2)), 1);
    assert_eq!(d.count(&Value::int_unchecked(99)), 0);
}

#[test]
fn test_index_of() {
    let mut d = Deque::new();
    d.append(Value::int_unchecked(10));
    d.append(Value::int_unchecked(20));
    d.append(Value::int_unchecked(30));

    assert_eq!(d.index_of(&Value::int_unchecked(20)), Some(1));
    assert_eq!(d.index_of(&Value::int_unchecked(99)), None);
}

// =========================================================================
// Iterator Tests
// =========================================================================

#[test]
fn test_iter() {
    let mut d = Deque::new();
    d.append(Value::int_unchecked(1));
    d.append(Value::int_unchecked(2));
    d.append(Value::int_unchecked(3));

    let vals: Vec<i64> = d.iter().filter_map(|v| v.as_int()).collect();
    assert_eq!(vals, vec![1, 2, 3]);
}

#[test]
fn test_iter_reverse() {
    let mut d = Deque::new();
    d.append(Value::int_unchecked(1));
    d.append(Value::int_unchecked(2));
    d.append(Value::int_unchecked(3));

    let vals: Vec<i64> = d.iter().rev().filter_map(|v| v.as_int()).collect();
    assert_eq!(vals, vec![3, 2, 1]);
}

#[test]
fn test_into_iter() {
    let mut d = Deque::new();
    d.append(Value::int_unchecked(1));
    d.append(Value::int_unchecked(2));

    let vals: Vec<i64> = d.into_iter().filter_map(|v| v.as_int()).collect();
    assert_eq!(vals, vec![1, 2]);
}

// =========================================================================
// Extend Tests
// =========================================================================

#[test]
fn test_extend() {
    let mut d = Deque::new();
    d.append(Value::int_unchecked(1));

    d.extend(vec![Value::int_unchecked(2), Value::int_unchecked(3)]);

    assert_eq!(d.len(), 3);
    assert_eq!(d.get(2).and_then(|v| v.as_int()), Some(3));
}

#[test]
fn test_extendleft() {
    let mut d = Deque::new();
    d.append(Value::int_unchecked(3));

    d.extendleft(vec![Value::int_unchecked(2), Value::int_unchecked(1)]);

    // Note: extendleft adds in reverse order (like Python)
    assert_eq!(d.len(), 3);
    assert_eq!(d.get(0).and_then(|v| v.as_int()), Some(1));
}

// =========================================================================
// Growth Tests
// =========================================================================

#[test]
fn test_growth_on_overflow() {
    let mut d = Deque::with_capacity(16);
    let initial_cap = d.capacity();

    for i in 0..20 {
        d.append(Value::int_unchecked(i));
    }

    assert!(d.capacity() > initial_cap);
    assert_eq!(d.len(), 20);

    // Verify all elements are correct
    for i in 0..20 {
        assert_eq!(d.get(i as isize).and_then(|v| v.as_int()), Some(i));
    }
}

#[test]
fn test_growth_preserves_order() {
    let mut d = Deque::with_capacity(16);

    // Add elements from both ends
    for i in 0..10 {
        d.append(Value::int_unchecked(i));
        d.appendleft(Value::int_unchecked(-(i + 1)));
    }

    // Force growth
    for i in 10..30 {
        d.append(Value::int_unchecked(i));
    }

    // Verify order is preserved
    assert_eq!(d.get(0).and_then(|v| v.as_int()), Some(-10));
    assert_eq!(d.get(10).and_then(|v| v.as_int()), Some(0));
}

// =========================================================================
// Equality Tests
// =========================================================================

#[test]
fn test_equality() {
    let mut d1 = Deque::new();
    let mut d2 = Deque::new();

    for i in 0..5 {
        d1.append(Value::int_unchecked(i));
        d2.append(Value::int_unchecked(i));
    }

    assert_eq!(d1, d2);
}

#[test]
fn test_inequality_different_length() {
    let mut d1 = Deque::new();
    let mut d2 = Deque::new();

    d1.append(Value::int_unchecked(1));
    d2.append(Value::int_unchecked(1));
    d2.append(Value::int_unchecked(2));

    assert_ne!(d1, d2);
}

#[test]
fn test_inequality_different_values() {
    let mut d1 = Deque::new();
    let mut d2 = Deque::new();

    d1.append(Value::int_unchecked(1));
    d2.append(Value::int_unchecked(2));

    assert_ne!(d1, d2);
}

#[test]
fn test_equality_with_identical_nan_values() {
    let nan = Value::float(f64::NAN);
    let mut d1 = Deque::new();
    let mut d2 = Deque::new();
    d1.append(nan);
    d2.append(nan);

    assert_eq!(d1, d2);
}

// =========================================================================
// Index Trait Tests
// =========================================================================

#[test]
fn test_index_trait() {
    let mut d = Deque::new();
    d.append(Value::int_unchecked(10));
    d.append(Value::int_unchecked(20));

    assert_eq!(d[0].as_int(), Some(10));
    assert_eq!(d[1].as_int(), Some(20));
}

#[test]
#[should_panic(expected = "deque index out of bounds")]
fn test_index_out_of_bounds_panics() {
    let d = Deque::new();
    let _ = &d[0];
}

// =========================================================================
// Stress Tests
// =========================================================================

#[test]
fn test_stress_alternating_operations() {
    let mut d = Deque::new();

    for i in 0..1000 {
        if i % 2 == 0 {
            d.append(Value::int_unchecked(i));
        } else {
            d.appendleft(Value::int_unchecked(i));
        }
    }

    for _ in 0..500 {
        d.pop();
        d.popleft();
    }

    assert!(d.is_empty());
}

#[test]
fn test_stress_rotate_many() {
    let mut d = Deque::new();
    for i in 0..100 {
        d.append(Value::int_unchecked(i));
    }

    // Rotate by full length should result in same order
    d.rotate(100);

    for i in 0..100 {
        assert_eq!(d.get(i).and_then(|v| v.as_int()), Some(i as i64));
    }
}
