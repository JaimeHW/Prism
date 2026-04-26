use super::*;
use prism_core::Value;

fn int(i: i64) -> Value {
    Value::int_unchecked(i)
}

fn vals(ints: &[i64]) -> Vec<Value> {
    ints.iter().map(|&i| int(i)).collect()
}

fn to_ints(v: Vec<Value>) -> Vec<i64> {
    v.into_iter().filter_map(|v| v.as_int()).collect()
}

// =========================================================================
// Chain tests
// =========================================================================

#[test]
fn test_chain_basic() {
    let a = vals(&[1, 2, 3]);
    let b = vals(&[4, 5, 6]);
    let result = to_ints(Chain::new(a.into_iter(), b.into_iter()).collect());
    assert_eq!(result, vec![1, 2, 3, 4, 5, 6]);
}

#[test]
fn test_chain_empty_first() {
    let a: Vec<Value> = vec![];
    let b = vals(&[1, 2, 3]);
    let result = to_ints(Chain::new(a.into_iter(), b.into_iter()).collect());
    assert_eq!(result, vec![1, 2, 3]);
}

#[test]
fn test_chain_empty_second() {
    let a = vals(&[1, 2, 3]);
    let b: Vec<Value> = vec![];
    let result = to_ints(Chain::new(a.into_iter(), b.into_iter()).collect());
    assert_eq!(result, vec![1, 2, 3]);
}

#[test]
fn test_chain_both_empty() {
    let a: Vec<Value> = vec![];
    let b: Vec<Value> = vec![];
    let result: Vec<Value> = Chain::new(a.into_iter(), b.into_iter()).collect();
    assert!(result.is_empty());
}

#[test]
fn test_chain_single_elements() {
    let a = vals(&[1]);
    let b = vals(&[2]);
    let result = to_ints(Chain::new(a.into_iter(), b.into_iter()).collect());
    assert_eq!(result, vec![1, 2]);
}

#[test]
fn test_chain_size_hint() {
    let a = vals(&[1, 2, 3]);
    let b = vals(&[4, 5]);
    let c = Chain::new(a.into_iter(), b.into_iter());
    assert_eq!(c.size_hint(), (5, Some(5)));
}

#[test]
fn test_chain_from_iterable() {
    let iterables = vec![vals(&[1, 2]), vals(&[3]), vals(&[4, 5, 6])];
    let result = to_ints(chain_from_iterable(iterables).collect());
    assert_eq!(result, vec![1, 2, 3, 4, 5, 6]);
}

#[test]
fn test_chain_from_iterable_empty() {
    let iterables: Vec<Vec<Value>> = vec![vec![], vec![], vec![]];
    let result: Vec<Value> = chain_from_iterable(iterables).collect();
    assert!(result.is_empty());
}

#[test]
fn test_chain_large() {
    let a: Vec<Value> = (0..500).map(|i| int(i)).collect();
    let b: Vec<Value> = (500..1000).map(|i| int(i)).collect();
    let result = to_ints(Chain::new(a.into_iter(), b.into_iter()).collect());
    assert_eq!(result.len(), 1000);
    for (i, v) in result.iter().enumerate() {
        assert_eq!(*v, i as i64);
    }
}

// =========================================================================
// Compress tests
// =========================================================================

#[test]
fn test_compress_basic() {
    let data = vals(&[1, 2, 3, 4, 5]);
    let selectors = vec![
        Value::bool(true),
        Value::bool(false),
        Value::bool(true),
        Value::bool(false),
        Value::bool(true),
    ];
    let result = to_ints(Compress::new(data.into_iter(), selectors.into_iter()).collect());
    assert_eq!(result, vec![1, 3, 5]);
}

#[test]
fn test_compress_with_int_selectors() {
    let data = vals(&[10, 20, 30, 40]);
    let selectors = vals(&[1, 0, 1, 0]);
    let result = to_ints(Compress::new(data.into_iter(), selectors.into_iter()).collect());
    assert_eq!(result, vec![10, 30]);
}

#[test]
fn test_compress_all_true() {
    let data = vals(&[1, 2, 3]);
    let selectors = vec![Value::bool(true), Value::bool(true), Value::bool(true)];
    let result = to_ints(Compress::new(data.into_iter(), selectors.into_iter()).collect());
    assert_eq!(result, vec![1, 2, 3]);
}

#[test]
fn test_compress_all_false() {
    let data = vals(&[1, 2, 3]);
    let selectors = vec![Value::bool(false), Value::bool(false), Value::bool(false)];
    let result: Vec<Value> = Compress::new(data.into_iter(), selectors.into_iter()).collect();
    assert!(result.is_empty());
}

#[test]
fn test_compress_empty_data() {
    let data: Vec<Value> = vec![];
    let selectors = vec![Value::bool(true)];
    let result: Vec<Value> = Compress::new(data.into_iter(), selectors.into_iter()).collect();
    assert!(result.is_empty());
}

#[test]
fn test_compress_shorter_selectors() {
    // Stops when selectors run out
    let data = vals(&[1, 2, 3, 4, 5]);
    let selectors = vec![Value::bool(true), Value::bool(true)];
    let result = to_ints(Compress::new(data.into_iter(), selectors.into_iter()).collect());
    assert_eq!(result, vec![1, 2]);
}

#[test]
fn test_compress_none_selector() {
    // None is falsy
    let data = vals(&[1, 2]);
    let selectors = vec![Value::none(), Value::bool(true)];
    let result = to_ints(Compress::new(data.into_iter(), selectors.into_iter()).collect());
    assert_eq!(result, vec![2]);
}

// =========================================================================
// DropWhile tests
// =========================================================================

#[test]
fn test_dropwhile_basic() {
    let data = vals(&[1, 2, 3, 4, 5, 1, 2]);
    let result = to_ints(DropWhile::new(data.into_iter(), |v| v.as_int().unwrap() < 3).collect());
    assert_eq!(result, vec![3, 4, 5, 1, 2]); // doesn't re-drop after start
}

#[test]
fn test_dropwhile_all_dropped() {
    let data = vals(&[1, 2, 3]);
    let result: Vec<Value> =
        DropWhile::new(data.into_iter(), |v| v.as_int().unwrap() < 10).collect();
    assert!(result.is_empty());
}

#[test]
fn test_dropwhile_none_dropped() {
    let data = vals(&[5, 4, 3, 2, 1]);
    let result = to_ints(DropWhile::new(data.into_iter(), |v| v.as_int().unwrap() > 10).collect());
    assert_eq!(result, vec![5, 4, 3, 2, 1]);
}

#[test]
fn test_dropwhile_empty() {
    let data: Vec<Value> = vec![];
    let result: Vec<Value> = DropWhile::new(data.into_iter(), |_| true).collect();
    assert!(result.is_empty());
}

#[test]
fn test_dropwhile_single_match() {
    let data = vals(&[1, 5, 3]);
    let result = to_ints(DropWhile::new(data.into_iter(), |v| v.as_int().unwrap() < 3).collect());
    assert_eq!(result, vec![5, 3]);
}

#[test]
fn test_dropwhile_predicate_not_re_evaluated() {
    // After the first false, remaining elements are passed through
    let data = vals(&[1, 4, 2, 5, 1]);
    let result = to_ints(DropWhile::new(data.into_iter(), |v| v.as_int().unwrap() < 3).collect());
    // Should yield 4, 2, 5, 1 — the 2 and 1 are NOT re-dropped
    assert_eq!(result, vec![4, 2, 5, 1]);
}

// =========================================================================
// TakeWhile tests
// =========================================================================

#[test]
fn test_takewhile_basic() {
    let data = vals(&[1, 2, 3, 4, 5]);
    let result = to_ints(TakeWhile::new(data.into_iter(), |v| v.as_int().unwrap() < 4).collect());
    assert_eq!(result, vec![1, 2, 3]);
}

#[test]
fn test_takewhile_all_taken() {
    let data = vals(&[1, 2, 3]);
    let result = to_ints(TakeWhile::new(data.into_iter(), |v| v.as_int().unwrap() < 10).collect());
    assert_eq!(result, vec![1, 2, 3]);
}

#[test]
fn test_takewhile_none_taken() {
    let data = vals(&[5, 4, 3]);
    let result: Vec<Value> =
        TakeWhile::new(data.into_iter(), |v| v.as_int().unwrap() < 0).collect();
    assert!(result.is_empty());
}

#[test]
fn test_takewhile_empty() {
    let data: Vec<Value> = vec![];
    let result: Vec<Value> = TakeWhile::new(data.into_iter(), |_| true).collect();
    assert!(result.is_empty());
}

#[test]
fn test_takewhile_stops_at_first_false() {
    // Even if later elements satisfy the predicate
    let data = vals(&[1, 2, 5, 3, 4]);
    let result = to_ints(TakeWhile::new(data.into_iter(), |v| v.as_int().unwrap() < 4).collect());
    assert_eq!(result, vec![1, 2]); // stops at 5
}

#[test]
fn test_takewhile_fused() {
    let data = vals(&[1, 5]);
    let mut tw = TakeWhile::new(data.into_iter(), |v| v.as_int().unwrap() < 3);
    assert!(tw.next().is_some()); // 1
    assert!(tw.next().is_none()); // 5 fails predicate
    assert!(tw.next().is_none()); // fused
}

// =========================================================================
// FilterFalse tests
// =========================================================================

#[test]
fn test_filterfalse_basic() {
    let data = vals(&[1, 2, 3, 4, 5, 6]);
    let result =
        to_ints(FilterFalse::new(data.into_iter(), |v| v.as_int().unwrap() % 2 == 0).collect());
    assert_eq!(result, vec![1, 3, 5]); // odd numbers (where "is even" is false)
}

#[test]
fn test_filterfalse_all_pass_predicate() {
    let data = vals(&[2, 4, 6]);
    let result: Vec<Value> =
        FilterFalse::new(data.into_iter(), |v| v.as_int().unwrap() % 2 == 0).collect();
    assert!(result.is_empty()); // all even, all filtered out
}

#[test]
fn test_filterfalse_none_pass_predicate() {
    let data = vals(&[1, 3, 5]);
    let result =
        to_ints(FilterFalse::new(data.into_iter(), |v| v.as_int().unwrap() % 2 == 0).collect());
    assert_eq!(result, vec![1, 3, 5]); // none even, all kept
}

#[test]
fn test_filterfalse_empty() {
    let data: Vec<Value> = vec![];
    let result: Vec<Value> = FilterFalse::new(data.into_iter(), |_| true).collect();
    assert!(result.is_empty());
}

#[test]
fn test_filterfalse_complement_of_filter() {
    let data = vals(&[1, 2, 3, 4, 5]);
    let pred = |v: &Value| v.as_int().unwrap() > 3;

    let filtered: Vec<i64> = data
        .iter()
        .cloned()
        .filter(|v| pred(v))
        .filter_map(|v| v.as_int())
        .collect();
    let false_filtered = to_ints(FilterFalse::new(data.into_iter(), pred).collect());

    // Together they should cover all elements
    assert_eq!(filtered, vec![4, 5]);
    assert_eq!(false_filtered, vec![1, 2, 3]);
}

// =========================================================================
// ISlice tests
// =========================================================================

#[test]
fn test_islice_stop_only() {
    let data = vals(&[0, 1, 2, 3, 4, 5]);
    let result = to_ints(ISlice::new_stop(data.into_iter(), 3).collect());
    assert_eq!(result, vec![0, 1, 2]);
}

#[test]
fn test_islice_start_stop() {
    let data = vals(&[0, 1, 2, 3, 4, 5]);
    let result = to_ints(ISlice::new(data.into_iter(), 2, Some(5), 1).collect());
    assert_eq!(result, vec![2, 3, 4]);
}

#[test]
fn test_islice_start_stop_step() {
    let data = vals(&[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
    let result = to_ints(ISlice::new(data.into_iter(), 1, Some(8), 2).collect());
    assert_eq!(result, vec![1, 3, 5, 7]);
}

#[test]
fn test_islice_stop_zero() {
    let data = vals(&[0, 1, 2]);
    let result: Vec<Value> = ISlice::new_stop(data.into_iter(), 0).collect();
    assert!(result.is_empty());
}

#[test]
fn test_islice_stop_beyond_length() {
    let data = vals(&[0, 1, 2]);
    let result = to_ints(ISlice::new_stop(data.into_iter(), 100).collect());
    assert_eq!(result, vec![0, 1, 2]); // just yields all
}

#[test]
fn test_islice_start_beyond_length() {
    let data = vals(&[0, 1, 2]);
    let result: Vec<Value> = ISlice::new(data.into_iter(), 10, Some(20), 1).collect();
    assert!(result.is_empty());
}

#[test]
fn test_islice_step_3() {
    let data: Vec<Value> = (0..20).map(|i| int(i)).collect();
    let result = to_ints(ISlice::new(data.into_iter(), 0, Some(20), 3).collect());
    assert_eq!(result, vec![0, 3, 6, 9, 12, 15, 18]);
}

#[test]
fn test_islice_no_stop() {
    let data: Vec<Value> = (0..10).map(|i| int(i)).collect();
    let result = to_ints(ISlice::new(data.into_iter(), 5, None, 1).collect());
    assert_eq!(result, vec![5, 6, 7, 8, 9]);
}

#[test]
fn test_islice_empty_input() {
    let data: Vec<Value> = vec![];
    let result: Vec<Value> = ISlice::new_stop(data.into_iter(), 5).collect();
    assert!(result.is_empty());
}

#[test]
fn test_islice_single_element() {
    let data = vals(&[42]);
    let result = to_ints(ISlice::new(data.into_iter(), 0, Some(1), 1).collect());
    assert_eq!(result, vec![42]);
}

#[test]
fn test_islice_fused() {
    let data = vals(&[1, 2, 3]);
    let mut s = ISlice::new_stop(data.into_iter(), 2);
    assert!(s.next().is_some());
    assert!(s.next().is_some());
    assert!(s.next().is_none());
    assert!(s.next().is_none()); // fused
}

// =========================================================================
// Starmap tests
// =========================================================================

#[test]
fn test_starmap_add() {
    let pairs = vec![(int(1), int(2)), (int(3), int(4)), (int(5), int(6))];
    let result = to_ints(
        Starmap::new(pairs.into_iter(), |a, b| {
            int(a.as_int().unwrap() + b.as_int().unwrap())
        })
        .collect(),
    );
    assert_eq!(result, vec![3, 7, 11]);
}

#[test]
fn test_starmap_mul() {
    let pairs = vec![(int(2), int(3)), (int(4), int(5))];
    let result = to_ints(
        Starmap::new(pairs.into_iter(), |a, b| {
            int(a.as_int().unwrap() * b.as_int().unwrap())
        })
        .collect(),
    );
    assert_eq!(result, vec![6, 20]);
}

#[test]
fn test_starmap_empty() {
    let pairs: Vec<(Value, Value)> = vec![];
    let result: Vec<Value> = Starmap::new(pairs.into_iter(), |a, b| {
        int(a.as_int().unwrap() + b.as_int().unwrap())
    })
    .collect();
    assert!(result.is_empty());
}

#[test]
fn test_starmap_pow() {
    let pairs = vec![(int(2), int(10)), (int(3), int(3))];
    let result = to_ints(
        Starmap::new(pairs.into_iter(), |a, b| {
            int(a.as_int().unwrap().pow(b.as_int().unwrap() as u32))
        })
        .collect(),
    );
    assert_eq!(result, vec![1024, 27]);
}

#[test]
fn test_starmap_size_hint() {
    let pairs = vec![(int(1), int(2)), (int(3), int(4))];
    let sm = Starmap::new(pairs.into_iter(), |a, b| {
        int(a.as_int().unwrap() + b.as_int().unwrap())
    });
    assert_eq!(sm.size_hint(), (2, Some(2)));
}

// =========================================================================
// ZipLongest tests
// =========================================================================

#[test]
fn test_zip_longest_equal_length() {
    let a = vals(&[1, 2, 3]);
    let b = vals(&[4, 5, 6]);
    let result: Vec<EitherOrBoth> = ZipLongest::new(a.into_iter(), b.into_iter()).collect();
    assert_eq!(result.len(), 3);
    assert!(result.iter().all(|e| e.has_both()));
}

#[test]
fn test_zip_longest_left_longer() {
    let a = vals(&[1, 2, 3, 4]);
    let b = vals(&[10, 20]);
    let result: Vec<EitherOrBoth> = ZipLongest::new(a.into_iter(), b.into_iter()).collect();
    assert_eq!(result.len(), 4);
    assert!(result[0].has_both());
    assert!(result[1].has_both());
    assert!(matches!(&result[2], EitherOrBoth::Left(_)));
    assert!(matches!(&result[3], EitherOrBoth::Left(_)));
}

#[test]
fn test_zip_longest_right_longer() {
    let a = vals(&[1]);
    let b = vals(&[10, 20, 30]);
    let result: Vec<EitherOrBoth> = ZipLongest::new(a.into_iter(), b.into_iter()).collect();
    assert_eq!(result.len(), 3);
    assert!(result[0].has_both());
    assert!(matches!(&result[1], EitherOrBoth::Right(_)));
    assert!(matches!(&result[2], EitherOrBoth::Right(_)));
}

#[test]
fn test_zip_longest_both_empty() {
    let a: Vec<Value> = vec![];
    let b: Vec<Value> = vec![];
    let result: Vec<EitherOrBoth> = ZipLongest::new(a.into_iter(), b.into_iter()).collect();
    assert!(result.is_empty());
}

#[test]
fn test_zip_longest_left_empty() {
    let a: Vec<Value> = vec![];
    let b = vals(&[1, 2]);
    let result: Vec<EitherOrBoth> = ZipLongest::new(a.into_iter(), b.into_iter()).collect();
    assert_eq!(result.len(), 2);
    assert!(result.iter().all(|e| matches!(e, EitherOrBoth::Right(_))));
}

#[test]
fn test_zip_longest_into_tuple() {
    let eob = EitherOrBoth::Left(int(5));
    let (l, r) = eob.into_tuple(int(0), int(99));
    assert_eq!(l.as_int(), Some(5));
    assert_eq!(r.as_int(), Some(99)); // fill value used
}

#[test]
fn test_zip_longest_left_right_accessors() {
    let both = EitherOrBoth::Both(int(1), int(2));
    assert_eq!(both.left().unwrap().as_int(), Some(1));
    assert_eq!(both.right().unwrap().as_int(), Some(2));

    let left = EitherOrBoth::Left(int(3));
    assert_eq!(left.left().unwrap().as_int(), Some(3));
    assert!(left.right().is_none());

    let right = EitherOrBoth::Right(int(4));
    assert!(right.left().is_none());
    assert_eq!(right.right().unwrap().as_int(), Some(4));
}

#[test]
fn test_zip_longest_size_hint() {
    let a = vals(&[1, 2, 3]);
    let b = vals(&[4, 5]);
    let z = ZipLongest::new(a.into_iter(), b.into_iter());
    let (lo, hi) = z.size_hint();
    assert_eq!(lo, 3); // max of 3 and 2
    assert_eq!(hi, Some(3));
}

// =========================================================================
// is_truthy tests
// =========================================================================

#[test]
fn test_truthy_none() {
    assert!(!is_truthy(&Value::none()));
}

#[test]
fn test_truthy_bool() {
    assert!(is_truthy(&Value::bool(true)));
    assert!(!is_truthy(&Value::bool(false)));
}

#[test]
fn test_truthy_int() {
    assert!(is_truthy(&int(1)));
    assert!(is_truthy(&int(-1)));
    assert!(!is_truthy(&int(0)));
}

#[test]
fn test_truthy_float() {
    assert!(is_truthy(&Value::float(1.0)));
    assert!(!is_truthy(&Value::float(0.0)));
}

// =========================================================================
// Stress tests
// =========================================================================

#[test]
fn test_chain_stress_10k() {
    let a: Vec<Value> = (0..5000).map(|i| int(i)).collect();
    let b: Vec<Value> = (5000..10000).map(|i| int(i)).collect();
    let result = to_ints(Chain::new(a.into_iter(), b.into_iter()).collect());
    assert_eq!(result.len(), 10_000);
    assert_eq!(result[0], 0);
    assert_eq!(result[9999], 9999);
}

#[test]
fn test_dropwhile_stress() {
    let data: Vec<Value> = (0..10_000).map(|i| int(i)).collect();
    let result =
        to_ints(DropWhile::new(data.into_iter(), |v| v.as_int().unwrap() < 9000).collect());
    assert_eq!(result.len(), 1000);
    assert_eq!(result[0], 9000);
}

#[test]
fn test_islice_stress_step() {
    let data: Vec<Value> = (0..10_000).map(|i| int(i)).collect();
    let result = to_ints(ISlice::new(data.into_iter(), 0, Some(10_000), 10).collect());
    assert_eq!(result.len(), 1000);
    for (i, v) in result.iter().enumerate() {
        assert_eq!(*v, (i * 10) as i64);
    }
}
