use super::*;

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
// GroupBy tests
// =========================================================================

#[test]
fn test_groupby_identity() {
    let data = vals(&[1, 1, 2, 2, 2, 3, 1, 1]);
    let result: Vec<(Value, Vec<Value>)> = GroupBy::new(data.into_iter(), identity_key).collect();
    assert_eq!(result.len(), 4);
    assert_eq!(result[0].0.as_int(), Some(1));
    assert_eq!(to_ints(result[0].1.clone()), vec![1, 1]);
    assert_eq!(result[1].0.as_int(), Some(2));
    assert_eq!(to_ints(result[1].1.clone()), vec![2, 2, 2]);
    assert_eq!(result[2].0.as_int(), Some(3));
    assert_eq!(to_ints(result[2].1.clone()), vec![3]);
    assert_eq!(result[3].0.as_int(), Some(1));
    assert_eq!(to_ints(result[3].1.clone()), vec![1, 1]);
}

#[test]
fn test_groupby_with_key() {
    // Group by even/odd
    let data = vals(&[1, 3, 5, 2, 4, 1, 3]);
    let result: Vec<(Value, Vec<Value>)> =
        GroupBy::new(data.into_iter(), |v| int(v.as_int().unwrap() % 2)).collect();
    // 1,3,5 → key 1; 2,4 → key 0; 1,3 → key 1
    assert_eq!(result.len(), 3);
    assert_eq!(result[0].0.as_int(), Some(1)); // odd
    assert_eq!(to_ints(result[0].1.clone()), vec![1, 3, 5]);
    assert_eq!(result[1].0.as_int(), Some(0)); // even
    assert_eq!(to_ints(result[1].1.clone()), vec![2, 4]);
    assert_eq!(result[2].0.as_int(), Some(1)); // odd again
    assert_eq!(to_ints(result[2].1.clone()), vec![1, 3]);
}

#[test]
fn test_groupby_single_element() {
    let data = vals(&[42]);
    let result: Vec<(Value, Vec<Value>)> = GroupBy::new(data.into_iter(), identity_key).collect();
    assert_eq!(result.len(), 1);
    assert_eq!(to_ints(result[0].1.clone()), vec![42]);
}

#[test]
fn test_groupby_all_same() {
    let data = vals(&[5, 5, 5, 5]);
    let result: Vec<(Value, Vec<Value>)> = GroupBy::new(data.into_iter(), identity_key).collect();
    assert_eq!(result.len(), 1);
    assert_eq!(to_ints(result[0].1.clone()), vec![5, 5, 5, 5]);
}

#[test]
fn test_groupby_all_different() {
    let data = vals(&[1, 2, 3, 4]);
    let result: Vec<(Value, Vec<Value>)> = GroupBy::new(data.into_iter(), identity_key).collect();
    assert_eq!(result.len(), 4);
    for (i, (_, group)) in result.iter().enumerate() {
        assert_eq!(group.len(), 1);
        assert_eq!(group[0].as_int(), Some((i + 1) as i64));
    }
}

#[test]
fn test_groupby_empty() {
    let data: Vec<Value> = vec![];
    let result: Vec<(Value, Vec<Value>)> = GroupBy::new(data.into_iter(), identity_key).collect();
    assert!(result.is_empty());
}

#[test]
fn test_groupby_preserves_order() {
    let data = vals(&[3, 3, 1, 1, 2, 2, 1, 1]);
    let keys: Vec<i64> = GroupBy::new(data.into_iter(), identity_key)
        .map(|(k, _)| k.as_int().unwrap())
        .collect();
    assert_eq!(keys, vec![3, 1, 2, 1]);
}

#[test]
fn test_groupby_key_division() {
    // Group by integer division: key = x / 10
    let data = vals(&[1, 5, 9, 10, 15, 20, 25, 30]);
    let result: Vec<(Value, Vec<Value>)> =
        GroupBy::new(data.into_iter(), |v| int(v.as_int().unwrap() / 10)).collect();
    assert_eq!(result[0].0.as_int(), Some(0)); // 1, 5, 9
    assert_eq!(to_ints(result[0].1.clone()), vec![1, 5, 9]);
    assert_eq!(result[1].0.as_int(), Some(1)); // 10, 15
    assert_eq!(to_ints(result[1].1.clone()), vec![10, 15]);
    assert_eq!(result[2].0.as_int(), Some(2)); // 20, 25
    assert_eq!(to_ints(result[2].1.clone()), vec![20, 25]);
    assert_eq!(result[3].0.as_int(), Some(3)); // 30
    assert_eq!(to_ints(result[3].1.clone()), vec![30]);
}

#[test]
fn test_groupby_bool_key() {
    // Group by > 3
    let data = vals(&[1, 2, 5, 6, 3, 8]);
    let result: Vec<(Value, Vec<Value>)> =
        GroupBy::new(data.into_iter(), |v| Value::bool(v.as_int().unwrap() > 3)).collect();
    assert_eq!(result.len(), 4);
    // First group: false (1, 2)
    assert_eq!(result[0].0.as_bool(), Some(false));
    assert_eq!(to_ints(result[0].1.clone()), vec![1, 2]);
    // Second group: true (5, 6)
    assert_eq!(result[1].0.as_bool(), Some(true));
    assert_eq!(to_ints(result[1].1.clone()), vec![5, 6]);
    // Third group: false (3)
    assert_eq!(result[2].0.as_bool(), Some(false));
    assert_eq!(to_ints(result[2].1.clone()), vec![3]);
    // Fourth group: true (8)
    assert_eq!(result[3].0.as_bool(), Some(true));
    assert_eq!(to_ints(result[3].1.clone()), vec![8]);
}

// =========================================================================
// Pairwise tests
// =========================================================================

#[test]
fn test_pairwise_basic() {
    let data = vals(&[1, 2, 3, 4]);
    let result: Vec<(i64, i64)> = Pairwise::new(data.into_iter())
        .map(|(a, b)| (a.as_int().unwrap(), b.as_int().unwrap()))
        .collect();
    assert_eq!(result, vec![(1, 2), (2, 3), (3, 4)]);
}

#[test]
fn test_pairwise_empty() {
    let data: Vec<Value> = vec![];
    let result: Vec<(Value, Value)> = Pairwise::new(data.into_iter()).collect();
    assert!(result.is_empty());
}

#[test]
fn test_pairwise_single() {
    let data = vals(&[1]);
    let result: Vec<(Value, Value)> = Pairwise::new(data.into_iter()).collect();
    assert!(result.is_empty());
}

#[test]
fn test_pairwise_two_elements() {
    let data = vals(&[10, 20]);
    let result: Vec<(i64, i64)> = Pairwise::new(data.into_iter())
        .map(|(a, b)| (a.as_int().unwrap(), b.as_int().unwrap()))
        .collect();
    assert_eq!(result, vec![(10, 20)]);
}

#[test]
fn test_pairwise_five_elements() {
    let data = vals(&[1, 2, 3, 4, 5]);
    let result: Vec<(i64, i64)> = Pairwise::new(data.into_iter())
        .map(|(a, b)| (a.as_int().unwrap(), b.as_int().unwrap()))
        .collect();
    assert_eq!(result.len(), 4);
    assert_eq!(result, vec![(1, 2), (2, 3), (3, 4), (4, 5)]);
}

#[test]
fn test_pairwise_fused() {
    let data = vals(&[1, 2]);
    let mut p = Pairwise::new(data.into_iter());
    assert!(p.next().is_some());
    assert!(p.next().is_none());
    assert!(p.next().is_none());
}

#[test]
fn test_pairwise_stress() {
    let data: Vec<Value> = (0..1000).map(|i| int(i)).collect();
    let result: Vec<(Value, Value)> = Pairwise::new(data.into_iter()).collect();
    assert_eq!(result.len(), 999);
}

// =========================================================================
// Batched tests
// =========================================================================

#[test]
fn test_batched_even_split() {
    let data = vals(&[1, 2, 3, 4, 5, 6]);
    let result: Vec<Vec<i64>> = Batched::new(data.into_iter(), 2)
        .map(|batch| to_ints(batch))
        .collect();
    assert_eq!(result, vec![vec![1, 2], vec![3, 4], vec![5, 6]]);
}

#[test]
fn test_batched_uneven_split() {
    let data = vals(&[1, 2, 3, 4, 5]);
    let result: Vec<Vec<i64>> = Batched::new(data.into_iter(), 2)
        .map(|batch| to_ints(batch))
        .collect();
    assert_eq!(result, vec![vec![1, 2], vec![3, 4], vec![5]]);
}

#[test]
fn test_batched_size_1() {
    let data = vals(&[1, 2, 3]);
    let result: Vec<Vec<i64>> = Batched::new(data.into_iter(), 1)
        .map(|batch| to_ints(batch))
        .collect();
    assert_eq!(result, vec![vec![1], vec![2], vec![3]]);
}

#[test]
fn test_batched_size_larger_than_input() {
    let data = vals(&[1, 2, 3]);
    let result: Vec<Vec<i64>> = Batched::new(data.into_iter(), 10)
        .map(|batch| to_ints(batch))
        .collect();
    assert_eq!(result, vec![vec![1, 2, 3]]);
}

#[test]
fn test_batched_exact_fit() {
    let data = vals(&[1, 2, 3, 4, 5, 6, 7, 8, 9]);
    let result: Vec<Vec<i64>> = Batched::new(data.into_iter(), 3)
        .map(|batch| to_ints(batch))
        .collect();
    assert_eq!(result, vec![vec![1, 2, 3], vec![4, 5, 6], vec![7, 8, 9]]);
}

#[test]
fn test_batched_empty() {
    let data: Vec<Value> = vec![];
    let result: Vec<Vec<Value>> = Batched::new(data.into_iter(), 3).collect();
    assert!(result.is_empty());
}

#[test]
#[should_panic(expected = "batch_size must be >= 1")]
fn test_batched_zero_panics() {
    let data = vals(&[1]);
    let _ = Batched::new(data.into_iter(), 0);
}

#[test]
fn test_batched_fused() {
    let data = vals(&[1, 2, 3]);
    let mut b = Batched::new(data.into_iter(), 5);
    assert!(b.next().is_some());
    assert!(b.next().is_none());
    assert!(b.next().is_none());
}

#[test]
fn test_batched_size_hint() {
    let data = vals(&[1, 2, 3, 4, 5]);
    let b = Batched::new(data.into_iter(), 2);
    let (lo, hi) = b.size_hint();
    assert_eq!(lo, 3); // ceil(5/2)
    assert_eq!(hi, Some(3));
}

#[test]
fn test_batched_stress() {
    let data: Vec<Value> = (0..10_000).map(|i| int(i)).collect();
    let result: Vec<Vec<Value>> = Batched::new(data.into_iter(), 100).collect();
    assert_eq!(result.len(), 100);
    for batch in &result {
        assert_eq!(batch.len(), 100);
    }
}

// =========================================================================
// Triplewise tests
// =========================================================================

#[test]
fn test_triplewise_basic() {
    let data = vals(&[1, 2, 3, 4, 5]);
    let result: Vec<(i64, i64, i64)> = Triplewise::new(data.into_iter())
        .map(|(a, b, c)| {
            (
                a.as_int().unwrap(),
                b.as_int().unwrap(),
                c.as_int().unwrap(),
            )
        })
        .collect();
    assert_eq!(result, vec![(1, 2, 3), (2, 3, 4), (3, 4, 5)]);
}

#[test]
fn test_triplewise_empty() {
    let data: Vec<Value> = vec![];
    let result: Vec<(Value, Value, Value)> = Triplewise::new(data.into_iter()).collect();
    assert!(result.is_empty());
}

#[test]
fn test_triplewise_one() {
    let data = vals(&[1]);
    let result: Vec<(Value, Value, Value)> = Triplewise::new(data.into_iter()).collect();
    assert!(result.is_empty());
}

#[test]
fn test_triplewise_two() {
    let data = vals(&[1, 2]);
    let result: Vec<(Value, Value, Value)> = Triplewise::new(data.into_iter()).collect();
    assert!(result.is_empty());
}

#[test]
fn test_triplewise_three() {
    let data = vals(&[10, 20, 30]);
    let result: Vec<(i64, i64, i64)> = Triplewise::new(data.into_iter())
        .map(|(a, b, c)| {
            (
                a.as_int().unwrap(),
                b.as_int().unwrap(),
                c.as_int().unwrap(),
            )
        })
        .collect();
    assert_eq!(result, vec![(10, 20, 30)]);
}

#[test]
fn test_triplewise_stress() {
    let data: Vec<Value> = (0..1000).map(|i| int(i)).collect();
    let result: Vec<(Value, Value, Value)> = Triplewise::new(data.into_iter()).collect();
    assert_eq!(result.len(), 998);
}

// =========================================================================
// values_equal tests
// =========================================================================

#[test]
fn test_values_equal_same_int() {
    assert!(values_equal(&int(42), &int(42)));
}

#[test]
fn test_values_equal_different_int() {
    assert!(!values_equal(&int(1), &int(2)));
}

#[test]
fn test_values_equal_same_float() {
    assert!(values_equal(&Value::float(3.14), &Value::float(3.14)));
}

#[test]
fn test_values_equal_int_float_same() {
    assert!(values_equal(&int(5), &Value::float(5.0)));
}

#[test]
fn test_values_equal_none() {
    assert!(values_equal(&Value::none(), &Value::none()));
}

#[test]
fn test_values_equal_bool() {
    assert!(values_equal(&Value::bool(true), &Value::bool(true)));
    assert!(!values_equal(&Value::bool(true), &Value::bool(false)));
}
