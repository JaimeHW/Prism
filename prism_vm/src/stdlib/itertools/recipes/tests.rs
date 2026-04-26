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
// Flatten tests
// =========================================================================

#[test]
fn test_flatten_basic() {
    let data = vec![vals(&[1, 2]), vals(&[3, 4]), vals(&[5])];
    let result = to_ints(Flatten::new(data.into_iter()).collect());
    assert_eq!(result, vec![1, 2, 3, 4, 5]);
}

#[test]
fn test_flatten_empty_inner() {
    let data: Vec<Vec<Value>> = vec![vec![], vals(&[1]), vec![], vals(&[2])];
    let result = to_ints(Flatten::new(data.into_iter()).collect());
    assert_eq!(result, vec![1, 2]);
}

#[test]
fn test_flatten_empty_outer() {
    let data: Vec<Vec<Value>> = vec![];
    let result: Vec<Value> = Flatten::new(data.into_iter()).collect();
    assert!(result.is_empty());
}

#[test]
fn test_flatten_all_empty_inner() {
    let data: Vec<Vec<Value>> = vec![vec![], vec![], vec![]];
    let result: Vec<Value> = Flatten::new(data.into_iter()).collect();
    assert!(result.is_empty());
}

#[test]
fn test_flatten_single() {
    let data = vec![vals(&[1, 2, 3])];
    let result = to_ints(Flatten::new(data.into_iter()).collect());
    assert_eq!(result, vec![1, 2, 3]);
}

#[test]
fn test_flatten_stress() {
    let data: Vec<Vec<Value>> = (0..100).map(|i| vec![int(i)]).collect();
    let result = to_ints(Flatten::new(data.into_iter()).collect());
    assert_eq!(result.len(), 100);
}

// =========================================================================
// UniqueEverseen tests
// =========================================================================

#[test]
fn test_unique_everseen_basic() {
    let data = vals(&[1, 2, 3, 1, 2, 4, 3, 5]);
    let result = to_ints(UniqueEverseen::new(data.into_iter()).collect());
    assert_eq!(result, vec![1, 2, 3, 4, 5]);
}

#[test]
fn test_unique_everseen_all_unique() {
    let data = vals(&[1, 2, 3, 4, 5]);
    let result = to_ints(UniqueEverseen::new(data.into_iter()).collect());
    assert_eq!(result, vec![1, 2, 3, 4, 5]);
}

#[test]
fn test_unique_everseen_all_same() {
    let data = vals(&[1, 1, 1, 1]);
    let result = to_ints(UniqueEverseen::new(data.into_iter()).collect());
    assert_eq!(result, vec![1]);
}

#[test]
fn test_unique_everseen_empty() {
    let data: Vec<Value> = vec![];
    let result: Vec<Value> = UniqueEverseen::new(data.into_iter()).collect();
    assert!(result.is_empty());
}

#[test]
fn test_unique_everseen_preserves_first_seen_order() {
    let data = vals(&[5, 3, 1, 3, 5, 2, 1, 4]);
    let result = to_ints(UniqueEverseen::new(data.into_iter()).collect());
    assert_eq!(result, vec![5, 3, 1, 2, 4]);
}

#[test]
fn test_unique_everseen_with_none() {
    let data = vec![Value::none(), int(1), Value::none(), int(2)];
    let result: Vec<Value> = UniqueEverseen::new(data.into_iter()).collect();
    assert_eq!(result.len(), 3); // None, 1, 2
}

#[test]
fn test_unique_everseen_stress() {
    // 10k elements with lots of duplicates
    let data: Vec<Value> = (0..10_000).map(|i| int(i % 100)).collect();
    let result = to_ints(UniqueEverseen::new(data.into_iter()).collect());
    assert_eq!(result.len(), 100);
}

// =========================================================================
// UniqueJustseen tests
// =========================================================================

#[test]
fn test_unique_justseen_basic() {
    let data = vals(&[1, 1, 2, 2, 2, 3, 1, 1]);
    let result = to_ints(UniqueJustseen::new(data.into_iter()).collect());
    assert_eq!(result, vec![1, 2, 3, 1]); // note: 1 appears twice (non-consecutive)
}

#[test]
fn test_unique_justseen_all_unique() {
    let data = vals(&[1, 2, 3, 4, 5]);
    let result = to_ints(UniqueJustseen::new(data.into_iter()).collect());
    assert_eq!(result, vec![1, 2, 3, 4, 5]);
}

#[test]
fn test_unique_justseen_all_same() {
    let data = vals(&[7, 7, 7, 7, 7]);
    let result = to_ints(UniqueJustseen::new(data.into_iter()).collect());
    assert_eq!(result, vec![7]);
}

#[test]
fn test_unique_justseen_empty() {
    let data: Vec<Value> = vec![];
    let result: Vec<Value> = UniqueJustseen::new(data.into_iter()).collect();
    assert!(result.is_empty());
}

#[test]
fn test_unique_justseen_single() {
    let data = vals(&[42]);
    let result = to_ints(UniqueJustseen::new(data.into_iter()).collect());
    assert_eq!(result, vec![42]);
}

#[test]
fn test_unique_justseen_alternating() {
    let data = vals(&[1, 2, 1, 2, 1, 2]);
    let result = to_ints(UniqueJustseen::new(data.into_iter()).collect());
    assert_eq!(result, vec![1, 2, 1, 2, 1, 2]); // no consecutive dupes
}

// =========================================================================
// SlidingWindow tests
// =========================================================================

#[test]
fn test_sliding_window_size_3() {
    let data = vals(&[1, 2, 3, 4, 5]);
    let result: Vec<Vec<i64>> = SlidingWindow::new(data.into_iter(), 3)
        .map(|w| to_ints(w))
        .collect();
    assert_eq!(result, vec![vec![1, 2, 3], vec![2, 3, 4], vec![3, 4, 5]]);
}

#[test]
fn test_sliding_window_size_1() {
    let data = vals(&[1, 2, 3]);
    let result: Vec<Vec<i64>> = SlidingWindow::new(data.into_iter(), 1)
        .map(|w| to_ints(w))
        .collect();
    assert_eq!(result, vec![vec![1], vec![2], vec![3]]);
}

#[test]
fn test_sliding_window_size_equals_length() {
    let data = vals(&[1, 2, 3]);
    let result: Vec<Vec<i64>> = SlidingWindow::new(data.into_iter(), 3)
        .map(|w| to_ints(w))
        .collect();
    assert_eq!(result, vec![vec![1, 2, 3]]);
}

#[test]
fn test_sliding_window_size_exceeds_length() {
    let data = vals(&[1, 2]);
    let result: Vec<Vec<Value>> = SlidingWindow::new(data.into_iter(), 5).collect();
    assert!(result.is_empty()); // not enough elements
}

#[test]
fn test_sliding_window_empty() {
    let data: Vec<Value> = vec![];
    let result: Vec<Vec<Value>> = SlidingWindow::new(data.into_iter(), 3).collect();
    assert!(result.is_empty());
}

#[test]
#[should_panic(expected = "size must be >= 1")]
fn test_sliding_window_zero_panics() {
    let data = vals(&[1]);
    let _ = SlidingWindow::new(data.into_iter(), 0);
}

#[test]
fn test_sliding_window_size_2_matches_pairwise() {
    // sliding_window(data, 2) should produce same as pairwise
    let data1 = vals(&[1, 2, 3, 4, 5]);
    let data2 = data1.clone();

    let sw: Vec<Vec<i64>> = SlidingWindow::new(data1.into_iter(), 2)
        .map(|w| to_ints(w))
        .collect();

    let pw: Vec<Vec<i64>> = crate::stdlib::itertools::grouping::Pairwise::new(data2.into_iter())
        .map(|(a, b)| vec![a.as_int().unwrap(), b.as_int().unwrap()])
        .collect();

    assert_eq!(sw, pw);
}

#[test]
fn test_sliding_window_stress() {
    let data: Vec<Value> = (0..1000).map(|i| int(i)).collect();
    let result: Vec<Vec<Value>> = SlidingWindow::new(data.into_iter(), 10).collect();
    assert_eq!(result.len(), 991); // 1000 - 10 + 1
}

// =========================================================================
// RoundRobin tests
// =========================================================================

#[test]
fn test_roundrobin_equal_length() {
    let iters = vec![vals(&[1, 2, 3]), vals(&[4, 5, 6])];
    let result = to_ints(RoundRobin::new(iters).collect());
    assert_eq!(result, vec![1, 4, 2, 5, 3, 6]);
}

#[test]
fn test_roundrobin_unequal_length() {
    let iters = vec![vals(&[1, 2, 3]), vals(&[4, 5]), vals(&[6, 7, 8, 9])];
    let result = to_ints(RoundRobin::new(iters).collect());
    // 1, 4, 6, 2, 5, 7, 3, 8, 9
    assert_eq!(result, vec![1, 4, 6, 2, 5, 7, 3, 8, 9]);
}

#[test]
fn test_roundrobin_single_iter() {
    let iters = vec![vals(&[1, 2, 3])];
    let result = to_ints(RoundRobin::new(iters).collect());
    assert_eq!(result, vec![1, 2, 3]);
}

#[test]
fn test_roundrobin_empty() {
    let iters: Vec<Vec<Value>> = vec![];
    let result: Vec<Value> = RoundRobin::new(iters).collect();
    assert!(result.is_empty());
}

#[test]
fn test_roundrobin_all_empty() {
    let iters: Vec<Vec<Value>> = vec![vec![], vec![], vec![]];
    let result: Vec<Value> = RoundRobin::new(iters).collect();
    assert!(result.is_empty());
}

#[test]
fn test_roundrobin_one_empty() {
    let iters = vec![vals(&[1, 2]), vec![], vals(&[3, 4])];
    let result = to_ints(RoundRobin::new(iters).collect());
    assert_eq!(result, vec![1, 3, 2, 4]);
}

#[test]
fn test_roundrobin_exact_size() {
    let iters = vec![vals(&[1, 2, 3]), vals(&[4, 5])];
    let rr = RoundRobin::new(iters);
    assert_eq!(rr.len(), 5);
}

// =========================================================================
// Accumulate tests
// =========================================================================

#[test]
fn test_accumulate_sum() {
    let data = vals(&[1, 2, 3, 4, 5]);
    let result = to_ints(
        Accumulate::new(data.into_iter(), |a, b| {
            int(a.as_int().unwrap() + b.as_int().unwrap())
        })
        .collect(),
    );
    assert_eq!(result, vec![1, 3, 6, 10, 15]);
}

#[test]
fn test_accumulate_product() {
    let data = vals(&[1, 2, 3, 4, 5]);
    let result = to_ints(
        Accumulate::new(data.into_iter(), |a, b| {
            int(a.as_int().unwrap() * b.as_int().unwrap())
        })
        .collect(),
    );
    assert_eq!(result, vec![1, 2, 6, 24, 120]);
}

#[test]
fn test_accumulate_with_initial() {
    let data = vals(&[1, 2, 3]);
    let result = to_ints(
        Accumulate::with_initial(
            data.into_iter(),
            |a, b| int(a.as_int().unwrap() + b.as_int().unwrap()),
            int(100),
        )
        .collect(),
    );
    assert_eq!(result, vec![100, 101, 103, 106]);
}

#[test]
fn test_accumulate_empty() {
    let data: Vec<Value> = vec![];
    let result: Vec<Value> = Accumulate::new(data.into_iter(), |a, b| {
        int(a.as_int().unwrap() + b.as_int().unwrap())
    })
    .collect();
    assert!(result.is_empty());
}

#[test]
fn test_accumulate_single() {
    let data = vals(&[42]);
    let result = to_ints(
        Accumulate::new(data.into_iter(), |a, b| {
            int(a.as_int().unwrap() + b.as_int().unwrap())
        })
        .collect(),
    );
    assert_eq!(result, vec![42]);
}

#[test]
fn test_accumulate_max() {
    let data = vals(&[3, 1, 4, 1, 5, 9, 2, 6]);
    let result = to_ints(
        Accumulate::new(data.into_iter(), |a, b| {
            int(a.as_int().unwrap().max(b.as_int().unwrap()))
        })
        .collect(),
    );
    assert_eq!(result, vec![3, 3, 4, 4, 5, 9, 9, 9]);
}

#[test]
fn test_accumulate_with_initial_empty() {
    let data: Vec<Value> = vec![];
    let result = to_ints(
        Accumulate::with_initial(
            data.into_iter(),
            |a, b| int(a.as_int().unwrap() + b.as_int().unwrap()),
            int(0),
        )
        .collect(),
    );
    assert_eq!(result, vec![0]); // just the initial
}

// =========================================================================
// partition tests
// =========================================================================

#[test]
fn test_partition_basic() {
    let data = vals(&[1, 2, 3, 4, 5, 6]);
    let (odds, evens) = partition(data.into_iter(), |v| v.as_int().unwrap() % 2 == 0);
    assert_eq!(to_ints(odds), vec![1, 3, 5]); // falses (not even)
    assert_eq!(to_ints(evens), vec![2, 4, 6]); // trues (even)
}

#[test]
fn test_partition_all_true() {
    let data = vals(&[2, 4, 6]);
    let (falses, trues) = partition(data.into_iter(), |v| v.as_int().unwrap() % 2 == 0);
    assert!(falses.is_empty());
    assert_eq!(to_ints(trues), vec![2, 4, 6]);
}

#[test]
fn test_partition_all_false() {
    let data = vals(&[1, 3, 5]);
    let (falses, trues) = partition(data.into_iter(), |v| v.as_int().unwrap() % 2 == 0);
    assert_eq!(to_ints(falses), vec![1, 3, 5]);
    assert!(trues.is_empty());
}

#[test]
fn test_partition_empty() {
    let data: Vec<Value> = vec![];
    let (falses, trues) = partition(data.into_iter(), |_| true);
    assert!(falses.is_empty());
    assert!(trues.is_empty());
}

#[test]
fn test_partition_preserves_order() {
    let data = vals(&[5, 3, 1, 4, 2]);
    let (small, big) = partition(data.into_iter(), |v| v.as_int().unwrap() > 3);
    assert_eq!(to_ints(small), vec![3, 1, 2]);
    assert_eq!(to_ints(big), vec![5, 4]);
}

// =========================================================================
// quantify tests
// =========================================================================

#[test]
fn test_quantify_basic() {
    let data = vals(&[1, 2, 3, 4, 5, 6]);
    let count = quantify(data.into_iter(), |v| v.as_int().unwrap() % 2 == 0);
    assert_eq!(count, 3);
}

#[test]
fn test_quantify_all_true() {
    let data = vals(&[2, 4, 6]);
    let count = quantify(data.into_iter(), |v| v.as_int().unwrap() % 2 == 0);
    assert_eq!(count, 3);
}

#[test]
fn test_quantify_none_true() {
    let data = vals(&[1, 3, 5]);
    let count = quantify(data.into_iter(), |v| v.as_int().unwrap() % 2 == 0);
    assert_eq!(count, 0);
}

#[test]
fn test_quantify_empty() {
    let data: Vec<Value> = vec![];
    let count = quantify(data.into_iter(), |_| true);
    assert_eq!(count, 0);
}

// =========================================================================
// head_tail tests
// =========================================================================

#[test]
fn test_head_tail_basic() {
    let data: Vec<Value> = (0..10).map(|i| int(i)).collect();
    let (head, tail) = head_tail(data.into_iter(), 3);
    assert_eq!(to_ints(head), vec![0, 1, 2]);
    assert_eq!(to_ints(tail), vec![7, 8, 9]);
}

#[test]
fn test_head_tail_short_input() {
    let data = vals(&[1, 2]);
    let (head, tail) = head_tail(data.into_iter(), 5);
    assert_eq!(to_ints(head), vec![1, 2]);
    assert_eq!(to_ints(tail), vec![1, 2]);
}

#[test]
fn test_head_tail_empty() {
    let data: Vec<Value> = vec![];
    let (head, tail) = head_tail(data.into_iter(), 3);
    assert!(head.is_empty());
    assert!(tail.is_empty());
}

#[test]
fn test_head_tail_n_is_zero() {
    let data = vals(&[1, 2, 3]);
    let (head, tail) = head_tail(data.into_iter(), 0);
    assert!(head.is_empty());
    assert!(tail.is_empty());
}

// =========================================================================
// Stress tests
// =========================================================================

#[test]
fn test_unique_everseen_1m_small_alphabet() {
    // 100k elements from alphabet of 50
    let data: Vec<Value> = (0..100_000).map(|i| int(i % 50)).collect();
    let result = to_ints(UniqueEverseen::new(data.into_iter()).collect());
    assert_eq!(result.len(), 50);
}

#[test]
fn test_accumulate_stress() {
    let data: Vec<Value> = (1..=1000).map(|i| int(i)).collect();
    let result = to_ints(
        Accumulate::new(data.into_iter(), |a, b| {
            int(a.as_int().unwrap() + b.as_int().unwrap())
        })
        .collect(),
    );
    assert_eq!(result.len(), 1000);
    // Last element should be sum(1..=1000) = 500500
    assert_eq!(*result.last().unwrap(), 500_500);
}
