use super::*;

fn int(i: i64) -> Value {
    Value::int_unchecked(i)
}

fn to_int_vecs(v: Vec<Vec<Value>>) -> Vec<Vec<i64>> {
    v.into_iter()
        .map(|row| row.into_iter().filter_map(|v| v.as_int()).collect())
        .collect()
}

// =========================================================================
// Product tests
// =========================================================================

#[test]
fn test_product_two_pools() {
    let p = Product::new(vec![vec![int(1), int(2)], vec![int(3), int(4)]]);
    let result = to_int_vecs(p.collect());
    assert_eq!(result, vec![vec![1, 3], vec![1, 4], vec![2, 3], vec![2, 4]]);
}

#[test]
fn test_product_three_pools() {
    let p = Product::new(vec![
        vec![int(0), int(1)],
        vec![int(0), int(1)],
        vec![int(0), int(1)],
    ]);
    let result = to_int_vecs(p.collect());
    assert_eq!(result.len(), 8); // 2^3
    // First should be [0,0,0], last [1,1,1]
    assert_eq!(result[0], vec![0, 0, 0]);
    assert_eq!(result[7], vec![1, 1, 1]);
}

#[test]
fn test_product_single_pool() {
    let p = Product::new(vec![vec![int(1), int(2), int(3)]]);
    let result = to_int_vecs(p.collect());
    assert_eq!(result, vec![vec![1], vec![2], vec![3]]);
}

#[test]
fn test_product_empty_pool() {
    let p = Product::new(vec![vec![int(1)], vec![]]);
    let result: Vec<Vec<Value>> = p.collect();
    assert!(result.is_empty());
}

#[test]
fn test_product_no_pools() {
    let p = Product::new(vec![]);
    let result: Vec<Vec<Value>> = p.collect();
    assert_eq!(result.len(), 1); // one empty tuple
    assert!(result[0].is_empty());
}

#[test]
fn test_product_with_repeat() {
    let p = Product::with_repeat(vec![int(0), int(1)], 2);
    let result = to_int_vecs(p.collect());
    assert_eq!(result, vec![vec![0, 0], vec![0, 1], vec![1, 0], vec![1, 1]]);
}

#[test]
fn test_product_repeat_3() {
    let p = Product::with_repeat(vec![int(0), int(1)], 3);
    let result = to_int_vecs(p.collect());
    assert_eq!(result.len(), 8);
}

#[test]
fn test_product_total_size() {
    let p = Product::new(vec![vec![int(1), int(2)], vec![int(3), int(4), int(5)]]);
    assert_eq!(p.total_size(), 6); // 2 * 3
}

#[test]
fn test_product_asymmetric_pools() {
    let p = Product::new(vec![
        vec![int(1)],
        vec![int(2), int(3)],
        vec![int(4), int(5), int(6)],
    ]);
    let result = to_int_vecs(p.collect());
    assert_eq!(result.len(), 6); // 1 * 2 * 3
}

#[test]
fn test_product_fused() {
    let mut p = Product::new(vec![vec![int(1)], vec![int(2)]]);
    assert!(p.next().is_some());
    assert!(p.next().is_none());
    assert!(p.next().is_none());
}

#[test]
fn test_product_lexicographic_order() {
    // Verify output matches CPython's order exactly
    let p = Product::new(vec![
        vec![int(1), int(2)],
        vec![int(3), int(4)],
        vec![int(5), int(6)],
    ]);
    let result = to_int_vecs(p.collect());
    assert_eq!(result[0], vec![1, 3, 5]);
    assert_eq!(result[1], vec![1, 3, 6]);
    assert_eq!(result[2], vec![1, 4, 5]);
    assert_eq!(result[3], vec![1, 4, 6]);
    assert_eq!(result[4], vec![2, 3, 5]);
    assert_eq!(result[5], vec![2, 3, 6]);
    assert_eq!(result[6], vec![2, 4, 5]);
    assert_eq!(result[7], vec![2, 4, 6]);
}

// =========================================================================
// Permutations tests
// =========================================================================

#[test]
fn test_permutations_full() {
    let p = Permutations::full(vec![int(1), int(2), int(3)]);
    let result = to_int_vecs(p.collect());
    assert_eq!(result.len(), 6); // 3!
    // Verify all are distinct
    let mut sorted = result.clone();
    sorted.sort();
    sorted.dedup();
    assert_eq!(sorted.len(), 6);
}

#[test]
fn test_permutations_r2_from_3() {
    let p = Permutations::new(vec![int(1), int(2), int(3)], 2);
    let result = to_int_vecs(p.collect());
    assert_eq!(result.len(), 6); // 3!/(3-2)! = 6
}

#[test]
fn test_permutations_r1() {
    let p = Permutations::new(vec![int(1), int(2), int(3)], 1);
    let result = to_int_vecs(p.collect());
    assert_eq!(result, vec![vec![1], vec![2], vec![3]]);
}

#[test]
fn test_permutations_r0() {
    let p = Permutations::new(vec![int(1), int(2)], 0);
    let result: Vec<Vec<Value>> = p.collect();
    // r=0 means 0 cycles, first iteration returns empty tuple, then done
    assert!(!result.is_empty());
}

#[test]
fn test_permutations_r_exceeds_n() {
    let p = Permutations::new(vec![int(1), int(2)], 5);
    let result: Vec<Vec<Value>> = p.collect();
    assert!(result.is_empty());
}

#[test]
fn test_permutations_single_element() {
    let p = Permutations::full(vec![int(42)]);
    let result = to_int_vecs(p.collect());
    assert_eq!(result, vec![vec![42]]);
}

#[test]
fn test_permutations_two_elements() {
    let p = Permutations::full(vec![int(1), int(2)]);
    let result = to_int_vecs(p.collect());
    assert_eq!(result.len(), 2);
    assert!(result.contains(&vec![1, 2]));
    assert!(result.contains(&vec![2, 1]));
}

#[test]
fn test_permutations_four_elements() {
    let p = Permutations::full(vec![int(1), int(2), int(3), int(4)]);
    let result = to_int_vecs(p.collect());
    assert_eq!(result.len(), 24); // 4!
}

#[test]
fn test_permutations_no_duplicates() {
    let p = Permutations::full(vec![int(1), int(2), int(3)]);
    let result = to_int_vecs(p.collect());
    let mut sorted = result.clone();
    sorted.sort();
    sorted.dedup();
    assert_eq!(sorted.len(), result.len());
}

#[test]
fn test_permutations_fused() {
    let mut p = Permutations::full(vec![int(1)]);
    assert!(p.next().is_some());
    assert!(p.next().is_none());
    assert!(p.next().is_none());
}

// =========================================================================
// Combinations tests
// =========================================================================

#[test]
fn test_combinations_basic() {
    let c = Combinations::new(vec![int(1), int(2), int(3), int(4)], 2);
    let result = to_int_vecs(c.collect());
    assert_eq!(
        result,
        vec![
            vec![1, 2],
            vec![1, 3],
            vec![1, 4],
            vec![2, 3],
            vec![2, 4],
            vec![3, 4],
        ]
    );
}

#[test]
fn test_combinations_r_equals_n() {
    let c = Combinations::new(vec![int(1), int(2), int(3)], 3);
    let result = to_int_vecs(c.collect());
    assert_eq!(result, vec![vec![1, 2, 3]]);
}

#[test]
fn test_combinations_r1() {
    let c = Combinations::new(vec![int(10), int(20), int(30)], 1);
    let result = to_int_vecs(c.collect());
    assert_eq!(result, vec![vec![10], vec![20], vec![30]]);
}

#[test]
fn test_combinations_r0() {
    let c = Combinations::new(vec![int(1), int(2)], 0);
    let result: Vec<Vec<Value>> = c.collect();
    assert_eq!(result.len(), 1); // one empty combination
    assert!(result[0].is_empty());
}

#[test]
fn test_combinations_r_exceeds_n() {
    let c = Combinations::new(vec![int(1), int(2)], 5);
    let result: Vec<Vec<Value>> = c.collect();
    assert!(result.is_empty());
}

#[test]
fn test_combinations_count() {
    // C(5, 3) = 10
    let c = Combinations::new(vec![int(1), int(2), int(3), int(4), int(5)], 3);
    let result: Vec<Vec<Value>> = c.collect();
    assert_eq!(result.len(), 10);
}

#[test]
fn test_combinations_c6_2() {
    // C(6, 2) = 15
    let pool: Vec<Value> = (1..=6).map(|i| int(i)).collect();
    let c = Combinations::new(pool, 2);
    let result: Vec<Vec<Value>> = c.collect();
    assert_eq!(result.len(), 15);
}

#[test]
fn test_combinations_lexicographic_order() {
    let c = Combinations::new(vec![int(1), int(2), int(3), int(4)], 2);
    let result = to_int_vecs(c.collect());
    // Should be in ascending order
    for window in result.windows(2) {
        assert!(window[0] < window[1], "Not in lexicographic order");
    }
}

#[test]
fn test_combinations_no_duplicates() {
    let c = Combinations::new(vec![int(1), int(2), int(3), int(4)], 2);
    let result = to_int_vecs(c.collect());
    let mut sorted = result.clone();
    sorted.sort();
    sorted.dedup();
    assert_eq!(sorted.len(), result.len());
}

#[test]
fn test_combinations_fused() {
    let mut c = Combinations::new(vec![int(1)], 1);
    assert!(c.next().is_some());
    assert!(c.next().is_none());
    assert!(c.next().is_none());
}

// =========================================================================
// CombinationsWithReplacement tests
// =========================================================================

#[test]
fn test_cwr_basic() {
    let c = CombinationsWithReplacement::new(vec![int(1), int(2), int(3)], 2);
    let result = to_int_vecs(c.collect());
    assert_eq!(
        result,
        vec![
            vec![1, 1],
            vec![1, 2],
            vec![1, 3],
            vec![2, 2],
            vec![2, 3],
            vec![3, 3],
        ]
    );
}

#[test]
fn test_cwr_r1() {
    let c = CombinationsWithReplacement::new(vec![int(1), int(2)], 1);
    let result = to_int_vecs(c.collect());
    assert_eq!(result, vec![vec![1], vec![2]]);
}

#[test]
fn test_cwr_r0() {
    let c = CombinationsWithReplacement::new(vec![int(1), int(2)], 0);
    let result: Vec<Vec<Value>> = c.collect();
    assert_eq!(result.len(), 1);
    assert!(result[0].is_empty());
}

#[test]
fn test_cwr_empty_pool() {
    let c = CombinationsWithReplacement::new(vec![], 2);
    let result: Vec<Vec<Value>> = c.collect();
    assert!(result.is_empty());
}

#[test]
fn test_cwr_single_element() {
    let c = CombinationsWithReplacement::new(vec![int(1)], 3);
    let result = to_int_vecs(c.collect());
    assert_eq!(result, vec![vec![1, 1, 1]]);
}

#[test]
fn test_cwr_count() {
    // C(n+r-1, r) = C(3+2-1, 2) = C(4,2) = 6
    let c = CombinationsWithReplacement::new(vec![int(1), int(2), int(3)], 2);
    let result: Vec<Vec<Value>> = c.collect();
    assert_eq!(result.len(), 6);
}

#[test]
fn test_cwr_index_non_decreasing() {
    let c = CombinationsWithReplacement::new(vec![int(1), int(2), int(3)], 3);
    let result = to_int_vecs(c.collect());
    for row in &result {
        for window in row.windows(2) {
            assert!(window[0] <= window[1], "Indices not non-decreasing");
        }
    }
}

#[test]
fn test_cwr_fused() {
    let mut c = CombinationsWithReplacement::new(vec![int(1)], 1);
    assert!(c.next().is_some());
    assert!(c.next().is_none());
    assert!(c.next().is_none());
}

#[test]
fn test_cwr_larger() {
    // C(4+3-1, 3) = C(6, 3) = 20
    let pool: Vec<Value> = (1..=4).map(|i| int(i)).collect();
    let c = CombinationsWithReplacement::new(pool, 3);
    let result: Vec<Vec<Value>> = c.collect();
    assert_eq!(result.len(), 20);
}

// =========================================================================
// Stress tests
// =========================================================================

#[test]
fn test_product_stress() {
    // product(range(10), range(10)) = 100 elements
    let pool: Vec<Value> = (0..10).map(|i| int(i)).collect();
    let p = Product::new(vec![pool.clone(), pool]);
    let result: Vec<Vec<Value>> = p.collect();
    assert_eq!(result.len(), 100);
}

#[test]
fn test_permutations_stress_5() {
    // 5! = 120
    let pool: Vec<Value> = (0..5).map(|i| int(i)).collect();
    let p = Permutations::full(pool);
    let result: Vec<Vec<Value>> = p.collect();
    assert_eq!(result.len(), 120);
}

#[test]
fn test_combinations_stress_c10_3() {
    // C(10, 3) = 120
    let pool: Vec<Value> = (0..10).map(|i| int(i)).collect();
    let c = Combinations::new(pool, 3);
    let result: Vec<Vec<Value>> = c.collect();
    assert_eq!(result.len(), 120);
}

#[test]
fn test_cwr_stress() {
    // C(5+3-1, 3) = C(7, 3) = 35
    let pool: Vec<Value> = (0..5).map(|i| int(i)).collect();
    let c = CombinationsWithReplacement::new(pool, 3);
    let result: Vec<Vec<Value>> = c.collect();
    assert_eq!(result.len(), 35);
}
