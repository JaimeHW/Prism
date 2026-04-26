use super::*;

// =========================================================================
// Pattern Detection Tests
// =========================================================================

#[test]
fn test_power_of_two_plus_one() {
    assert_eq!(is_power_of_two_plus_one(3), Some(1)); // 2^1 + 1
    assert_eq!(is_power_of_two_plus_one(5), Some(2)); // 2^2 + 1
    assert_eq!(is_power_of_two_plus_one(9), Some(3)); // 2^3 + 1
    assert_eq!(is_power_of_two_plus_one(17), Some(4)); // 2^4 + 1
    assert_eq!(is_power_of_two_plus_one(7), None);
    assert_eq!(is_power_of_two_plus_one(10), None);
}

#[test]
fn test_power_of_two_minus_one() {
    assert_eq!(is_power_of_two_minus_one(1), Some(1)); // 2^1 - 1
    assert_eq!(is_power_of_two_minus_one(3), Some(2)); // 2^2 - 1
    assert_eq!(is_power_of_two_minus_one(7), Some(3)); // 2^3 - 1
    assert_eq!(is_power_of_two_minus_one(15), Some(4)); // 2^4 - 1
    assert_eq!(is_power_of_two_minus_one(31), Some(5)); // 2^5 - 1
    assert_eq!(is_power_of_two_minus_one(10), None);
}

#[test]
fn test_sum_of_two_powers() {
    assert_eq!(is_sum_of_two_powers(3), Some((1, 0))); // 2^1 + 2^0
    assert_eq!(is_sum_of_two_powers(5), Some((2, 0))); // 2^2 + 2^0
    assert_eq!(is_sum_of_two_powers(6), Some((2, 1))); // 2^2 + 2^1
    assert_eq!(is_sum_of_two_powers(10), Some((3, 1))); // 2^3 + 2^1
    assert_eq!(is_sum_of_two_powers(7), None); // 3 bits set
    assert_eq!(is_sum_of_two_powers(8), None); // 1 bit set
}

#[test]
fn test_diff_of_two_powers() {
    assert_eq!(is_diff_of_two_powers(2), Some((2, 1))); // 4 - 2
    assert_eq!(is_diff_of_two_powers(6), Some((3, 1))); // 8 - 2
    assert_eq!(is_diff_of_two_powers(12), Some((4, 2))); // 16 - 4
    assert_eq!(is_diff_of_two_powers(14), Some((4, 1))); // 16 - 2
    assert_eq!(is_diff_of_two_powers(5), None);
}

// =========================================================================
// NAF Conversion Tests
// =========================================================================

#[test]
fn test_naf_simple() {
    // 3 = 4 - 1 = 2^2 - 2^0, NAF = [-1, 0, 1]
    let naf = to_naf(3);
    assert_eq!(naf, vec![-1, 0, 1]);

    // 7 = 8 - 1, NAF = [-1, 0, 0, 1]
    let naf = to_naf(7);
    assert_eq!(naf, vec![-1, 0, 0, 1]);
}

#[test]
fn test_naf_no_adjacent_nonzero() {
    for n in 1u64..1000 {
        let naf = to_naf(n);

        // Verify no adjacent non-zero digits
        for i in 0..naf.len().saturating_sub(1) {
            if naf[i] != 0 && naf[i + 1] != 0 {
                panic!("NAF has adjacent non-zero digits for n={}: {:?}", n, naf);
            }
        }

        // Verify it computes the right value
        let mut computed = 0i64;
        for (i, &d) in naf.iter().enumerate() {
            computed += (d as i64) << i;
        }
        assert_eq!(computed, n as i64, "NAF mismatch for n={}", n);
    }
}

// =========================================================================
// Decomposition Tests
// =========================================================================

#[test]
fn test_decompose_special_cases() {
    let config = DecompConfig::default();

    // x * 0 - not decomposed here
    assert!(decompose_multiply(0, &config).is_none());

    // x * 1 - identity
    assert!(decompose_multiply(1, &config).is_none());

    // x * -1 - negate
    let decomp = decompose_multiply(-1, &config).unwrap();
    assert_eq!(decomp.ops, vec![DecompOp::Negate]);
}

#[test]
fn test_decompose_powers_of_two() {
    let config = DecompConfig::default();

    for shift in 1u8..62 {
        let n = 1i64 << shift;
        let decomp = decompose_multiply(n, &config).unwrap();
        assert_eq!(decomp.ops, vec![DecompOp::Shift(shift)]);
        assert!(!decomp.negate_result);

        // Verify correctness (use wrapping to avoid overflow)
        for x in [1i64, 7, 100, -50] {
            assert_eq!(decomp.apply(x), x.wrapping_mul(n));
        }
    }
}

#[test]
fn test_decompose_small_constants() {
    let config = DecompConfig::default();

    // Test 2-20
    for n in 2i64..=20 {
        let decomp_opt = decompose_multiply(n, &config);

        // We expect most small constants to decompose
        if let Some(decomp) = decomp_opt {
            // Verify correctness
            for x in [0i64, 1, -1, 7, -7, 100, -100, 12345] {
                let expected = x.wrapping_mul(n);
                let actual = decomp.apply(x);
                assert_eq!(
                    actual, expected,
                    "Decomposition failed for x={}, n={}: got {}, expected {}",
                    x, n, actual, expected
                );
            }
        }
    }
}

#[test]
fn test_decompose_negative_constants() {
    let config = DecompConfig::default();

    for n in [-2i64, -3, -5, -7, -10, -15] {
        let decomp = decompose_multiply(n, &config).unwrap();
        assert!(decomp.negate_result);

        for x in [1i64, 7, -7, 100] {
            let expected = x.wrapping_mul(n);
            let actual = decomp.apply(x);
            assert_eq!(actual, expected, "Failed for x={}, n={}", x, n);
        }
    }
}

#[test]
fn test_decompose_specific_patterns() {
    let config = DecompConfig::default();

    // x * 3 = x + (x << 1)
    let d3 = decompose_multiply(3, &config).unwrap();
    assert!(d3.cost() <= 2);
    assert_eq!(d3.apply(10), 30);

    // x * 5 = x + (x << 2)
    let d5 = decompose_multiply(5, &config).unwrap();
    assert!(d5.cost() <= 2);
    assert_eq!(d5.apply(10), 50);

    // x * 7 = (x << 3) - x
    let d7 = decompose_multiply(7, &config).unwrap();
    assert!(d7.cost() <= 2);
    assert_eq!(d7.apply(10), 70);

    // x * 9 = x + (x << 3)
    let d9 = decompose_multiply(9, &config).unwrap();
    assert!(d9.cost() <= 2);
    assert_eq!(d9.apply(10), 90);

    // x * 15 = (x << 4) - x
    let d15 = decompose_multiply(15, &config).unwrap();
    assert!(d15.cost() <= 2);
    assert_eq!(d15.apply(10), 150);
}

#[test]
fn test_decompose_cost() {
    let config = DecompConfig::default();

    // Power of 2 should be cheapest (1 op)
    let d2 = decompose_multiply(2, &config).unwrap();
    assert_eq!(d2.cost(), 1);

    // 2^k + 1 should be 2 ops
    let d3 = decompose_multiply(3, &config).unwrap();
    assert!(d3.cost() <= 2);

    // 2^k - 1 should be 2 ops
    let d7 = decompose_multiply(7, &config).unwrap();
    assert!(d7.cost() <= 2);
}

#[test]
fn test_decompose_profitability() {
    // With low multiply cost, fewer decompositions are profitable
    let low_cost = DecompConfig {
        max_ops: 4,
        multiply_cost: 2,
        allow_negate: true,
    };

    // Power of 2 still profitable
    assert!(decompose_multiply(4, &low_cost).is_some());

    // More complex might not be
    let d = decompose_multiply(10, &low_cost);
    if let Some(decomp) = d {
        assert!(decomp.is_profitable(2));
    }
}

#[test]
fn test_decompose_comprehensive() {
    let config = DecompConfig::default();

    // Test many constants
    for n in 2i64..500 {
        if let Some(decomp) = decompose_multiply(n, &config) {
            // Verify correctness on multiple inputs
            let test_values = [0i64, 1, -1, 10, -10, 127, -128, 1000, -1000];

            for x in test_values {
                let expected = x.wrapping_mul(n);
                let actual = decomp.apply(x);
                assert_eq!(
                    actual, expected,
                    "Decomposition failed for x={}, n={}: got {}, expected {}. Decomp: {:?}",
                    x, n, actual, expected, decomp
                );
            }
        }
    }
}

#[test]
fn test_decompose_config_low_latency() {
    let config = DecompConfig::low_latency();

    // Should only allow very simple decompositions
    assert_eq!(config.max_ops, 2);

    // Power of 2 works
    assert!(decompose_multiply(4, &config).is_some());

    // Simple patterns work
    assert!(decompose_multiply(3, &config).is_some());
}

#[test]
fn test_decompose_config_aggressive() {
    let config = DecompConfig::aggressive();

    // Should allow more complex decompositions
    assert!(config.max_ops >= 6);

    // More constants should decompose
    let count = (2i64..100)
        .filter(|&n| decompose_multiply(n, &config).is_some())
        .count();

    // Most should decompose with aggressive settings
    assert!(count > 50);
}

#[test]
fn test_decomp_op_coverage() {
    let config = DecompConfig::aggressive();

    // Find a decomposition that uses each op type
    let mut found_add_shift = false;
    let mut found_sub_shift = false;

    for n in 2i64..1000 {
        if let Some(decomp) = decompose_multiply(n, &config) {
            for op in &decomp.ops {
                match op {
                    DecompOp::AddShift(_) => found_add_shift = true,
                    DecompOp::SubShift(_) => found_sub_shift = true,
                    _ => {}
                }
            }
        }
    }

    assert!(found_add_shift, "No AddShift operations found");
    assert!(found_sub_shift, "No SubShift operations found");
}
