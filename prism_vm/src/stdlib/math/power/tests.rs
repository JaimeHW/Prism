use super::*;

const EPSILON: f64 = 1e-10;

// =========================================================================
// pow() Tests
// =========================================================================

#[test]
fn test_pow_zero_exp() {
    assert!((pow(5.0, 0.0).unwrap() - 1.0).abs() < EPSILON);
    assert!((pow(0.0, 0.0).unwrap() - 1.0).abs() < EPSILON); // Python convention
}

#[test]
fn test_pow_positive_int() {
    assert!((pow(2.0, 3.0).unwrap() - 8.0).abs() < EPSILON);
    assert!((pow(3.0, 4.0).unwrap() - 81.0).abs() < EPSILON);
}

#[test]
fn test_pow_negative_int() {
    assert!((pow(2.0, -1.0).unwrap() - 0.5).abs() < EPSILON);
    assert!((pow(2.0, -3.0).unwrap() - 0.125).abs() < EPSILON);
}

#[test]
fn test_pow_fractional() {
    assert!((pow(4.0, 0.5).unwrap() - 2.0).abs() < EPSILON);
    assert!((pow(8.0, 1.0 / 3.0).unwrap() - 2.0).abs() < EPSILON);
}

#[test]
fn test_pow_zero_base_positive_exp() {
    assert!((pow(0.0, 5.0).unwrap()).abs() < EPSILON);
}

#[test]
fn test_pow_zero_base_negative_exp() {
    assert!(pow(0.0, -1.0).is_err());
}

#[test]
fn test_pow_negative_base_fractional_exp() {
    assert!(pow(-2.0, 0.5).is_err());
}

#[test]
fn test_pow_negative_base_integer_exp() {
    assert!((pow(-2.0, 3.0).unwrap() - (-8.0)).abs() < EPSILON);
    assert!((pow(-2.0, 4.0).unwrap() - 16.0).abs() < EPSILON);
}

#[test]
fn test_pow_large_exponent() {
    assert!((pow(2.0, 10.0).unwrap() - 1024.0).abs() < EPSILON);
    assert!((pow(2.0, 20.0).unwrap() - 1048576.0).abs() < EPSILON);
}

// =========================================================================
// pow_int() Tests (internal)
// =========================================================================

#[test]
fn test_pow_int_basic() {
    assert!((pow_int(2.0, 0) - 1.0).abs() < EPSILON);
    assert!((pow_int(2.0, 1) - 2.0).abs() < EPSILON);
    assert!((pow_int(2.0, 10) - 1024.0).abs() < EPSILON);
}

#[test]
fn test_pow_int_negative_exp() {
    assert!((pow_int(2.0, -1) - 0.5).abs() < EPSILON);
    assert!((pow_int(2.0, -2) - 0.25).abs() < EPSILON);
}

#[test]
fn test_pow_int_large() {
    // 2^30 = 1073741824
    assert!((pow_int(2.0, 30) - 1073741824.0).abs() < EPSILON);
}

// =========================================================================
// sqrt() Tests
// =========================================================================

#[test]
fn test_sqrt_perfect_squares() {
    assert!((sqrt(0.0).unwrap()).abs() < EPSILON);
    assert!((sqrt(1.0).unwrap() - 1.0).abs() < EPSILON);
    assert!((sqrt(4.0).unwrap() - 2.0).abs() < EPSILON);
    assert!((sqrt(9.0).unwrap() - 3.0).abs() < EPSILON);
    assert!((sqrt(16.0).unwrap() - 4.0).abs() < EPSILON);
}

#[test]
fn test_sqrt_non_perfect() {
    assert!((sqrt(2.0).unwrap() - std::f64::consts::SQRT_2).abs() < EPSILON);
}

#[test]
fn test_sqrt_domain_error() {
    assert!(sqrt(-1.0).is_err());
    assert!(sqrt(-0.001).is_err());
}

#[test]
fn test_sqrt_special() {
    assert!(sqrt(f64::INFINITY).unwrap().is_infinite());
}

// =========================================================================
// isqrt() Tests
// =========================================================================

#[test]
fn test_isqrt_zero() {
    assert_eq!(isqrt(0).unwrap(), 0);
}

#[test]
fn test_isqrt_one() {
    assert_eq!(isqrt(1).unwrap(), 1);
}

#[test]
fn test_isqrt_perfect_squares() {
    assert_eq!(isqrt(4).unwrap(), 2);
    assert_eq!(isqrt(9).unwrap(), 3);
    assert_eq!(isqrt(16).unwrap(), 4);
    assert_eq!(isqrt(100).unwrap(), 10);
    assert_eq!(isqrt(10000).unwrap(), 100);
}

#[test]
fn test_isqrt_non_perfect() {
    assert_eq!(isqrt(2).unwrap(), 1);
    assert_eq!(isqrt(3).unwrap(), 1);
    assert_eq!(isqrt(5).unwrap(), 2);
    assert_eq!(isqrt(8).unwrap(), 2);
    assert_eq!(isqrt(15).unwrap(), 3);
    assert_eq!(isqrt(17).unwrap(), 4);
}

#[test]
fn test_isqrt_large() {
    assert_eq!(isqrt(1000000).unwrap(), 1000);
    assert_eq!(isqrt(999999).unwrap(), 999);
    assert_eq!(isqrt(1000001).unwrap(), 1000);
}

#[test]
fn test_isqrt_very_large() {
    // Near i64::MAX
    let n: i64 = 9_000_000_000_000_000_000;
    let result = isqrt(n).unwrap();
    assert!(result * result <= n);
    assert!((result + 1) * (result + 1) > n);
}

#[test]
fn test_isqrt_negative() {
    assert!(isqrt(-1).is_err());
}

// =========================================================================
// hypot() Tests
// =========================================================================

#[test]
fn test_hypot_pythagorean_triples() {
    assert!((hypot(3.0, 4.0) - 5.0).abs() < EPSILON);
    assert!((hypot(5.0, 12.0) - 13.0).abs() < EPSILON);
    assert!((hypot(8.0, 15.0) - 17.0).abs() < EPSILON);
}

#[test]
fn test_hypot_zero() {
    assert!((hypot(0.0, 0.0)).abs() < EPSILON);
    assert!((hypot(5.0, 0.0) - 5.0).abs() < EPSILON);
    assert!((hypot(0.0, 5.0) - 5.0).abs() < EPSILON);
}

#[test]
fn test_hypot_negative() {
    assert!((hypot(-3.0, 4.0) - 5.0).abs() < EPSILON);
    assert!((hypot(3.0, -4.0) - 5.0).abs() < EPSILON);
    assert!((hypot(-3.0, -4.0) - 5.0).abs() < EPSILON);
}

#[test]
fn test_hypot_infinity() {
    assert!(hypot(f64::INFINITY, 1.0).is_infinite());
    assert!(hypot(1.0, f64::INFINITY).is_infinite());
}

// =========================================================================
// cbrt() Tests
// =========================================================================

#[test]
fn test_cbrt_positive() {
    assert!((cbrt(8.0) - 2.0).abs() < EPSILON);
    assert!((cbrt(27.0) - 3.0).abs() < EPSILON);
    assert!((cbrt(64.0) - 4.0).abs() < EPSILON);
}

#[test]
fn test_cbrt_negative() {
    assert!((cbrt(-8.0) - (-2.0)).abs() < EPSILON);
    assert!((cbrt(-27.0) - (-3.0)).abs() < EPSILON);
}

#[test]
fn test_cbrt_zero() {
    assert!((cbrt(0.0)).abs() < EPSILON);
}

#[test]
fn test_cbrt_one() {
    assert!((cbrt(1.0) - 1.0).abs() < EPSILON);
}

// =========================================================================
// Identity Tests
// =========================================================================

#[test]
fn test_sqrt_pow_identity() {
    for x in [1.0, 2.0, 4.0, 9.0, 16.0, 100.0] {
        let s = sqrt(x).unwrap();
        let p = pow(x, 0.5).unwrap();
        assert!(
            (s - p).abs() < EPSILON,
            "sqrt(x) ≠ pow(x, 0.5) for x = {}",
            x
        );
    }
}

#[test]
fn test_pow_roundtrip() {
    for x in [2.0, 3.0, 5.0, 10.0] {
        for n in [2, 3, 5, 10] {
            let p = pow(x, n as f64).unwrap();
            let root = pow(p, 1.0 / n as f64).unwrap();
            assert!(
                (root - x).abs() < EPSILON,
                "pow(pow(x, n), 1/n) ≠ x for x = {}, n = {}",
                x,
                n
            );
        }
    }
}
