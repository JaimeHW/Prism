use super::*;

// =========================================================================
// isinf() Tests
// =========================================================================

#[test]
fn test_isinf_positive() {
    assert!(isinf(f64::INFINITY));
}

#[test]
fn test_isinf_negative() {
    assert!(isinf(f64::NEG_INFINITY));
}

#[test]
fn test_isinf_finite() {
    assert!(!isinf(0.0));
    assert!(!isinf(1.0));
    assert!(!isinf(-1e308));
    assert!(!isinf(1e308));
}

#[test]
fn test_isinf_nan() {
    assert!(!isinf(f64::NAN));
}

// =========================================================================
// isnan() Tests
// =========================================================================

#[test]
fn test_isnan_nan() {
    assert!(isnan(f64::NAN));
}

#[test]
fn test_isnan_finite() {
    assert!(!isnan(0.0));
    assert!(!isnan(1.0));
    assert!(!isnan(-1.0));
}

#[test]
fn test_isnan_infinity() {
    assert!(!isnan(f64::INFINITY));
    assert!(!isnan(f64::NEG_INFINITY));
}

// =========================================================================
// isfinite() Tests
// =========================================================================

#[test]
fn test_isfinite_regular() {
    assert!(isfinite(0.0));
    assert!(isfinite(1.0));
    assert!(isfinite(-1e308));
    assert!(isfinite(1e308));
}

#[test]
fn test_isfinite_infinity() {
    assert!(!isfinite(f64::INFINITY));
    assert!(!isfinite(f64::NEG_INFINITY));
}

#[test]
fn test_isfinite_nan() {
    assert!(!isfinite(f64::NAN));
}

// =========================================================================
// isnormal() Tests
// =========================================================================

#[test]
fn test_isnormal_regular() {
    assert!(isnormal(1.0));
    assert!(isnormal(-1.0));
    assert!(isnormal(1e-307));
}

#[test]
fn test_isnormal_zero() {
    assert!(!isnormal(0.0));
    assert!(!isnormal(-0.0));
}

#[test]
fn test_isnormal_subnormal() {
    // Smallest subnormal is ~5e-324
    assert!(!isnormal(5e-324));
}

#[test]
fn test_isnormal_special() {
    assert!(!isnormal(f64::INFINITY));
    assert!(!isnormal(f64::NAN));
}

// =========================================================================
// isclose() Tests
// =========================================================================

#[test]
fn test_isclose_equal() {
    assert!(isclose(1.0, 1.0, 1e-9, 0.0));
    assert!(isclose(0.0, 0.0, 1e-9, 0.0));
}

#[test]
fn test_isclose_relative() {
    assert!(isclose(1e10, 1.00001e10, 1e-4, 0.0));
    assert!(!isclose(1e10, 1.001e10, 1e-4, 0.0));
}

#[test]
fn test_isclose_absolute() {
    assert!(isclose(1e-10, 2e-10, 0.0, 1e-9));
    assert!(!isclose(1e-10, 2e-10, 0.0, 1e-11));
}

#[test]
fn test_isclose_zero() {
    // Near zero, need absolute tolerance
    assert!(!isclose(0.0, 1e-10, 1e-9, 0.0));
    assert!(isclose(0.0, 1e-10, 0.0, 1e-9));
}

#[test]
fn test_isclose_nan() {
    assert!(!isclose(f64::NAN, f64::NAN, 1e-9, 0.0));
    assert!(!isclose(1.0, f64::NAN, 1e-9, 0.0));
}

#[test]
fn test_isclose_infinity() {
    assert!(isclose(f64::INFINITY, f64::INFINITY, 1e-9, 0.0));
    assert!(!isclose(f64::INFINITY, 1e308, 1e-9, 0.0));
}

// =========================================================================
// nextafter() Tests
// =========================================================================

#[test]
fn test_nextafter_up() {
    let x = 1.0;
    let next = nextafter(x, f64::INFINITY);
    assert!(next > x);
    assert!(next - x < 1e-15); // Very close
}

#[test]
fn test_nextafter_down() {
    let x = 1.0;
    let prev = nextafter(x, 0.0);
    assert!(prev < x);
    assert!(x - prev < 1e-15);
}

#[test]
fn test_nextafter_equal() {
    assert_eq!(nextafter(1.0, 1.0), 1.0);
}

#[test]
fn test_nextafter_zero_up() {
    let next = nextafter(0.0, 1.0);
    assert!(next > 0.0);
    assert!(next < 1e-300); // Very small positive
}

#[test]
fn test_nextafter_nan() {
    assert!(nextafter(f64::NAN, 1.0).is_nan());
    assert!(nextafter(1.0, f64::NAN).is_nan());
}

// =========================================================================
// ulp() Tests
// =========================================================================

#[test]
fn test_ulp_one() {
    let u = ulp(1.0);
    assert!(u > 0.0);
    assert!(u < 1e-15);
}

#[test]
fn test_ulp_large() {
    let u_small = ulp(1.0);
    let u_large = ulp(1e100);
    assert!(u_large > u_small); // ULP grows with magnitude
}

#[test]
fn test_ulp_small() {
    let u_one = ulp(1.0);
    let u_small = ulp(1e-10);
    assert!(u_small < u_one); // ULP shrinks with magnitude
}

#[test]
fn test_ulp_special() {
    assert!(ulp(f64::NAN).is_nan());
    assert!(ulp(f64::INFINITY).is_infinite());
}

// =========================================================================
// Identity Tests
// =========================================================================

#[test]
fn test_finite_classification() {
    // A number is exactly one of: normal, subnormal, zero, infinite, or NaN
    let values = [0.0, 1.0, 5e-324, f64::INFINITY, f64::NAN];
    for x in values {
        let classifications = [
            x.is_normal(),
            x.is_subnormal(),
            x == 0.0,
            x.is_infinite(),
            x.is_nan(),
        ];
        let count: usize = classifications.iter().filter(|&&b| b).count();
        assert_eq!(count, 1, "Expected exactly one classification for {}", x);
    }
}

#[test]
fn test_isfinite_equals_not_inf_not_nan() {
    for x in [0.0, 1.0, -1.0, 1e308, f64::INFINITY, f64::NAN] {
        assert_eq!(isfinite(x), !isinf(x) && !isnan(x));
    }
}
