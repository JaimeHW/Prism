use super::*;

// =========================================================================
// ceil() Tests
// =========================================================================

#[test]
fn test_ceil_positive() {
    assert_eq!(ceil(2.1), 3.0);
    assert_eq!(ceil(2.9), 3.0);
    assert_eq!(ceil(2.0), 2.0);
}

#[test]
fn test_ceil_negative() {
    assert_eq!(ceil(-2.1), -2.0);
    assert_eq!(ceil(-2.9), -2.0);
    assert_eq!(ceil(-2.0), -2.0);
}

#[test]
fn test_ceil_zero() {
    assert_eq!(ceil(0.0), 0.0);
    assert_eq!(ceil(-0.0), -0.0);
}

#[test]
fn test_ceil_special() {
    assert!(ceil(f64::INFINITY).is_infinite());
    assert!(ceil(f64::NAN).is_nan());
}

// =========================================================================
// floor() Tests
// =========================================================================

#[test]
fn test_floor_positive() {
    assert_eq!(floor(2.1), 2.0);
    assert_eq!(floor(2.9), 2.0);
    assert_eq!(floor(2.0), 2.0);
}

#[test]
fn test_floor_negative() {
    assert_eq!(floor(-2.1), -3.0);
    assert_eq!(floor(-2.9), -3.0);
    assert_eq!(floor(-2.0), -2.0);
}

#[test]
fn test_floor_special() {
    assert!(floor(f64::INFINITY).is_infinite());
    assert!(floor(f64::NAN).is_nan());
}

// =========================================================================
// trunc() Tests
// =========================================================================

#[test]
fn test_trunc_positive() {
    assert_eq!(trunc(2.1), 2.0);
    assert_eq!(trunc(2.9), 2.0);
}

#[test]
fn test_trunc_negative() {
    assert_eq!(trunc(-2.1), -2.0);
    assert_eq!(trunc(-2.9), -2.0);
}

#[test]
fn test_trunc_special() {
    assert!(trunc(f64::INFINITY).is_infinite());
    assert!(trunc(f64::NAN).is_nan());
}

// =========================================================================
// fabs() Tests
// =========================================================================

#[test]
fn test_fabs_positive() {
    assert_eq!(fabs(3.14), 3.14);
}

#[test]
fn test_fabs_negative() {
    assert_eq!(fabs(-3.14), 3.14);
}

#[test]
fn test_fabs_zero() {
    assert_eq!(fabs(0.0), 0.0);
    assert_eq!(fabs(-0.0), 0.0);
}

#[test]
fn test_fabs_infinity() {
    assert_eq!(fabs(f64::INFINITY), f64::INFINITY);
    assert_eq!(fabs(f64::NEG_INFINITY), f64::INFINITY);
}

#[test]
fn test_fabs_nan() {
    assert!(fabs(f64::NAN).is_nan());
}

// =========================================================================
// copysign() Tests
// =========================================================================

#[test]
fn test_copysign_positive_to_positive() {
    assert_eq!(copysign(1.0, 1.0), 1.0);
}

#[test]
fn test_copysign_positive_to_negative() {
    assert_eq!(copysign(1.0, -1.0), -1.0);
}

#[test]
fn test_copysign_negative_to_positive() {
    assert_eq!(copysign(-1.0, 1.0), 1.0);
}

#[test]
fn test_copysign_with_zero() {
    assert_eq!(copysign(1.0, 0.0), 1.0);
    assert_eq!(copysign(1.0, -0.0), -1.0);
}

#[test]
fn test_copysign_infinity() {
    assert_eq!(copysign(f64::INFINITY, -1.0), f64::NEG_INFINITY);
}

#[test]
fn test_copysign_nan() {
    let result = copysign(f64::NAN, -1.0);
    assert!(result.is_nan());
    assert!(result.is_sign_negative());
}

// =========================================================================
// fmod() Tests
// =========================================================================

#[test]
fn test_fmod_basic() {
    assert!((fmod(5.0, 3.0).unwrap() - 2.0).abs() < 1e-15);
    assert!((fmod(7.0, 4.0).unwrap() - 3.0).abs() < 1e-15);
}

#[test]
fn test_fmod_negative() {
    assert!((fmod(-5.0, 3.0).unwrap() - (-2.0)).abs() < 1e-15);
    assert!((fmod(5.0, -3.0).unwrap() - 2.0).abs() < 1e-15);
}

#[test]
fn test_fmod_division_by_zero() {
    assert!(fmod(5.0, 0.0).is_err());
}

// =========================================================================
// modf() Tests
// =========================================================================

#[test]
fn test_modf_positive() {
    let (frac, int) = modf(3.5);
    assert!((frac - 0.5).abs() < 1e-15);
    assert!((int - 3.0).abs() < 1e-15);
}

#[test]
fn test_modf_negative() {
    let (frac, int) = modf(-3.5);
    assert!((frac - (-0.5)).abs() < 1e-15);
    assert!((int - (-3.0)).abs() < 1e-15);
}

#[test]
fn test_modf_integer() {
    let (frac, int) = modf(5.0);
    assert!((frac).abs() < 1e-15);
    assert!((int - 5.0).abs() < 1e-15);
}

// =========================================================================
// remainder() Tests
// =========================================================================

#[test]
fn test_remainder_basic() {
    // 5 / 3 = 1.67, rounds to 2, so 5 - 2*3 = -1
    assert!((remainder(5.0, 3.0).unwrap() - (-1.0)).abs() < 1e-15);
}

#[test]
fn test_remainder_division_by_zero() {
    assert!(remainder(5.0, 0.0).is_err());
}

#[test]
fn test_remainder_infinite_x() {
    assert!(remainder(f64::INFINITY, 1.0).is_err());
}

// =========================================================================
// frexp() Tests
// =========================================================================

#[test]
fn test_frexp_power_of_two() {
    let (m, e) = frexp(8.0);
    assert!((m - 0.5).abs() < 1e-15);
    assert_eq!(e, 4);
}

#[test]
fn test_frexp_non_power() {
    let (m, e) = frexp(3.0);
    assert!((m - 0.75).abs() < 1e-15);
    assert_eq!(e, 2);
}

#[test]
fn test_frexp_zero() {
    let (m, e) = frexp(0.0);
    assert_eq!(m, 0.0);
    assert_eq!(e, 0);
}

#[test]
fn test_frexp_negative() {
    let (m, e) = frexp(-4.0);
    assert!((m - (-0.5)).abs() < 1e-15);
    assert_eq!(e, 3);
}

// =========================================================================
// ldexp() Tests
// =========================================================================

#[test]
fn test_ldexp_basic() {
    assert!((ldexp(0.5, 4) - 8.0).abs() < 1e-15);
}

#[test]
fn test_ldexp_negative_exp() {
    assert!((ldexp(2.0, -1) - 1.0).abs() < 1e-15);
}

#[test]
fn test_ldexp_overflow() {
    assert!(ldexp(1.0, 1024).is_infinite());
}

// =========================================================================
// Round-trip Tests
// =========================================================================

#[test]
fn test_frexp_ldexp_roundtrip() {
    for x in [1.0, 2.0, 3.14159, 100.0, 0.001, -42.5] {
        let (m, e) = frexp(x);
        let result = ldexp(m, e);
        assert!(
            (result - x).abs() < 1e-15,
            "frexp/ldexp roundtrip failed for {}: got {}",
            x,
            result
        );
    }
}
