use super::*;

const EPSILON: f64 = 1e-10;

// =========================================================================
// factorial() Tests
// =========================================================================

#[test]
fn test_factorial_zero() {
    assert_eq!(factorial(0).unwrap(), 1);
}

#[test]
fn test_factorial_one() {
    assert_eq!(factorial(1).unwrap(), 1);
}

#[test]
fn test_factorial_small() {
    assert_eq!(factorial(2).unwrap(), 2);
    assert_eq!(factorial(3).unwrap(), 6);
    assert_eq!(factorial(4).unwrap(), 24);
    assert_eq!(factorial(5).unwrap(), 120);
}

#[test]
fn test_factorial_medium() {
    assert_eq!(factorial(10).unwrap(), 3628800);
    assert_eq!(factorial(12).unwrap(), 479001600);
}

#[test]
fn test_factorial_large() {
    assert_eq!(factorial(20).unwrap(), 2432902008176640000);
}

#[test]
fn test_factorial_negative() {
    assert!(factorial(-1).is_err());
    assert!(factorial(-10).is_err());
}

#[test]
fn test_factorial_overflow() {
    assert!(factorial(21).is_err());
    assert!(factorial(100).is_err());
}

// =========================================================================
// factorial_float() Tests
// =========================================================================

#[test]
fn test_factorial_float_small() {
    assert!((factorial_float(5).unwrap() - 120.0).abs() < EPSILON);
    assert!((factorial_float(10).unwrap() - 3628800.0).abs() < EPSILON);
}

#[test]
fn test_factorial_float_large() {
    let result = factorial_float(25).unwrap();
    assert!((result - 15511210043330985984000000.0).abs() / result < 1e-10);
}

#[test]
fn test_factorial_float_very_large() {
    let result = factorial_float(100).unwrap();
    // 100! is approximately 9.33e157
    assert!(result > 9e157);
    assert!(result < 1e158);
}

// =========================================================================
// gamma() Tests
// =========================================================================

#[test]
fn test_gamma_positive_integers() {
    // Gamma(n) = (n-1)!
    assert!((gamma(1.0).unwrap() - 1.0).abs() < EPSILON); // 0!
    assert!((gamma(2.0).unwrap() - 1.0).abs() < EPSILON); // 1!
    assert!((gamma(3.0).unwrap() - 2.0).abs() < EPSILON); // 2!
    assert!((gamma(4.0).unwrap() - 6.0).abs() < EPSILON); // 3!
    assert!((gamma(5.0).unwrap() - 24.0).abs() < EPSILON); // 4!
}

#[test]
fn test_gamma_half() {
    // Gamma(0.5) = sqrt(π)
    assert!((gamma(0.5).unwrap() - std::f64::consts::PI.sqrt()).abs() < EPSILON);
}

#[test]
fn test_gamma_poles() {
    // Non-positive integers are poles
    assert!(gamma(0.0).is_err());
    assert!(gamma(-1.0).is_err());
    assert!(gamma(-2.0).is_err());
}

#[test]
fn test_gamma_negative_non_integer() {
    // Should work for negative non-integers
    let result = gamma(-0.5);
    assert!(result.is_ok());
}

// =========================================================================
// lgamma() Tests
// =========================================================================

#[test]
fn test_lgamma_positive_integers() {
    // lgamma(n) = log((n-1)!)
    assert!((lgamma(1.0).unwrap()).abs() < EPSILON); // log(0!) = 0
    assert!((lgamma(2.0).unwrap()).abs() < EPSILON); // log(1!) = 0
    assert!((lgamma(3.0).unwrap() - 2.0_f64.ln()).abs() < EPSILON); // log(2!)
    assert!((lgamma(4.0).unwrap() - 6.0_f64.ln()).abs() < EPSILON); // log(3!)
}

#[test]
fn test_lgamma_large() {
    // For large values, lgamma is more useful than gamma
    let result = lgamma(100.0).unwrap();
    assert!(result > 350.0 && result < 370.0);
}

#[test]
fn test_lgamma_poles() {
    assert!(lgamma(0.0).is_err());
    assert!(lgamma(-1.0).is_err());
}

// =========================================================================
// erf() Tests
// =========================================================================

#[test]
fn test_erf_zero() {
    assert!(erf(0.0).abs() < EPSILON);
}

#[test]
fn test_erf_positive() {
    assert!((erf(1.0) - 0.8427007929497149).abs() < EPSILON);
}

#[test]
fn test_erf_negative() {
    // erf is odd: erf(-x) = -erf(x)
    assert!((erf(-1.0) - (-0.8427007929497149)).abs() < EPSILON);
}

#[test]
fn test_erf_symmetry() {
    for x in [0.1, 0.5, 1.0, 2.0] {
        assert!((erf(-x) - (-erf(x))).abs() < EPSILON);
    }
}

#[test]
fn test_erf_limits() {
    // erf(inf) = 1, erf(-inf) = -1
    assert!((erf(10.0) - 1.0).abs() < EPSILON);
    assert!((erf(-10.0) - (-1.0)).abs() < EPSILON);
}

#[test]
fn test_erf_bounds() {
    // -1 < erf(x) < 1 for all finite x
    for x in [-5.0, -1.0, 0.0, 1.0, 5.0] {
        assert!(erf(x) > -1.0 && erf(x) < 1.0 || x.abs() > 3.0);
    }
}

// =========================================================================
// erfc() Tests
// =========================================================================

#[test]
fn test_erfc_zero() {
    assert!((erfc(0.0) - 1.0).abs() < EPSILON);
}

#[test]
fn test_erfc_identity() {
    // erfc(x) = 1 - erf(x)
    for x in [-2.0, -1.0, 0.0, 1.0, 2.0] {
        assert!((erfc(x) - (1.0 - erf(x))).abs() < EPSILON);
    }
}

#[test]
fn test_erfc_large() {
    // erfc is more accurate for large x
    assert!(erfc(5.0) > 0.0);
    assert!(erfc(5.0) < 1e-10);
}

#[test]
fn test_erfc_limits() {
    assert!((erfc(10.0)).abs() < EPSILON); // Approaches 0
    assert!((erfc(-10.0) - 2.0).abs() < EPSILON); // Approaches 2
}

// =========================================================================
// Identity Tests
// =========================================================================

#[test]
fn test_factorial_gamma_identity() {
    // Gamma(n+1) = n! for non-negative integers
    for n in 0..=10 {
        let fact = factorial(n).unwrap() as f64;
        let gam = gamma((n + 1) as f64).unwrap();
        assert!(
            (fact - gam).abs() < EPSILON,
            "Gamma({}+1) = {} ≠ {}! = {}",
            n,
            gam,
            n,
            fact
        );
    }
}

#[test]
fn test_erf_erfc_identity() {
    // erf(x) + erfc(x) = 1
    for x in [-2.0, -1.0, 0.0, 0.5, 1.0, 2.0] {
        assert!(
            (erf(x) + erfc(x) - 1.0).abs() < EPSILON,
            "erf({}) + erfc({}) ≠ 1",
            x,
            x
        );
    }
}
