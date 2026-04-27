//! Special mathematical functions.
//!
//! Includes factorial with const lookup table, gamma, lgamma, erf, erfc.

use super::super::ModuleError;

// =============================================================================
// Factorial Lookup Table
// =============================================================================

/// Precomputed factorial values for n = 0..20.
/// These are the only values that fit in i64.
const FACTORIAL_TABLE: [i64; 21] = [
    1,                   // 0!
    1,                   // 1!
    2,                   // 2!
    6,                   // 3!
    24,                  // 4!
    120,                 // 5!
    720,                 // 6!
    5040,                // 7!
    40320,               // 8!
    362880,              // 9!
    3628800,             // 10!
    39916800,            // 11!
    479001600,           // 12!
    6227020800,          // 13!
    87178291200,         // 14!
    1307674368000,       // 15!
    20922789888000,      // 16!
    355687428096000,     // 17!
    6402373705728000,    // 18!
    121645100408832000,  // 19!
    2432902008176640000, // 20!
];

// =============================================================================
// Factorial
// =============================================================================

/// Return n factorial as an integer.
///
/// # Algorithm
/// - For n <= 20: const lookup table (O(1))
/// - For n > 20: overflow
///
/// # Errors
/// - ValueError if n < 0
/// - OverflowError if n > 20 (for i64 result)
pub fn factorial(n: i64) -> Result<i64, ModuleError> {
    if n < 0 {
        return Err(ModuleError::ValueError(
            "factorial() not defined for negative values".to_string(),
        ));
    }

    if n > 20 {
        return Err(ModuleError::MathRangeError(
            "factorial() result too large for i64".to_string(),
        ));
    }

    Ok(FACTORIAL_TABLE[n as usize])
}

/// Return n factorial as a float (for larger values).
///
/// Uses Stirling's approximation for n > 170.
pub fn factorial_float(n: i64) -> Result<f64, ModuleError> {
    if n < 0 {
        return Err(ModuleError::ValueError(
            "factorial() not defined for negative values".to_string(),
        ));
    }

    if n <= 20 {
        return Ok(FACTORIAL_TABLE[n as usize] as f64);
    }

    // For 21 <= n <= 170, compute via gamma
    if n <= 170 {
        return Ok(gamma_unchecked((n + 1) as f64));
    }

    // n > 170: Use Stirling's approximation
    // n! ≈ sqrt(2πn) * (n/e)^n
    let n = n as f64;
    let log_factorial =
        0.5 * (2.0 * std::f64::consts::PI * n).ln() + n * (n / std::f64::consts::E).ln();

    let result = log_factorial.exp();
    if result.is_infinite() {
        return Err(ModuleError::MathRangeError(
            "factorial() result too large".to_string(),
        ));
    }

    Ok(result)
}

// =============================================================================
// Gamma Function
// =============================================================================

/// Return the Gamma function at x.
///
/// Gamma(n) = (n-1)! for positive integers.
///
/// # Errors
/// - MathDomainError for non-positive integers
#[inline]
pub fn gamma(x: f64) -> Result<f64, ModuleError> {
    // Non-positive integers are poles
    if x <= 0.0 && x.fract() == 0.0 {
        return Err(ModuleError::MathDomainError(
            "math domain error".to_string(),
        ));
    }

    Ok(gamma_unchecked(x))
}

/// Unchecked gamma - assumes valid input.
#[inline]
fn gamma_unchecked(x: f64) -> f64 {
    libm::tgamma(x)
}

/// Return the natural log of the absolute value of Gamma(x).
///
/// # Errors
/// - MathDomainError for non-positive integers
#[inline]
pub fn lgamma(x: f64) -> Result<f64, ModuleError> {
    // Non-positive integers are poles
    if x <= 0.0 && x.fract() == 0.0 {
        return Err(ModuleError::MathDomainError(
            "math domain error".to_string(),
        ));
    }

    Ok(libm::lgamma_r(x).0)
}

// =============================================================================
// Error Function
// =============================================================================

/// Return the error function at x.
///
/// erf(x) = 2/sqrt(π) * integral(0 to x) of exp(-t²) dt
#[inline]
pub fn erf(x: f64) -> f64 {
    libm::erf(x)
}

/// Return the complementary error function at x.
///
/// erfc(x) = 1 - erf(x)
///
/// More accurate than `1 - erf(x)` for large x.
#[inline]
pub fn erfc(x: f64) -> f64 {
    libm::erfc(x)
}
