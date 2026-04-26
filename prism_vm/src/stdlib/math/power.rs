//! Power and root functions.
//!
//! Includes optimized integer exponentiation and Newton-Raphson isqrt.

use super::super::ModuleError;

// =============================================================================
// Power Functions
// =============================================================================

/// Return x raised to the power y.
///
/// # Performance
/// Uses binary exponentiation for integer y, libm for fractional.
#[inline]
pub fn pow(x: f64, y: f64) -> Result<f64, ModuleError> {
    // Handle special cases
    if x == 0.0 && y < 0.0 {
        return Err(ModuleError::MathDomainError(
            "0.0 cannot be raised to a negative power".to_string(),
        ));
    }
    if x < 0.0 && y.fract() != 0.0 {
        return Err(ModuleError::MathDomainError(
            "negative number cannot be raised to a fractional power".to_string(),
        ));
    }

    // Integer exponent fast path
    if y.fract() == 0.0 && y.abs() <= i32::MAX as f64 {
        let exp = y as i32;
        return Ok(pow_int(x, exp));
    }

    Ok(x.powf(y))
}

/// Fast integer exponentiation using binary exponentiation.
///
/// O(log n) multiplications, no transcendental function calls.
#[inline]
fn pow_int(mut base: f64, mut exp: i32) -> f64 {
    if exp == 0 {
        return 1.0;
    }

    let negative = exp < 0;
    if negative {
        exp = -exp;
    }

    let mut result = 1.0;

    // Binary exponentiation: square and multiply
    while exp > 0 {
        if exp & 1 == 1 {
            result *= base;
        }
        base *= base;
        exp >>= 1;
    }

    if negative { 1.0 / result } else { result }
}

// =============================================================================
// Root Functions
// =============================================================================

/// Return the square root of x.
///
/// # Errors
/// - MathDomainError if x < 0
#[inline]
pub fn sqrt(x: f64) -> Result<f64, ModuleError> {
    if x < 0.0 {
        return Err(ModuleError::MathDomainError(
            "math domain error".to_string(),
        ));
    }
    Ok(x.sqrt())
}

/// Return the integer square root of n.
///
/// This is the floor of the exact square root.
///
/// # Algorithm
/// Uses Newton-Raphson with fast convergence (~5 iterations for 64-bit).
///
/// # Errors
/// - ValueError if n < 0
pub fn isqrt(n: i64) -> Result<i64, ModuleError> {
    if n < 0 {
        return Err(ModuleError::ValueError(
            "isqrt() argument must be nonnegative".to_string(),
        ));
    }

    if n == 0 {
        return Ok(0);
    }

    // Newton-Raphson: x_{n+1} = (x_n + n/x_n) / 2
    // Start with a good initial guess using bit length
    let mut x = 1i64 << ((64 - n.leading_zeros() + 1) / 2);

    loop {
        let x1 = (x + n / x) / 2;
        if x1 >= x {
            break;
        }
        x = x1;
    }

    Ok(x)
}

/// Return the Euclidean distance, sqrt(x² + y²).
///
/// Handles overflow by scaling.
#[inline]
pub fn hypot(x: f64, y: f64) -> f64 {
    x.hypot(y)
}

/// Return the cube root of x.
#[inline]
pub fn cbrt(x: f64) -> f64 {
    x.cbrt()
}
