//! Basic mathematical functions.
//!
//! These compile directly to hardware instructions on x64.
//! All functions are `#[inline]` for maximum performance.

use super::super::ModuleError;

// =============================================================================
// Rounding Functions
// =============================================================================

/// Return the ceiling of x, the smallest integer greater than or equal to x.
///
/// # Python Semantics
/// - `math.ceil(2.1)` → 3
/// - `math.ceil(-2.1)` → -2
/// - `math.ceil(inf)` → inf
/// - `math.ceil(nan)` → nan
#[inline]
pub fn ceil(x: f64) -> f64 {
    x.ceil()
}

/// Return the floor of x, the largest integer less than or equal to x.
///
/// # Python Semantics
/// - `math.floor(2.9)` → 2
/// - `math.floor(-2.9)` → -3
/// - `math.floor(inf)` → inf
/// - `math.floor(nan)` → nan
#[inline]
pub fn floor(x: f64) -> f64 {
    x.floor()
}

/// Return x with the fractional part removed.
///
/// # Python Semantics
/// - `math.trunc(2.9)` → 2
/// - `math.trunc(-2.9)` → -2
/// - `math.trunc(inf)` → inf
/// - `math.trunc(nan)` → nan
#[inline]
pub fn trunc(x: f64) -> f64 {
    x.trunc()
}

// =============================================================================
// Absolute Value and Sign
// =============================================================================

/// Return the absolute value of x.
///
/// # Performance
/// Compiles to single x64 instruction (ANDPS to clear sign bit).
#[inline]
pub fn fabs(x: f64) -> f64 {
    x.abs()
}

/// Return x with the sign of y.
///
/// # Python Semantics
/// - `math.copysign(1.0, -0.0)` → -1.0
/// - `math.copysign(inf, -1.0)` → -inf
/// - `math.copysign(nan, -1.0)` → nan (with negative sign)
#[inline]
pub fn copysign(x: f64, y: f64) -> f64 {
    x.copysign(y)
}

// =============================================================================
// Modulo and Remainder
// =============================================================================

/// Return x modulo y.
///
/// # Python Semantics
/// - `math.fmod(5.0, 3.0)` → 2.0
/// - `math.fmod(-5.0, 3.0)` → -2.0
/// - Result has same sign as x (unlike Python's % operator)
///
/// # Errors
/// - ValueError if y is 0
#[inline]
pub fn fmod(x: f64, y: f64) -> Result<f64, ModuleError> {
    if y == 0.0 {
        return Err(ModuleError::MathDomainError(
            "fmod() division by zero".to_string(),
        ));
    }
    Ok(x % y)
}

/// Return the fractional and integer parts of x.
///
/// Both results carry the sign of x.
///
/// # Python Semantics
/// - `math.modf(3.5)` → (0.5, 3.0)
/// - `math.modf(-3.5)` → (-0.5, -3.0)
///
/// # Returns
/// (fractional_part, integer_part)
#[inline]
pub fn modf(x: f64) -> (f64, f64) {
    let int_part = x.trunc();
    let frac_part = x - int_part;
    (frac_part, int_part)
}

/// Return IEEE 754-style remainder of x with respect to y.
///
/// # Python Semantics
/// - `math.remainder(5.0, 3.0)` → -1.0 (rounds to nearest)
/// - Unlike fmod, chooses closest value to 0
#[inline]
pub fn remainder(x: f64, y: f64) -> Result<f64, ModuleError> {
    if y == 0.0 {
        return Err(ModuleError::MathDomainError(
            "remainder() division by zero".to_string(),
        ));
    }
    if x.is_infinite() {
        return Err(ModuleError::MathDomainError(
            "remainder() not defined for infinite x".to_string(),
        ));
    }
    if y.is_nan() || x.is_nan() {
        return Ok(f64::NAN);
    }

    // IEEE 754 remainder: x - n*y where n = round(x/y)
    let n = (x / y).round();
    Ok(x - n * y)
}

/// Return (m, e) such that x = m * 2**e exactly.
///
/// m is normalized so 0.5 <= abs(m) < 1.0 (except for 0, inf, nan).
///
/// # Python Semantics
/// - `math.frexp(8.0)` → (0.5, 4)  because 8 = 0.5 * 2^4
/// - `math.frexp(0.0)` → (0.0, 0)
#[inline]
pub fn frexp(x: f64) -> (f64, i32) {
    if x == 0.0 || x.is_nan() || x.is_infinite() {
        return (x, 0);
    }

    // Use bit manipulation for maximum performance
    let bits = x.to_bits();
    let sign = bits & 0x8000_0000_0000_0000;
    let exponent = ((bits >> 52) & 0x7FF) as i32;
    let mantissa = bits & 0x000F_FFFF_FFFF_FFFF;

    if exponent == 0 {
        // Subnormal number - normalize first
        let normalized = x * (1u64 << 52) as f64;
        let (m, e) = frexp(normalized);
        return (m, e - 52);
    }

    // Reconstruct with exponent = -1 (so value is in [0.5, 1.0))
    let new_bits = sign | (0x3FE << 52) | mantissa;
    let m = f64::from_bits(new_bits);
    let e = exponent - 0x3FE;

    (m, e)
}

/// Return m * 2**e (inverse of frexp).
///
/// # Python Semantics
/// - `math.ldexp(0.5, 4)` → 8.0
/// - `math.ldexp(1.0, 1024)` → inf (overflow)
#[inline]
pub fn ldexp(m: f64, e: i32) -> f64 {
    // Use built-in for correctness with edge cases
    libm::ldexp(m, e)
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests;
