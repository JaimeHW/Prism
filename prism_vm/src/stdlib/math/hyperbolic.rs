//! Hyperbolic functions.
//!
//! All functions compile directly to hardware instructions or optimized libm.

use super::super::ModuleError;

// =============================================================================
// Hyperbolic Functions
// =============================================================================

/// Return the hyperbolic sine of x.
///
/// sinh(x) = (exp(x) - exp(-x)) / 2
#[inline]
pub fn sinh(x: f64) -> f64 {
    x.sinh()
}

/// Return the hyperbolic cosine of x.
///
/// cosh(x) = (exp(x) + exp(-x)) / 2
#[inline]
pub fn cosh(x: f64) -> f64 {
    x.cosh()
}

/// Return the hyperbolic tangent of x.
///
/// tanh(x) = sinh(x) / cosh(x)
#[inline]
pub fn tanh(x: f64) -> f64 {
    x.tanh()
}

// =============================================================================
// Inverse Hyperbolic Functions
// =============================================================================

/// Return the inverse hyperbolic sine of x.
///
/// asinh(x) = ln(x + sqrt(x² + 1))
#[inline]
pub fn asinh(x: f64) -> f64 {
    x.asinh()
}

/// Return the inverse hyperbolic cosine of x.
///
/// acosh(x) = ln(x + sqrt(x² - 1))
///
/// # Errors
/// - MathDomainError if x < 1
#[inline]
pub fn acosh(x: f64) -> Result<f64, ModuleError> {
    if x < 1.0 {
        return Err(ModuleError::MathDomainError(
            "math domain error".to_string(),
        ));
    }
    Ok(x.acosh())
}

/// Return the inverse hyperbolic tangent of x.
///
/// atanh(x) = 0.5 * ln((1 + x) / (1 - x))
///
/// # Errors
/// - MathDomainError if |x| >= 1
#[inline]
pub fn atanh(x: f64) -> Result<f64, ModuleError> {
    if x <= -1.0 || x >= 1.0 {
        return Err(ModuleError::MathDomainError(
            "math domain error".to_string(),
        ));
    }
    Ok(x.atanh())
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests;
