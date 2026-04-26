//! Exponential and logarithmic functions.
//!
//! All functions compile directly to hardware instructions where available.
//! Handles special cases (inf, nan, domain errors) per Python 3.12.

use super::super::ModuleError;

// =============================================================================
// Exponential Functions
// =============================================================================

/// Return e raised to the power x.
///
/// # Performance
/// Compiles to optimized libm call.
#[inline]
pub fn exp(x: f64) -> f64 {
    x.exp()
}

/// Return 2 raised to the power x.
#[inline]
pub fn exp2(x: f64) -> f64 {
    x.exp2()
}

/// Return exp(x) - 1.
///
/// More accurate than `exp(x) - 1` for small x.
#[inline]
pub fn expm1(x: f64) -> f64 {
    x.exp_m1()
}

// =============================================================================
// Logarithmic Functions
// =============================================================================

/// Return the natural logarithm of x (base e).
///
/// # Errors
/// - MathDomainError if x <= 0
#[inline]
pub fn log(x: f64) -> Result<f64, ModuleError> {
    if x <= 0.0 {
        return Err(ModuleError::MathDomainError(
            "math domain error".to_string(),
        ));
    }
    Ok(x.ln())
}

/// Return the natural logarithm of x with optional base.
///
/// log(x, base) = ln(x) / ln(base)
#[inline]
pub fn log_base(x: f64, base: f64) -> Result<f64, ModuleError> {
    if x <= 0.0 || base <= 0.0 {
        return Err(ModuleError::MathDomainError(
            "math domain error".to_string(),
        ));
    }
    if base == 1.0 {
        return Err(ModuleError::MathDomainError(
            "math domain error".to_string(),
        ));
    }
    Ok(x.ln() / base.ln())
}

/// Return the base-2 logarithm of x.
///
/// # Errors
/// - MathDomainError if x <= 0
#[inline]
pub fn log2(x: f64) -> Result<f64, ModuleError> {
    if x <= 0.0 {
        return Err(ModuleError::MathDomainError(
            "math domain error".to_string(),
        ));
    }
    Ok(x.log2())
}

/// Return the base-10 logarithm of x.
///
/// # Errors
/// - MathDomainError if x <= 0
#[inline]
pub fn log10(x: f64) -> Result<f64, ModuleError> {
    if x <= 0.0 {
        return Err(ModuleError::MathDomainError(
            "math domain error".to_string(),
        ));
    }
    Ok(x.log10())
}

/// Return ln(1 + x).
///
/// More accurate than `log(1 + x)` for small x.
///
/// # Errors
/// - MathDomainError if x <= -1
#[inline]
pub fn log1p(x: f64) -> Result<f64, ModuleError> {
    if x <= -1.0 {
        return Err(ModuleError::MathDomainError(
            "math domain error".to_string(),
        ));
    }
    Ok(x.ln_1p())
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests;
