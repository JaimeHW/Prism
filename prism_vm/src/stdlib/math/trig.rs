//! Trigonometric functions.
//!
//! All functions compile to hardware sin/cos/tan instructions via libm.
//! Angles are in radians.

use super::super::ModuleError;

// =============================================================================
// Basic Trigonometric Functions
// =============================================================================

/// Return the sine of x (in radians).
///
/// # Performance
/// Compiles to FSIN instruction on x87 or calls optimized libm.
#[inline]
pub fn sin(x: f64) -> f64 {
    x.sin()
}

/// Return the cosine of x (in radians).
///
/// # Performance
/// Compiles to FCOS instruction on x87 or calls optimized libm.
#[inline]
pub fn cos(x: f64) -> f64 {
    x.cos()
}

/// Return the tangent of x (in radians).
///
/// # Performance
/// Compiles to FPTAN instruction on x87 or calls optimized libm.
#[inline]
pub fn tan(x: f64) -> f64 {
    x.tan()
}

// =============================================================================
// Inverse Trigonometric Functions
// =============================================================================

/// Return the arc sine of x, in radians.
///
/// Result is in [-π/2, π/2].
///
/// # Errors
/// - MathDomainError if x < -1 or x > 1
#[inline]
pub fn asin(x: f64) -> Result<f64, ModuleError> {
    if x < -1.0 || x > 1.0 {
        return Err(ModuleError::MathDomainError(
            "math domain error".to_string(),
        ));
    }
    Ok(x.asin())
}

/// Return the arc cosine of x, in radians.
///
/// Result is in [0, π].
///
/// # Errors
/// - MathDomainError if x < -1 or x > 1
#[inline]
pub fn acos(x: f64) -> Result<f64, ModuleError> {
    if x < -1.0 || x > 1.0 {
        return Err(ModuleError::MathDomainError(
            "math domain error".to_string(),
        ));
    }
    Ok(x.acos())
}

/// Return the arc tangent of x, in radians.
///
/// Result is in [-π/2, π/2].
#[inline]
pub fn atan(x: f64) -> f64 {
    x.atan()
}

/// Return atan(y/x), in radians.
///
/// Result is in [-π, π]. Unlike atan(y/x), atan2 gives correct
/// quadrant based on signs of both arguments.
///
/// # Python Semantics
/// - `math.atan2(1, 1)` → π/4
/// - `math.atan2(-1, -1)` → -3π/4
/// - `math.atan2(0, -1)` → π
/// - `math.atan2(0, 1)` → 0
#[inline]
pub fn atan2(y: f64, x: f64) -> f64 {
    y.atan2(x)
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests;
