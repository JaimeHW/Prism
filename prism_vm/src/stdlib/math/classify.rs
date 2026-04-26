//! Classification functions.
//!
//! Uses branch-free bit pattern checks where possible.

// =============================================================================
// Classification Functions
// =============================================================================

/// Return True if x is positive or negative infinity.
///
/// # Performance
/// Uses `f64::is_infinite()` which compiles to bit pattern check.
#[inline]
pub fn isinf(x: f64) -> bool {
    x.is_infinite()
}

/// Return True if x is NaN (not a number).
///
/// # Performance
/// Uses `f64::is_nan()` which compiles to bit pattern check.
#[inline]
pub fn isnan(x: f64) -> bool {
    x.is_nan()
}

/// Return True if x is neither infinity nor NaN.
///
/// # Performance
/// Uses `f64::is_finite()` which compiles to bit pattern check.
#[inline]
pub fn isfinite(x: f64) -> bool {
    x.is_finite()
}

/// Return True if x is a normalized floating-point number.
///
/// A number is normal if it's not zero, subnormal, infinity, or NaN.
#[inline]
pub fn isnormal(x: f64) -> bool {
    x.is_normal()
}

/// Return True if x and y are close in value.
///
/// rel_tol is maximum relative difference, abs_tol is minimum absolute.
///
/// # Python Semantics
/// - `math.isclose(1e10, 1.00001e10)` → True with default rel_tol
/// - NaN is not close to anything, including NaN
#[inline]
pub fn isclose(a: f64, b: f64, rel_tol: f64, abs_tol: f64) -> bool {
    // Handle NaN
    if a.is_nan() || b.is_nan() {
        return false;
    }

    // Handle infinities
    if a == b {
        return true; // Handles inf == inf
    }

    if a.is_infinite() || b.is_infinite() {
        return false; // Different infinities or one finite
    }

    // Standard closeness check
    let diff = (a - b).abs();
    diff <= rel_tol * a.abs().max(b.abs()) || diff <= abs_tol
}

/// Return the next float after x towards y.
///
/// Returns y if x == y.
#[inline]
pub fn nextafter(x: f64, y: f64) -> f64 {
    if x == y {
        return y;
    }
    if x.is_nan() || y.is_nan() {
        return f64::NAN;
    }

    let bits = x.to_bits() as i64;

    let next_bits = if (y > x) == (bits >= 0) {
        bits + 1
    } else {
        bits - 1
    };

    f64::from_bits(next_bits as u64)
}

/// Return the difference between x and the next representable float.
///
/// ulp = Unit in the Last Place
#[inline]
pub fn ulp(x: f64) -> f64 {
    if x.is_nan() {
        return x;
    }
    if x.is_infinite() {
        return f64::INFINITY;
    }

    let x_abs = x.abs();
    let next = nextafter(x_abs, f64::INFINITY);
    next - x_abs
}
