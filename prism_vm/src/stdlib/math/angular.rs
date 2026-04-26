//! Angular conversion functions.
//!
//! Simple multiply operations, inlined for zero overhead.

use super::constants::{DEGREES_PER_RADIAN, RADIANS_PER_DEGREE};

// =============================================================================
// Angular Conversion
// =============================================================================

/// Convert angle x from radians to degrees.
///
/// # Formula
/// degrees = radians * 180 / π
#[inline]
pub fn degrees(x: f64) -> f64 {
    x * DEGREES_PER_RADIAN
}

/// Convert angle x from degrees to radians.
///
/// # Formula
/// radians = degrees * π / 180
#[inline]
pub fn radians(x: f64) -> f64 {
    x * RADIANS_PER_DEGREE
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests;
