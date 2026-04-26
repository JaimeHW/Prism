//! Mathematical constants.
//!
//! All constants are defined as `const` for zero-cost inlining.
//! Values match Python 3.12 `math` module exactly.

/// π (pi) - Ratio of circle circumference to diameter.
pub const PI: f64 = std::f64::consts::PI;

/// e (Euler's number) - Base of natural logarithm.
pub const E: f64 = std::f64::consts::E;

/// τ (tau) - Circle constant, 2π.
pub const TAU: f64 = std::f64::consts::TAU;

/// Positive infinity.
pub const INFINITY: f64 = f64::INFINITY;

/// Negative infinity.
pub const NEG_INFINITY: f64 = f64::NEG_INFINITY;

/// Not a Number (NaN).
pub const NAN: f64 = f64::NAN;

// =============================================================================
// Additional Constants for Internal Use
// =============================================================================

/// 180/π for degrees conversion.
pub(crate) const DEGREES_PER_RADIAN: f64 = 180.0 / PI;

/// π/180 for radians conversion.
pub(crate) const RADIANS_PER_DEGREE: f64 = PI / 180.0;

/// ln(2) for log2 conversion.
pub(crate) const LN_2: f64 = std::f64::consts::LN_2;

/// ln(10) for log10 conversion.
pub(crate) const LN_10: f64 = std::f64::consts::LN_10;

/// sqrt(2)
pub(crate) const SQRT_2: f64 = std::f64::consts::SQRT_2;

/// 1/sqrt(2)
pub(crate) const FRAC_1_SQRT_2: f64 = std::f64::consts::FRAC_1_SQRT_2;

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests;
