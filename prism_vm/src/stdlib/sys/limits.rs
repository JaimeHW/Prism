//! System limits and configuration constants.
//!
//! Provides access to interpreter limits with proper
//! range validation.

use super::super::ModuleError;
use prism_core::value::SMALL_INT_MAX;

// =============================================================================
// Size Limits
// =============================================================================

/// Maximum size for containers (sys.maxsize).
/// This is the largest positive integer that can be stored inline in a Value.
pub const MAX_SIZE: i64 = SMALL_INT_MAX;

/// Maximum Unicode code point (sys.maxunicode).
pub const MAX_UNICODE: u32 = 0x10FFFF;

/// Default hash seed (0 means random).
pub const DEFAULT_HASH_SEED: u64 = 0;

// =============================================================================
// Recursion Limit
// =============================================================================

/// Default recursion limit (matching Python's default).
pub const DEFAULT_RECURSION_LIMIT: u32 = 1000;

/// Minimum allowed recursion limit.
pub const MIN_RECURSION_LIMIT: u32 = 10;

/// Maximum allowed recursion limit.
pub const MAX_RECURSION_LIMIT: u32 = 1_000_000;

/// Manages the recursion limit for the interpreter.
#[derive(Debug, Clone)]
pub struct RecursionLimit {
    limit: u32,
}

impl RecursionLimit {
    /// Create with default limit.
    #[inline]
    pub fn new() -> Self {
        Self {
            limit: DEFAULT_RECURSION_LIMIT,
        }
    }

    /// Create with custom limit.
    #[inline]
    pub fn with_limit(limit: u32) -> Result<Self, ModuleError> {
        if limit < MIN_RECURSION_LIMIT {
            return Err(ModuleError::ValueError(format!(
                "recursion limit must be at least {}",
                MIN_RECURSION_LIMIT
            )));
        }
        if limit > MAX_RECURSION_LIMIT {
            return Err(ModuleError::ValueError(format!(
                "recursion limit must be at most {}",
                MAX_RECURSION_LIMIT
            )));
        }
        Ok(Self { limit })
    }

    /// Get the current limit.
    #[inline]
    pub fn get(&self) -> u32 {
        self.limit
    }

    /// Set a new limit.
    #[inline]
    pub fn set(&mut self, limit: u32) -> Result<(), ModuleError> {
        if limit < MIN_RECURSION_LIMIT {
            return Err(ModuleError::ValueError(format!(
                "recursion limit must be at least {}",
                MIN_RECURSION_LIMIT
            )));
        }
        if limit > MAX_RECURSION_LIMIT {
            return Err(ModuleError::ValueError(format!(
                "recursion limit must be at most {}",
                MAX_RECURSION_LIMIT
            )));
        }
        self.limit = limit;
        Ok(())
    }

    /// Reset to default.
    #[inline]
    pub fn reset(&mut self) {
        self.limit = DEFAULT_RECURSION_LIMIT;
    }
}

impl Default for RecursionLimit {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// Switch Interval
// =============================================================================

/// Default switch interval in seconds for thread switching.
pub const DEFAULT_SWITCH_INTERVAL: f64 = 0.005;

/// Minimum switch interval.
pub const MIN_SWITCH_INTERVAL: f64 = 1e-9;

/// Manages the thread switch interval.
#[derive(Debug, Clone)]
pub struct SwitchInterval {
    interval: f64,
}

impl SwitchInterval {
    /// Create with default interval.
    #[inline]
    pub fn new() -> Self {
        Self {
            interval: DEFAULT_SWITCH_INTERVAL,
        }
    }

    /// Get the current interval.
    #[inline]
    pub fn get(&self) -> f64 {
        self.interval
    }

    /// Set a new interval.
    #[inline]
    pub fn set(&mut self, interval: f64) -> Result<(), ModuleError> {
        if interval < MIN_SWITCH_INTERVAL {
            return Err(ModuleError::ValueError(format!(
                "switch interval must be at least {} seconds",
                MIN_SWITCH_INTERVAL
            )));
        }
        if interval.is_nan() || interval.is_infinite() {
            return Err(ModuleError::ValueError(
                "switch interval must be a finite number".to_string(),
            ));
        }
        self.interval = interval;
        Ok(())
    }
}

impl Default for SwitchInterval {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // Size Limit Tests
    // =========================================================================

    #[test]
    fn test_max_size() {
        assert_eq!(MAX_SIZE, SMALL_INT_MAX);
    }

    #[test]
    fn test_max_unicode() {
        assert_eq!(MAX_UNICODE, 0x10FFFF);
    }

    #[test]
    fn test_max_unicode_valid_code_point() {
        // Should be a valid Unicode code point
        assert!(char::from_u32(MAX_UNICODE).is_some());
    }

    // =========================================================================
    // RecursionLimit Construction Tests
    // =========================================================================

    #[test]
    fn test_recursion_limit_new() {
        let limit = RecursionLimit::new();
        assert_eq!(limit.get(), DEFAULT_RECURSION_LIMIT);
    }

    #[test]
    fn test_recursion_limit_default() {
        let limit = RecursionLimit::default();
        assert_eq!(limit.get(), DEFAULT_RECURSION_LIMIT);
    }

    #[test]
    fn test_recursion_limit_with_limit() {
        let limit = RecursionLimit::with_limit(500).unwrap();
        assert_eq!(limit.get(), 500);
    }

    #[test]
    fn test_recursion_limit_with_limit_too_low() {
        let result = RecursionLimit::with_limit(5);
        assert!(result.is_err());
    }

    #[test]
    fn test_recursion_limit_with_limit_too_high() {
        let result = RecursionLimit::with_limit(MAX_RECURSION_LIMIT + 1);
        assert!(result.is_err());
    }

    #[test]
    fn test_recursion_limit_with_min() {
        let limit = RecursionLimit::with_limit(MIN_RECURSION_LIMIT).unwrap();
        assert_eq!(limit.get(), MIN_RECURSION_LIMIT);
    }

    #[test]
    fn test_recursion_limit_with_max() {
        let limit = RecursionLimit::with_limit(MAX_RECURSION_LIMIT).unwrap();
        assert_eq!(limit.get(), MAX_RECURSION_LIMIT);
    }

    // =========================================================================
    // RecursionLimit Set Tests
    // =========================================================================

    #[test]
    fn test_recursion_limit_set() {
        let mut limit = RecursionLimit::new();
        limit.set(2000).unwrap();
        assert_eq!(limit.get(), 2000);
    }

    #[test]
    fn test_recursion_limit_set_too_low() {
        let mut limit = RecursionLimit::new();
        let result = limit.set(1);
        assert!(result.is_err());
        // Should not have changed
        assert_eq!(limit.get(), DEFAULT_RECURSION_LIMIT);
    }

    #[test]
    fn test_recursion_limit_set_too_high() {
        let mut limit = RecursionLimit::new();
        let result = limit.set(MAX_RECURSION_LIMIT + 1);
        assert!(result.is_err());
    }

    #[test]
    fn test_recursion_limit_reset() {
        let mut limit = RecursionLimit::new();
        limit.set(5000).unwrap();
        limit.reset();
        assert_eq!(limit.get(), DEFAULT_RECURSION_LIMIT);
    }

    // =========================================================================
    // SwitchInterval Tests
    // =========================================================================

    #[test]
    fn test_switch_interval_new() {
        let interval = SwitchInterval::new();
        assert_eq!(interval.get(), DEFAULT_SWITCH_INTERVAL);
    }

    #[test]
    fn test_switch_interval_default() {
        let interval = SwitchInterval::default();
        assert_eq!(interval.get(), DEFAULT_SWITCH_INTERVAL);
    }

    #[test]
    fn test_switch_interval_set() {
        let mut interval = SwitchInterval::new();
        interval.set(0.01).unwrap();
        assert_eq!(interval.get(), 0.01);
    }

    #[test]
    fn test_switch_interval_set_min() {
        let mut interval = SwitchInterval::new();
        interval.set(MIN_SWITCH_INTERVAL).unwrap();
        assert_eq!(interval.get(), MIN_SWITCH_INTERVAL);
    }

    #[test]
    fn test_switch_interval_set_too_low() {
        let mut interval = SwitchInterval::new();
        let result = interval.set(0.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_switch_interval_set_negative() {
        let mut interval = SwitchInterval::new();
        let result = interval.set(-1.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_switch_interval_set_nan() {
        let mut interval = SwitchInterval::new();
        let result = interval.set(f64::NAN);
        assert!(result.is_err());
    }

    #[test]
    fn test_switch_interval_set_infinity() {
        let mut interval = SwitchInterval::new();
        let result = interval.set(f64::INFINITY);
        assert!(result.is_err());
    }

    // =========================================================================
    // Clone Tests
    // =========================================================================

    #[test]
    fn test_recursion_limit_clone() {
        let mut limit = RecursionLimit::new();
        limit.set(3000).unwrap();
        let cloned = limit.clone();
        assert_eq!(cloned.get(), 3000);
    }

    #[test]
    fn test_switch_interval_clone() {
        let mut interval = SwitchInterval::new();
        interval.set(0.1).unwrap();
        let cloned = interval.clone();
        assert_eq!(cloned.get(), 0.1);
    }
}
