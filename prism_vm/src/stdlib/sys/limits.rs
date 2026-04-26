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
mod tests;
