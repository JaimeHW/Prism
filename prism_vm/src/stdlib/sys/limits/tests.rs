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
