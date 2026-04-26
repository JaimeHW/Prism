use super::*;

// =========================================================================
// PatternMatch Tests
// =========================================================================

#[test]
fn test_pattern_match_new() {
    let m = PatternMatch::new(NodeId::new(1), "test");
    assert_eq!(m.target(), NodeId::new(1));
    assert_eq!(m.replacement(), None);
    assert_eq!(m.pattern_name(), "test");
    assert!(!m.eliminated());
}

#[test]
fn test_pattern_match_replace() {
    let m = PatternMatch::replace(NodeId::new(1), NodeId::new(2), "test");
    assert_eq!(m.target(), NodeId::new(1));
    assert_eq!(m.replacement(), Some(NodeId::new(2)));
    assert!(m.eliminated());
}

#[test]
fn test_pattern_match_created_new() {
    let mut m = PatternMatch::new(NodeId::new(1), "test");
    assert!(!m.created_new());
    m.created_new = true;
    assert!(m.created_new());
}

// =========================================================================
// Pattern Tests
// =========================================================================

#[test]
fn test_pattern_new() {
    let p = Pattern::new(PatternCategory::Arithmetic);
    assert_eq!(p.category(), PatternCategory::Arithmetic);
    assert_eq!(p.name(), "unknown");
}

#[test]
fn test_pattern_named() {
    let p = Pattern::named(PatternCategory::Bitwise, "and_zero");
    assert_eq!(p.category(), PatternCategory::Bitwise);
    assert_eq!(p.name(), "and_zero");
}

// =========================================================================
// PatternRegistry Tests
// =========================================================================

#[test]
fn test_registry_new() {
    let reg = PatternRegistry::new();
    assert!(!reg.is_empty());
    assert_eq!(reg.len(), 5);
}

#[test]
fn test_registry_default() {
    let reg = PatternRegistry::default();
    assert!(!reg.is_empty());
}

#[test]
fn test_registry_iter() {
    let reg = PatternRegistry::new();
    let count = reg.iter().count();
    assert_eq!(count, 5);
}

#[test]
fn test_registry_categories() {
    let reg = PatternRegistry::new();
    let categories: Vec<_> = reg.iter().map(|p| p.category()).collect();

    assert!(categories.contains(&PatternCategory::Arithmetic));
    assert!(categories.contains(&PatternCategory::Bitwise));
    assert!(categories.contains(&PatternCategory::Comparison));
    assert!(categories.contains(&PatternCategory::Memory));
    assert!(categories.contains(&PatternCategory::Control));
}
