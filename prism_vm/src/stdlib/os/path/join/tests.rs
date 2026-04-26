use super::*;

#[test]
fn test_join_basic() {
    let p = join("a", "b");
    assert!(p.contains("a") && p.contains("b"));
}

#[test]
fn test_join_empty_base() {
    let p = join("", "b");
    assert_eq!(p, "b");
}

#[test]
fn test_join_empty_part() {
    let p = join("a", "");
    assert!(p.contains("a"));
}

#[test]
fn test_join_many_empty() {
    let parts: Vec<&str> = vec![];
    assert_eq!(join_many(&parts), "");
}

#[test]
fn test_join_many_single() {
    let parts = vec!["a"];
    assert_eq!(join_many(&parts), "a");
}

#[test]
fn test_join_many_multiple() {
    let parts = vec!["a", "b", "c"];
    let p = join_many(&parts);
    assert!(p.contains("a") && p.contains("b") && p.contains("c"));
}

#[test]
fn test_stack_path_new() {
    let sp = StackPath::new();
    assert!(sp.is_empty());
    assert_eq!(sp.len(), 0);
}

#[test]
fn test_stack_path_from_str() {
    let sp = StackPath::from_str("hello").unwrap();
    assert_eq!(sp.as_str(), "hello");
    assert_eq!(sp.len(), 5);
}

#[test]
fn test_stack_path_push() {
    let mut sp = StackPath::from_str("a").unwrap();
    assert!(sp.push("b"));
    let s = sp.as_str();
    assert!(s.contains("a") && s.contains("b"));
}

#[test]
fn test_stack_path_too_long() {
    let long_str = "x".repeat(MAX_STACK_PATH + 1);
    assert!(StackPath::from_str(&long_str).is_none());
}
